"""Lodestar Analyst: a DeepSeek-powered agent that explains the forecast.

Design rule: the language model is the *analyst*, never the *forecaster*.
Probabilities come from the simulation; the model chooses which tools to call,
reads their JSON, and writes the explanation. A grounding check rejects any
percentage in the answer that doesn't appear in a tool result.

Protocol is plain JSON in the message content (works identically on the DeepSeek
API, Ollama-served R1 distills, and vLLM/llama.cpp servers):
    {"tool": "get_race", "args": {"query": "Maine Senate"}}
    {"answer": "Jackson is a modest favorite ..."}
"""
from __future__ import annotations

import json
import re

from .llm import LLM
from .tools import TOOLS, State, call_tool, resolve_race

MAX_STEPS = 5

SYSTEM = """You are Lodestar Analyst, the analyst for the Lodestar 2026 U.S. midterm forecast
(a model built on the 20 best-rated active pollsters, fundamentals and handicapper ratings,
simulated 40,000 times with correlated errors).

RULES
1. Never estimate a probability, margin or seat count yourself. Every number you state must
   appear in a TOOL_RESULT in this conversation. If no tool gives it, say so.
2. Reply with ONE JSON object and nothing else:
     {"tool": "<tool name>", "args": {...}}   to call a tool, or
     {"answer": "<reply to the user>"}        when you have what you need.
3. Answers lead with the bottom line, then 2-5 sentences of why, using plain language.
   Margins are Democratic-side minus Republican-side (positive = Democrat ahead).
   Mention the uncertainty (80% range) when giving a forecast. Use race ids like senate:ME.
4. Be strictly neutral: describe what the model says; never advocate for a party or candidate.
5. Probabilities are not predictions of certainty: a 70% favorite loses 3 times in 10.

TOOLS
""" + "\n".join(f"- {k}: {v[1]}" for k, v in TOOLS.items())

PCT_RE = re.compile(r"(\d{1,3}(?:\.\d)?)\s?%")
# a coverage level carried in a field name, e.g. margin_80pct_range or d_seats_80pct
KEY_PCT_RE = re.compile(r"(\d{1,3})\s?(?:pct|percent)", re.I)


def _first_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _numbers_in(obj, acc=None, key: str = "") -> set:
    """Every figure a tool result licenses the model to state as a percentage:
    the value itself, a 0-1 probability scaled to percent (and its complement),
    the complement of a field that is already a percent/probability, and a coverage
    level named in a field such as `margin_80pct_range` (which licenses "80% range")."""
    acc = set() if acc is None else acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            for m in KEY_PCT_RE.finditer(str(k)):
                acc.add(round(float(m.group(1)), 1))
            _numbers_in(v, acc, str(k))
    elif isinstance(obj, list):
        for v in obj:
            _numbers_in(v, acc, key)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        x = float(obj)
        k = key.lower()
        pctlike = any(t in k for t in ("pct", "p_", "share", "prob", "weight", "chance"))
        cands = [x]
        if 0 <= x <= 1:
            cands += [x * 100, 100 - x * 100]
        if pctlike and 0 <= x <= 100:
            cands.append(100 - x)
        for y in cands:
            acc.add(round(y, 1))
    return acc


def ungrounded_percentages(answer: str, tool_results: list) -> list[str]:
    allowed = _numbers_in(tool_results)
    allowed_int = {round(v) for v in allowed}
    bad = []
    for m in PCT_RE.finditer(answer):
        txt = m.group(1)
        v = float(txt)
        ok = (round(v, 1) in allowed) if "." in txt else (round(v) in allowed_int)
        if not ok:
            bad.append(m.group(0))
    return bad


class Analyst:
    def __init__(self, state: State, llm: LLM | None = None):
        self.state = state
        self.llm = llm or LLM()

    # ------------------------------------------------------------ main entry
    def ask(self, message: str, history: list[dict] | None = None) -> dict:
        if not self.llm.enabled:
            return self.offline(message)
        try:
            return self._agent_loop(message, history or [])
        except Exception as e:  # never leave the dashboard without an answer
            out = self.offline(message)
            out["note"] = f"DeepSeek unavailable ({type(e).__name__}); answered from model templates."
            return out

    def _agent_loop(self, message: str, history: list[dict]) -> dict:
        msgs = [{"role": "system", "content": SYSTEM}]
        for h in history[-6:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                msgs.append({"role": h["role"], "content": str(h["content"])[:2000]})
        msgs.append({"role": "user", "content": message})
        trace, results = [], []
        repaired = False
        for _ in range(MAX_STEPS + 1):
            r = self.llm.chat(msgs, json_mode=True)
            obj = _first_json(r["content"]) or {"answer": r["content"]}
            if "tool" in obj:
                name, args = str(obj["tool"]), obj.get("args") or {}
                res = call_tool(self.state, name, args)
                trace.append({"tool": name, "args": args, "result": res})
                results.append(res)
                msgs.append({"role": "assistant", "content": json.dumps(obj)})
                msgs.append({"role": "user", "content": f"TOOL_RESULT {name}: {json.dumps(res)[:6000]}"})
                continue
            answer = str(obj.get("answer") or r["content"]).strip()
            bad = ungrounded_percentages(answer, results)
            if bad and not repaired:
                repaired = True
                msgs.append({"role": "assistant", "content": json.dumps(obj)})
                msgs.append({"role": "user", "content":
                             f"These figures are not in any TOOL_RESULT: {', '.join(bad)}. Call a tool for them "
                             f"or rewrite the answer using only numbers from tool results."})
                continue
            return {"answer": answer, "trace": trace, "grounded": not bad, "unverified": bad,
                    "provider": self.llm.describe()}
        return {"answer": "I couldn't finish that analysis within the step limit.", "trace": trace,
                "grounded": False, "provider": self.llm.describe()}

    # ------------------------------------------------------------ no-LLM mode
    def offline(self, message: str) -> dict:
        low = message.lower()
        trace = []

        def run(name, **args):
            res = call_tool(self.state, name, args)
            trace.append({"tool": name, "args": args, "result": res})
            return res

        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:points?|pts?)", low)
        if m and any(k in low for k in ("what if", "miss", "off by", "underestimat", "overestimat", "shift")):
            s = abs(float(m.group(1)))
            pro_r = (any(k in low for k in ("toward republican", "towards republican", "toward the gop", "gop beats",
                                            "republicans beat", "republicans outperform", "r+"))
                     or ("underestimat" in low and ("republican" in low or "gop" in low))
                     or ("overestimat" in low and "democrat" in low))
            if pro_r:
                s = -s
            ch = "house" if "house" in low else "senate"
            w = run("what_if", shift=s, chamber=ch)
            flips = ", ".join(x["race_id"] for x in w["races_that_change_favorite"]) or "none of the races"
            ans = (f"If the polls miss by {abs(s):g} points toward {'Democrats' if s > 0 else 'Republicans'}, "
                   f"Democrats' chance of {ch} control moves from {w['p_dem_control_pct_now']}% to "
                   f"{w['p_dem_control_pct_after']}%. Races whose favorite would flip: {flips}.")
            return {"answer": ans, "trace": trace, "grounded": True, "provider": self.llm.describe(), "mode": "offline"}
        race = resolve_race(self.state, message)
        if race and not any(k in low for k in ("control", "majority", "chamber", "overall")):
            c = run("get_race", query=race["id"])
            fav = race["d_cand"] if race["p_d"] >= 0.5 else race["r_cand"]
            p = max(c["p_dem_side_win_pct"], c["p_rep_side_win_pct"])
            lo, hi = c["margin_80pct_range"]
            pa = c["polling_average"]
            poll_txt = (f" The adjusted polling average is {pa['adjusted_margin']:+.1f} across {pa['n_polls']} polls "
                        f"({pa['top20_polls']} from Top-20 pollsters).") if pa else " There is little public polling here."
            ans = (f"{fav} is favored in {race['id']} with a {p}% chance. Projected margin "
                   f"{c['projected_margin_dem_minus_rep']:+.1f} (80% range {lo:+.0f} to {hi:+.0f}).{poll_txt}"
                   f" Handicappers: {c['handicappers_consensus'] or 'n/a'}.")
            if c.get("live") and (c["live"]["pct_reporting"] or c["live"]["status"] == "called"):
                lv = c["live"]
                ans += (f" Live: {lv['pct_reporting']}% reporting, counted margin {lv['counted_margin']}, "
                        f"live chance for the Democratic side {lv['live_p_dem_side_pct']}%.")
            return {"answer": ans, "trace": trace, "grounded": True, "provider": self.llm.describe(), "mode": "offline"}
        ch = "house" if "house" in low else "governor" if "governor" in low else "senate"
        s = run("chamber_summary", chamber=ch)
        lr = run("list_races", chamber="senate" if ch == "senate" else ch, limit=5)
        close = ", ".join(f"{x['race_id']} ({x['p_dem_side_pct']}% D)" for x in lr["races"])
        if ch == "governor":
            ans = (f"Democrats have a {s['p_dem_majority_of_governorships_pct']}% chance of holding a majority of "
                   f"governorships (median {s['dem_governors_median']:.0f}). Closest: {close}.")
        else:
            ans = (f"Democrats have a {s['p_dem_control_pct']}% chance of controlling the {ch}; median "
                   f"{s['dem_seats_median']:.0f} seats (80% range {s['dem_seats_80pct'][0]:.0f}-"
                   f"{s['dem_seats_80pct'][1]:.0f}). Closest races: {close}.")
        return {"answer": ans, "trace": trace, "grounded": True, "provider": self.llm.describe(), "mode": "offline"}

    # ------------------------------------------------------------ results parsing
    def parse_results(self, text: str) -> dict:
        """Turn pasted results text (a TV chyron, a tweet, an SOS page) into result rows
        for human review. Nothing is applied until the user confirms in the dashboard."""
        rows = []
        if self.llm.enabled:
            prompt = ("Extract election results from the text. Return JSON {\"rows\": [{\"race\": \"<state + office or "
                      "district, e.g. 'Maine Senate' or 'PA-07'>\", \"dem_name\": str|null, \"dem_pct\": number|null, "
                      "\"rep_name\": str|null, \"rep_pct\": number|null, \"dem_votes\": int|null, \"rep_votes\": int|null, "
                      "\"pct_reporting\": number|null, \"called_for\": \"D\"|\"R\"|null}]}. Use only numbers in the text.")
            try:
                r = self.llm.chat([{"role": "system", "content": prompt}, {"role": "user", "content": text[:6000]}],
                                  json_mode=True)
                obj = _first_json(r["content"]) or {}
                rows = obj.get("rows") or []
            except Exception:
                rows = []
        if not rows:   # regex fallback: "Maine Senate: Jackson 51.2, Collins 47.9, 82% reporting"
            for line in text.splitlines():
                label, _, body = line.partition(":")
                nums = re.findall(r"(\d+(?:\.\d+)?)", body if body else line)
                if len(nums) >= 2:
                    rows.append({"race": label, "dem_pct": float(nums[0]), "rep_pct": float(nums[1]),
                                 "pct_reporting": float(nums[2]) if len(nums) > 2 else None, "called_for": None})
        out = []
        for row in rows:
            race = resolve_race(self.state, str(row.get("race", "")))
            if not race:
                out.append({"input": row, "error": "could not match a race"})
                continue
            dv = row.get("dem_votes") or (row.get("dem_pct") or 0) * 1000
            rv = row.get("rep_votes") or (row.get("rep_pct") or 0) * 1000
            out.append({"race_id": race["id"], "d_votes": int(dv), "r_votes": int(rv), "other_votes": 0,
                        "pct_reporting": float(row.get("pct_reporting") or 0),
                        "status": "called" if row.get("called_for") in ("D", "R") else "counting",
                        "winner": row.get("called_for"), "source": "pasted"})
        return {"rows": out, "provider": self.llm.describe()}
