"""Deterministic tools the analyst agent may call. The LLM never computes a
probability itself: every number it reports comes out of one of these functions,
which read the model's own outputs (forecast.json / the live state)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from lodestar.config import PARAMS, STATE_NAMES

S = PARAMS.sim
NON_NATIONAL_SD = math.sqrt(S.sd_region ** 2 + S.sd_state ** 2 + 1.2 ** 2)
ABBR = {v.lower(): k for k, v in STATE_NAMES.items()}


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _pct(p):
    return None if p is None else round(100 * float(p), 1)


@dataclass
class State:
    forecast: dict
    live: dict | None = None
    results: dict = field(default_factory=dict)

    @property
    def races(self):
        return self.forecast["races"]

    def race(self, rid):
        return next((r for r in self.races if r["id"] == rid), None)


def resolve_race(state: State, query: str) -> dict | None:
    q = (query or "").strip()
    if not q:
        return None
    if state.race(q):
        return state.race(q)
    low = q.lower()
    m = re.search(r"\b([a-z]{2})[- ]?(\d{1,2}|al)\b", low)
    if m and m.group(1).upper() in STATE_NAMES and ("house" in low or "district" in low or "-" in q or m.group(2)):
        dist = m.group(2).upper()
        rid = f"house:{m.group(1).upper()}-{'AL' if dist == 'AL' else dist.zfill(2)}"
        if state.race(rid):
            return state.race(rid)
    chamber = "governor" if re.search(r"gov", low) else "house" if re.search(r"house|district", low) else \
        "senate" if "senat" in low else None
    st = None
    for name, ab in sorted(ABBR.items(), key=lambda x: -len(x[0])):
        if name in low:
            st = ab
            break
    if st is None:   # bare postal codes only when typed in capitals ("ME", "TX") — "me", "or", "in" are words
        for t in re.findall(r"\b[A-Z]{2}\b", q):
            if t in STATE_NAMES:
                st = t
                break
    best, score = None, 0
    for r in state.races:
        s = 0
        if st and r["state"] == st:
            s += 3
        if chamber and r["chamber"] == chamber:
            s += 2
        elif chamber is None and r["chamber"] == "senate":
            s += 1
        for c in (r.get("d_cand"), r.get("r_cand"), r.get("incumbent")):
            if c and c.split()[-1].lower() in low and len(c.split()[-1]) > 2:
                s += 5
        if "special" in low and r.get("special"):
            s += 1
        if s > score:
            best, score = r, s
    return best if score >= 3 else None


def race_card(r: dict, live: dict | None = None) -> dict:
    card = {
        "race_id": r["id"], "office": r["chamber"], "state": r["state_name"], "label": r["label"],
        "democratic_side": f"{r['d_cand']} ({r['d_side']})", "republican_side": f"{r['r_cand']} ({r['r_side']})",
        "incumbent": r.get("incumbent"), "seat_held_by": r.get("holder"),
        "p_dem_side_win_pct": _pct(r["p_d"]), "p_rep_side_win_pct": _pct(r["p_r"]),
        "projected_margin_dem_minus_rep": r["margin"]["mean"],
        "margin_80pct_range": [r["margin"]["p10"], r["margin"]["p90"]],
        "poll_weight_in_estimate_pct": _pct(r.get("poll_weight")),
        "polling_average": None if not r.get("polls") else {
            "adjusted_margin": r["polls"]["avg"], "n_polls": r["polls"]["n_polls"],
            "top20_only_margin": r["polls"]["core_avg"], "top20_polls": r["polls"]["core_n"],
            "undecided_pct": r["polls"]["undecided"], "latest_poll": r["polls"]["latest"]},
        "fundamentals_margin": r["fundamentals"]["mu"],
        "handicappers_consensus": r["ratings"]["consensus"] if r.get("ratings") else None,
        "p_seat_flips_pct": _pct(r.get("p_flip")),
    }
    nc = (live or {}).get("races", {}).get(r["id"]) if live else None
    if nc and (nc.get("pct_reporting") or nc.get("status") == "called"):
        card["live"] = {"status": nc["status"], "pct_reporting": nc["pct_reporting"],
                        "counted_margin": nc.get("counted_margin"), "projected_final_margin": nc.get("est_margin"),
                        "live_p_dem_side_pct": _pct(nc.get("p_d")), "winner": nc.get("winner")}
    return card


# ----------------------------------------------------------------- the tools
def t_chamber_summary(state: State, chamber: str = "senate") -> dict:
    ch = state.forecast["chambers"].get(chamber)
    if ch is None:
        return {"error": f"unknown chamber {chamber}; use senate, house or governor"}
    if chamber == "governor":
        out = {"p_dem_majority_of_governorships_pct": _pct(ch["p_d_majority"]),
               "dem_governors_median": ch["d_seats"]["median"],
               "dem_governors_80pct": [ch["d_seats"]["p10"], ch["d_seats"]["p90"]]}
    else:
        out = {"p_dem_control_pct": _pct(ch["p_d_control"]), "p_rep_control_pct": _pct(ch["p_r_control"]),
               "dem_seats_median": ch["d_seats"]["median"], "dem_seats_80pct": [ch["d_seats"]["p10"], ch["d_seats"]["p90"]]}
        if chamber == "senate":
            out["p_independents_hold_balance_pct"] = _pct(ch["p_independents_decide"])
            out["tipping_point_races"] = ch["tipping_point"][:5]
            out["rule"] = "Democrats need 51 seats (VP Vance breaks ties); 34 D/I seats are not up."
        else:
            out["rule"] = "218 seats for a majority."
    if state.live:
        lc = state.live["chambers"].get(chamber, {})
        out["live_now"] = {k: (_pct(v) if k.startswith("p_") else v) for k, v in lc.items()}
    return out


def t_list_races(state: State, chamber: str = "senate", competitive_only: bool = True, limit: int = 12) -> dict:
    rs = [r for r in state.races if r["chamber"] == chamber and not r.get("fixed")]
    if competitive_only:
        rs = [r for r in rs if 0.05 <= r["p_d"] <= 0.95]
    rs.sort(key=lambda r: abs(r["p_d"] - 0.5))
    return {"races": [{"race_id": r["id"], "matchup": f"{r['d_cand']} vs {r['r_cand']}",
                       "p_dem_side_pct": _pct(r["p_d"]), "margin": r["margin"]["mean"],
                       "holder": r.get("holder")} for r in rs[: int(limit)]]}


def t_get_race(state: State, query: str) -> dict:
    r = resolve_race(state, query)
    if not r:
        return {"error": f"no race matches '{query}'. Try e.g. 'senate:ME', 'Texas Senate', 'PA-07'."}
    return race_card(r, state.live)


def t_get_polls(state: State, query: str, limit: int = 8) -> dict:
    r = resolve_race(state, query)
    if not r:
        return {"error": f"no race matches '{query}'"}
    polls = r.get("poll_list") or []
    return {"race_id": r["id"], "polls": [{
        "pollster": p["pollster"], "top20_rank": p["core_rank"], "grade": p["grade"], "dates": f"{p['start']}..{p['end']}",
        "n": p["n"], "pop": p["pop"], "dem": p["d"], "rep": p["r"], "margin": p["margin"],
        "adjusted_margin": p["adj_margin"], "weight_share_pct": _pct(p["weight_share"]),
        "partisan_sponsor": p["partisan"]} for p in polls[: int(limit)]]}


def t_pollster(state: State, name: str) -> dict:
    low = (name or "").lower()
    hits = [p for p in state.forecast["pollsters"] if low and low in p["pollster"].lower()]
    if not hits:
        return {"error": f"no pollster matches '{name}'"}
    p = hits[0]
    return {"pollster": p["pollster"], "top20_rank": p["core_rank"], "grade": p["grade"],
            "predictive_plus_minus": p["ppm"], "rating_source": p["rating_source"], "polls_2026": p["n_polls"],
            "house_effect_dem_minus_rep": p["house_effect"], "weight_multiplier": p["weight_multiplier"]}


def t_what_if(state: State, shift: float, chamber: str = "senate", top: int = 8) -> dict:
    """Uniform polling miss: shift>0 means Democrats beat their polls by `shift` points of margin."""
    shift = float(max(-8, min(8, shift)))
    ch = state.forecast["chambers"].get(chamber)
    if not ch or "sensitivity" not in ch:
        return {"error": "what_if supports senate and house"}
    sens = ch["sensitivity"]
    lo = max([s for s in sens if s["shift"] <= shift], key=lambda s: s["shift"])
    hi = min([s for s in sens if s["shift"] >= shift], key=lambda s: s["shift"])
    t = 0 if hi["shift"] == lo["shift"] else (shift - lo["shift"]) / (hi["shift"] - lo["shift"])
    pd_ = lo["p_d"] + t * (hi["p_d"] - lo["p_d"])
    moved = []
    el = PARAMS.fundamentals.elasticity[chamber]
    for r in state.races:
        if r["chamber"] != chamber or r.get("fixed"):
            continue
        p1 = _phi((r["margin"]["mean"] + el * shift) / math.hypot(r["posterior_sd"], NON_NATIONAL_SD))
        if (p1 >= 0.5) != (r["p_d"] >= 0.5):
            moved.append({"race_id": r["id"], "p_dem_before_pct": _pct(r["p_d"]), "p_dem_after_pct": _pct(p1)})
    return {"shift_points_toward_dem": shift, "chamber": chamber,
            "p_dem_control_pct_after": _pct(pd_), "p_dem_control_pct_now": _pct(ch["p_d_control"]),
            "races_that_change_favorite": moved[: int(top)],
            "note": "Chamber odds come from the full simulation re-scored with the shift; race odds are approximations."}


def t_live_status(state: State, query: str | None = None) -> dict:
    if not state.live:
        return {"status": "no live results loaded yet"}
    if query:
        r = resolve_race(state, query)
        if not r:
            return {"error": f"no race matches '{query}'"}
        return race_card(r, state.live)
    L = state.live
    return {"chambers": {k: {kk: (_pct(vv) if kk.startswith("p_") else vv) for kk, vv in v.items()}
                         for k, v in L["chambers"].items()},
            "shared_polling_error_learned": L["shared_error"], "scorecard_vs_forecast": L["scorecard"]}


TOOLS = {
    "chamber_summary": (t_chamber_summary, "Control odds and seat ranges. args: chamber in {senate, house, governor}"),
    "list_races": (t_list_races, "Closest races. args: chamber, competitive_only (bool), limit (int)"),
    "get_race": (t_get_race, "One race's forecast card. args: query (race id like 'senate:ME', or text like 'Texas Senate', 'PA-07', a candidate surname)"),
    "get_polls": (t_get_polls, "Polls used in a race with weights. args: query, limit"),
    "pollster": (t_pollster, "A pollster's rating, Top-20 rank and house effect. args: name"),
    "what_if": (t_what_if, "Scenario: polls miss uniformly by `shift` points (positive = Democrats beat polls). args: shift (float, -8..8), chamber (senate|house)"),
    "live_status": (t_live_status, "Election-night state, overall or one race. args: query (optional)"),
}


def call_tool(state: State, name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"unknown tool {name}; available: {', '.join(TOOLS)}"}
    fn = TOOLS[name][0]
    try:
        return fn(state, **(args or {}))
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
