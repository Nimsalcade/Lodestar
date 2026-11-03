"""Smoke + invariant tests. Run: python -m pytest -q  (or python tests/test_model.py)"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agent import Analyst, LLM, LLMConfig, State  # noqa: E402
from agent.agent import _first_json, ungrounded_percentages  # noqa: E402
from lodestar.data import _collapse_variants, canonical_pollster  # noqa: E402
from lodestar.fundamentals import blend, combine_prior  # noqa: E402
from lodestar.live import LiveParams, nowcast_race, run_live  # noqa: E402
from lodestar.simulate import chamber_outcomes, simulate_margins  # noqa: E402
from lodestar.sources.civicapi import demo, map_civic  # noqa: E402

FORECAST = json.loads((ROOT / "output" / "forecast.json").read_text())


def test_canonical_names():
    assert canonical_pollster("New York Times/Siena University") == "The New York Times/Siena College"
    assert canonical_pollster("ReconMR/Siena University") == "Siena College"
    assert canonical_pollster("Texas Public Opinion Research") == "Texas Public Opinion Research"
    assert canonical_pollster("Hart Research Associates (D)/ Public Opinion Strategies (R)") == \
        "Hart Research Associates/Public Opinion Strategies"
    assert canonical_pollster("KSTP/SurveyUSA") == "SurveyUSA"


def test_variant_collapse_prefers_lv_and_final_round():
    df = pd.DataFrame([
        dict(race="x", pollster_canon="A", pollster="A", start="2026-10-01", end="2026-10-02", population="LV", d=42, r=25,
             other=None, undecided=17, n=600, partisan=None, d_label="", r_label="", url=""),
        dict(race="x", pollster_canon="A", pollster="A", start="2026-10-01", end="2026-10-02", population="LV", d=52, r=48,
             other=None, undecided=None, n=600, partisan=None, d_label="", r_label="", url=""),
        dict(race="x", pollster_canon="A", pollster="A", start="2026-10-01", end="2026-10-02", population="RV", d=60, r=40,
             other=None, undecided=None, n=900, partisan=None, d_label="", r_label="", url=""),
    ])
    out = _collapse_variants(df)
    assert len(out) == 1 and out.iloc[0].d == 52 and out.iloc[0].r == 48


def test_blend_moves_toward_more_precise_source():
    mu, sd = combine_prior({"mu": 10.0, "sd": 7.0}, {"mu": 0.0, "sd": 7.0})
    assert abs(mu - 5.0) < 1e-9
    post = blend(mu, sd, {"avg": 2.0, "sd": 3.0, "undecided": 0.0}, "senate")
    assert 2.0 < post["mu"] < mu and post["poll_weight"] > 0.5


def test_forecast_invariants():
    f = FORECAST
    assert len(f["races"]) == 506
    for r in f["races"]:
        assert 0.0 <= r["p_d"] <= 1.0 and abs(r["p_d"] + r["p_r"] - 1) < 1e-6
        assert r["margin"]["p10"] <= r["margin"]["p50"] <= r["margin"]["p90"]
        if r["fixed"]:
            assert r["p_d"] in (0.0, 1.0)
    s = f["chambers"]["senate"]
    assert abs(s["p_d_control"] + s["p_r_control"] + s["p_independents_decide"] - 1) < 1e-6
    assert abs(sum(s["hist_d"]["p"]) - 1) < 1e-3
    # sensitivity must be monotone: Democrats beating their polls can't lower their odds
    for ch in ("senate", "house"):
        ps = [x["p_d"] for x in f["chambers"][ch]["sensitivity"]]
        assert all(b >= a - 1e-9 for a, b in zip(ps, ps[1:]))
    core = [p for p in f["pollsters"] if p["is_core"]]
    assert len(core) == 20 and sorted(p["core_rank"] for p in core) == list(range(1, 21))


def test_simulation_probability_matches_normal_approx():
    """With only idiosyncratic error the simulated win prob ~= normal CDF (t-tails aside)."""
    races = pd.DataFrame([{"state": "OH", "chamber": "senate", "mu": 2.0, "sd": 4.0, "elasticity": 1.0,
                           "d_side": "D", "r_side": "R", "fixed": ""}])
    m, e = simulate_margins(races, n=60000, seed=1)
    p = (m[:, 0] > 0).mean()
    tot = math.sqrt(4 ** 2 + 3.6 ** 2 + 2 ** 2 + 1.5 ** 2 + 1.0 ** 2 + (0.04 * 4) ** 2 + (0.12 * 3) ** 2)
    approx = 0.5 * (1 + math.erf(2.0 / tot / math.sqrt(2)))
    assert abs(p - approx) < 0.02


def test_live_reproduces_pre_election_at_zero_reporting():
    L = run_live(FORECAST, {"senate:ME": {"d_votes": 0, "r_votes": 0, "pct_reporting": 0, "status": "awaiting"}},
                 LiveParams(sims=20000))
    assert abs(L["chambers"]["senate"]["p_d_control"] - FORECAST["chambers"]["senate"]["p_d_control"]) < 0.03


def test_nowcast_converges_to_count():
    r = next(x for x in FORECAST["races"] if x["id"] == "senate:ME")
    nc = nowcast_race(r, {"d_votes": 480000, "r_votes": 520000, "pct_reporting": 99.5, "status": "counting"},
                      LiveParams())
    assert nc["est_margin"] < -3.5 and nc["p_d"] < 0.05


def test_live_learns_national_error_from_demo_night():
    L = run_live(FORECAST, demo(FORECAST, 1.0, seed=3))
    assert L["shared_error"]["n_obs"] > 100 and L["scorecard"]["called"] > 300


def test_civic_mapping():
    civic = [{"type": "US Senate", "province": "OH", "percent_reporting": 50,
              "candidates": [{"name": "Sherrod Brown", "votes": 1000, "winner": False},
                             {"name": "Jon Husted", "votes": 900}, {"name": "Bill Redpath", "votes": 20}]}]
    out = map_civic(civic, FORECAST)
    assert out["senate:OH-special"]["d_votes"] == 1000 and out["senate:OH-special"]["r_votes"] == 900


class FakeLLM(LLM):
    """Scripted DeepSeek stand-in: calls a tool, then answers (once with a made-up %)."""

    def __init__(self, script):
        super().__init__(LLMConfig(provider="deepseek", api_key="test", model="deepseek-v4-flash"))
        self.script, self.calls = list(script), []

    def chat(self, messages, json_mode=False, max_tokens=None):
        self.calls.append(messages[-1]["content"])
        return {"content": self.script.pop(0), "reasoning": None, "latency_s": 0.0}


def test_agent_tool_loop_and_grounding_repair():
    me = next(x for x in FORECAST["races"] if x["id"] == "senate:ME")
    good = round(100 * me["p_d"], 1)
    llm = FakeLLM(['{"tool": "get_race", "args": {"query": "Maine Senate"}}',
                   '{"answer": "Jackson has a 91.3% chance."}',          # fabricated -> repair round
                   json.dumps({"answer": f"Jackson has a {good}% chance in senate:ME."})])
    out = Analyst(State(FORECAST), llm).ask("Who wins Maine?")
    assert out["grounded"] and str(good) in out["answer"] and out["trace"][0]["tool"] == "get_race"
    assert any("not in any TOOL_RESULT" in c for c in llm.calls)


def test_json_extraction_and_grounding():
    assert _first_json('<think>x</think> sure: {"tool":"x","args":{}} trailing') == {"tool": "x", "args": {}}
    assert ungrounded_percentages("It is 75.0% likely", [{"p": 0.75}]) == []
    assert ungrounded_percentages("It is 12% likely", [{"p": 0.75}]) == ["12%"]


def test_offline_analyst_resolves_races():
    a = Analyst(State(FORECAST), LLM(LLMConfig(provider="none")))
    assert "senate:TX" in a.ask("TX Senate")["answer"]
    assert "house:PA-07" in a.ask("Tell me about PA-07")["answer"]
    w = a.ask("What if polls overestimate Democrats by 3 points?")
    assert "toward Republicans" in w["answer"]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # pragma: no cover
                fails += 1
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
