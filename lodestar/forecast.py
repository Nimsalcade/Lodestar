"""Run the full Lodestar pipeline and write output/forecast.json."""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .averaging import (fit_house_effects, generic_current, generic_series, race_averages,
                        simple_average)
from .config import (CHAMBERS, FORECAST_AS_OF, MODEL_NAME, MODEL_VERSION, PARAMS, STATE_NAMES,
                     region_of)
from .fundamentals import blend, combine_prior, fundamentals_prior, ratings_prior, scale_label
from .pollsters import RatingBook, build_pollster_table
from .simulate import chamber_outcomes, simulate_margins

OUT = D.ROOT / "output"
S = PARAMS.sim
SHARED_SD = math.sqrt(S.sd_national ** 2 + S.sd_region ** 2 + S.sd_state ** 2 + 1.2 ** 2 + 0.8 ** 2)


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _r(x, k=2):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), k)


def build(as_of: str = FORECAST_AS_OF, n_sims: int | None = None, verbose: bool = True) -> dict:
    t0 = time.time()
    now = pd.Timestamp(as_of)
    rp, gp = D.load_polls()
    rp, gp = rp[rp["end"] <= now], gp[gp["end"] <= now]
    races = D.load_races()
    ratings = D.load_ratings()
    fin = D.load_finance()
    book = RatingBook(D.load_pollster_ratings())
    ptable = build_pollster_table(rp, gp, book)
    house, _ = fit_house_effects(rp, gp, ptable, now)
    ptable["house_effect"] = ptable["pollster"].map(lambda p: house.get(p, (0.0, 0))[0])
    ptable["house_effect_n"] = ptable["pollster"].map(lambda p: house.get(p, (0.0, 0))[1])

    gs = generic_series(gp, ptable, house, now)
    gcur = generic_current(gp, ptable, house, now)
    env = gcur["margin"]
    avgs, frame = race_averages(rp, ptable, house, gs, env, now)
    avgs_core, _ = race_averages(rp, ptable, house, gs, env, now, core_only=True)
    simple = simple_average(rp, now)
    if verbose:
        print(f"[lodestar] polls: {len(rp)} race + {len(gp)} generic | generic ballot D{env:+.2f} "
              f"| house effects for {len(house)} firms | {time.time()-t0:.1f}s")

    rows, extra = [], {}
    for row in races.itertuples():
        fund = fundamentals_prior(row, env, fin)
        rat = ratings_prior(row.id, row.chamber, ratings, fund["mu"])
        pmu, psd = combine_prior(fund, rat)
        poll = avgs.get(row.id)
        post = blend(pmu, psd, poll, row.chamber)
        post_core = blend(pmu, psd, avgs_core.get(row.id), row.chamber)
        fixed = row.fixed if isinstance(row.fixed, str) else None
        mu, sd = post["mu"], post["sd"]
        if fixed:
            mu, sd = (60.0 if fixed == "D" else -60.0), 0.01
        el = PARAMS.fundamentals.elasticity[row.chamber]
        rows.append({"id": row.id, "chamber": row.chamber, "state": row.state, "mu": mu, "sd": sd,
                     "elasticity": el, "d_side": row.d_side, "r_side": row.r_side, "fixed": fixed or ""})
        tot = math.sqrt(sd ** 2 + (el * SHARED_SD) ** 2)
        rat_mu = rat["mu"] if rat else fund["mu"]
        extra[row.id] = {
            "row": row, "fund": fund, "rat": rat, "prior": (pmu, psd), "poll": poll, "post": post,
            "variants": {
                "ratings_only": 1.0 if fixed == "D" else 0.0 if fixed == "R" else _phi(rat_mu / 8.0),
                "fundamentals_only": 1.0 if fixed == "D" else 0.0 if fixed == "R" else
                _phi(fund["mu"] / math.sqrt(fund["sd"] ** 2 + 4.5 ** 2)),
                "polls_simple": (1.0 if fixed == "D" else 0.0 if fixed == "R" else
                                 _phi(simple[row.id] / 6.5) if row.id in simple else _phi(rat_mu / 8.0)),
                "top20_only": 1.0 if fixed == "D" else 0.0 if fixed == "R" else
                _phi(post_core["mu"] / math.sqrt(post_core["sd"] ** 2 + (el * SHARED_SD) ** 2)),
                "top20_only_margin": post_core["mu"],
                "normal_approx": _phi(mu / tot) if not fixed else (1.0 if fixed == "D" else 0.0),
            },
            "core_avg": avgs_core.get(row.id),
            "simple_avg": simple.get(row.id),
        }
    rdf = pd.DataFrame(rows)
    margins, _ = simulate_margins(rdf, n=n_sims)
    chambers = chamber_outcomes(rdf, margins)
    if verbose:
        print(f"[lodestar] simulated {margins.shape[0]:,} x {margins.shape[1]} races | {time.time()-t0:.1f}s")

    pmat_win = (margins > 0).mean(0)
    q = np.percentile(margins, [10, 50, 90], axis=0)

    # polls per race (for the detail drawer)
    frame = frame.copy()
    frame["share"] = frame["w"] / frame.groupby("race")["w"].transform("sum")
    ptab = ptable.set_index("pollster")
    polls_by_race = {}
    for race, g in frame.groupby("race"):
        g = g.sort_values("end", ascending=False)
        polls_by_race[race] = [{
            "pollster": r.pollster_canon, "core_rank": None if pd.isna(ptab["core_rank"].get(r.pollster_canon, np.nan))
            else int(ptab["core_rank"].get(r.pollster_canon)),
            "grade": ptab["grade"].get(r.pollster_canon, "—"),
            "start": r.start.strftime("%Y-%m-%d"), "end": r.end.strftime("%Y-%m-%d"),
            "n": None if pd.isna(r.n) else int(r.n), "pop": r.population, "partisan": r.partisan,
            "d": _r(r.d, 1), "r": _r(r.r, 1), "margin": _r(r.margin, 1), "adj_margin": _r(r.adj_margin, 1),
            "house_adj": _r(r.house_adj, 2), "trend_adj": _r(r.trend_adj, 2), "partisan_adj": _r(r.partisan_adj, 1),
            "weight_share": _r(r.share, 4), "url": r.url if isinstance(r.url, str) else None,
        } for r in g.itertuples()]

    race_out = []
    for i, rr in enumerate(rows):
        x = extra[rr["id"]]
        row = x["row"]
        p_d = float(pmat_win[i])
        fund, rat, poll, post = x["fund"], x["rat"], x["poll"], x["post"]
        holder = row.inc_party if isinstance(row.inc_party, str) else None
        winner_side = "D" if p_d >= 0.5 else "R"
        flip = None
        if holder in ("D", "R"):
            flip = p_d if holder == "R" else 1 - p_d
        race_out.append({
            "id": rr["id"], "chamber": rr["chamber"], "state": row.state,
            "state_name": STATE_NAMES.get(row.state, row.state), "label": row.label,
            "region": region_of(row.state), "special": rr["id"].endswith("-special"),
            "incumbent": row.incumbent if isinstance(row.incumbent, str) else None,
            "holder": holder, "inc_running": bool(row.inc_running),
            "d_cand": row.d_cand if isinstance(row.d_cand, str) else None, "d_side": row.d_side,
            "r_cand": row.r_cand if isinstance(row.r_cand, str) else None, "r_side": row.r_side,
            "fixed": rr["fixed"] or None,
            "p_d": round(p_d, 4), "p_r": round(1 - round(p_d, 4), 4), "favored": winner_side,
            "p_flip": None if flip is None else round(flip, 4),
            "margin": {"mean": _r(rr["mu"], 2), "p10": _r(q[0, i], 1), "p50": _r(q[1, i], 1), "p90": _r(q[2, i], 1)},
            "posterior_sd": _r(rr["sd"], 2),
            "poll_weight": _r(post["poll_weight"], 3),
            "polls": None if poll is None else {
                "avg": _r(poll["avg"], 2), "raw_avg": _r(poll["raw_avg"], 2), "n_polls": poll["n_polls"],
                "n_eff": _r(poll["n_eff"], 1), "sd": _r(poll["sd"], 2), "undecided": _r(poll["undecided"], 1),
                "latest": poll["latest"], "d": _r(poll["d"], 1), "r": _r(poll["r"], 1),
                "core_avg": _r(x["core_avg"]["avg"], 2) if x["core_avg"] else None,
                "core_n": x["core_avg"]["n_polls"] if x["core_avg"] else 0,
                "simple_avg": _r(x["simple_avg"], 2)},
            "fundamentals": {"mu": _r(fund["mu"], 2), "sd": _r(fund["sd"], 1), "lean": _r(fund["lean"], 2),
                             "environment": _r(fund["env_component"], 2), "incumbency": _r(fund["incumbency"], 2),
                             "finance": _r(fund["finance"], 2), "independent_adj": _r(fund["independent_adj"], 1)},
            "ratings": None if rat is None else {"mu": _r(rat["mu"], 2), "consensus": scale_label(rat["consensus_scale"]),
                                                 "scale": _r(rat["consensus_scale"], 2), "by_rater": rat["by_rater"]},
            "prior": {"mu": _r(x["prior"][0], 2), "sd": _r(x["prior"][1], 2)},
            "variants": {k: _r(v, 4) for k, v in x["variants"].items()},
            "poll_list": polls_by_race.get(rr["id"], []) if rr["chamber"] != "house" or poll else [],
        })

    # pollster leaderboard
    pl = []
    for r in ptable.itertuples():
        pl.append({"pollster": r.pollster, "core_rank": None if pd.isna(r.core_rank) else int(r.core_rank),
                   "is_core": bool(r.is_core), "grade": r.grade, "ppm": _r(r.ppm, 3), "rating_source": r.source,
                   "n_polls": int(r.n_polls), "n_races": int(r.n_races), "last_poll": r.last_poll,
                   "share_partisan": r.share_partisan, "house_effect": _r(r.house_effect, 2),
                   "house_effect_n": int(r.house_effect_n), "weight_multiplier": _r(r.weight_multiplier, 3)})
    core_mask = frame["pollster_canon"].isin(ptable[ptable.is_core].pollster)
    core_share = float(frame.loc[core_mask, "w"].sum() / frame["w"].sum())
    core_count_share = float(core_mask.mean())

    gtrend = gs.reset_index()
    gtrend = gtrend[gtrend["date"] >= pd.Timestamp("2026-01-01")]
    out = {
        "meta": {
            "model": MODEL_NAME, "version": MODEL_VERSION, "as_of": as_of,
            "election_date": "2026-11-03",
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_sims": int(margins.shape[0]), "n_races": len(race_out),
            "n_polls_race": int(len(rp)), "n_polls_generic": int(len(gp)),
            "latest_poll": max(rp["end"].max(), gp["end"].max()).strftime("%Y-%m-%d"),
            "core_weight_share": round(core_share, 3), "core_count_share": round(core_count_share, 3),
            "data_snapshot": {"repo": "github.com/KaraZajac/MIDTERMS2026 (data/ only, CC BY-SA 4.0)",
                              "commit": D.snapshot_commit()},
            "attribution": ["Pollster ratings: Silver Bulletin (Jan 2026); FiveThirtyEight (2024, CC BY 4.0)",
                            "Polls, nominees, ratings: Wikipedia contributors (CC BY-SA 4.0) via MIDTERMS2026",
                            "Some polls: VoteHub (CC BY 4.0)", "District lean: The Downballot",
                            "Campaign finance: FEC"],
            "params": {"sd_national": S.sd_national, "t_df": S.t_df, "top_n": PARAMS.pollsters.top_n,
                       "half_life_state": PARAMS.averaging.half_life_days_state,
                       "half_life_generic": PARAMS.averaging.half_life_days_generic},
            "runtime_s": round(time.time() - t0, 1),
        },
        "national": {
            "generic_ballot": {"margin": _r(env, 2), "d": _r(gcur["d"], 1), "r": _r(gcur["r"], 1),
                               "n_polls": gcur["n_polls"], "n_eff": _r(gcur["n_eff"], 1),
                               "top_polls": gcur["top_polls"]},
            "generic_trend": [{"date": d.strftime("%Y-%m-%d"), "avg": _r(a, 2), "smooth": _r(s, 2)}
                              for d, a, s in zip(gtrend["date"], gtrend["avg"], gtrend["smooth"])],
        },
        "chambers": chambers,
        "control_rules": CHAMBERS,
        "pollsters": pl,
        "races": race_out,
    }
    return out


def clean(x):
    """Strict JSON: NaN/inf -> null (browsers' JSON.parse and Starlette reject NaN)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def write(out: dict, path: Path | None = None) -> Path:
    path = path or OUT / "forecast.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(out), separators=(",", ":"), default=str, allow_nan=False))
    return path


def main():
    ap = argparse.ArgumentParser(description="Run the Lodestar 2026 midterm forecast")
    ap.add_argument("--as-of", default=FORECAST_AS_OF)
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = build(a.as_of, a.sims)
    p = write(out, Path(a.out) if a.out else None)
    s, h, g = out["chambers"]["senate"], out["chambers"]["house"], out["chambers"]["governor"]
    print(f"[lodestar] Senate  P(D control)={s['p_d_control']:.3f}  P(R)={s['p_r_control']:.3f}  "
          f"D seats median {s['d_seats']['median']:.0f} (80%: {s['d_seats']['p10']:.0f}-{s['d_seats']['p90']:.0f})")
    print(f"[lodestar] House   P(D control)={h['p_d_control']:.3f}  D seats median {h['d_seats']['median']:.0f} "
          f"(80%: {h['d_seats']['p10']:.0f}-{h['d_seats']['p90']:.0f})")
    print(f"[lodestar] Govs    P(D majority)={g['p_d_majority']:.3f}  D govs median {g['d_seats']['median']:.0f}")
    print(f"[lodestar] wrote {p}")


if __name__ == "__main__":
    main()
