"""Election-night engine.

Three steps, re-run on every results refresh:
  1. Nowcast each counting race: blend the counted margin with the model's
     expectation for the uncounted vote (more weight to the count as it grows).
  2. Learn the night's shared polling error: races that are (nearly) done tell us
     how far the polls missed nationally and by region; shift every race that
     hasn't reported by that amount (a Bayesian update, shrunk toward zero).
  3. Re-simulate the chambers with called races fixed, counting races drawn from
     their nowcast, and everything else from the updated forecast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CHAMBERS, PARAMS, REGIONS, region_of
from .simulate import REGION_NAMES, draw_errors, simulate_margins

S = PARAMS.sim


@dataclass
class LiveParams:
    # Expected count-order skew of the REMAINING vote vs. the counted vote (points of
    # D-minus-R margin). Positive = late vote bluer (mail-heavy late counts).
    # Defaults are 0: set per state on the night if you trust a pattern.
    count_order_bias: dict = field(default_factory=dict)
    min_report_for_learning: float = 0.90      # share of expected vote counted
    # The uncounted vote can differ sharply from the counted vote (mail vs. Election
    # Day, a big county still out), and "% of expected vote" is itself an estimate.
    remaining_sd_floor: float = 10.0          # SD of the remaining vote's margin (points)
    expected_vote_cv: float = 0.35            # relative uncertainty in how much vote remains
    sims: int = 12000
    seed: int = 7


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def nowcast_race(race: dict, res: dict, lp: LiveParams, shift: float = 0.0) -> dict:
    """race: forecast race dict; res: {d_votes, r_votes, other_votes, pct_reporting, status, winner}."""
    mu0 = float(race["margin"]["mean"]) + shift
    sd0 = float(race.get("posterior_sd") or 5.0)
    status = res.get("status", "counting")
    winner = res.get("winner")
    d, r = float(res.get("d_votes") or 0), float(res.get("r_votes") or 0)
    p = min(max(float(res.get("pct_reporting") or 0) / 100.0, 0.0), 1.0)
    out = {"id": race["id"], "status": status, "pct_reporting": round(100 * p, 1), "winner": winner,
           "d_votes": int(d), "r_votes": int(r), "other_votes": int(res.get("other_votes") or 0)}
    if status == "called" and winner in ("D", "R", "IND", "OTHER"):
        side_d = winner == "D" or (winner == "IND" and race.get("d_side") == "IND")
        m_obs = 100 * (d - r) / (d + r) if d + r > 0 else (mu0 if side_d else -mu0)
        out.update(p_d=1.0 if side_d else 0.0, est_margin=round(m_obs, 2), est_sd=0.0, counted_margin=round(m_obs, 2))
        return out
    if d + r <= 0 or p <= 0:
        out.update(p_d=_phi(mu0 / math.hypot(sd0, S.sd_national)), est_margin=round(mu0, 2),
                   est_sd=round(math.hypot(sd0, S.sd_national), 2), counted_margin=None)
        return out
    m_obs = 100 * (d - r) / (d + r)
    w = math.sqrt(p)                                  # how representative the count is
    bias = lp.count_order_bias.get(race["state"], 0.0)
    mu_rem = w * m_obs + (1 - w) * mu0 + bias
    sd_rem = math.hypot((1 - w) * sd0, lp.remaining_sd_floor)
    rem = 1 - p
    est = p * m_obs + rem * mu_rem
    # final = (1-r')*counted + r'*remaining, r' uncertain around rem
    s_r = lp.expected_vote_cv * rem + 0.015            # +1.5 pts of total turnout uncertainty
    sd = math.sqrt((rem ** 2 + s_r ** 2) * sd_rem ** 2 + (s_r * (mu_rem - m_obs)) ** 2)
    sd = max(sd, 0.2)
    out.update(p_d=_phi(est / sd), est_margin=round(est, 2), est_sd=round(sd, 2), counted_margin=round(m_obs, 2))
    return out


def learn_shared_error(forecast: dict, nowcasts: dict, lp: LiveParams) -> dict:
    """Posterior of the national and regional polling errors given (nearly) complete races."""
    races = {r["id"]: r for r in forecast["races"]}
    obs = []
    for rid, nc in nowcasts.items():
        r = races.get(rid)
        if r is None or r.get("fixed"):
            continue
        done = nc["status"] == "called" and nc.get("counted_margin") is not None and nc["pct_reporting"] >= 75
        if done or (nc["pct_reporting"] >= 100 * lp.min_report_for_learning and nc.get("est_margin") is not None):
            el = PARAMS.fundamentals.elasticity[r["chamber"]]
            resid = (float(nc["est_margin"]) - float(r["margin"]["mean"]))
            v = float(r.get("posterior_sd") or 5.0) ** 2 + S.sd_state ** 2 + S.sd_chamber[r["chamber"]] ** 2
            obs.append((rid, region_of(r["state"]), el, resid, v))
    if not obs:
        return {"national": 0.0, "national_sd": S.sd_national, "regions": {}, "n_obs": 0}
    # national (regions marginalized into the noise for this step)
    prec0 = 1 / S.sd_national ** 2
    num = sum(el * res / (v + S.sd_region ** 2) for _, _, el, res, v in obs)
    den = prec0 + sum(el * el / (v + S.sd_region ** 2) for _, _, el, _, v in obs)
    nat, nat_sd = num / den, math.sqrt(1 / den)
    regions = {}
    for reg in REGIONS:
        sub = [(el, res - el * nat, v) for _, rg, el, res, v in obs if rg == reg]
        if not sub:
            continue
        p0 = 1 / S.sd_region ** 2
        num_r = sum(res / v for _, res, v in sub)
        den_r = p0 + sum(1 / v for _, _, v in sub)
        regions[reg] = {"shift": num_r / den_r, "sd": math.sqrt(1 / den_r), "n": len(sub)}
    return {"national": nat, "national_sd": nat_sd, "regions": regions, "n_obs": len(obs)}


def run_live(forecast: dict, results: dict, lp: LiveParams | None = None) -> dict:
    lp = lp or LiveParams()
    races = forecast["races"]
    # pass 1: nowcast with no shift, learn the shared error
    first = {rid: nowcast_race(next(r for r in races if r["id"] == rid), res, lp)
             for rid, res in results.items() if any(r["id"] == rid for r in races)}
    shared = learn_shared_error(forecast, first, lp)
    nat, regs = shared["national"], shared["regions"]

    def shift_for(r):
        el = PARAMS.fundamentals.elasticity[r["chamber"]]
        return el * nat + regs.get(region_of(r["state"]), {}).get("shift", 0.0)

    # pass 2: nowcast again with the learned shift applied to each race's prior
    now = {}
    for r in races:
        if r["id"] in results:
            now[r["id"]] = nowcast_race(r, results[r["id"]], lp, shift=shift_for(r))
    # ------------------------------------------------ re-simulate chambers
    # Same error structure as the pre-election simulator (so at 0% reporting the
    # live odds reproduce the forecast), with the national and regional components
    # replaced by their posteriors once results arrive.
    rng = np.random.default_rng(lp.seed)
    n = lp.sims
    e = draw_errors(n, rng)
    if shared["n_obs"] > 0:
        e["nat"] = rng.normal(0, shared["national_sd"], n)       # shift is added to mu below
        for k, g in enumerate(REGION_NAMES):
            if g in regs:
                e["region"][:, k] = rng.normal(0, regs[g]["sd"], n)
    rdf = pd.DataFrame([{"state": r["state"], "chamber": r["chamber"],
                         "mu": float(r["margin"]["mean"]) + (shift_for(r) if shared["n_obs"] else 0.0),
                         "sd": float(r.get("posterior_sd") or 5.0),
                         "elasticity": PARAMS.fundamentals.elasticity[r["chamber"]]} for r in races])
    margins, _ = simulate_margins(rdf, n=n, seed=lp.seed + 1, errors=e)
    for j, r in enumerate(races):
        if r.get("fixed"):
            margins[:, j] = 60.0 if r["fixed"] == "D" else -60.0
            continue
        nc = now.get(r["id"])
        if nc and nc["status"] == "called":
            margins[:, j] = 50.0 if nc["p_d"] >= 0.5 else -50.0
        elif nc and nc.get("counted_margin") is not None:
            margins[:, j] = rng.normal(nc["est_margin"], max(nc["est_sd"], 0.25), n)
    win = margins > 0
    ids = [r["id"] for r in races]
    chamber = np.array([r["chamber"] for r in races])
    d_side = np.array([r["d_side"] for r in races])
    r_side = np.array([r["r_side"] for r in races])
    fixed = np.array([r.get("fixed") or "" for r in races])
    sen = chamber == "senate"
    ds = CHAMBERS["senate"]["not_up"]["D"] + (win[:, sen] & (d_side[sen] == "D")).sum(1)
    rs = CHAMBERS["senate"]["not_up"]["R"] + (~win[:, sen]).sum(1)
    hou = chamber == "house"
    free = hou & (fixed == "")
    dh = (win[:, free] & (d_side[free] == "D")).sum(1) + ((fixed == "D") & hou).sum()
    rh = (~win[:, free] & (r_side[free] == "R")).sum(1) + ((fixed == "R") & hou).sum()
    gov = chamber == "governor"
    dg = CHAMBERS["governor"]["not_up"]["D"] + (win[:, gov] & (d_side[gov] == "D")).sum(1)

    # scorecard vs the locked pre-election forecast
    called = [(rid, nc) for rid, nc in now.items() if nc["status"] == "called"]
    pre = {r["id"]: r["p_d"] for r in races}
    hits, brier = 0, 0.0
    for rid, nc in called:
        y = 1.0 if nc["p_d"] >= 0.5 else 0.0
        hits += int((pre[rid] >= 0.5) == (y == 1.0))
        brier += (pre[rid] - y) ** 2
    return {
        "shared_error": {"national": round(nat, 2), "national_sd": round(shared["national_sd"], 2),
                         "regions": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in regs.items()},
                         "n_obs": shared["n_obs"],
                         "explain": "Positive = Democrats running ahead of the pre-election forecast."},
        "chambers": {
            "senate": {"p_d_control": float((ds >= 51).mean()), "p_r_control": float((rs >= 50).mean()),
                       "d_median": float(np.median(ds)), "d_p10": float(np.percentile(ds, 10)),
                       "d_p90": float(np.percentile(ds, 90))},
            "house": {"p_d_control": float((dh >= 218).mean()), "p_r_control": float((rh >= 218).mean()),
                      "d_median": float(np.median(dh)), "d_p10": float(np.percentile(dh, 10)),
                      "d_p90": float(np.percentile(dh, 90))},
            "governor": {"p_d_majority": float((dg >= 26).mean()), "d_median": float(np.median(dg))},
        },
        "races": now,
        "race_p_d": {rid: round(float(win[:, j].mean()), 4) for j, rid in enumerate(ids)},
        "scorecard": {"called": len(called), "correct": hits,
                      "brier": round(brier / len(called), 4) if called else None},
    }
