"""Priors: partisan lean x national environment + incumbency + money, and the
qualitative handicappers' ratings; then the Bayesian blend with the polls."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PARAMS

F = PARAMS.fundamentals
R = PARAMS.ratings

APPOINTED = {"senate:OH-special", "senate:FL-special", "senate:SC"}   # Husted, Moody, D. Graham
SUCCEEDED_GOV = {"governor:SD"}                                      # Rhoden succeeded Noem (2025)

# Incumbent "personal vote": the incumbent's last general-election margin minus the
# presidential margin in that state for the same/closest cycle (D minus R, points).
# Collins and Phil Scott are the canonical cases: partisanship alone misses them by
# 18 and 80+ points. Recorded only for incumbents in races where it can matter.
PERSONAL_VOTE = {
    # Senate (last race, presidential reference)
    "senate:ME": -17.7,   # Collins 2020 R+8.6 vs Biden +9.1
    "senate:AK": -2.6,    # Sullivan 2020 R+12.7 vs Trump +10.1
    "senate:KS": +3.2,    # Marshall 2020 R+11.4 vs Trump +14.6
    "senate:NE": -4.7,    # Ricketts 2024 R+25.2 vs Trump +20.5
    "senate:MS": +6.5,    # Hyde-Smith 2020 R+10.0 vs Trump +16.5
    "senate:GA": +1.0,    # Ossoff 2021 runoff D+1.2 vs Biden +0.2
    "senate:SD": -5.2,    # Rounds 2020 R+31.4 vs Trump +26.2
    # Governors
    "governor:VT": -83.0,  # Scott 2024 R+51.5 vs Harris +31.5
    "governor:NH": -12.3,  # Ayotte 2024 R+9.5 vs Harris +2.8
    "governor:NV": -3.9,   # Lombardo 2022 R+1.5 vs Biden 2020 +2.4
    "governor:AZ": +0.4,   # Hobbs 2022 D+0.7 vs Biden 2020 +0.3
    "governor:TX": -5.3,   # Abbott 2022 R+10.9 vs Trump 2020 +5.6
    "governor:PA": +13.6,  # Shapiro 2022 D+14.8 vs Biden 2020 +1.2
    "governor:NY": -16.7,  # Hochul 2022 D+6.4 vs Biden 2020 +23.1
    "governor:OR": -12.7,  # Kotek 2022 D+3.4 vs Biden 2020 +16.1
    "governor:NE": -4.6,   # Pillen 2022 R+23.7 vs Trump 2020 +19.1
    "governor:IL": -4.5,   # Pritzker 2022 D+12.5 vs Biden 2020 +17.0
    "governor:CT": -7.2,   # Lamont 2022 D+12.9 vs Biden 2020 +20.1
}
PERSONAL_WEIGHT = {"senate": 0.40, "governor": 0.35, "house": 0.30}
PERSONAL_CAP = {"senate": 8.0, "governor": 18.0, "house": 6.0}


def national_lean_baseline() -> float:
    return F.w24 * F.nat_pres24 + (1 - F.w24) * F.nat_pres20


def incumbency(row) -> float:
    if not row.inc_running or row.inc_party not in ("D", "R"):
        return 0.0
    nominee = row.d_cand if row.inc_party == "D" else row.r_cand
    if not isinstance(nominee, str) or not isinstance(row.incumbent, str):
        return 0.0
    if row.incumbent.split()[-1].lower() not in nominee.lower():
        return 0.0          # incumbent lost the primary / seat holder isn't the nominee
    adv = F.inc_adv[row.chamber]
    if row.id in APPOINTED:
        adv = F.appointed_inc_adv
    if row.id in SUCCEEDED_GOV:
        adv *= 0.5
    if row.chamber == "house" and row.state in F.redrawn_states:
        adv *= F.redrawn_inc_factor
    generic = adv if row.inc_party == "D" else -adv
    if row.id in PERSONAL_VOTE:
        pv = 0.5 * generic + PERSONAL_WEIGHT[row.chamber] * PERSONAL_VOTE[row.id]
        cap = PERSONAL_CAP[row.chamber]
        return float(np.clip(pv, -cap, cap))
    return generic


def finance_edge(race_id: str, fin: pd.DataFrame) -> float:
    f = fin[fin.race == race_id]
    d = f[f.side == "D"]["total_receipts"].sum()
    r = f[f.side == "R"]["total_receipts"].sum()
    if d <= 0 or r <= 0:
        return 0.0
    return float(np.clip(F.finance_coef * np.log(d / r), -F.finance_cap, F.finance_cap))


def fundamentals_prior(row, env: float, fin: pd.DataFrame) -> dict:
    lean = float(row.lean) - national_lean_baseline()
    el = F.elasticity[row.chamber]
    base = el * (lean + env)
    indie = 0.0
    if row.d_side == "IND":
        base = 0.8 * base
        indie = F.independent_bonus.get(row.state, F.independent_bonus["default"])
    if row.r_side == "IND":       # deep-blue seat where the R-side is an independent
        base = 0.8 * base
        indie = -F.independent_bonus["default"]
    inc = incumbency(row)
    money = finance_edge(row.id, fin)
    mu = base + indie + inc + money
    sd = F.sd_fund[row.chamber] + (2.5 if indie else 0.0)
    return {"mu": mu, "sd": sd, "lean": lean, "env_component": el * env, "incumbency": inc,
            "finance": money, "independent_adj": indie}


def ratings_prior(race_id: str, chamber: str, ratings: pd.DataFrame, mu_fund: float) -> dict | None:
    r = ratings[(ratings.race == race_id) & (ratings.rater.isin(R.raters))]
    if r.empty:
        return None
    r = r.sort_values("asof").groupby("rater").tail(1)
    vals, ws, labels = [], [], {}
    for x in r.itertuples():
        s = int(np.clip(round(float(x.scale)), -3, 3))
        mag = R.scale_to_margin[abs(s)]
        if abs(s) == 3 and np.sign(mu_fund) == np.sign(s):
            mag = max(mag, abs(mu_fund))
        vals.append(np.sign(s) * mag)
        ws.append(R.rater_weight.get(x.rater, R.other_rater_weight))
        labels[x.rater] = x.rating
    ws = np.array(ws)
    mu = float(np.dot(ws, vals) / ws.sum())
    cons_scale = float(np.dot(ws, [float(v) for v in r["scale"]]) / ws.sum())
    return {"mu": mu, "sd": R.sd_rating[chamber], "consensus_scale": cons_scale, "by_rater": labels}


def scale_label(s: float) -> str:
    a = abs(s)
    side = "D" if s > 0 else "R"
    if a < 0.5:
        return "Toss-up"
    if a < 1.5:
        return f"Lean {side}"
    if a < 2.5:
        return f"Likely {side}"
    return f"Safe {side}"


def combine_prior(fund: dict, rat: dict | None) -> tuple[float, float]:
    if rat is None:
        return fund["mu"], fund["sd"]
    pf, pr = 1 / fund["sd"] ** 2, 1 / rat["sd"] ** 2
    mu = (fund["mu"] * pf + rat["mu"] * pr) / (pf + pr)
    sd = np.sqrt(1 / (pf + pr)) * R.prior_corr_inflation
    return float(mu), float(sd)


def blend(prior_mu: float, prior_sd: float, poll: dict | None, chamber: str) -> dict:
    """Precision-weighted posterior of the race's true margin (before the shared
    national/regional/state errors, which the simulator adds)."""
    A = PARAMS.averaging
    if poll is None:
        extra = PARAMS.sim.sd_district_extra if chamber == "house" else 0.0
        sd = float(np.sqrt(prior_sd ** 2 + extra ** 2))
        return {"mu": prior_mu, "sd": sd, "poll_weight": 0.0}
    pp, pa = 1 / prior_sd ** 2, 1 / poll["sd"] ** 2
    mu = (prior_mu * pp + poll["avg"] * pa) / (pp + pa)
    sd = np.sqrt(1 / (pp + pa))
    und = A.undecided_sd_per_pt * poll.get("undecided", 0.0)
    sd = float(np.sqrt(sd ** 2 + und ** 2))
    return {"mu": float(mu), "sd": sd, "poll_weight": float(pa / (pp + pa))}
