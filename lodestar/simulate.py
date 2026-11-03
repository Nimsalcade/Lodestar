"""Correlated, fat-tailed Monte Carlo over every race at once."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BLACK_SHARE, CHAMBERS, HISPANIC_SHARE, PARAMS, REGIONS, region_of

S = PARAMS.sim
STATES = sorted({s for v in REGIONS.values() for s in v})
REGION_NAMES = list(REGIONS)
CHAMBER_IDX = {"senate": 0, "house": 1, "governor": 2}


def _t(rng, size, df):
    """Student-t scaled to unit variance."""
    return rng.standard_t(df, size=size) / np.sqrt(df / (df - 2))


def draw_errors(n: int, rng: np.random.Generator) -> dict:
    return {
        "nat": _t(rng, n, S.t_df) * S.sd_national + S.national_bias,
        "region": rng.normal(0, S.sd_region, (n, len(REGION_NAMES))),
        "state": rng.normal(0, S.sd_state, (n, len(STATES))),
        "hisp": rng.normal(0, S.sd_hispanic, n),
        "black": rng.normal(0, S.sd_black, n),
        "chamber": np.stack([rng.normal(0, S.sd_chamber[c], n) for c in CHAMBER_IDX], axis=1),
    }


def simulate_margins(races: pd.DataFrame, n: int | None = None, seed: int | None = None,
                     errors: dict | None = None) -> tuple[np.ndarray, dict]:
    """races needs: state, chamber, mu, sd, elasticity. Returns (n x races) float32
    margins (D-side minus R-side) and the shared error draws."""
    n = n or S.n_sims
    rng = np.random.default_rng(S.seed if seed is None else seed)
    e = errors or draw_errors(n, rng)
    st_idx = races["state"].map({s: i for i, s in enumerate(STATES)}).to_numpy()
    rg_idx = races["state"].map(lambda s: REGION_NAMES.index(region_of(s))).to_numpy()
    ch_idx = races["chamber"].map(CHAMBER_IDX).to_numpy()
    hisp = races["state"].map(HISPANIC_SHARE).fillna(0.08).to_numpy()
    black = races["state"].map(BLACK_SHARE).fillna(0.08).to_numpy()
    el = races["elasticity"].to_numpy()
    mu = races["mu"].to_numpy()
    sd = races["sd"].to_numpy()
    shared = (e["nat"][:, None] * el[None, :]
              + e["region"][:, rg_idx] + e["state"][:, st_idx]
              + e["hisp"][:, None] * hisp[None, :] + e["black"][:, None] * black[None, :]
              + e["chamber"][:, ch_idx])
    idio = _t(rng, (n, len(races)), S.t_df) * sd[None, :]
    m = (mu[None, :] + shared + idio).astype(np.float32)
    return m, e


def _hist(x: np.ndarray, lo: int | None = None, hi: int | None = None) -> dict:
    lo = int(x.min()) if lo is None else lo
    hi = int(x.max()) if hi is None else hi
    counts = np.bincount(np.clip(x, lo, hi) - lo, minlength=hi - lo + 1)[: hi - lo + 1]
    return {"min": lo, "max": hi, "p": (counts / len(x)).round(5).tolist()}


def _summ(x: np.ndarray) -> dict:
    q = np.percentile(x, [5, 10, 25, 50, 75, 90, 95])
    return {"mean": float(x.mean()), "median": float(q[3]), "p05": float(q[0]), "p10": float(q[1]),
            "p25": float(q[2]), "p75": float(q[4]), "p90": float(q[5]), "p95": float(q[6])}


def chamber_outcomes(races: pd.DataFrame, margins: np.ndarray, shifts=range(-8, 9)) -> dict:
    """Seat totals, control probabilities, tipping points and a polling-error
    sensitivity curve for each chamber."""
    out = {}
    win = margins > 0
    # ---------------- Senate
    sen = races.index[races.chamber == "senate"].to_numpy()
    if len(sen):
        rs = races.loc[sen]
        is_dparty = (rs.d_side == "D").to_numpy()
        is_ind = (rs.d_side == "IND").to_numpy()
        w = win[:, sen]
        d = CHAMBERS["senate"]["not_up"]["D"] + (w & is_dparty).sum(1)
        ind = (w & is_ind).sum(1)
        r = CHAMBERS["senate"]["not_up"]["R"] + (~w).sum(1)
        p_d = float((d >= 51).mean())
        p_r = float((r >= 50).mean())
        need = 51 - CHAMBERS["senate"]["not_up"]["D"]
        mm = margins[:, sen][:, is_dparty]
        order = np.argsort(-mm, axis=1)
        tip = order[:, need - 1]
        tip_ids = rs.index[is_dparty].to_numpy()[tip]
        tip_freq = pd.Series(races.loc[tip_ids, "id"].to_numpy()).value_counts(normalize=True)
        el = races.loc[sen, "elasticity"].to_numpy()
        sens = []
        for s in shifts:
            ws = (margins[:, sen] + s * el[None, :]) > 0
            dd = CHAMBERS["senate"]["not_up"]["D"] + (ws & is_dparty).sum(1)
            rr = CHAMBERS["senate"]["not_up"]["R"] + (~ws).sum(1)
            sens.append({"shift": s, "p_d": float((dd >= 51).mean()), "p_r": float((rr >= 50).mean())})
        out["senate"] = {
            "p_d_control": p_d, "p_r_control": p_r, "p_independents_decide": float(1 - p_d - p_r),
            "d_seats": _summ(d), "r_seats": _summ(r), "ind_seats": _summ(ind),
            "p_any_independent": float((ind > 0).mean()),
            "hist_d": _hist(d, 40, 62), "hist_r": _hist(r, 38, 60),
            "tipping_point": [{"id": k, "p": round(float(v), 4)} for k, v in tip_freq.head(8).items()],
            "sensitivity": sens,
        }
    # ---------------- House
    hou = races.index[races.chamber == "house"].to_numpy()
    if len(hou):
        rh = races.loc[hou]
        w = win[:, hou]
        fixed = rh["fixed"].to_numpy()
        is_fixed_d, is_fixed_r = fixed == "D", fixed == "R"
        dpart = (rh.d_side == "D").to_numpy() & ~is_fixed_d & ~is_fixed_r
        rpart = (rh.r_side == "R").to_numpy() & ~is_fixed_d & ~is_fixed_r
        d = (w & dpart).sum(1) + is_fixed_d.sum()
        r = (~w & rpart).sum(1) + is_fixed_r.sum()
        oth = 435 - d - r
        el = rh["elasticity"].to_numpy()
        sens = []
        for s in shifts:
            ws = (margins[:, hou] + s * el[None, :]) > 0
            dd = (ws & dpart).sum(1) + is_fixed_d.sum()
            rr = (~ws & rpart).sum(1) + is_fixed_r.sum()
            sens.append({"shift": s, "p_d": float((dd >= 218).mean()), "p_r": float((rr >= 218).mean()),
                         "d_median": float(np.median(dd))})
        house_d, house_r = d, r
        out["house"] = {
            "p_d_control": float((d >= 218).mean()), "p_r_control": float((r >= 218).mean()),
            "d_seats": _summ(d), "r_seats": _summ(r), "other_seats": _summ(oth),
            "hist_d": _hist(d, 175, 300), "sensitivity": sens,
        }
    # ---------------- Governors
    gov = races.index[races.chamber == "governor"].to_numpy()
    if len(gov):
        w = win[:, gov]
        rg = races.loc[gov]
        d = CHAMBERS["governor"]["not_up"]["D"] + (w & (rg.d_side == "D").to_numpy()).sum(1)
        r = CHAMBERS["governor"]["not_up"]["R"] + (~w).sum(1)
        out["governor"] = {"p_d_majority": float((d >= 26).mean()), "p_r_majority": float((r >= 26).mean()),
                           "d_seats": _summ(d), "r_seats": _summ(r), "hist_d": _hist(d, 10, 40)}
    # joint
    if len(sen) and len(hou):
        rs_ = races.loc[sen]
        ds = CHAMBERS["senate"]["not_up"]["D"] + (win[:, sen] & (rs_.d_side == "D").to_numpy()).sum(1)
        rs = CHAMBERS["senate"]["not_up"]["R"] + (~win[:, sen]).sum(1)
        dh, rh_ = house_d >= 218, house_r >= 218
        out["congress"] = {
            "d_both": float(((ds >= 51) & dh).mean()),
            "d_house_r_senate": float(((rs >= 50) & dh).mean()),
            "r_both": float(((rs >= 50) & rh_).mean()),
            "r_house_d_senate": float(((ds >= 51) & rh_).mean()),
        }
    return out
