"""Poll averaging: weights, partisan and house-effect adjustments, national trend
adjustment, the generic-ballot series, and per-race averages."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PARAMS
from .pollsters import estimate_house_effects

A = PARAMS.averaging
P = PARAMS.pollsters


def _sample_factor(n: pd.Series, population: pd.Series) -> pd.Series:
    fill = population.map({"LV": 600, "RV": 800, "A": 1000, "V": 600}).fillna(600)
    n = n.fillna(fill).clip(lower=150, upper=A.sample_cap)
    return np.sqrt(n / A.sample_ref)


def base_weights(df: pd.DataFrame, ptable: pd.DataFrame, core_only: bool = False) -> pd.Series:
    """Everything except recency: pollster quality x core/non-core, sample size,
    population, partisan sponsorship."""
    pt = ptable.set_index("pollster")
    mult = df["pollster_canon"].map(pt["weight_multiplier"]).fillna(
        P.noncore_multiplier * P.unrated_multiplier * 0.85)
    if core_only:
        is_core = df["pollster_canon"].map(pt["is_core"]).fillna(False).astype(bool)
        mult = mult.where(is_core, 0.0)
    popw = df["population"].map(A.pop_weight).fillna(0.6)
    partw = np.where(df["partisan"].notna(), P.partisan_weight, 1.0)
    return mult * _sample_factor(df["n"], df["population"]) * popw * partw


def partisan_adjust(df: pd.DataFrame) -> pd.Series:
    """Sponsored polls run hot for the sponsor: subtract from D-sponsored, add to R-sponsored."""
    return np.select([df["partisan"].eq("D"), df["partisan"].eq("R")],
                     [-P.partisan_shift, P.partisan_shift], 0.0)


def _decay(age_days, half_life):
    return np.power(0.5, np.clip(age_days, 0, None) / half_life)


# ----------------------------------------------------------------- generic ballot
def generic_series(gp: pd.DataFrame, ptable: pd.DataFrame, house: dict, now: pd.Timestamp,
                   start: str = "2025-11-01") -> pd.DataFrame:
    """Daily one-sided average (what you'd have known that day) plus a two-sided
    smooth (best estimate of the true value on that day, for trend adjustment)."""
    g = gp.copy()
    g["adj"] = g["margin"] + partisan_adjust(g) - g["pollster_canon"].map(
        lambda p: house.get(p, (0.0, 0))[0])
    g["w0"] = base_weights(g, ptable)
    g = g[g["w0"] > 0]
    days = pd.date_range(start, now, freq="D")
    mids = g["mid"].to_numpy(dtype="datetime64[D]")
    adj, w0 = g["adj"].to_numpy(), g["w0"].to_numpy()
    rows = []
    for d in days:
        dd = np.datetime64(d.date(), "D")
        age = (dd - mids).astype(float)
        m1 = (age >= 0) & (age <= 60)
        w1 = w0[m1] * _decay(age[m1], A.half_life_days_generic)
        k = np.exp(-0.5 * (age / 8.0) ** 2) * (np.abs(age) <= 30)
        w2 = w0 * k
        one = float((w1 * adj[m1]).sum() / w1.sum()) if w1.sum() > 0 else np.nan
        two = float((w2 * adj).sum() / w2.sum()) if w2.sum() > 0 else np.nan
        neff = float(w1.sum() ** 2 / (w1 ** 2).sum()) if w1.sum() > 0 else 0.0
        rows.append({"date": d, "avg": one, "smooth": two, "n_eff": neff})
    s = pd.DataFrame(rows).set_index("date")
    s["smooth"] = s["smooth"].interpolate(limit_direction="both")
    s["avg"] = s["avg"].ffill()
    return s


def generic_current(gp: pd.DataFrame, ptable: pd.DataFrame, house: dict, now: pd.Timestamp) -> dict:
    g = gp.copy()
    g["adj"] = g["margin"] + partisan_adjust(g) - g["pollster_canon"].map(lambda p: house.get(p, (0.0, 0))[0])
    age = (now - g["mid"]).dt.days.to_numpy(dtype=float)
    g["w"] = base_weights(g, ptable) * _decay(age, A.half_life_days_generic) * (age <= 45)
    g = g[g["w"] > 0]
    w = g["w"].to_numpy()
    avg = float((w * g["adj"]).sum() / w.sum())
    neff = float(w.sum() ** 2 / (w ** 2).sum())
    d_share = float((w * g["d"]).sum() / w.sum())
    r_share = float((w * g["r"]).sum() / w.sum())
    g["share"] = w / w.sum()
    recent = g.sort_values("w", ascending=False).head(25)
    polls = [{"pollster": r.pollster_canon, "end": r.end.strftime("%Y-%m-%d"), "n": None if pd.isna(r.n) else int(r.n),
              "pop": r.population, "d": round(r.d, 1), "r": round(r.r, 1), "margin": round(r.margin, 1),
              "adjusted": round(r.adj, 1), "weight_share": round(r.share, 4)} for r in recent.itertuples()]
    return {"margin": avg, "d": d_share, "r": r_share, "n_eff": neff, "n_polls": int(len(g)), "top_polls": polls}


# ----------------------------------------------------------------------- races
def _race_frame(rp: pd.DataFrame, ptable: pd.DataFrame, house: dict, gseries: pd.DataFrame,
                gb_now: float, now: pd.Timestamp, core_only: bool = False) -> pd.DataFrame:
    df = rp.copy()
    age = (now - df["mid"]).dt.days.astype(float)
    df = df[age <= A.window_days].copy()
    age = (now - df["mid"]).dt.days.astype(float)
    smooth = gseries["smooth"]
    gb_then = df["mid"].dt.normalize().map(lambda d: smooth.get(d, np.nan)).astype(float)
    gb_then = gb_then.fillna(smooth.iloc[0])
    chamber = df["race"].str.split(":").str[0]
    elast = chamber.map(PARAMS.fundamentals.elasticity).fillna(1.0)
    df["trend_adj"] = A.trend_elasticity * elast * (gb_now - gb_then)
    df["partisan_adj"] = partisan_adjust(df)
    df["house_adj"] = -df["pollster_canon"].map(lambda p: house.get(p, (0.0, 0))[0])
    df["adj_margin"] = df["margin"] + df["partisan_adj"] + df["house_adj"] + df["trend_adj"]
    df["age_days"] = age
    w = base_weights(df, ptable, core_only=core_only) * _decay(age, A.half_life_days_state)
    # a prolific firm shouldn't own a race: divide by sqrt(#its polls in this race, last 60 days)
    recent = df[age <= 60].groupby(["race", "pollster_canon"]).size()
    k = df.set_index(["race", "pollster_canon"]).index.map(lambda ix: recent.get(ix, 1)).to_numpy()
    df["w"] = (w / np.sqrt(np.maximum(k, 1))).to_numpy()
    und = df["undecided"]
    implied = (100 - df["d"] - df["r"] - df["other"].fillna(0)).clip(lower=0)
    df["und"] = und.fillna(implied).clip(0, 40)
    return df


def race_averages(rp: pd.DataFrame, ptable: pd.DataFrame, house: dict, gseries: pd.DataFrame,
                  gb_now: float, now: pd.Timestamp, core_only: bool = False) -> tuple[dict, pd.DataFrame]:
    df = _race_frame(rp, ptable, house, gseries, gb_now, now, core_only)
    out = {}
    for race, g in df.groupby("race"):
        g = g[g["w"] > 0]
        if g.empty:
            continue
        w = g["w"].to_numpy()
        avg = float((w * g["adj_margin"]).sum() / w.sum())
        raw = float((w * g["margin"]).sum() / w.sum())
        neff = float(w.sum() ** 2 / (w ** 2).sum())
        und = float((w * g["und"]).sum() / w.sum())
        sd = float(np.sqrt(A.sd_per_poll ** 2 / neff + A.sd_race_floor ** 2))
        out[race] = {"avg": avg, "raw_avg": raw, "n_eff": neff, "sd": sd, "undecided": und,
                     "n_polls": int(len(g)), "latest": g["end"].max().strftime("%Y-%m-%d"),
                     "d": float((w * g["d"]).sum() / w.sum()), "r": float((w * g["r"]).sum() / w.sum())}
    return out, df


def simple_average(rp: pd.DataFrame, now: pd.Timestamp) -> dict:
    """Baseline: unweighted mean of the last 30 days of polls (else last 3 polls)."""
    out = {}
    for race, g in rp.groupby("race"):
        g = g.sort_values("end")
        recent = g[(now - g["end"]).dt.days <= 30]
        use = recent if len(recent) else g.tail(3)
        if len(use):
            out[race] = float(use["margin"].mean())
    return out


def fit_house_effects(rp: pd.DataFrame, gp: pd.DataFrame, ptable: pd.DataFrame,
                      now: pd.Timestamp, iterations: int = 3) -> tuple[dict, pd.DataFrame]:
    """Alternate between averages and pollster house effects."""
    house: dict = {}
    for _ in range(iterations):
        gs = generic_series(gp, ptable, house, now)
        gb_now = generic_current(gp, ptable, house, now)["margin"]
        _, frame = race_averages(rp, ptable, house, gs, gb_now, now)
        resid = []
        # race polls: residual vs the weighted average of OTHER firms in the race
        for race, g in frame.groupby("race"):
            if g["pollster_canon"].nunique() < 2:
                continue
            base = g["margin"] + g["partisan_adj"] + g["trend_adj"]
            for firm, gf in g.groupby("pollster_canon"):
                other = g[g["pollster_canon"] != firm]
                ob = (other["margin"] + other["partisan_adj"] + other["trend_adj"] + other["house_adj"])
                ow = other["w"].to_numpy()
                if ow.sum() <= 0:
                    continue
                cons = float((ow * ob).sum() / ow.sum())
                for v in base[gf.index]:
                    resid.append((firm, v - cons))
        # generic polls: residual vs two-sided smooth that day
        g2 = gp[(now - gp["mid"]).dt.days <= 300].copy()
        sm = gs["smooth"]
        cons = g2["mid"].dt.normalize().map(lambda d: sm.get(d, np.nan)).astype(float)
        for firm, v, c in zip(g2["pollster_canon"], g2["margin"] + partisan_adjust(g2), cons):
            if np.isfinite(c):
                resid.append((firm, v - c))
        house = estimate_house_effects(pd.DataFrame(resid, columns=["pollster_canon", "resid"]))
    return house, frame
