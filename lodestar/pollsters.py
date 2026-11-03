"""Pollster ratings, the Top-20 core panel, quality weights and house effects."""
from __future__ import annotations

import difflib
import re

import numpy as np
import pandas as pd

from .config import PARAMS
from .data import canonical_pollster


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(the|inc|llc|co|corp|university|college|research|polling|group|associates|strategies|"
               r"insights|analytics|survey|center|institute|poll)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


class RatingBook:
    """Silver Bulletin (Jan 2026) first; FiveThirtyEight's final (2024) ratings,
    re-scaled onto Silver Bulletin's Predictive Plus-Minus, for firms SB doesn't rate."""

    def __init__(self, ratings: pd.DataFrame):
        sb = ratings[ratings.source == "silver-bulletin-2026-01"].copy()
        fte = ratings[ratings.source == "538-2024"].copy()
        self.sb = {r.pollster: r for r in sb.itertuples()}
        self.fte = {r.pollster: r for r in fte.itertuples()}
        # Linear map 538 ppm -> SB ppm on firms rated by both
        both = sb.merge(fte, on="pollster", suffixes=("_sb", "_538")).dropna(subset=["ppm_sb", "ppm_538"])
        if len(both) >= 10:
            b, a = np.polyfit(both["ppm_538"], both["ppm_sb"], 1)
        else:
            a, b = 0.0, 0.6
        self.map_a, self.map_b, self.n_overlap = float(a), float(b), int(len(both))
        self._sb_norm = {_norm(k): k for k in self.sb}
        self._fte_norm = {_norm(k): k for k in self.fte}

    def lookup(self, canon: str) -> dict:
        def from_sb(name):
            r = self.sb[name]
            return {"rated_as": name, "source": "Silver Bulletin 2026", "grade": str(r.grade),
                    "ppm": float(r.ppm), "banned": isinstance(r.banned, str) and r.banned.strip() != ""}

        def from_fte(name):
            r = self.fte[name]
            ppm = self.map_a + self.map_b * float(r.ppm)
            return {"rated_as": name, "source": "FiveThirtyEight 2024 (rescaled)", "grade": f"538 {r.grade}",
                    "ppm": float(ppm), "banned": False}

        if canon in self.sb:
            return from_sb(canon)
        if canon in self.fte and pd.notna(self.fte[canon].ppm):
            return from_fte(canon)
        n = _norm(canon)
        if n in self._sb_norm:
            return from_sb(self._sb_norm[n])
        hit = difflib.get_close_matches(n, list(self._sb_norm), n=1, cutoff=0.92)
        if hit:
            return from_sb(self._sb_norm[hit[0]])
        hit = difflib.get_close_matches(n, list(self._fte_norm), n=1, cutoff=0.92)
        if hit and pd.notna(self.fte[self._fte_norm[hit[0]]].ppm):
            return from_fte(self._fte_norm[hit[0]])
        return {"rated_as": None, "source": "unrated", "grade": "—", "ppm": PARAMS.pollsters.unrated_ppm,
                "banned": False}


def quality_weight(ppm: float) -> float:
    s = PARAMS.pollsters.sigma_ref
    return float((s / max(s + ppm, 1.0)) ** 2)


def build_pollster_table(race_polls: pd.DataFrame, generic_polls: pd.DataFrame,
                         book: RatingBook) -> pd.DataFrame:
    """One row per canonical pollster active in 2026, with its rating, Top-20
    rank (if core), and the multiplier every one of its polls receives."""
    P = PARAMS.pollsters
    allp = pd.concat([race_polls.assign(kind="race"), generic_polls.assign(kind="generic")])
    active = allp[allp["end"] >= pd.Timestamp(P.active_since)]
    g = (active.groupby("pollster_canon")
               .agg(n_polls=("margin", "size"), n_races=("race", "nunique"),
                    last_poll=("end", "max"),
                    share_partisan=("partisan", lambda s: float(s.notna().mean())))
               .reset_index())
    rows = []
    for r in g.itertuples():
        info = book.lookup(r.pollster_canon)
        rows.append({"pollster": r.pollster_canon, "n_polls": int(r.n_polls), "n_races": int(r.n_races),
                     "last_poll": r.last_poll.strftime("%Y-%m-%d"),
                     "share_partisan": round(r.share_partisan, 2), **info})
    t = pd.DataFrame(rows)
    eligible = t[(t.source != "unrated") & (~t.banned) & (t.n_polls >= P.min_polls_for_core)]
    core = eligible.sort_values("ppm").head(P.top_n)
    t["core_rank"] = t["pollster"].map({p: i + 1 for i, p in enumerate(core["pollster"])})
    t["is_core"] = t["core_rank"].notna()
    t["quality_weight"] = t["ppm"].map(quality_weight)
    mult = np.where(t.is_core, P.core_multiplier,
                    np.where(t.source == "unrated", P.noncore_multiplier * P.unrated_multiplier,
                             P.noncore_multiplier))
    t["weight_multiplier"] = np.where(t.banned, 0.0, t["quality_weight"] * mult)
    t["house_effect"] = 0.0
    t["house_effect_n"] = 0
    return t.sort_values(["is_core", "ppm"], ascending=[False, True]).reset_index(drop=True)


def estimate_house_effects(residuals: pd.DataFrame) -> dict[str, tuple[float, int]]:
    """residuals: columns pollster_canon, resid (poll margin minus consensus excluding
    that firm). Shrunk mean: sum(resid) / (n + k). Positive = firm leans Democratic."""
    P = PARAMS.pollsters
    out = {}
    for name, grp in residuals.groupby("pollster_canon"):
        r = grp["resid"].to_numpy()
        r = r[np.isfinite(r)]
        if len(r) == 0:
            continue
        h = float(np.clip(r.sum() / (len(r) + P.house_shrink_k), -P.house_max, P.house_max))
        out[name] = (h, int(len(r)))
    return out


__all__ = ["RatingBook", "build_pollster_table", "estimate_house_effects", "quality_weight",
           "canonical_pollster"]
