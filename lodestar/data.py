"""Load the raw snapshot, canonicalize pollster names, collapse poll variants."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# (regex on lower-cased pollster string, canonical name). Order matters: first hit wins.
# Canonical names match Silver Bulletin's rating table wherever the firm is rated there.
CANON_RULES: list[tuple[str, str]] = [
    (r"(new york times|nyt).*siena|siena.*(new york times|nyt)", "The New York Times/Siena College"),
    (r"siena", "Siena College"),
    (r"washington post|schar school|\bsspg\b", "The Washington Post"),
    (r"marquette", "Marquette University Law School"),
    (r"research & polling", "Research & Polling Inc."),
    (r"onmessage", "OnMessage Inc."),
    (r"tipp", "TIPP Insights"),
    (r"cygnal", "Cygnal"),
    (r"surveyusa", "SurveyUSA"),
    (r"hart research.*public opinion strategies|nbc news", "Hart Research Associates/Public Opinion Strategies"),
    (r"^public opini?on strategies", "Public Opinion Strategies"),
    (r"atlas ?intel", "AtlasIntel"),
    (r"emerson", "Emerson College"),
    (r"muhlenberg", "Muhlenberg College"),
    (r"echelon", "Echelon Insights"),
    (r"north florida", "University of North Florida"),
    (r"beacon research.*shaw|fox news", "Beacon Research/Shaw & Co. Research"),
    (r"massinc", "MassINC Polling Group"),
    (r"suffolk", "Suffolk University"),
    (r"susquehanna", "Susquehanna Polling & Research Inc."),
    (r"mason[- ]dixon", "Mason-Dixon Polling & Strategy"),
    (r"marist", "Marist College"),
    (r"noble predictive", "Noble Predictive Insights"),
    (r"angus reid", "Angus Reid Global"),
    (r"winston group", "The Winston Group"),
    (r"concord public opinion", "Concord Public Opinion Partners"),
    (r"christopher newport", "Christopher Newport University"),
    (r"tarrance", "Tarrance Group"),
    (r"alaska survey research", "Alaska Survey Research"),
    (r"impact research.*national research|national research", "National Research"),
    (r"quinnipiac", "Quinnipiac University"),
    (r"franklin (&|and) marshall", "Franklin & Marshall College"),
    (r"fabrizio", "Fabrizio, Lee & Associates"),
    (r"saint anselm|st\.? anselm", "Saint Anselm College"),
    (r"university of new hampshire|\bunh\b", "University of New Hampshire"),
    (r"\bcnn\b|ssrs", "SSRS"),
    (r"yougov", "YouGov"),
    (r"ipsos", "Ipsos"),
    (r"data for progress", "Data for Progress"),
    (r"trafalgar", "Trafalgar Group"),
    (r"insider ?advantage", "InsiderAdvantage"),
    (r"rasmussen", "Rasmussen Reports"),
    (r"quantus", "Quantus Insights"),
    (r"activote", "ActiVote"),
    (r"morning consult", "Morning Consult"),
    (r"harrisx|harris insights|harvard.?harris", "Harris Insights & Analytics"),
    (r"^impact research", "Impact Research"),
    (r"gbao", "GBAO"),
    (r"global strategy group", "Global Strategy Group"),
    (r"public policy polling|^ppp\b", "Public Policy Polling"),
    (r"co/efficient", "co/efficient"),
    (r"high point", "High Point University"),
    (r"\belon\b", "Elon University"),
    (r"east carolina", "East Carolina University"),
    (r"citadel", "The Citadel"),
    (r"winthrop", "Winthrop University"),
    (r"^research co", "Research Co."),
    (r"mitchell research", "Mitchell Research & Communications"),
    (r"targoz", "Targoz Market Research"),
    (r"patinkin", "Patinkin Research Strategies"),
    (r"national public affairs", "National Public Affairs"),
    (r"a2 insights", "A2 Insights"),
    (r"redfield", "Redfield & Wilton Strategies"),
    (r"selzer", "Selzer & Co."),
    (r"monmouth", "Monmouth University"),
    (r"elway", "Elway Research"),
    (r"stockton", "Stockton University"),
    (r"fairleigh", "Fairleigh Dickinson University"),
    (r"university of texas|texas politics project", "University of Texas/Texas Politics Project"),
    (r"texas public opinion research", "Texas Public Opinion Research"),
]
_CANON = [(re.compile(p), name) for p, name in CANON_RULES]


def canonical_pollster(raw: str) -> str:
    s = str(raw or "").strip()
    low = s.lower()
    for pat, name in _CANON:
        if pat.search(low):
            return name
    return re.sub(r"\s+", " ", s)


POP_RANK = {"LV": 0, "V": 1, "RV": 2, "A": 3}


def _collapse_variants(df: pd.DataFrame) -> pd.DataFrame:
    """One row per poll. Wikipedia lists several versions of many polls (LV and RV,
    with/without leaners, two-way and full-field, ranked-choice final round vs first
    choice). Keep the best population (LV > V > RV > A); then keep the version with
    the most decided D+R share -- the RCV final round (Alaska, Maine) or the
    leaners-pushed version -- because that is the one that resembles a ballot.
    Exact ties are averaged."""
    df = df.copy()
    df["pop_rank"] = df["population"].map(POP_RANK).fillna(4)
    key = ["race", "pollster_canon", "start", "end"]
    best = df.groupby(key)["pop_rank"].transform("min")
    df = df[df["pop_rank"] == best]
    df["decided"] = df["d"] + df["r"]
    top = df.groupby(key)["decided"].transform("max")
    df = df[df["decided"] >= top - 1e-9]
    agg = (df.groupby(key, dropna=False)
             .agg(pollster=("pollster", "first"), partisan=("partisan", "first"),
                  n=("n", "max"), population=("population", "first"),
                  d=("d", "mean"), r=("r", "mean"), other=("other", "mean"),
                  undecided=("undecided", "mean"), d_label=("d_label", "first"),
                  r_label=("r_label", "first"), url=("url", "first"), variants=("d", "size"))
             .reset_index())
    return agg


RCV_STATES = {"AK"}   # ranked-choice general elections (Maine uses RCV only for federal races)


def _drop_first_choice_only(p: pd.DataFrame) -> pd.DataFrame:
    """In ranked-choice states, a poll published only as first choices (e.g. Alaska governor
    42-25 with the Republican vote split three ways) is not comparable with final-round
    polls (53-47). Where a race has final-round polls, drop first-choice-only ones."""
    st = p["race"].str.split(":").str[1].str[:2]
    decided = p["d"] + p["r"]
    rcv = st.isin(RCV_STATES)
    n_final = decided.where(rcv & (decided >= 95)).groupby(p["race"]).transform("count")
    drop = rcv & (n_final >= 2) & (decided < 80)
    return p[~drop]


def load_polls() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (race_polls, generic_polls), one row per distinct poll."""
    out = []
    for name in ("polls_races.csv", "polls_generic.csv"):
        p = pd.read_csv(RAW / name)
        p = p.dropna(subset=["d", "r", "end"])
        p["pollster_canon"] = p["pollster"].map(canonical_pollster)
        p["start"] = pd.to_datetime(p["start"], errors="coerce")
        p["end"] = pd.to_datetime(p["end"], errors="coerce")
        p["start"] = p["start"].fillna(p["end"])
        p = _collapse_variants(p)
        if name == "polls_races.csv":
            p = _drop_first_choice_only(p)
        p["margin"] = p["d"] - p["r"]
        p["mid"] = p["start"] + (p["end"] - p["start"]) / 2
        p["partisan"] = p["partisan"].where(p["partisan"].isin(["D", "R"]), None)
        out.append(p)
    return out[0], out[1]


def load_races() -> pd.DataFrame:
    frames = []
    for chamber in ("senate", "governor", "house"):
        r = pd.read_csv(RAW / f"{chamber}.csv")
        r["chamber"] = chamber
        frames.append(r)
    races = pd.concat(frames, ignore_index=True)
    races["inc_running"] = races["inc_running"].astype(str).str.lower().eq("yes")
    return races


def load_ratings() -> pd.DataFrame:
    r = pd.read_csv(RAW / "ratings.csv")
    r["asof"] = pd.to_datetime(r["asof"], errors="coerce")
    return r


def load_pollster_ratings() -> pd.DataFrame:
    return pd.read_csv(RAW / "pollster_ratings.csv")


def load_finance() -> pd.DataFrame:
    return pd.read_csv(RAW / "finance.csv")


def load_published_averages() -> pd.DataFrame:
    return pd.read_csv(RAW / "averages.csv")


def snapshot_commit() -> str:
    f = RAW / "SNAPSHOT_COMMIT"
    return f.read_text().strip() if f.exists() else "unknown"


def finite(x, default=None):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default
