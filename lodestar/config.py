"""Lodestar 2026 — every tunable number in one place.

Each parameter carries the reason it has the value it has. Values marked
[backtest] come from the 2014-2024 Senate/governor polling-error record
(323 races; see data/SOURCES.md); [lit] from the published forecasting
literature / standard practice; [judgment] are calls made for this cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ELECTION_DATE = "2026-11-03"
FORECAST_AS_OF = "2026-11-02"
MODEL_NAME = "Lodestar"
MODEL_VERSION = "1.0.0"


@dataclass(frozen=True)
class PollsterParams:
    top_n: int = 20                    # the "core" panel the user asked for
    min_polls_for_core: int = 3        # a firm must actually be polling 2026 to be core
    active_since: str = "2026-01-01"
    # Quality weight = (sigma_ref / (sigma_ref + ppm))^2: inverse-variance in
    # expected error. ppm = Silver Bulletin Predictive Plus-Minus (lower = better).
    sigma_ref: float = 6.0
    core_multiplier: float = 1.00
    noncore_multiplier: float = 0.50   # non-top-20 polls still count, at half weight
    unrated_ppm: float = 0.55          # unrated firms ~ an average B/C pollster [lit]
    unrated_multiplier: float = 0.70   # ... and a further haircut for no track record
    # House effects: residual vs. consensus, shrunk toward zero
    house_shrink_k: float = 6.0        # pseudo-polls of prior at 0 [judgment]
    house_max: float = 6.0             # cap any single firm's adjustment (points of margin)
    # Partisan-sponsored polls (Wikipedia (D)/(R) flag)
    partisan_shift: float = 1.5        # points of margin against the sponsor's side [lit: internals run 2-4 pts hot]
    partisan_weight: float = 0.60


@dataclass(frozen=True)
class AveragingParams:
    half_life_days_state: float = 16.0   # statewide/district polls
    half_life_days_generic: float = 9.0  # national generic ballot moves faster, polled far more
    window_days: int = 150               # ignore polls older than this
    sample_cap: int = 3000               # n beyond this adds ~nothing (non-sampling error dominates)
    sample_ref: int = 600
    pop_weight: dict = field(default_factory=lambda: {"LV": 1.0, "V": 1.0, "RV": 0.80, "A": 0.55})
    # Trend adjustment: move an old state poll by the national swing since it was taken
    trend_elasticity: float = 0.85
    # Per-poll random error SD on the margin (sampling + design + mode), and the
    # race-level structural floor that more polls cannot average away.
    sd_per_poll: float = 7.0             # [backtest: fitted 8.75 incl. house noise; we remove house effects]
    sd_race_floor: float = 3.0           # [backtest]
    undecided_sd_per_pt: float = 0.30    # each point undecided adds 0.3 pt SD (in quadrature)


@dataclass(frozen=True)
class FundamentalsParams:
    # National weighted presidential margin (0.75*2024 + 0.25*2020, D minus R)
    nat_pres24: float = -1.47
    nat_pres20: float = 4.45
    w24: float = 0.75
    elasticity: dict = field(default_factory=lambda: {"senate": 1.0, "house": 1.0, "governor": 0.65})
    inc_adv: dict = field(default_factory=lambda: {"senate": 2.0, "house": 2.5, "governor": 4.0})
    appointed_inc_adv: float = 0.5       # appointed senators historically get little [lit]
    redrawn_inc_factor: float = 0.5      # incumbents running on new 2025-26 lines get half
    redrawn_states: tuple = ("AL", "CA", "FL", "LA", "MO", "NC", "OH", "TN", "TX", "UT")
    independent_bonus: dict = field(default_factory=lambda: {"NE": 7.0, "default": 4.0})
    finance_coef: float = 0.9            # points per log(D receipts / R receipts) [judgment, capped]
    finance_cap: float = 2.5
    # governors decouple from federal partisanship, so their fundamentals are
    # deliberately loose: polls and handicappers carry those races
    sd_fund: dict = field(default_factory=lambda: {"senate": 7.0, "house": 6.5, "governor": 11.0})


@dataclass(frozen=True)
class RatingsParams:
    # Only qualitative handicappers; quantitative models (DDHQ, Silver Bulletin,
    # Economist, FiftyPlusOne, Split Ticket) are excluded so Lodestar stays an
    # independent forecast and does not double-count the same polls.
    raters: tuple = ("Cook", "Sabato", "Inside Elections", "NBC News", "Fox News",
                     "CBS News", "ABC News", "MS NOW", "RCP")
    rater_weight: dict = field(default_factory=lambda: {"Cook": 1.0, "Sabato": 1.0, "Inside Elections": 1.0})
    other_rater_weight: float = 0.5
    # rating scale -3..3 -> implied margin (D minus R) [lit: historical average margins by category]
    scale_to_margin: dict = field(default_factory=lambda: {0: 0.0, 1: 4.5, 2: 10.0, 3: 20.0})
    sd_rating: dict = field(default_factory=lambda: {"senate": 7.0, "house": 6.0, "governor": 7.5})
    prior_corr_inflation: float = 1.20   # fundamentals and ratings are not independent


@dataclass(frozen=True)
class SimParams:
    n_sims: int = 40000
    seed: int = 20261103
    t_df: float = 5.0                     # fat tails
    sd_national: float = 3.6              # [backtest: midterm-like cycle-level miss SD 3.59]
    sd_region: float = 2.0
    sd_state: float = 1.5
    sd_hispanic: float = 4.0              # scaled by state Hispanic share (0-0.5)
    sd_black: float = 3.0                 # scaled by state Black share
    sd_chamber: dict = field(default_factory=lambda: {"senate": 1.0, "house": 1.2, "governor": 1.6})
    sd_district_extra: float = 2.5        # House districts: idiosyncratic beyond the prior
    national_bias: float = 0.0            # NO directional correction — see METHODOLOGY.md


@dataclass(frozen=True)
class Params:
    pollsters: PollsterParams = field(default_factory=PollsterParams)
    averaging: AveragingParams = field(default_factory=AveragingParams)
    fundamentals: FundamentalsParams = field(default_factory=FundamentalsParams)
    ratings: RatingsParams = field(default_factory=RatingsParams)
    sim: SimParams = field(default_factory=SimParams)


PARAMS = Params()

# Seats not on the ballot (119th Congress / governors), and control rules.
CHAMBERS = {
    "senate": {"majority": 51, "tie_breaker": "R", "not_up": {"D": 34, "R": 31},
               "note": "D total includes Sanders (VT) and King (ME), who caucus with Democrats. "
                       "Vice President Vance breaks 50-50 ties for Republicans."},
    "governor": {"majority": 26, "not_up": {"D": 6, "R": 8}},
    "house": {"majority": 218, "seats": 435},
}

REGIONS = {
    "New England": ["ME", "NH", "VT", "MA", "RI", "CT"],
    "Mid-Atlantic": ["NY", "NJ", "PA", "DE", "MD", "DC"],
    "Great Lakes": ["OH", "MI", "WI", "MN", "IA", "IL", "IN"],
    "Plains": ["KS", "NE", "SD", "ND", "MO", "OK"],
    "South": ["GA", "NC", "SC", "FL", "VA", "AL", "MS", "LA", "AR", "TN", "KY", "WV", "TX"],
    "Mountain": ["AZ", "NV", "CO", "NM", "UT", "ID", "MT", "WY"],
    "Pacific": ["CA", "OR", "WA", "AK", "HI"],
}

# Approximate shares of the voting-age population (Census ACS 2023, rounded).
# Used only to scale two demographic polling-error factors in the simulation.
HISPANIC_SHARE = {
    "NM": .45, "CA": .36, "TX": .36, "AZ": .28, "NV": .26, "FL": .25, "CO": .20, "NJ": .19,
    "NY": .17, "IL": .16, "CT": .15, "RI": .15, "UT": .13, "OR": .12, "WA": .12, "ID": .12,
    "KS": .11, "MA": .11, "OK": .10, "NE": .10, "DE": .09, "MD": .09, "GA": .09, "HI": .09,
    "VA": .09, "NC": .09, "WY": .09, "PA": .08, "AR": .07, "IA": .06, "IN": .07, "WI": .07,
    "MI": .05, "MN": .05, "SC": .05, "LA": .05, "AK": .07, "TN": .05, "AL": .04, "MO": .04,
    "OH": .04, "SD": .04, "NH": .04, "MS": .03, "KY": .04, "MT": .04, "ND": .04, "ME": .02,
    "VT": .02, "WV": .02,
}
BLACK_SHARE = {
    "MS": .37, "LA": .32, "GA": .32, "MD": .30, "SC": .26, "AL": .26, "NC": .21, "DE": .22,
    "VA": .19, "TN": .16, "FL": .15, "AR": .15, "NY": .15, "IL": .14, "MI": .14, "NJ": .13,
    "OH": .12, "TX": .12, "PA": .11, "MO": .11, "CT": .11, "IN": .09, "KY": .08, "NV": .09,
    "MA": .08, "OK": .07, "WI": .06, "KS": .06, "MN": .07, "RI": .06, "CA": .06, "NE": .05,
    "WV": .04, "CO": .04, "AZ": .05, "WA": .04, "IA": .04, "AK": .03, "NM": .02, "OR": .02,
    "HI": .02, "NH": .02, "UT": .01, "ME": .02, "SD": .02, "ND": .03, "VT": .01, "ID": .01,
    "MT": .01, "WY": .01,
}

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


def region_of(state: str) -> str:
    for name, states in REGIONS.items():
        if state in states:
            return name
    return "Other"
