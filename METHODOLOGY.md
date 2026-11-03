# Lodestar 2026 — methodology

Margins are always **Democratic-side minus Republican-side**, in points. Where an
independent is the main anti-Republican candidate (Nebraska, Montana, Idaho, South Dakota),
that independent is the "D-side".

## 1. The poll ledger

There are 1,504 nominee-matched head-to-head polls for 179 races and 541 generic-ballot
polls, current through 2026-11-01. They were compiled from Wikipedia's race pages and
VoteHub (snapshot of the CC BY-SA `data/` tables of github.com/KaraZajac/MIDTERMS2026;
none of that project's code is used).

Wikipedia often lists several versions of one poll. Lodestar keeps exactly one row per poll:

1. **Best population:** LV > V > RV > A.
2. **The most "ballot-like" version:** the one with the largest decided D+R share. This is the
   ranked-choice final round in Alaska and Maine, or the leaners-pushed version.
3. **Alaska first-choice-only polls are dropped** when a race has final-round polls. In the
   governor's race, first choices split the Republican vote three ways (42–25) while final rounds
   run 52–48 to 55–45, so mixing them overstated the Democrat by about 5 points.

Pollster names are canonicalized (for example, "New York Times/Siena University" becomes
"The New York Times/Siena College", and "KSTP/SurveyUSA" becomes "SurveyUSA"), so a firm's
polls and its rating line up.

## 2. The Top-20 panel

Ratings come from **Silver Bulletin's January 2026 pollster ratings** (Predictive Plus-Minus,
or PPM: the expected error relative to an average pollster, lower is better). Firms Silver
Bulletin doesn't rate fall back to **FiveThirtyEight's final 2024 ratings**, linearly rescaled
onto the PPM scale using firms rated by both.

The **core panel** is the 20 lowest-PPM firms that are *actually polling 2026* (at least 3
polls this year, not banned). Ranking all 540 rated firms would include pollsters that have
stopped polling, and they add nothing to a 2026 average.

Each poll's quality weight is inverse-variance in expected error:

    w_quality = (6 / (6 + PPM))^2          # PPM -1.09 → 1.49 ;  PPM +1.0 → 0.73

| tier | multiplier on w_quality |
|---|---|
| Top-20 core | 1.00 |
| other rated firms | 0.50 |
| unrated firms | 0.35 |
| banned | 0 |

In practice a core poll counts about 2–3× a rated non-core poll of the same size and age.
The core panel conducted **22%** of race polls but carries **51%** of the weight.

**Why not use only the Top 20?** In many states the panel polled once or never. A Top-20-only
average there is one poll's noise. The core-only variant is computed anyway, locked and graded
after the election, so the choice can be tested rather than asserted.

## 3. Adjustments to each poll

- **Partisan sponsor** (Wikipedia's (D)/(R) flag): 1.5 points against the sponsor, and the poll
  counts at 0.6× weight. Internal and allied polls historically run 2–4 points hot.
- **House effects:** for each firm, residual = its polls minus the weighted average of *other*
  firms in the same races (generic-ballot polls are compared with the daily smoothed average).
  The effect is shrunk toward zero, `h = Σresid / (n + 6)`, capped at ±6, and subtracted from the
  firm's polls. Alternating averages and house effects converges in 3 iterations.
- **National trend:** a state poll taken when the generic ballot was D+6 is moved by
  0.85 × elasticity × (today's generic ballot − the generic ballot then).

## 4. Averages

    w = w_quality × tier × sqrt(min(n, 3000)/600) × population (LV 1, RV 0.8, A 0.55)
        × sponsor (0.6) × 0.5^(age/16 days) / sqrt(# polls by the firm in that race, last 60 days)

- The generic ballot uses a 9-day half-life over a 45-day window.
- The average's own uncertainty is `sqrt(7^2 / n_eff + 3^2)` points: per-poll noise averages
  away, but a race-level floor doesn't.
- Undecideds add `0.3 × undecided%` to the race's standard deviation.

**Generic ballot (Lodestar average): D+8.7** (D 51.4 – R 43.2).

## 5. Priors

**Fundamentals**

    μ_fund = elasticity × (lean + environment) + incumbency + money

- `lean` = 0.75 × 2024 + 0.25 × 2020 presidential margin, relative to the nation. House
  districts use their 2026 lines.
- `environment` = the generic-ballot average (D+8.7).
- `elasticity` is 1.0 for Senate and House and 0.65 for governors, who decouple from federal
  partisanship.
- **Incumbency:** Senate ±2.0, House ±2.5, governors ±4.0. Appointed senators get ±0.5. House
  incumbents on new 2025–26 lines get half.
- **Personal vote:** where an incumbent's last margin diverged from their state's presidential
  vote, the term becomes `0.5 × generic + w × (personal over/under-performance)`, with
  w = 0.40 Senate and 0.35 governors, capped at 8 and 18 points. Collins (ME, +17.7 R
  over-performance in 2020) and Phil Scott (VT, +83) are why this exists. Without it,
  partisanship alone calls both races badly wrong.
- **Money:** `0.9 × ln(D receipts / R receipts)`, capped at ±2.5 (FEC filings). This is mostly a
  candidate-quality signal.
- **Independents:** base × 0.8, plus +7 for Osborn (NE, ran 14 points ahead of the 2024
  presidential baseline) and +4 for others.

SD: Senate 7, House 6.5, governor 11 (deliberately loose).

**Handicappers.** Only *qualitative* raters are used: Cook, Sabato, Inside Elections (weight 1)
and NBC, Fox, CBS, ABC, MS NOW and RCP (weight 0.5). Quantitative models (Silver Bulletin,
DDHQ, the Economist, FiftyPlusOne, Split Ticket) are excluded so Lodestar stays independent
and doesn't double-count the same polls. Ratings map to margins as Toss-up 0, Lean ±4.5,
Likely ±10, Safe ±max(20, |μ_fund|). SD 7 (House 6).

The two priors are precision-weighted, and their SD is inflated 1.2× because they aren't
independent.

## 6. Posterior

    μ = (μ_prior/σ_prior² + poll_avg/σ_polls²) / (1/σ_prior² + 1/σ_polls²)

In well-polled races, polls get between two-thirds and three-quarters of the weight. The prior pulls hardest on
thinly-polled, deep-red races where Democratic candidates poll close (Louisiana, Mississippi):
red-state Democratic polling leads have a long history of evaporating (Kansas 2014 and 2020,
for example).

## 7. Simulation (40,000 draws)

    margin_i = μ_i + λ_i·e_national + e_region + e_state + h_i·e_hispanic + b_i·e_black + e_chamber + σ_i·t5

| component | SD | why |
|---|---|---|
| national (Student-t, 5 df) | 3.6 | measured cycle-level miss in midterm-like years, 2014–2025 |
| region (7) | 2.0 | Rust Belt / Sun Belt style correlated misses |
| state | 1.5 | Senate, governor and House races in one state share errors |
| Hispanic share × factor | 4.0 | 2020–24 misses concentrated among Hispanic voters |
| Black share × factor | 3.0 | |
| chamber | 1.0–1.6 | governor races share candidate-driven error |
| race (t, 5 df) | posterior SD | |

**No directional correction.** In the 2014–2025 Senate and governor record, polls overstated
Democrats in 2014, 2016, 2020, 2022 and 2024 (by 2–7 points on the margin), were accurate in
2018, and *understated* Democrats by about 7 points in the 2025 off-year governor races.
Two mechanisms argue against assuming 2024's pro-Democratic bias carries over:

- Midterm electorates are the high-propensity voters polls reach best.
- Weighting on recalled 2024 vote can over-represent Trump-coalition voters who skip midterms.

So Lodestar widens the national error instead of guessing its sign. This is the single most
consequential judgment in the model: if this cycle's polls carry the usual 3-point pro-Democratic
bias, the Senate odds fall from 75% to about 45% (see the sensitivity curve).

**Chamber math.** The Senate has 34 Democratic/independent seats not up (Sanders and King caucus
with Democrats) and 31 Republican. Democrats need 51; the Vice President gives Republicans
control at 50. Independent winners (Osborn and others) are *not* counted for either party, so
"independents hold the balance" is reported separately. Governors: 6 D and 8 R not up. House:
218, with 17 seats fixed (uncontested, or same-party top-two).

The model also exports tipping points (the race that delivers the 51st seat in each simulation)
and a sensitivity curve (control odds if every race misses by the same −8…+8 points).

## 8. Election night (`lodestar/live.py`)

**Per race:** with share p of expected vote counted and counted margin m,

    remaining margin  ~ N(√p·m + (1−√p)·μ_prior + count_order_bias, sd² = ((1−√p)σ)² + 10²)
    final             = p·m + (1−p)·remaining

The remaining share itself is uncertain: ±(0.35·(1−p) + 1.5 points).

**Shared error:** races at least 90% counted (or called with at least 75%) give a posterior
for the national error, with prior SD 3.6, and for each region's extra error. Uncounted races
shift by the posterior means, and the chambers are re-simulated with the reduced variance.

At 0% reporting the live engine reproduces the pre-election odds (tested).

## 9. The DeepSeek analyst (`agent/`)

The language model is the *analyst*, never the *forecaster*. It answers through seven
deterministic tools that read the model's outputs: `chamber_summary`, `list_races`, `get_race`,
`get_polls`, `pollster`, `what_if` and `live_status`. It speaks a plain-JSON tool protocol that
works the same on the DeepSeek API, Ollama-served R1 distills and vLLM. Reasoning traces
(`<think>` blocks or `reasoning_content`) are stripped.

A grounding check extracts every percentage in the answer and rejects any that doesn't
appear in a tool result. The model gets one rewrite; after that, the answer is flagged. LLMs are
poorly calibrated numeric forecasters, so this split keeps the numbers honest and the prose
useful.

## 10. Known limits

- Parameters come from the published 2014–2024 polling-error record and standard practice,
  not from an end-to-end backtest of this exact code, which is the main open item.
- Ranked-choice and multi-candidate races are reduced to the two leading sides.
- Most House districts have no public polls. They rest on lean, incumbency, money and
  handicappers, with an extra 2.5 points of SD.
- Wikipedia-compiled polls can contain transcription errors. Spot checks against
  RealClearPolitics' generic-ballot list and news reports matched.
- "World's most accurate" can't be promised in advance by any model. What this one does
  promise is calibrated probabilities, locked before the vote and graded in public against
  four simpler baselines.
