# Lodestar 2026 — midterm forecast, live tracker and DeepSeek analyst

Lodestar is a probabilistic forecast of every 2026 U.S. Senate, House and governor
race. It's built on the **20 best-rated pollsters actively polling this cycle**, adds
fundamentals and handicapper ratings, and simulates the election 40,000 times with
correlated errors. A dashboard shows the forecast and, on election night, updates it
from live results. An AI analyst powered by **DeepSeek** (run locally through Ollama,
or through the DeepSeek API when deployed) explains it all, but never makes up a number.

**Locked forecast (as of 2026-11-02, re-locked 16:08 UTC after an Alaska ranked-choice fix; 40,000 simulations):**

| | Democrats | Republicans | Median D seats (80% range) |
|---|---:|---:|---:|
| Senate control (51 for D; VP breaks ties for R) | **74.8%** | 23.9% | 53 (48–56) |
| House control (218) | **87.4%** | 12.5% | 236 (215–256) |
| Governorships majority (26 of 50) | **79.7%** | 12.8% | 28 (24–32) |
| Both chambers | **72.8%** | 10.3% | |

Lock hashes (in `output/locked/LOCK.txt`). Post them somewhere public before polls close so
anyone can check the forecast wasn't changed afterward:

```
forecast    7e5771e66ef971a06d33d7cdc004ded52e7900c70f6c76acb5c883f27d638a68  forecast_2026-11-02.json
predictions c3ba904ee3a4e4f2055abcf8c2a2a0e7a76f76a3526b3abb547823b6997664ac  predictions_2026-11-02.csv
```

Check them at any time with `python -m lodestar.lock --verify` (or `GET /api/lock` on a
running server). Re-locking the same bytes is a no-op; replacing a published lock requires
`--force`.

---

## Quick start (no AI model needed)

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
python -m lodestar.forecast            # ~20 s: writes output/forecast.json
uvicorn server.app:app --port 8000     # open http://localhost:8000
```

The forecast is already built and hash-locked in `output/`, so you can skip the
`lodestar.forecast` step to serve the published numbers as-is. Without a model
configured, the AI analyst tab answers from deterministic templates.

## Run the DeepSeek analyst locally (Ollama)

```bash
# 1. install Ollama: https://ollama.com/download
ollama pull deepseek-r1:14b            # ~9 GB, needs 16-32 GB RAM. deepseek-r1:8b for 16 GB laptops.
# 2. start Lodestar; LLM_PROVIDER=auto finds the running Ollama by itself
uvicorn server.app:app --port 8000
```

`deepseek-v4-flash` also runs through Ollama but needs a ~256 GB-RAM workstation. Any
OpenAI-compatible server hosting DeepSeek weights (vLLM, llama.cpp `llama-server`, LM Studio)
works with `LLM_PROVIDER=openai_compat LLM_BASE_URL=http://host:port/v1 LLM_MODEL=<name>`.

`LLM_PROVIDER=auto` is the default: it uses `DEEPSEEK_API_KEY` when a real key is present,
otherwise a local Ollama, otherwise the templates. A placeholder copied from `.env.example`
counts as "not configured" rather than failing every call.

## Deploy with the DeepSeek API

```bash
cp .env.example .env                   # set DEEPSEEK_API_KEY (platform.deepseek.com)
docker compose up --build              # http://localhost:8000
```

`.env` ships `LLM_PROVIDER=auto` and `DEEPSEEK_MODEL=deepseek-v4-flash`; switch to
`deepseek-v4-pro` for harder questions. If a model name is retired, the client falls back
through the provider's candidates once and remembers the one that answered. To run fully
self-contained with a local model instead: `docker compose --profile local up --build`
(adds an Ollama container; set `LLM_PROVIDER=ollama` and
`OLLAMA_BASE_URL=http://ollama:11434/v1` in `.env`, then
`docker compose exec ollama ollama pull deepseek-r1:14b`).

Confirm whichever route you chose before relying on the analyst tab:

```bash
python scripts/check_agent.py                      # prints the resolved provider, pings it, asks one grounded question
python scripts/check_agent.py "Who wins Maine?"
```

The image is a plain FastAPI app on port 8000, so it deploys anywhere that runs containers
(Fly.io, Render, Railway, Cloud Run, a VPS). Keep the API key in the platform's secret store.

## Deploy to Vercel

Vercel's **FastAPI preset** is zero-config: it looks for a `FastAPI` instance named
`app` in `app.py`, `index.py`, `server.py`, `main.py`, `wsgi.py` or `asgi.py` (or the
same names in `src/` or `app/`). This repo's `main.py` re-exports `server.app:app`, so
importing the repository is enough — Vercel detects FastAPI, routes every path to the
app, and deploys it as one Function. `vercel.json` only raises `maxDuration` to 60 s so a
cold civicAPI query has room.

Set the environment variables in the project:

| variable | value |
|---|---|
| `LIVE_SOURCE` | `civicapi` for live results (default `none` serves the locked forecast) |
| `DEEPSEEK_API_KEY` | optional; the analyst runs from templates without it |
| `LLM_PROVIDER` | `auto` (default) |

Keep the **FastAPI** application preset. If Vercel shows a warning about a
`services` block, the deployment is using an older config — this repo no longer
declares one.

What changes on a serverless host:

- there is **no background poller** (it is disabled when Vercel's `VERCEL` env is set);
  `GET /api/live` refreshes on demand when its numbers are older than
  `LIVE_AUTOREFRESH_SECONDS` (default 15 s), which is cheap once civicAPI is warm;
- the bundle is read-only, so `live.json` / `live_state.json` go to `/tmp` and are
  per-instance (civicAPI refreshes replace the whole snapshot, so instances converge);
- `POST /api/results` is therefore instance-local, not a shared board;
- civicAPI's first **uncached** query takes ~10–15 s while warm calls take ~0.2 s (the
  client requests gzip). On Vercel the feed allows 30 s for that first query with a
  single attempt, so the first live poll can take ~15 s and later ones on the same
  instance are instant; raise `CIVICAPI_TIMEOUT` if you change the Function budget;
- `/api/forecast` is ~1 MB, comfortably under Vercel's response limit.

For uninterrupted 30-second polling on election night, the container deployment is the
better host; Vercel is fine for the forecast, the analyst, and on-demand live updates.

## Election night

In the dashboard's **Live results** tab, pick a source:

- **civicAPI (real results)** — free and keyless, with counts from secretaries of state and race
  calls. Verified against this forecast: all **506 of 506** races match by id, including the
  Ohio and Florida Senate specials. Lodestar polls it every 30 s (`LIVE_POLL_SECONDS`).
  Attribution is required: "Results and race calls: civicAPI (civicapi.org)".
  Or start with `LIVE_SOURCE=civicapi`.
- **Demo night**: a simulated night drawn from the forecast, for rehearsing.
- **Paste results**: paste any text (a chyron, a tweet, a results page). DeepSeek turns it into
  rows; you review them before they're applied.
- **API**: `POST /api/results` with
  `[{"race_id": "senate:ME", "d_votes": 412000, "r_votes": 398000, "pct_reporting": 81, "status": "counting"}]`
  (`status`: `counting` | `called` | `awaiting`; `winner`: `D` | `R` | `IND` when called).

The dashboard fetches `/api/live` on page load, every 15 seconds, and whenever you open
the **Live results** tab; the server pulls civicAPI whenever its own snapshot is older
than `LIVE_AUTOREFRESH_SECONDS` (15 s), so a visitor sees results at most a few seconds
behind the feed.

On each refresh, the live engine:
1. Nowcasts each race by blending the counted margin with the expected margin of the uncounted vote.
2. Learns the night's shared polling error from races that are nearly fully counted.
3. Shifts races still out by that error and re-simulates control of each chamber.

The night itself is treated as a system that can fail:

- the two civicAPI searches run **concurrently** and the client asks for **gzip**, so a
  warm refresh is ~0.2 s instead of ~14 s (418 KB uncompressed), and each request retries
  with backoff (`CIVICAPI_RETRIES`, `CIVICAPI_TIMEOUT`);
- refreshes never overlap, and results are mutated under one lock, so a slow fetch cannot race
  the poller or a `POST /api/results`;
- results are written to `output/live_state.json` after every refresh, so a crash or redeploy
  resumes with the night's returns instead of an empty board;
- a candidate is matched by surname, then by party when the surname is unknown, so votes are
  not dropped into "other";
- `/api/live` reports `age_seconds` / `stale` and `/api/health` reports the lock check, so the
  dashboard says "Live results (stale)" instead of showing old odds as current.

If you know the count order in a state (for example, late mail that leans Democratic), either set
`LiveParams.count_order_bias` in code or post it with the source:
`POST /api/live/source {"source":"civicapi","count_order_bias":{"ME":1.5}}`.

## Testing the model after Tuesday

```bash
python -m lodestar.evaluate --from-live output/live.json        # straight from election-night calls
python -m lodestar.evaluate --results results_2026.csv           # or from final certified results
```

`results_2026.csv` has the columns `race_id,d_pct,r_pct,winner`. `race_id` uses the forecast's
ids (`senate:OH-special`, `house:PA-07`, `governor:AZ`), and d/r follow the forecast's D-side and
R-side nominees. The scorecard grades five models side by side:

| model | what it is |
|---|---|
| `lodestar` | the full model (published probabilities) |
| `top20_only` | the same pipeline using **only** Top-20-pollster polls |
| `polls_simple` | unweighted 30-day poll average, RealClearPolitics-style |
| `ratings_only` | handicapper consensus alone |
| `fundamentals_only` | partisan lean + national environment + incumbency + money |

Metrics: Brier score, log loss, accuracy, margin MAE/RMSE/bias, 80%-interval coverage and a
calibration table. A good probabilistic forecast isn't one that "calls every race": favorites
given 70% should lose about 3 times in 10.

## Refreshing before the lock

The data snapshot is in `data/raw/` (`SNAPSHOT_COMMIT` records its source revision). To re-run
with newer polls before polls close:

```bash
git clone --depth 1 https://github.com/KaraZajac/MIDTERMS2026 /tmp/m26
cp /tmp/m26/data/polls/races.csv data/raw/polls_races.csv
cp /tmp/m26/data/polls/generic.csv data/raw/polls_generic.csv
cp /tmp/m26/data/races/{senate,governor,house}.csv /tmp/m26/data/{ratings,finance,averages}.csv data/raw/
python -m lodestar.forecast && python -m lodestar.lock --force && python scripts/build_static.py
```

`python -m lodestar.lock` refuses to replace a published lock; pass `--force` only when you
mean to publish new hashes, then tell people the new ones. `python -m lodestar.lock --verify`
re-hashes the locked files and reports whether anything moved (`GET /api/lock` does the same
for a running server).

Only the `data/` tables are used. No code from that project runs here.

## Layout

```
lodestar/      the model: data.py (load, pollster names, poll variants) · pollsters.py (ratings,
               Top-20 panel, house effects) · averaging.py · fundamentals.py (priors + blend) ·
               simulate.py · forecast.py · live.py · lock.py (immutable forecast lock) ·
               evaluate.py · sources/civicapi.py (live results + demo feed)
agent/         DeepSeek analyst: llm.py (auto / Ollama / DeepSeek API / OpenAI-compatible) ·
               tools.py (deterministic tools) · agent.py (tool loop + grounding check)
server/app.py  FastAPI: dashboard, /api/forecast, /api/live, /api/results, /api/agent/*,
               /api/lock, /api/health; persists night results, never overlaps a refresh
main.py        Vercel entrypoint (re-exports the FastAPI app); vercel.json sets maxDuration
dashboard/     index.html: the whole UI, one file, no external dependencies
scripts/       build_static.py (single-file dist/ with the forecast embedded) ·
               check_agent.py (smoke-test the configured DeepSeek model)
output/        forecast.json, live.json, live_state.json, locked/ (hash-locked forecast + CSV)
tests/         python -m pytest -q   (model, live engine, feed mapping, agent over HTTP, lock,
               server persistence)
```

## API

| method | path | |
|---|---|---|
| GET | `/api/forecast` | full forecast JSON |
| GET | `/api/live` | live nowcast, chamber odds, learned polling error, scorecard, `stale`/`age_seconds` |
| POST | `/api/results` | add or replace result rows |
| POST | `/api/live/source` | `{"source": "none" \| "civicapi" \| "demo", "count_order_bias": {"ME": 1.5}}` |
| POST | `/api/agent/chat` | `{"message": "...", "history": [...]}` returns the answer + tool trace |
| POST | `/api/agent/parse-results` | `{"text": "..."}` returns proposed result rows (not applied) |
| GET | `/api/agent/ping` | checks that the configured DeepSeek model answers |
| GET | `/api/scorecard` | grades the locked forecast against races called so far |
| GET | `/api/lock` | re-hashes the published forecast and reports whether it still matches |

## Data, licenses, attribution

Code: MIT. Derived data in `data/` and `output/` is **CC BY-SA 4.0** because it builds on
Wikipedia content (see `DATA-NOTICE.md`). Pollster ratings © Silver Bulletin (free to use with
attribution) and FiveThirtyEight (CC BY 4.0). Live results come from civicAPI (attribution required).
See `METHODOLOGY.md` for the full algorithm.
