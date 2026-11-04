"""Lodestar server: dashboard + forecast API + live results + DeepSeek analyst.

Run:  uvicorn server.app:app --host 0.0.0.0 --port 8000
Env:  LLM_PROVIDER=auto|ollama|deepseek|openai_compat|none
      DEEPSEEK_API_KEY=...   LIVE_SOURCE=none|civicapi|demo   LIVE_POLL_SECONDS=30

Election-night behaviours that matter here:
  * results survive a restart (output/live_state.json), so a crashed or redeployed
    process does not lose the night's returns;
  * results are mutated under one lock and refreshes never overlap, so a slow
    civicAPI call cannot race the poller or an incoming POST /api/results;
  * /api/live reports the age of its numbers, and /api/health reports the lock
    check, so the dashboard can say "stale" instead of silently showing old odds.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import Analyst, LLM, State
from lodestar.evaluate import evaluate
from lodestar.live import LiveParams, run_live
from lodestar.sources import civicapi as feeds

ROOT = Path(__file__).resolve().parent.parent
FORECAST = Path(os.getenv("FORECAST_PATH", ROOT / "output" / "forecast.json"))
DASH = ROOT / "dashboard" / "index.html"
# Serverless platforms mount the bundle read-only; /tmp is the only writable place.
# Docker and local runs keep writing next to the other output artifacts.
ON_VERCEL = bool(os.getenv("VERCEL") or os.getenv("NOW_REGION"))
WRITABLE = Path("/tmp") if ON_VERCEL else ROOT / "output"
LIVE_OUT = Path(os.getenv("LIVE_OUT_PATH", WRITABLE / "live.json"))
STATE_OUT = Path(os.getenv("LIVE_STATE_PATH", WRITABLE / "live_state.json"))
POLL_SECONDS = int(os.getenv("LIVE_POLL_SECONDS", "30"))
# civicAPI has taken 20-60 s per query tonight, so a 15 s cache would start a new
# slow refresh before the previous one could be served. Cache for two minutes on
# serverless and mark the feed stale only after five.
STALE_AFTER = float(os.getenv("LIVE_STALE_SECONDS", "600" if ON_VERCEL else str(max(90, 3 * POLL_SECONDS))))
AUTOREFRESH_SECONDS = int(os.getenv("LIVE_AUTOREFRESH_SECONDS", "180" if ON_VERCEL else str(min(POLL_SECONDS, 15))))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_atomic(path: Path, text: str) -> None:
    """Readers (and a restart) see the old file or the new one, never half of each."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _load_forecast(path: Path) -> dict:
    """Read the served forecast, falling back to the locked copy.

    Bundled deployments (Vercel, slim containers) sometimes ship only one of the two;
    the locked file is the same bytes, so serving it is always correct.
    """
    for candidate in [path, *sorted((ROOT / "output" / "locked").glob("forecast_*.json"))]:
        if candidate.exists():
            return json.loads(candidate.read_text())
    raise RuntimeError(f"no forecast found at {path} or output/locked/ — run python -m lodestar.forecast")


class Hub:
    def __init__(self):
        self.lock = threading.RLock()
        self.forecast = _load_forecast(FORECAST)
        self.results: dict = {}
        self.live: dict | None = None
        self.source = os.getenv("LIVE_SOURCE", "none")
        self.demo_progress = 0.0
        self.last_fetch: float | None = None
        self.last_error: str | None = None
        self.fetch_meta: dict = {}
        self._refresh_guard = threading.Lock()
        self.params = LiveParams()
        self.llm = LLM()
        self.state = State(self.forecast)
        self.analyst = Analyst(self.state, self.llm)
        self.lock_report: dict = {}
        self._load_state()

    # ---------------------------------------------------------------- state io
    def _load_state(self):
        if not STATE_OUT.exists():
            return
        try:
            saved = json.loads(STATE_OUT.read_text())
        except (OSError, json.JSONDecodeError):
            return
        self.results = saved.get("results") or {}
        self.source = saved.get("source") or self.source
        self.demo_progress = float(saved.get("demo_progress") or 0.0)
        self.last_fetch = saved.get("last_fetch_epoch")
        self.fetch_meta = saved.get("fetch_meta") or {}
        if self.results:
            self.recompute()

    def _save_state(self):
        payload = {"source": self.source, "results": self.results, "demo_progress": self.demo_progress,
                   "last_fetch_epoch": self.last_fetch, "fetch_meta": self.fetch_meta, "saved_utc": _now()}
        try:
            _write_atomic(STATE_OUT, json.dumps(payload))
        except OSError as e:  # a read-only volume must not take the night down
            self.last_error = f"could not persist live state: {type(e).__name__}: {e}"

    # ---------------------------------------------------------------- compute
    def recompute(self):
        with self.lock:
            results = dict(self.results)
        try:
            live = run_live(self.forecast, results, self.params) if results else None
        except Exception as e:  # keep the previous odds rather than serving a 500
            self.last_error = f"live model failed: {type(e).__name__}: {e}"
            return
        if live:
            live["source"] = self.source
            live["updated"] = _now()
            live["age_seconds"] = 0
            if self.source == "civicapi":
                live["attribution"] = feeds.ATTRIBUTION
            if self.fetch_meta:
                live["fetch"] = self.fetch_meta
            try:
                _write_atomic(LIVE_OUT, json.dumps(live))
            except OSError:
                pass
        with self.lock:
            self.live = live
            self.state.live = live

    def refresh(self):
        """Fetch the configured feed. Never overlaps an in-flight refresh."""
        if not self._refresh_guard.acquire(blocking=False):
            return
        try:
            meta = {}
            if self.source == "civicapi":
                fresh, meta = feeds.civicapi_with_meta(self.forecast)
                with self.lock:
                    self.results.update(fresh)
            elif self.source == "demo":
                self.demo_progress = min(1.0, self.demo_progress + 0.04)
                with self.lock:
                    self.results = feeds.demo(self.forecast, self.demo_progress)
            with self.lock:
                self.fetch_meta = meta
                self.last_fetch = time.time()
                # Partial coverage is not a failure: keep the results, but surface
                # which query is lagging so the UI can say so.
                self.last_error = ("partial: " + "; ".join(meta.get("errors") or [])[:280]) if meta.get("partial") else None
            self.recompute()
            self._save_state()
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
        finally:
            self._refresh_guard.release()

    # ---------------------------------------------------------------- health
    def set_source(self, source: str):
        with self.lock:
            changed = source != self.source
            if changed or source == "none":
                # Never blend feeds: demo results surviving into a civicAPI count
                # would look like real votes.
                self.results, self.live, self.fetch_meta = {}, None, {}
                self.state.live = None
            self.source = source
        if source == "none":
            self._save_state()
        else:
            self.refresh()

    def live_age(self) -> float | None:
        if not self.last_fetch:
            return None
        return round(time.time() - self.last_fetch, 1)

    def ensure_fresh(self) -> None:
        """Pull the feed when the numbers are older than AUTOREFRESH_SECONDS.

        Docker/local runs already poll in the background, so this normally does
        nothing. On a serverless host it is the only refresh path: a request for
        /api/live pays for the fetch (well under a second once civicAPI is warm)
        instead of the page waiting for a poller that does not exist.
        """
        if self.source == "none":
            return
        age = self.live_age()
        if age is not None and age < AUTOREFRESH_SECONDS:
            return
        self.refresh()

    def status(self) -> dict:
        report = self.lock_report or {}
        return {
            "ok": True, "model": self.forecast["meta"]["model"], "version": self.forecast["meta"]["version"],
            "as_of": self.forecast["meta"]["as_of"], "llm": self.llm.describe(), "live_source": self.source,
            "results_loaded": len(self.results), "last_fetch": self.last_fetch, "last_error": self.last_error,
            "live_age_seconds": self.live_age(), "stale": bool(self.last_fetch and (self.live_age() or 0) > STALE_AFTER),
            "attribution": feeds.ATTRIBUTION if self.source == "civicapi" else None,
            "fetch": self.fetch_meta, "lock": {k: report.get(k) for k in ("ok", "locked_utc", "forecast_sha256")} if report else None,
        }


HUB = Hub()


def _verify_lock():
    """Cheap integrity check of the published forecast; never fatal."""
    try:
        from lodestar.lock import verify as verify_lock
        HUB.lock_report = verify_lock(FORECAST)
    except Exception as e:  # noqa: BLE001 - a missing lock must not stop the server
        HUB.lock_report = {"ok": False, "error": f"{type(e).__name__}: {e}"}


async def poller():
    while True:
        if HUB.source in ("civicapi", "demo"):
            await asyncio.to_thread(HUB.refresh)
        await asyncio.sleep(POLL_SECONDS if HUB.source == "civicapi" else 6)


@asynccontextmanager
async def lifespan(app):
    _verify_lock()
    tasks = []
    # A serverless invocation must not leave a polling loop running: /api/live
    # refreshes on demand there instead (see Hub.ensure_fresh).
    if not ON_VERCEL:
        tasks.append(asyncio.create_task(poller()))
        if HUB.source != "none":
            tasks.append(asyncio.create_task(asyncio.to_thread(HUB.refresh)))
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="Lodestar 2026", version="1.0.0", lifespan=lifespan)

# Candidate portraits live in public/ so Vercel serves them straight from its CDN;
# this mount is what makes them work locally and in the container.
PUBLIC = ROOT / "public"
if (PUBLIC / "portraits").is_dir():
    app.mount("/portraits", StaticFiles(directory=PUBLIC / "portraits"), name="portraits")


@app.get("/")
def dashboard():
    for candidate in (DASH, ROOT / "dist" / "index.html"):
        if candidate.exists():
            return FileResponse(candidate)
    raise HTTPException(503, "dashboard file missing from this bundle")


@app.get("/api/health")
def health():
    return HUB.status()


@app.get("/api/forecast")
def forecast():
    return JSONResponse(HUB.forecast)


@app.get("/api/lock")
def lock_status():
    _verify_lock()
    return HUB.lock_report


@app.get("/api/live")
def live():
    HUB.ensure_fresh()
    if not HUB.live:
        return {"status": "no results yet", "source": HUB.source, "last_error": HUB.last_error,
                "results_loaded": len(HUB.results), "fetch": HUB.fetch_meta,
                "attribution": feeds.ATTRIBUTION if HUB.source == "civicapi" else None}
    out = dict(HUB.live)
    age = HUB.live_age()
    out["age_seconds"] = age
    out["stale"] = bool(age is not None and age > STALE_AFTER)
    out["results_loaded"] = len(HUB.results)
    out["last_error"] = HUB.last_error
    out["fetch"] = HUB.fetch_meta
    return out


class SourceReq(BaseModel):
    source: Literal["none", "civicapi", "demo"]
    demo_progress: float | None = None
    # LiveParams overrides are deliberately limited: state -> points of expected
    # count-order skew for the remaining vote (positive = late vote bluer).
    count_order_bias: dict[str, float] | None = None


@app.post("/api/live/source")
def set_source(req: SourceReq):
    if req.demo_progress is not None:
        HUB.demo_progress = max(0.0, min(1.0, req.demo_progress))
    if req.count_order_bias is not None:
        HUB.params.count_order_bias = {k.upper(): float(v) for k, v in req.count_order_bias.items()}
    HUB.set_source(req.source)
    return HUB.status()


class ResultRow(BaseModel):
    race_id: str
    d_votes: int = 0
    r_votes: int = 0
    other_votes: int = 0
    pct_reporting: float = 0
    status: Literal["counting", "called", "awaiting"] = "counting"     # counting | called | awaiting
    winner: Literal["D", "R", "IND", "OTHER"] | None = None            # when called


@app.post("/api/results")
def post_results(rows: list[ResultRow]):
    ids = {r["id"] for r in HUB.forecast["races"]}
    bad = [r.race_id for r in rows if r.race_id not in ids]
    if bad:
        raise HTTPException(400, f"unknown race ids: {bad[:10]}")
    with HUB.lock:
        for r in rows:
            HUB.results[r.race_id] = r.model_dump()
    HUB.recompute()
    HUB._save_state()
    return {"ok": True, "results_loaded": len(HUB.results),
            "chambers": HUB.live["chambers"] if HUB.live else None}


class ChatReq(BaseModel):
    message: str
    history: list[dict] = []


@app.post("/api/agent/chat")
async def chat(req: ChatReq):
    return await asyncio.to_thread(HUB.analyst.ask, req.message, req.history)


class ParseReq(BaseModel):
    text: str


@app.post("/api/agent/parse-results")
async def parse_results(req: ParseReq):
    return await asyncio.to_thread(HUB.analyst.parse_results, req.text)


@app.get("/api/agent/ping")
async def ping():
    return await asyncio.to_thread(HUB.llm.ping) if HUB.llm.enabled else {"ok": False, "error": HUB.llm.describe()["reason"]}


@app.get("/api/scorecard")
def scorecard():
    """Grade the locked forecast against every race called so far."""
    if not HUB.live:
        return {"status": "no results yet"}
    res = {}
    for rid, nc in HUB.live["races"].items():
        if nc["status"] == "called":
            d, r, o = nc["d_votes"], nc["r_votes"], nc.get("other_votes", 0)
            tot = d + r + o
            res[rid] = {"winner": "D" if nc["p_d"] >= 0.5 else "R", "margin": 100 * (d - r) / tot if tot else None}
    locked = sorted((ROOT / "output" / "locked").glob("forecast_*.json"))
    return evaluate(res, locked[-1] if locked else FORECAST)
