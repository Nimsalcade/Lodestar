"""Election-night and DeepSeek-agent tests that exercise real I/O paths.

The agent tests run a throwaway OpenAI-compatible server, so they prove the actual
code path the DeepSeek API and Ollama use (the openai SDK, JSON mode, model
fallback) rather than a scripted fake object. Everything listens on 127.0.0.1 and
is torn down afterwards.
"""
from __future__ import annotations

import importlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import Analyst, LLM, LLMConfig, State  # noqa: E402
from lodestar import lock as lockmod  # noqa: E402
from lodestar.sources import civicapi  # noqa: E402

FORECAST = json.loads((ROOT / "output" / "forecast.json").read_text())


# --------------------------------------------------------------- mock provider
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # Ollama probe + a scripted JSON endpoint for the feed retry test
        srv = self.server
        if self.path.startswith("/api/tags"):
            self._send(200, {"models": [{"name": "deepseek-r1:14b"}]})
            return
        with srv.mu:  # type: ignore[attr-defined]
            if srv.fail_times > 0:  # type: ignore[attr-defined]
                srv.fail_times -= 1  # type: ignore[attr-defined]
                fail = True
            else:
                fail = False
        if fail:
            self._send(503, {"error": "upstream unavailable"})
            return
        self._send(200, {"ok": True})

    def do_POST(self):
        srv = self.server
        length = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(length) or b"{}")
        with srv.mu:  # type: ignore[attr-defined]
            srv.requests.append(req)  # type: ignore[attr-defined]
            if srv.fail_times > 0:  # type: ignore[attr-defined]
                srv.fail_times -= 1  # type: ignore[attr-defined]
                fail = True
            else:
                fail = False
        if fail:
            self._send(503, {"error": {"message": "upstream unavailable"}})
            return
        model = req.get("model", "")
        if model in srv.unknown_models:  # type: ignore[attr-defined]
            self._send(404, {"error": {"message": f"model '{model}' does not exist",
                                       "type": "invalid_request_error", "code": "model_not_found"}})
            return
        if srv.reject_json_mode and "response_format" in req:  # type: ignore[attr-defined]
            self._send(400, {"error": {"message": "response_format is not supported for this model",
                                       "type": "invalid_request_error"}})
            return
        with srv.mu:  # a rejected request must not consume a scripted reply
            pos = srv.pos  # type: ignore[attr-defined]
            srv.pos += 1  # type: ignore[attr-defined]
        script = srv.script  # type: ignore[attr-defined]
        content = script[pos] if pos < len(script) else json.dumps({"answer": "done"})
        self._send(200, {"id": "mock", "object": "chat.completion", "created": 1, "model": model,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                      "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


@pytest.fixture
def mock_openai():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.mu = threading.Lock()
    srv.requests, srv.script, srv.pos = [], [], 0
    srv.unknown_models, srv.reject_json_mode, srv.fail_times = set(), False, 0
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1", srv
    finally:
        srv.shutdown()
        srv.server_close()


def _llm(base: str, model: str = "deepseek-v4-flash", models: list[str] | None = None) -> LLM:
    cfg = LLMConfig(provider="openai_compat", model=model, base_url=base, api_key="test",
                    models=models or [model], timeout=10)
    return LLM(cfg)


# ------------------------------------------------------------------ the agent
def test_agent_end_to_end_over_http(mock_openai):
    """Tool call -> TOOL_RESULT -> grounded answer, through the openai SDK.

    The answer names the 80% interval, whose coverage level only appears in the tool
    result's field name; that is licensed, not an invented figure.
    """
    base, srv = mock_openai
    me = next(r for r in FORECAST["races"] if r["id"] == "senate:ME")
    pct = round(100 * me["p_d"], 1)
    srv.script = ['{"tool": "get_race", "args": {"query": "Maine Senate"}}',
                  json.dumps({"answer": f"Jackson has a {pct}% chance; the 80% range is "
                                        f"{me['margin']['p10']:.0f} to {me['margin']['p90']:.0f} points."})]
    out = Analyst(State(FORECAST), _llm(base)).ask("Who wins Maine?")
    assert out["grounded"] is True and out["trace"][0]["tool"] == "get_race"
    assert f"{pct}%" in out["answer"] and out.get("unverified") == []
    assert len(srv.requests) == 2
    assert all(r.get("response_format", {}).get("type") == "json_object" for r in srv.requests)


def test_agent_repairs_an_ungrounded_number(mock_openai):
    base, srv = mock_openai
    me = next(r for r in FORECAST["races"] if r["id"] == "senate:ME")
    pct = round(100 * me["p_d"], 1)
    srv.script = ['{"tool": "get_race", "args": {"query": "Maine Senate"}}',
                  '{"answer": "Jackson is a 93.7% favorite."}',       # invented
                  json.dumps({"answer": f"Jackson has a {pct}% chance."})]
    out = Analyst(State(FORECAST), _llm(base)).ask("Who wins Maine?")
    assert out["grounded"] is True and "93.7" not in out["answer"]
    assert "not in any TOOL_RESULT" in srv.requests[-1]["messages"][-1]["content"]


def test_model_fallback_remembers_what_answered(mock_openai):
    base, srv = mock_openai
    srv.unknown_models = {"retired-model"}
    srv.script = ['{"answer": "ready"}']
    llm = _llm(base, model="retired-model", models=["retired-model", "deepseek-v4-flash"])
    out = llm.chat([{"role": "user", "content": "hi"}])
    assert out["content"] == '{"answer": "ready"}'
    assert llm.cfg.model == "deepseek-v4-flash"
    assert "retired-model" in llm.tried_models


def test_json_mode_rejection_falls_back_to_plain(mock_openai):
    base, srv = mock_openai
    srv.reject_json_mode = True
    srv.script = ['{"answer": "ok"}']
    llm = _llm(base)
    out = llm.chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert out["content"] == '{"answer": "ok"}'
    assert "response_format" in srv.requests[0] and "response_format" not in srv.requests[1]


def test_transient_http_errors_are_retried(mock_openai):
    base, srv = mock_openai
    srv.fail_times = 2
    assert civicapi._get(f"{base}/race/search", timeout=5, retries=4) == {"ok": True}
    assert srv.fail_times == 0


def test_grounding_licenses_interval_coverage_from_field_names():
    from agent.agent import ungrounded_percentages
    result = {"margin_80pct_range": [-3.0, 10.0], "p_dem_win_pct": 76.8}
    assert ungrounded_percentages("A 76.8% chance; the 80% range runs -3 to +10.", [result]) == []
    assert ungrounded_percentages("A 76.8% chance; the 95% range runs -3 to +10.", [result]) == ["95%"]
    assert ungrounded_percentages("A 51.2% chance.", [result]) == ["51.2%"]


# ------------------------------------------------------------- live feed rules
def test_civic_mapping_matches_party_when_surname_is_unknown():
    civic = [{"type": "US Senate", "province": "ME", "percent_reporting": 42,
              "candidates": [{"name": "T. Jackson", "party": "Democratic", "votes": 1200},
                             {"name": "S. Collins", "party": "Republican", "votes": 900},
                             {"name": "Someone Else", "party": "Green", "votes": 30}]}]
    out = civicapi.map_civic(civic, FORECAST)["senate:ME"]
    assert (out["d_votes"], out["r_votes"], out["other_votes"]) == (1200, 900, 30)
    assert out["status"] == "counting" and out["pct_reporting"] == 42


def test_civic_mapping_keeps_the_highest_count_and_reads_calls():
    civic = [{"type": "US Senate", "province": "ME", "percent_reporting": 30,
              "candidates": [{"name": "Troy Jackson", "votes": 1000}, {"name": "Susan Collins", "votes": 900}]},
             {"type": "US Senate", "province": "ME", "percent_reporting": 90,
              "candidates": [{"name": "Troy Jackson", "votes": 5000, "winner": True},
                             {"name": "Susan Collins", "votes": 4000}]}]
    out = civicapi.map_civic(civic, FORECAST)["senate:ME"]
    assert out["d_votes"] == 5000 and out["status"] == "called" and out["winner"] == "D"


def test_civic_mapping_ignores_races_the_model_does_not_cover():
    civic = [{"type": "Delegate", "province": "MP", "candidates": [{"name": "X", "votes": 1}]},
             {"type": "US Senate", "province": "US", "candidates": [{"name": "Y", "votes": 1}]}]
    assert civicapi.map_civic(civic, FORECAST) == {}


# --------------------------------------------------------------- lock integrity
def _mini_forecast(p_d: float = 0.6) -> dict:
    return {"meta": {"as_of": "2026-11-02", "model": "Lodestar", "version": "1.0.0"},
            "races": [{"id": "senate:ME", "chamber": "senate", "state": "ME", "d_cand": "A B",
                       "r_cand": "C D", "favored": "D", "p_d": p_d,
                       "margin": {"mean": 1.0, "p10": -4.0, "p90": 6.0},
                       "variants": {"top20_only": 0.5, "polls_simple": 0.5,
                                    "ratings_only": 0.5, "fundamentals_only": 0.5}}],
            "chambers": {"senate": {"p_d_control": 0.5, "d_seats": {"median": 50}},
                         "house": {"p_d_control": 0.5, "d_seats": {"median": 200}},
                         "governor": {"p_d_majority": 0.5, "d_seats": {"median": 25}}}}


def test_lock_is_idempotent_and_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(lockmod, "OUT", tmp_path)
    monkeypatch.setattr(lockmod, "LOCKED", tmp_path / "locked")
    src = tmp_path / "forecast.json"
    src.write_text(json.dumps(_mini_forecast()))
    first = lockmod.lock(src)
    again = lockmod.lock(src)
    assert again["unchanged"] is True and again["locked_utc"] == first["locked_utc"]
    assert lockmod.verify(src)["ok"] is True

    src.write_text(json.dumps(_mini_forecast(0.7)))     # a different forecast
    with pytest.raises(SystemExit):
        lockmod.lock(src)
    forced = lockmod.lock(src, force=True)
    assert not forced.get("unchanged") and forced["replaced"]["previous_forecast_sha256"] == first["forecast_sha256"]
    assert lockmod.sha256(src) == forced["forecast_sha256"]


# ------------------------------------------------------------------- the server
def test_server_persists_results_and_reloads_them(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("LIVE_OUT_PATH", str(tmp_path / "live.json"))
    monkeypatch.setenv("LIVE_SOURCE", "none")
    from fastapi.testclient import TestClient

    import server.app as sapp
    sapp = importlib.reload(sapp)
    with TestClient(sapp.app) as client:
        assert client.get("/api/health").json()["ok"] is True
        assert client.get("/api/lock").json()["ok"] is True
        assert client.post("/api/results", json=[{"race_id": "nope:none"}]).status_code == 400
        r = client.post("/api/results", json=[{"race_id": "senate:ME", "d_votes": 1000, "r_votes": 900,
                                               "pct_reporting": 50, "status": "counting"}])
        assert r.status_code == 200 and r.json()["results_loaded"] == 1
        live = client.get("/api/live").json()
        assert live["races"]["senate:ME"]["pct_reporting"] == 50
        assert live["stale"] is False
        assert client.get("/api/health").json()["results_loaded"] == 1
    assert (tmp_path / "state.json").exists()

    sapp = importlib.reload(sapp)          # a restart, as after a crash or deploy
    assert sapp.HUB.results["senate:ME"]["d_votes"] == 1000
    assert sapp.HUB.live is not None


def test_live_source_rejects_unknown_values(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("LIVE_OUT_PATH", str(tmp_path / "live.json"))
    monkeypatch.setenv("LIVE_SOURCE", "none")
    from fastapi.testclient import TestClient

    import server.app as sapp
    sapp = importlib.reload(sapp)
    with TestClient(sapp.app) as client:
        assert client.post("/api/live/source", json={"source": "twitter"}).status_code == 422
        assert client.post("/api/live/source", json={"source": "none"}).status_code == 200


# ------------------------------------------------------------ provider detection
def test_provider_detection(monkeypatch):
    monkeypatch.setattr("agent.llm.probe_ollama", lambda *a, **k: False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-live-abc123")
    cfg = LLMConfig.from_env()
    assert cfg.provider == "deepseek" and LLM(cfg).enabled and len(cfg.models) >= 2

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-your-key-here")   # the .env.example value
    cfg = LLMConfig.from_env()
    assert cfg.provider == "none" and not LLM(cfg).enabled
    assert "placeholder" in cfg.reason

    monkeypatch.delenv("DEEPSEEK_API_KEY")
    cfg = LLMConfig.from_env()
    assert cfg.provider == "none" and "no DEEPSEEK_API_KEY" in cfg.reason


def test_explicit_ollama_provider_is_used(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("OLLAMA_MODEL", "deepseek-r1:8b")
    cfg = LLMConfig.from_env()
    assert cfg.provider == "ollama" and cfg.model == "deepseek-r1:8b" and LLM(cfg).enabled
