"""DeepSeek client — one interface, four ways to run it.

  LLM_PROVIDER=auto       (default) DeepSeek API if a real DEEPSEEK_API_KEY is set,
                          else a local Ollama if one is listening, else templates.
  LLM_PROVIDER=ollama     local:   Ollama's OpenAI-compatible endpoint
                                   (default model deepseek-r1:14b; deepseek-r1:8b for 16 GB
                                   laptops; deepseek-v4-flash needs a ~256 GB workstation)
  LLM_PROVIDER=deepseek   deploy:  DeepSeek's hosted API, https://api.deepseek.com
                                   (default model deepseek-v4-flash; deepseek-v4-pro for harder work)
  LLM_PROVIDER=none       no model: the agent answers from deterministic templates

Any other OpenAI-compatible server hosting DeepSeek weights (vLLM, llama.cpp
server, LM Studio) works with LLM_PROVIDER=openai_compat + LLM_BASE_URL.

Two failure modes are handled here so the analyst degrades cleanly on the night:
a placeholder key pasted from .env.example counts as "not configured", and a
model name the server does not recognise falls back through the provider's
candidates once, remembering the one that answered.
"""
from __future__ import annotations

import os
import re
import time
import urllib.request
from dataclasses import dataclass, field

DEFAULTS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "deepseek-r1:14b", "api_key": "ollama",
               "models": ["deepseek-r1:14b", "deepseek-r1:8b", "deepseek-r1:32b", "deepseek-v4-flash"]},
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-v4-flash", "api_key": None,
                 "models": ["deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner", "deepseek-v4-pro"]},
    "openai_compat": {"base_url": "http://localhost:8000/v1", "model": "deepseek-ai/DeepSeek-V4-Flash",
                      "api_key": "local",
                      "models": ["deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-R1", "deepseek-chat"]},
}

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
# values people copy out of .env.example and forget to replace
PLACEHOLDERS = {"sk-your-key-here", "sk-...", "your-key", "changeme", "replace-me", "sk-proj-..."}


def _real_key(key: str | None) -> bool:
    k = (key or "").strip()
    return bool(k) and k.lower() not in PLACEHOLDERS and not k.lower().startswith("sk-your")


def probe_ollama(base_url: str, timeout: float = 0.6) -> bool:
    """True if an Ollama server answers at base_url (which may end in /v1)."""
    root = (base_url or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        req = urllib.request.Request(f"{root}/api/tags", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200 and b"models" in r.read(4096)
    except Exception:  # noqa: BLE001 - absence is the normal case
        return False


@dataclass
class LLMConfig:
    provider: str = "none"
    model: str = ""
    base_url: str = ""
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout: float = 90.0
    models: list[str] = field(default_factory=list)
    reason: str = ""

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY") or ""
        reason = ""
        if not provider or provider == "auto":
            ollama_url = os.getenv("OLLAMA_BASE_URL", DEFAULTS["ollama"]["base_url"])
            if _real_key(key):
                provider = "deepseek"
            elif probe_ollama(ollama_url):
                provider = "ollama"
            else:
                provider = "none"
                reason = (f"auto: DEEPSEEK_API_KEY looks like the .env.example placeholder and no Ollama "
                          f"server answered on {ollama_url}" if key else
                          f"auto: no DEEPSEEK_API_KEY and no Ollama server answered on {ollama_url}")
        d = DEFAULTS.get(provider, {})
        if provider == "deepseek":
            api_key = key or d.get("api_key")
            if not _real_key(api_key):
                reason = ("DEEPSEEK_API_KEY is unset or still the placeholder from .env.example "
                          "— set a real key from platform.deepseek.com")
        else:
            api_key = os.getenv("LLM_API_KEY") or d.get("api_key")
        model = (os.getenv("LLM_MODEL")
                 or os.getenv("DEEPSEEK_MODEL" if provider == "deepseek" else "OLLAMA_MODEL")
                 or d.get("model", ""))
        base = (os.getenv("LLM_BASE_URL")
                or (os.getenv("OLLAMA_BASE_URL") if provider == "ollama" else None)
                or d.get("base_url", ""))
        models = [m for m in [os.getenv("LLM_MODEL"), os.getenv("DEEPSEEK_MODEL" if provider == "deepseek"
                                                               else "OLLAMA_MODEL"), model,
                              *(d.get("models") or [])] if m]
        seen, ordered = set(), []
        for m in models:                       # keep order, drop duplicates
            if m not in seen:
                seen.add(m)
                ordered.append(m)
        return cls(provider=provider, model=model, base_url=base, api_key=api_key,
                   temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
                   max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1200")),
                   timeout=float(os.getenv("LLM_TIMEOUT", "90")),
                   models=ordered, reason=reason)


def _is_model_error(msg: str) -> bool:
    m = msg.lower()
    return any(t in m for t in ("model_not_found", "model not found", "does not exist", "invalid model",
                                "unknown model", "no such model", "model_not_supported",
                                "not a valid model")) or ("model" in m and "exist" in m)


class LLM:
    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLMConfig.from_env()
        self._client = None
        self.last_error: str | None = None
        self.tried_models: list[str] = []

    @property
    def enabled(self) -> bool:
        if self.cfg.provider == "none":
            return False
        if self.cfg.provider == "deepseek" and not _real_key(self.cfg.api_key):
            return False
        return True

    def describe(self) -> dict:
        return {"provider": self.cfg.provider, "model": self.cfg.model if self.enabled else None,
                "base_url": self.cfg.base_url if self.enabled else None, "enabled": self.enabled,
                "mode": "deepseek-api" if self.cfg.provider == "deepseek" else
                        "local-model" if self.cfg.provider in ("ollama", "openai_compat") else "templates",
                "reason": self.cfg.reason or ("" if self.enabled else
                                              "no language model configured (set LLM_PROVIDER / DEEPSEEK_API_KEY)"),
                "last_error": self.last_error, "models_tried": self.tried_models[-4:]}

    def client(self):
        if self._client is None:
            from openai import OpenAI  # OpenAI-compatible SDK; DeepSeek and Ollama both speak it
            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key or "none",
                                  timeout=self.cfg.timeout, max_retries=1)
        return self._client

    def _candidates(self) -> list[str]:
        cands = self.cfg.models or [self.cfg.model]
        return [m for m in cands if m]

    def chat(self, messages: list[dict], json_mode: bool = False, max_tokens: int | None = None) -> dict:
        """Returns {"content", "reasoning", "latency_s"}. Reasoning from deepseek-r1 style
        <think> blocks or the API's reasoning_content field is split off, never shown as the answer."""
        if not self.enabled:
            raise RuntimeError(self.describe()["reason"] or "LLM disabled")
        t = time.time()
        errors: list[str] = []
        for model in self._candidates():
            kw = {"model": model, "messages": messages, "temperature": self.cfg.temperature,
                  "max_tokens": max_tokens or self.cfg.max_tokens}
            attempts = [True, False] if json_mode else [False]
            for use_json in attempts:
                if use_json:
                    kw["response_format"] = {"type": "json_object"}
                else:
                    kw.pop("response_format", None)
                try:
                    resp = self.client().chat.completions.create(**kw)
                except Exception as e:  # noqa: BLE001
                    msg = f"{type(e).__name__}: {e}"
                    self.tried_models.append(model)
                    # a server that rejects JSON mode still answers without it
                    if use_json and ("response_format" in msg.lower() or "json" in msg.lower()):
                        continue
                    errors.append(f"{model}: {msg}")
                    self.last_error = errors[-1]
                    if _is_model_error(msg):
                        break            # try the next candidate model
                    raise
                else:
                    self.last_error = None
                    if model != self.cfg.model:
                        self.cfg.model = model      # remember what actually answered
                    msg = resp.choices[0].message
                    content = msg.content or ""
                    reasoning = getattr(msg, "reasoning_content", None)
                    thinks = THINK_RE.findall(content)
                    if thinks:
                        reasoning = (reasoning or "") + "\n".join(thinks)
                        content = THINK_RE.sub("", content).strip()
                    return {"content": content.strip(), "reasoning": reasoning,
                            "latency_s": round(time.time() - t, 2), "model": model}
        self.last_error = "; ".join(errors)[:400] or "no model answered"
        raise RuntimeError(self.last_error)

    def ping(self) -> dict:
        try:
            r = self.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=20)
            return {"ok": True, "reply": r["content"][:40], "latency_s": r["latency_s"],
                    "model": r.get("model"), "provider": self.cfg.provider}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "provider": self.cfg.provider}


