"""Smoke-test the configured DeepSeek analyst before election night.

    python scripts/check_agent.py                     # default question
    python scripts/check_agent.py "Who wins Maine?"

Prints how the provider was resolved (API key, local Ollama, or templates), pings
the model, then runs one grounded question and shows the tools it called. Exit
codes: 0 ok, 1 not configured, 2 model unreachable, 3 answer failed grounding.

This is the fastest way to confirm a key or an Ollama pull works before relying on
the analyst tab.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import Analyst, LLM, State  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    question = args[0] if args else "Who controls the Senate?"
    forecast = json.loads((ROOT / "output" / "forecast.json").read_text())
    llm = LLM()
    print("configured:", json.dumps(llm.describe(), indent=2))
    if not llm.enabled:
        print("\nNo model is configured, so the analyst answers from deterministic templates.")
        print("Set DEEPSEEK_API_KEY, or run Ollama and `ollama pull deepseek-r1:14b`, then try again.")
        return 1
    ping = llm.ping()
    print("ping:", json.dumps(ping, indent=2))
    if not ping.get("ok"):
        return 2
    out = Analyst(State(forecast), llm).ask(question)
    print(f"\nQ: {question}\nA: {out['answer']}")
    print("tools:", [t["tool"] for t in out.get("trace", [])])
    print("grounded:", out.get("grounded"), "provider:", (out.get("provider") or {}).get("provider"))
    return 0 if out.get("grounded") else 3


if __name__ == "__main__":
    sys.exit(main())
