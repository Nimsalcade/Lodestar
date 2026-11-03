"""Build dist/index.html: the dashboard with the forecast (and a simulated election
night for the Live tab) embedded, so it runs with no server — for static hosting
or sharing as a single file. The served app (server/app.py) doesn't need this."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lodestar.live import run_live  # noqa: E402
from lodestar.sources.civicapi import demo  # noqa: E402

FRAMES = [(0.0, "Before polls close"), (0.14, "≈7:30 pm ET · first East Coast counts"),
          (0.26, "≈8:30 pm ET"), (0.38, "≈9:30 pm ET"), (0.5, "≈10:30 pm ET"), (0.62, "≈11:30 pm ET"),
          (0.74, "≈12:30 am ET"), (0.86, "≈1:30 am ET · West Coast counting"), (1.0, "Morning after · counts complete")]


def compact_live(L: dict, forecast: dict) -> dict:
    keep = {r["id"] for r in forecast["races"]
            if r["chamber"] != "house" or (0.05 < r["p_d"] < 0.95)}
    races = {rid: {k: v for k, v in nc.items() if k in ("status", "pct_reporting", "counted_margin", "est_margin", "p_d", "winner")}
             for rid, nc in L["races"].items() if rid in keep}
    for v in races.values():
        v["p_d"] = round(v["p_d"], 4)
    return {"chambers": L["chambers"], "shared_error": L["shared_error"], "scorecard": L["scorecard"],
            "races": races, "source": "demo", "updated": "simulated"}


def main():
    f = json.loads((ROOT / "output" / "forecast.json").read_text())
    lock = ROOT / "output" / "locked" / "lock.json"
    if lock.exists():
        f["meta"]["lock"] = json.loads(lock.read_text())
    frames = []
    for prog, label in FRAMES:
        if prog == 0:
            frames.append({"label": label, "live": None})
            continue
        L = run_live(f, demo(f, prog, seed=2026))
        frames.append({"label": label, "live": compact_live(L, f)})
    for r in f["races"]:                      # keep the page light: newest 30 polls per race
        r["poll_list"] = r["poll_list"][:30]
    payload = json.dumps({"forecast": f, "demo_frames": frames}, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")   # never let data close the script tag
    html = (ROOT / "dashboard" / "index.html").read_text()
    marker = '<script>\n"use strict";'
    assert marker in html, "dashboard script marker not found"
    html = html.replace(marker, f"<script>window.__LODESTAR__={payload};</script>\n{marker}", 1)
    out = ROOT / "dist" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB, {len(frames)} demo frames)")


if __name__ == "__main__":
    main()
