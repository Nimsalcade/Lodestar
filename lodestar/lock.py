"""Freeze the pre-election forecast so it can be graded honestly.

Writes output/locked/:
  forecast_<as_of>.json      the full forecast, byte-for-byte what was published
  predictions_<as_of>.csv    one row per race: the numbers that will be scored
  LOCK.txt                   SHA-256 of both files + UTC timestamp

Post the SHA-256 somewhere public (a tweet, a git commit, an email to yourself)
before polls close: anyone can later re-hash the files and confirm nothing moved.

A lock is an audit trail, so this module refuses to replace one. Re-locking the
same bytes is a no-op that keeps the original timestamp; locking *different*
bytes requires an explicit --force, and reports what it replaced. Plain re-running
the forecast and then re-locking would otherwise silently rewrite the published
hashes — the exact thing a lock exists to catch.

Usage:
    python -m lodestar.lock                 # lock output/forecast.json (first time)
    python -m lodestar.lock --verify        # check the files still match their hashes
    python -m lodestar.lock --force         # deliberately replace an existing lock
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
LOCKED = OUT / "locked"

VARIANTS = ["top20_only", "polls_simple", "ratings_only", "fundamentals_only"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _predictions_csv(f: dict) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(["race_id", "chamber", "state", "d_cand", "r_cand", "favored", "p_d", "margin_mean",
                "margin_p10", "margin_p90", *[f"p_d_{v}" for v in VARIANTS], "top20_margin"])
    for r in f["races"]:
        v = r["variants"]
        w.writerow([r["id"], r["chamber"], r["state"], r["d_cand"], r["r_cand"], r["favored"], r["p_d"],
                    r["margin"]["mean"], r["margin"]["p10"], r["margin"]["p90"],
                    *[v.get(k) for k in VARIANTS], v.get("top20_only_margin")])
    return buf.getvalue().encode()


def _existing() -> dict | None:
    p = LOCKED / "lock.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def lock(src: Path | None = None, force: bool = False) -> dict:
    src = src or OUT / "forecast.json"
    raw = src.read_bytes()
    f = json.loads(raw)
    as_of = f["meta"]["as_of"]
    LOCKED.mkdir(parents=True, exist_ok=True)
    fj = LOCKED / f"forecast_{as_of}.json"
    pc = LOCKED / f"predictions_{as_of}.csv"

    new_forecast_sha = sha256_bytes(raw)
    new_predictions = _predictions_csv(f)
    new_predictions_sha = sha256_bytes(new_predictions)

    ch = f["chambers"]
    headline = {"senate_p_d_control": ch["senate"]["p_d_control"],
                "house_p_d_control": ch["house"]["p_d_control"],
                "governor_p_d_majority": ch["governor"]["p_d_majority"],
                "senate_d_median": ch["senate"]["d_seats"]["median"],
                "house_d_median": ch["house"]["d_seats"]["median"]}

    # ---- already locked with exactly these numbers? keep the original stamp.
    cur = _existing()
    if cur and cur.get("forecast_sha256") == new_forecast_sha \
            and cur.get("predictions_sha256") == new_predictions_sha:
        return {**cur, "unchanged": True}

    if cur and not force:
        raise SystemExit(
            "refusing to overwrite the existing lock — a published forecast must stay fixed.\n"
            f"  locked  : {cur.get('locked_utc')}  forecast {str(cur.get('forecast_sha256'))[:16]}…  "
            f"predictions {str(cur.get('predictions_sha256'))[:16]}…\n"
            f"  on disk : {src}  forecast {new_forecast_sha[:16]}…  predictions {new_predictions_sha[:16]}…\n"
            "If you are refreshing the forecast *before* polls close, that is a deliberate re-lock: "
            "run again with --force and publish the new hashes.\n"
            "To check the current files instead: python -m lodestar.lock --verify")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fj.write_bytes(raw)
    pc.write_bytes(new_predictions)
    info = {"locked_utc": stamp, "as_of": as_of, "model": f"{f['meta']['model']} {f['meta']['version']}",
            "forecast_sha256": new_forecast_sha, "predictions_sha256": new_predictions_sha,
            "headline": headline}
    (LOCKED / "LOCK.txt").write_text(
        f"Lodestar 2026 pre-election forecast lock\n"
        f"locked (UTC):        {stamp}\n"
        f"forecast as of:      {as_of}\n"
        f"forecast sha256:     {new_forecast_sha}  {fj.name}\n"
        f"predictions sha256:  {new_predictions_sha}  {pc.name}\n\n"
        f"Verify:  sha256sum output/locked/{fj.name} output/locked/{pc.name}\n")
    (LOCKED / "lock.json").write_text(json.dumps(info, indent=2))
    if cur:
        info["replaced"] = {"previous_locked_utc": cur.get("locked_utc"),
                            "previous_forecast_sha256": cur.get("forecast_sha256")}
    return info


def verify(src: Path | None = None) -> dict:
    """Check the locked files still match lock.json (and whether the served copy does)."""
    src = src or OUT / "forecast.json"
    info = _existing()
    if info is None:
        return {"ok": False, "error": "no lock found in output/locked — run `python -m lodestar.lock` first"}
    as_of = info.get("as_of")
    fj = LOCKED / f"forecast_{as_of}.json"
    pc = LOCKED / f"predictions_{as_of}.csv"
    checks: dict = {}
    problems: list[str] = []
    for name, path, want in (("locked_forecast", fj, info.get("forecast_sha256")),
                             ("locked_predictions", pc, info.get("predictions_sha256"))):
        if not path.exists():
            checks[name] = {"ok": False, "error": "missing"}
            problems.append(f"{path.name} is missing")
            continue
        have = sha256(path)
        checks[name] = {"ok": have == want, "sha256": have, "expected": want}
        if have != want:
            problems.append(f"{path.name} has changed since it was locked")
    checks["served_forecast"] = {"matches_lock": bool(src.exists()) and sha256(src) == info.get("forecast_sha256")}
    return {"ok": not problems, "locked_utc": info.get("locked_utc"), "as_of": as_of,
            "forecast_sha256": info.get("forecast_sha256"),
            "predictions_sha256": info.get("predictions_sha256"),
            "checks": checks, "problems": problems}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lock or verify the pre-election forecast")
    ap.add_argument("src", nargs="?", type=Path, help="forecast JSON (default output/forecast.json)")
    ap.add_argument("--force", action="store_true", help="replace an existing lock deliberately")
    ap.add_argument("--verify", action="store_true", help="check the lock instead of writing one")
    a = ap.parse_args(argv)
    if a.verify:
        report = verify(a.src)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    info = lock(a.src, force=a.force)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
