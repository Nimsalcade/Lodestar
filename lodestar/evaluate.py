"""Grade the locked forecast against actual results.

Usage:
    python -m lodestar.evaluate --results data/results_2026.csv
    python -m lodestar.evaluate --from-live output/live.json      # civicAPI counts/calls

results CSV columns (one row per race): race_id, d_pct, r_pct, winner
  - winner: D, R or IND/OTHER (who actually won; D-side/R-side per the forecast)
  - d_pct / r_pct: final vote shares of the forecast's D-side and R-side nominees

Scores every model variant stored in the lock, so you can see whether the Top-20
weighting, the fundamentals and the ratings each earned their place:
  lodestar (full model) | top20_only | polls_simple | ratings_only | fundamentals_only
Metrics: Brier score (lower is better), log loss, accuracy on the favored side,
margin MAE / RMSE / bias (D-minus-R; negative bias = model too Democratic),
80%-interval coverage (should be ~0.80), and a calibration table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCKED = ROOT / "output" / "locked"


def _load_results_csv(path: Path) -> dict:
    out = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            rid = row["race_id"].strip()
            d = float(row["d_pct"]) if row.get("d_pct") not in (None, "") else None
            r = float(row["r_pct"]) if row.get("r_pct") not in (None, "") else None
            win = (row.get("winner") or "").strip().upper() or None
            if win is None and d is not None and r is not None:
                win = "D" if d > r else "R"
            out[rid] = {"winner": win, "margin": (d - r) if d is not None and r is not None else None}
    return out


def _load_results_live(path: Path) -> dict:
    live = json.loads(path.read_text())
    out = {}
    for rid, nc in live.get("races", {}).items():
        if nc.get("status") != "called":
            continue
        d, r = nc.get("d_votes") or 0, nc.get("r_votes") or 0
        tot = d + r + (nc.get("other_votes") or 0)
        out[rid] = {"winner": "D" if nc["p_d"] >= 0.5 else "R",
                    "margin": 100 * (d - r) / tot if tot else None}
    return out


def _metrics(pairs):
    """pairs: list of (p_d, y) with y in {0,1}."""
    if not pairs:
        return None
    n = len(pairs)
    eps = 1e-4
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    ll = -sum(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps)) for p, y in pairs) / n
    acc = sum(int((p >= 0.5) == (y == 1)) for p, y in pairs) / n
    return {"n": n, "brier": round(brier, 4), "log_loss": round(ll, 4), "accuracy": round(acc, 4)}


def evaluate(results: dict, forecast_path: Path | None = None) -> dict:
    if forecast_path is None:
        cands = sorted(LOCKED.glob("forecast_*.json"))
        if not cands:
            raise SystemExit("no locked forecast found in output/locked — run `python -m lodestar.lock` first")
        forecast_path = cands[-1]
    f = json.loads(forecast_path.read_text())
    races = {r["id"]: r for r in f["races"]}
    models = ["lodestar", "top20_only", "polls_simple", "ratings_only", "fundamentals_only"]
    pairs = {m: {"all": [], "senate": [], "house": [], "governor": [], "competitive": []} for m in models}
    margin_err, top20_err, covered = [], [], []
    calib = [[0, 0.0, 0] for _ in range(10)]
    misses = []
    for rid, res in results.items():
        r = races.get(rid)
        if r is None or r.get("fixed") or res["winner"] is None:
            continue
        y = 1.0 if (res["winner"] == "D" or (res["winner"] == "IND" and r["d_side"] == "IND")) else 0.0
        comp = 0.05 < r["p_d"] < 0.95
        for m in models:
            p = r["p_d"] if m == "lodestar" else r["variants"].get(m)
            if p is None:
                continue
            for k in ("all", r["chamber"]) + (("competitive",) if comp else ()):
                pairs[m][k].append((float(p), y))
        b = min(int(r["p_d"] * 10), 9)
        calib[b][0] += 1
        calib[b][1] += r["p_d"]
        calib[b][2] += int(y)
        if (r["p_d"] >= 0.5) != (y == 1.0):
            misses.append({"id": rid, "p_d": r["p_d"], "winner": res["winner"]})
        if res["margin"] is not None:
            margin_err.append(res["margin"] - r["margin"]["mean"])
            if r["variants"].get("top20_only_margin") is not None:
                top20_err.append(res["margin"] - r["variants"]["top20_only_margin"])
            lo, hi = r["margin"]["p10"], r["margin"]["p90"]
            covered.append(int(lo <= res["margin"] <= hi))

    def merr(e):
        if not e:
            return None
        n = len(e)
        return {"n": n, "mae": round(sum(abs(x) for x in e) / n, 2),
                "rmse": round(math.sqrt(sum(x * x for x in e) / n), 2),
                "bias": round(sum(e) / n, 2)}

    return {
        "forecast": forecast_path.name,
        "scores": {m: {k: _metrics(v) for k, v in d.items()} for m, d in pairs.items()},
        "margin_error_lodestar": merr(margin_err),
        "margin_error_top20_only": merr(top20_err),
        "interval80_coverage": round(sum(covered) / len(covered), 3) if covered else None,
        "calibration": [{"bin": f"{10*i}-{10*i+10}%", "n": c[0],
                         "mean_forecast": round(c[1] / c[0], 3) if c[0] else None,
                         "observed_d_rate": round(c[2] / c[0], 3) if c[0] else None} for i, c in enumerate(calib)],
        "misses": sorted(misses, key=lambda x: -abs(x["p_d"] - 0.5)),
    }


def to_markdown(ev: dict) -> str:
    L = [f"# Lodestar 2026 — scorecard ({ev['forecast']})", "",
         "| model | races | Brier ↓ | log loss ↓ | accuracy | competitive Brier |", "|---|---:|---:|---:|---:|---:|"]
    for m, d in ev["scores"].items():
        a, c = d["all"], d["competitive"]
        if a:
            L.append(f"| {m} | {a['n']} | {a['brier']} | {a['log_loss']} | {a['accuracy']:.1%} | "
                     f"{c['brier'] if c else '—'} |")
    me = ev["margin_error_lodestar"]
    if me:
        L += ["", f"Margin error (Lodestar): MAE {me['mae']}, RMSE {me['rmse']}, bias {me['bias']:+} "
                  f"(positive = Democrats beat the forecast). 80% interval coverage: {ev['interval80_coverage']}"]
    L += ["", "| forecast bin | races | mean forecast | observed D win rate |", "|---|---:|---:|---:|"]
    for c in ev["calibration"]:
        if c["n"]:
            L.append(f"| {c['bin']} | {c['n']} | {c['mean_forecast']} | {c['observed_d_rate']} |")
    if ev["misses"]:
        L += ["", "Wrong-side calls: " + ", ".join(f"{m['id']} (P(D)={m['p_d']:.2f})" for m in ev["misses"])]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path)
    ap.add_argument("--from-live", type=Path)
    ap.add_argument("--forecast", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "scorecard.json")
    a = ap.parse_args()
    if a.results:
        res = _load_results_csv(a.results)
    elif a.from_live:
        res = _load_results_live(a.from_live)
    else:
        raise SystemExit("pass --results CSV or --from-live output/live.json")
    ev = evaluate(res, a.forecast)
    a.out.write_text(json.dumps(ev, indent=2))
    md = to_markdown(ev)
    a.out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
