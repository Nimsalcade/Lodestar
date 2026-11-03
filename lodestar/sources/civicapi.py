"""Live results adapters.

civicapi(): civicAPI (https://civicapi.org) — free, keyless; counts from secretaries
of state, refreshed every few seconds on election night, with its own race calls.
Terms require attribution: "Results and race calls: civicAPI (civicapi.org)".

demo(): synthetic results drawn from the forecast's own distribution, for testing
the dashboard before polls close. Always labelled DEMO.

Election-night reliability notes:
  * the two searches (federal general, statewide governors) run concurrently, which
    roughly halves the wall-clock cost of a refresh;
  * every request retries with backoff, so a single 502 from a county feed doesn't
    blank the live model for a whole poll interval;
  * a candidate is matched by surname first, then by party when the surname is
    unknown (a married name, a nickname), so votes are not silently dropped into
    "other" on the night.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SEARCH = "https://civicapi.org/api/v2/race/search"
try:  # keep the query date in one place with the model
    from ..config import ELECTION_DATE as ELECTION
except Exception:  # pragma: no cover - config is always importable in practice
    ELECTION = "2026-11-03"
ATTRIBUTION = "Results and race calls: civicAPI (civicapi.org)"
TYPES = {"US Senate": "senate", "House of Representatives": "house", "Governor": "governor"}
QUERIES = ({"election_type": "General"}, {"query": "Governor", "election_type": "Statewide"})

TIMEOUT = float(os.getenv("CIVICAPI_TIMEOUT", "20"))
RETRIES = int(os.getenv("CIVICAPI_RETRIES", "3"))


def _get(url: str, timeout: float | None = None, retries: int | None = None) -> dict:
    """GET JSON, retrying transient failures with backoff. Raises the last error.

    civicAPI compresses its responses; asking for gzip is the difference between a
    ~14 s refresh (418 KB uncompressed, bandwidth-bound) and a sub-second one.
    """
    timeout = TIMEOUT if timeout is None else timeout
    retries = RETRIES if retries is None else max(1, retries)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Lodestar-2026/1.0",
                                                      "Accept": "application/json",
                                                      "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                encoding = (r.headers.get("Content-Encoding") or "").lower()
            if encoding == "gzip":
                raw = gzip.decompress(raw)
            return json.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - network, TLS, JSON and HTTP errors alike
            last = e
            if attempt + 1 < retries:
                time.sleep(0.6 * (2 ** attempt))
    raise last  # type: ignore[misc]


def _surname(name: str) -> str:
    toks = re.sub(r"\(.*?\)|\b(jr|sr|ii|iii|iv)\b\.?", " ", (name or "").lower())
    toks = re.findall(r"[a-z][a-z'\-]+", toks)
    return toks[-1] if toks else ""


def _fetch_query(q: dict, limit: int = 1000) -> list[dict]:
    races: list[dict] = []
    offset = 0
    while True:
        params = {"startDate": ELECTION, "endDate": ELECTION, "country": "US", "limit": limit,
                  "offset": offset, **q}
        page = _get(f"{SEARCH}?{urllib.parse.urlencode(params)}")
        batch = page.get("races") or []
        races += batch
        offset += len(batch)
        if not batch or offset >= int(page.get("count") or 0):
            break
    return races


def fetch_civicapi(limit: int = 1000) -> list[dict]:
    """Every race in the election, de-duplicated by civicAPI id.

    The two searches are disjoint in practice (governors only appear under
    election_type=Statewide), but de-duplicating keeps a change on their side from
    double-counting a race.
    """
    with ThreadPoolExecutor(max_workers=len(QUERIES)) as pool:
        batches = list(pool.map(lambda q: _fetch_query(q, limit), QUERIES))
    seen: set = set()
    races: list[dict] = []
    for batch in batches:
        for r in batch:
            key = r.get("id")
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
            races.append(r)
    return races


def _party_side(c: dict, race: dict) -> str | None:
    """Side of the forecast matchup a candidate's party unambiguously belongs to."""
    p = (c.get("party") or "").strip().lower()
    if p == "republican" and race.get("r_side") == "R":
        return "R"
    if p == "democratic" and race.get("d_side") == "D":
        return "D"
    return None


def map_civic(civic_races: list[dict], forecast: dict) -> dict:
    """civicAPI races -> {race_id: result dict} on the forecast's D-side / R-side."""
    ours = {r["id"]: r for r in forecast["races"]}
    out = {}
    for c in civic_races:
        chamber = TYPES.get(c.get("type"))
        st = (c.get("province") or "").upper()
        if not chamber or len(st) != 2:
            continue
        if chamber == "house":
            rid = f"house:{(c.get('district') or '').upper()}"
        else:
            rid = f"{chamber}:{st}"
            if rid not in ours and f"{rid}-special" in ours:
                rid += "-special"
        race = ours.get(rid)
        if not race:
            continue
        cands = c.get("candidates") or []
        sd, sr = _surname(race.get("d_cand")), _surname(race.get("r_cand"))
        dv = rv = ov = 0
        winner = None
        for x in cands:
            v = int(x.get("votes") or 0)
            s = _surname(x.get("name", ""))
            if sd and s == sd:
                dv += v
                side = "D" if race.get("d_side") == "D" else "IND"
            elif sr and s == sr:
                rv += v
                side = "R"
            else:
                by_party = _party_side(x, race)
                if by_party == "D":
                    dv += v
                    side = "D"
                elif by_party == "R":
                    rv += v
                    side = "R"
                else:
                    ov += v
                    side = "OTHER"
            if x.get("winner") is True:
                winner = side
        pct = float(c.get("percent_reporting") or 0)
        status = "called" if winner else ("counting" if dv + rv > 0 else "awaiting")
        if rid in out and (out[rid]["d_votes"] + out[rid]["r_votes"]) >= dv + rv:
            continue
        out[rid] = {"d_votes": dv, "r_votes": rv, "other_votes": ov, "pct_reporting": pct,
                    "status": status, "winner": winner, "source": "civicAPI"}
    return out


def civicapi(forecast: dict) -> dict:
    return map_civic(fetch_civicapi(), forecast)


def civicapi_with_meta(forecast: dict) -> tuple[dict, dict]:
    """Results plus a small provenance record for /api/live and /api/health."""
    t0 = time.time()
    civic = fetch_civicapi()
    results = map_civic(civic, forecast)
    meta = {"source": "civicapi", "election_date": ELECTION, "elapsed_s": round(time.time() - t0, 1),
            "races_returned": len(civic), "races_matched": len(results),
            "attribution": ATTRIBUTION,
            "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return results, meta


# ------------------------------------------------------------------ demo feed
EAST_FIRST = ["GA", "IN", "KY", "SC", "VT", "VA", "FL", "NH", "ME", "MA", "CT", "RI", "NJ", "DE", "MD",
              "PA", "OH", "NC", "WV", "MI", "AL", "MS", "TN", "IL", "MO", "OK", "KS", "TX", "AR", "LA",
              "MN", "WI", "IA", "NE", "SD", "ND", "NY", "AZ", "CO", "NM", "WY", "MT", "UT", "ID", "NV",
              "OR", "WA", "CA", "HI", "AK"]


def demo(forecast: dict, progress: float, seed: int = 11) -> dict:
    """Synthetic night at `progress` in [0, 1]. One shared national/regional draw so
    the demo looks like a real correlated night, not coin flips."""
    rnd = random.Random(seed)
    nat = rnd.gauss(0, 3.0)
    reg = {g: rnd.gauss(0, 1.8) for g in sorted({r["region"] for r in forecast["races"]})}
    out = {}
    for r in forecast["races"]:
        # draw every random number up front so a race's "truth" never depends on progress
        g_race, g_count = rnd.gauss(0, 1), rnd.gauss(0, 1)
        if r.get("fixed"):
            continue
        st = r["state"]
        true = float(r["margin"]["mean"]) + nat + reg[r["region"]] + g_race * float(r["posterior_sd"] or 5)
        order = EAST_FIRST.index(st) / len(EAST_FIRST) if st in EAST_FIRST else 0.9
        start = 0.05 + 0.6 * order
        p = max(0.0, min(1.0, (progress - start) / 0.35))
        if p <= 0:
            out[r["id"]] = {"d_votes": 0, "r_votes": 0, "other_votes": 0, "pct_reporting": 0,
                            "status": "awaiting", "winner": None, "source": "DEMO"}
            continue
        noise = 6.0 * g_count * (1 - p) ** 1.2
        m = true + noise
        total = int(250000 * p * (1.5 if r["chamber"] != "house" else 0.4))
        d = int(total * (50 + m / 2) / 100)
        rv = total - d
        called = p >= 0.995 or (p > 0.6 and abs(m) > 8) or (p > 0.85 and abs(m) > 3.5)
        win = None
        if called:
            win = ("D" if r["d_side"] == "D" else "IND") if m > 0 else "R"
        out[r["id"]] = {"d_votes": max(d, 0), "r_votes": max(rv, 0), "other_votes": int(total * 0.02),
                        "pct_reporting": round(100 * p, 1), "status": "called" if called else "counting",
                        "winner": win, "source": "DEMO"}
    return out


__all__ = ["civicapi", "civicapi_with_meta", "demo", "map_civic", "fetch_civicapi", "ATTRIBUTION"]
_ = math
