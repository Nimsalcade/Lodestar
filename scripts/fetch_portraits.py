"""Fetch official/public portraits for every candidate from Wikipedia/Wikimedia.

Wikipedia's lead image for a politician is usually their official portrait, and every
file carries an author and licence that CC BY-SA requires us to show. This script:

  1. resolves each (candidate, state, office) to a Wikipedia article with a scored
     match — a surname plus a state/office mention is required, so an obscure
     challenger is left without a photo rather than given someone else's;
  2. downloads the lead image, crops it to a 4:5 portrait and writes a small WebP
     under public/portraits/;
  3. records the article, file, author and licence for attribution in
     public/portraits/index.json and public/portraits/ATTRIBUTION.md.

It is resumable: every resolution and download is cached in
output/portraits_cache.json, so a rerun only asks about candidates that changed.

    python scripts/fetch_portraits.py                 # all candidates
    python scripts/fetch_portraits.py --limit 40      # a quick sample
    python scripts/fetch_portraits.py --refresh       # ignore the cache

Requires Pillow (dev-only; the app itself does not need it):
    .venv/bin/pip install pillow

Be polite to Wikipedia: requests are paced and 429s are retried with backoff.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORECAST = ROOT / "output" / "forecast.json"
OUT_DIR = ROOT / "public" / "portraits"
CACHE = ROOT / "output" / "portraits_cache.json"
API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "Lodestar2026/1.0 (election forecast; https://github.com/Nimsalcade/Lodestar)",
      "Accept-Encoding": "gzip"}
THUMB_PX = 480
SIZE = (240, 300)
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
# A candidate with a real article normally scores at least this. Below it we prefer
# a clean initials avatar over a stranger's face.
MIN_SCORE = 5
OFFICE_WORD = {"senate": "Senate", "house": "House", "governor": "Governor"}


# --------------------------------------------------------------------- HTTP
def _get(params: dict, tries: int = 5, timeout: float = 30) -> dict:
    url = f"{API}?{urllib.parse.urlencode(params)}"
    delay = 1.0
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                raw = r.read()
                if (r.headers.get("Content-Encoding") or "") == "gzip":
                    raw = gzip.decompress(raw)
            return json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt + 1 < tries:
                wait = float(e.headers.get("Retry-After") or delay)
                time.sleep(min(wait, 30))
                delay = min(delay * 2, 16)
                continue
            raise
    raise RuntimeError("unreachable")


def _download(url: str, tries: int = 4) -> bytes:
    delay = 1.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"]})
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt + 1 < tries:
                time.sleep(min(float(e.headers.get("Retry-After") or delay), 20))
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- identity
def _tokens(name: str) -> list[str]:
    return [t for t in re.findall(r"[a-z'\-]+", (name or "").lower()) if t not in SUFFIXES]


def resolve(name: str, state_name: str, office: str) -> dict | None:
    """Best scored Wikipedia article for this candidate, or None if not confident."""
    query = f'"{name}" {state_name} {OFFICE_WORD[office]}'
    data = _get({"action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": 6,
                 "prop": "pageimages|extracts", "piprop": "thumbnail|name", "pithumbsize": THUMB_PX,
                 "exintro": 1, "explaintext": 1, "format": "json", "formatversion": 2, "redirects": 1})
    pages = ((data.get("query") or {}).get("pages")) or []
    toks = _tokens(name)
    if not toks:
        return None
    first, surname = toks[0], toks[-1]
    low = name.lower()
    best = None
    for p in pages:
        title = p.get("title", "")
        tl = title.lower()
        extract = (p.get("extract") or "").lower()
        if surname not in tl or (first not in tl and first[0] not in tl):
            continue
        if re.search(r"\b(19|20)\d\d\b", title) or "election" in tl or "district" in tl:
            continue
        thumb = (p.get("thumbnail") or {}).get("source")
        if not thumb:
            continue
        score = 0
        if tl == low:
            score += 5
        elif tl.startswith(low):
            score += 3
        if state_name.lower() in tl:
            score += 2
        if office.lower() in extract or any(w in extract for w in
                                           ("politician", "representative", "senator", "governor", "congress")):
            score += 2
        if state_name.lower() in extract:
            score += 1
        if "(politician" in tl:
            score += 1
        if score >= MIN_SCORE and (best is None or score > best["score"]):
            best = {"title": title, "score": score, "thumb": thumb,
                    "pageimage": p.get("pageimage") or "",
                    "extract": (p.get("extract") or "")[:220]}
    return best


def slugify(name: str, state: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48]
    digest = hashlib.sha1(f"{name}|{state}".encode()).hexdigest()[:8]
    return f"{base}-{digest}"


# -------------------------------------------------------------- attribution
def _clean_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def imageinfo(files: list[str]) -> dict:
    """Batch licence/author metadata for Wikimedia File: titles."""
    out: dict = {}
    for i in range(0, len(files), 45):
        batch = files[i:i + 45]
        data = _get({"action": "query", "titles": "|".join(f"File:{f}" for f in batch),
                     "prop": "imageinfo", "iiprop": "extmetadata|url", "format": "json",
                     "formatversion": 2, "redirects": 1})
        for p in ((data.get("query") or {}).get("pages")) or []:
            # the API returns spaces where pageimages gave underscores; normalise for lookup
            title = (p.get("title") or "").removeprefix("File:").replace(" ", "_")
            md = ((p.get("imageinfo") or [{}])[0].get("extmetadata")) or {}
            out[title] = {
                "author": _clean_html((md.get("Artist") or {}).get("value", ""))[:160],
                "license": _clean_html((md.get("LicenseShortName") or {}).get("value", ""))[:60],
                "license_url": (md.get("LicenseUrl") or {}).get("value", ""),
                "credit": _clean_html((md.get("Credit") or {}).get("value", ""))[:160],
            }
        time.sleep(0.4)
    return out


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--pace", type=float, default=0.35, help="seconds between Wikipedia queries")
    a = ap.parse_args(argv)

    forecast = json.loads(FORECAST.read_text())
    cache: dict = {} if a.refresh or not CACHE.exists() else json.loads(CACHE.read_text())
    cache.setdefault("resolutions", {})
    cache.setdefault("credits", {})

    wanted: list[dict] = []
    seen: set = set()
    for r in forecast["races"]:
        for side in ("d", "r"):
            name = r.get(f"{side}_cand")
            if not name:
                continue
            key = f"{name}|{r['state']}|{r['chamber']}"
            if key in seen:
                continue
            seen.add(key)
            wanted.append({"key": key, "name": name, "state": r["state"],
                           "state_name": r["state_name"], "office": r["chamber"]})
    if a.limit:
        wanted = wanted[: a.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[portraits] {len(wanted)} distinct candidates; cache has {len(cache['resolutions'])}", flush=True)

    resolved = 0
    for n, cand in enumerate(wanted, 1):
        key = cand["key"]
        if key not in cache["resolutions"]:
            try:
                cache["resolutions"][key] = resolve(cand["name"], cand["state_name"], cand["office"]) or {}
            except Exception as e:  # noqa: BLE001 - a network blip must not lose the run
                print(f"  ! {cand['name']}: {type(e).__name__}: {e}", flush=True)
                cache["resolutions"][key] = {"error": f"{type(e).__name__}"}
            time.sleep(a.pace)
            if n % 40 == 0:
                CACHE.write_text(json.dumps(cache))
                print(f"  {n}/{len(wanted)} resolved ({resolved} with images)", flush=True)
        hit = cache["resolutions"][key]
        if hit.get("thumb"):
            resolved += 1
    CACHE.write_text(json.dumps(cache))

    # ---- download + convert
    from PIL import Image, ImageOps

    files = {hit["pageimage"] for hit in cache["resolutions"].values() if hit.get("thumb") and hit.get("pageimage")}
    missing = [f for f in files if f not in cache["credits"]]
    if missing:
        print(f"[portraits] fetching licence metadata for {len(missing)} files", flush=True)
        cache["credits"].update(imageinfo(missing))
        CACHE.write_text(json.dumps(cache))

    index = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "source": "Wikipedia article lead images / Wikimedia Commons",
             "license_note": "Each image keeps its own licence (usually CC BY-SA or public domain); "
                             "author and licence are recorded per image.",
             "races": {}}
    downloaded = 0
    for cand in wanted:
        hit = cache["resolutions"].get(cand["key"]) or {}
        if not hit.get("thumb"):
            continue
        slug = slugify(cand["name"], cand["state"])
        path = OUT_DIR / f"{slug}.webp"
        if not path.exists():
            try:
                raw = _download(hit["thumb"])
                with Image.open(io.BytesIO(raw)) as im:
                    im = im.convert("RGB")
                    im = ImageOps.fit(im, SIZE, method=Image.LANCZOS, centering=(0.5, 0.38))
                    im.save(path, "WEBP", quality=82, method=6)
                downloaded += 1
                time.sleep(0.12)
            except Exception as e:  # noqa: BLE001
                print(f"  ! download {cand['name']}: {type(e).__name__}: {e}", flush=True)
                continue
        credit = cache["credits"].get(hit.get("pageimage", ""), {})
        index.setdefault("candidates", {})[cand["key"]] = {
            "name": cand["name"], "img": f"/portraits/{slug}.webp", "slug": slug,
            "article": hit["title"],
            "article_url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(hit["title"].replace(" ", "_")),
            "file": hit.get("pageimage", ""),
            "author": credit.get("author", ""), "license": credit.get("license", ""),
            "license_url": credit.get("license_url", ""), "score": hit.get("score"),
        }
    index["races"] = {}

    # ---- per-race lookup, guarded by the candidate name the forecast currently has
    by_key = index.get("candidates", {})
    for r in forecast["races"]:
        for side in ("d", "r"):
            name = r.get(f"{side}_cand")
            if not name:
                continue
            entry = by_key.get(f"{name}|{r['state']}|{r['chamber']}")
            if entry:
                index["races"].setdefault(r["id"], {})[side] = entry

    (OUT_DIR / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    with_photo = sum(1 for v in by_key.values() if v.get("img"))
    lines = ["# Portrait credits", "",
             "Candidate photos are the lead images of their Wikipedia articles, fetched from "
             "Wikimedia Commons. Each file keeps its original licence; author and licence are "
             "listed below. Cropped to 4:5 at 240x300 and re-encoded as WebP.", "",
             f"Generated {index['generated_utc']} · {with_photo} of {len(wanted)} candidates.", "",
             "| candidate | file | author | licence | article |", "|---|---|---|---|---|"]
    for entry in sorted(by_key.values(), key=lambda e: e["name"]):
        lines.append(f"| {entry['name']} | {entry['file']} | {entry['author'] or '—'} | "
                     f"{entry['license'] or '—'} | [wiki]({entry['article_url']}) |")
    (OUT_DIR / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n")

    print(f"[portraits] {with_photo}/{len(wanted)} candidates have a portrait "
          f"({downloaded} downloaded this run) -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
