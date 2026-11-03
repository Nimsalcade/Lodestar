# Data notice and attribution

| data | source | terms |
|---|---|---|
| Polls, nominees, incumbency, handicapper ratings (`data/raw/*.csv`) | Wikipedia contributors, compiled by github.com/KaraZajac/MIDTERMS2026 (snapshot commit in `data/raw/SNAPSHOT_COMMIT`) | CC BY-SA 4.0 |
| Some polls (rows whose `source` starts with `votehub:`) | VoteHub (votehub.com) | CC BY 4.0 |
| District partisan lean (2026 lines) | The Downballot, presidential results by congressional district | cite with link: https://www.the-downballot.com/p/data |
| Pollster ratings | Silver Bulletin pollster ratings, January 2026 | free to use with attribution to Silver Bulletin |
| Pollster ratings (fallback) | FiveThirtyEight pollster ratings, 2024 | CC BY 4.0 |
| Campaign finance | Federal Election Commission | public domain |
| Election-night results (when enabled) | civicAPI (civicapi.org) | attribution required: "Results and race calls: civicAPI (civicapi.org)" |
| Candidate portraits (`public/portraits/`) | Wikipedia article lead images, fetched from Wikimedia Commons | each image keeps its own licence (usually CC BY-SA or public domain); author and licence are recorded per image in `public/portraits/index.json` and `public/portraits/ATTRIBUTION.md` |

Portraits are cropped and re-encoded for display; the adapted images remain under their original
licences, and the credits file travels with them. Remove a candidate image if its subject asks.

Because the inputs include CC BY-SA material, Lodestar's derived data (`data/`, `output/`) is
released under **CC BY-SA 4.0**. The code is MIT-licensed (`LICENSE`).
