"""Collapsing listings of the same event into one card, at read time.

The database keeps every source's row on purpose: the ranker counts how many
sources list an event (rank.agreeing_sources), and each row carries its own
score and bookkeeping. What the reader sees shouldn't repeat itself, though,
and duplicates arrive several ways:

- two sources list the same show ("Kreayshawn" on 19hz and DoTheBay, or
  DoTheBay's "Goldenvoice Presents: Portola Week 2026 / Fatboy Slim" against
  19hz's "Fatboy Slim" at the same venue),
- one source lists it twice under different ids (DoTheBay's early and late
  comedy shows, 19hz's same party behind two ticket links),
- a recurring class runs back-to-back sessions (Rec & Park's 10:00 and 11:00
  dance fitness at UN Plaza).

Matching is deliberately conservative - a missed duplicate costs a repeated
card, a false match hides a real event:

- same start date, always;
- titles with the same content words, or one's words a subset of the
  other's (presenter prefixes and lineups pad titles differently per source).
  Subset matching needs at least two words on the short side, and never
  applies to curated annual/seasonal listings: "Castro Street Fair" is a
  subset of "Castro Street Fair After Party ft Octo Octa", and isn't it;
- venues that agree (one's words within the other's) where both are known.
  A one-word title ("Bilal", "Halloween") matches far too much, so it also
  needs both venues known and agreeing.
"""
from __future__ import annotations
import re

from .rank import title_tokens

# Umbrella listings - a whole festival, a citywide holiday - whose names turn
# up inside the titles of other events (after-parties, spin-offs).
UMBRELLA_SOURCES = {"annual_bay_area", "seasonal_bay_area"}

# Blanks on the kept card that another copy can fill in.
FILLABLE = (
    "venue", "address", "cost", "url", "description", "end_ts", "images", "categories",
)

# Street-name spellings, so "Folsom St" agrees with "Folsom Street, SoMa".
ABBREVIATIONS = {"st": "street", "ave": "avenue", "blvd": "boulevard", "rd": "road"}
YEAR_RE = re.compile(r"^20\d\d$")


def _title_tokens(title: str) -> set[str]:
    # "Folsom Street Fair 2026" is "Folsom Street Fair": the date already
    # says which year.
    return {w for w in title_tokens(title or "") if not YEAR_RE.match(w)}


def _venue_tokens(venue: str) -> set[str]:
    # 19hz appends the city: "DNA Lounge (San Francisco)" is "DNA Lounge".
    words = title_tokens(re.sub(r"\([^)]*\)", " ", venue or ""))
    return {ABBREVIATIONS.get(w, w) for w in words}


def _has_time(row: dict) -> bool:
    # Date-only sources (DoTheBay) store midnight.
    ts = row.get("start_ts") or ""
    return len(ts) > 10 and ts[11:16] != "00:00"


def same_event(a: dict, b: dict) -> bool:
    ta, tb = _title_tokens(a["title"]), _title_tokens(b["title"])
    if not ta or not tb:
        return False
    if ta != tb:
        if a["source"] in UMBRELLA_SOURCES or b["source"] in UMBRELLA_SOURCES:
            return False
        if min(len(ta), len(tb)) < 2 or not (ta <= tb or tb <= ta):
            return False
    va, vb = _venue_tokens(a.get("venue")), _venue_tokens(b.get("venue"))
    if va and vb:
        return va <= vb or vb <= va
    return min(len(ta), len(tb)) >= 2


def _preference(row: dict) -> tuple:
    """Which copy becomes the card: best score, then the most specific."""
    filled = sum(bool(row.get(f)) for f in ("venue", "address", "cost", "url", "images"))
    return (
        row.get("score") if row.get("score") is not None else -1.0,
        _has_time(row),
        filled,
        len(row.get("description") or ""),
    )


def _merge(group: list[dict]) -> dict:
    group = sorted(group, key=_preference, reverse=True)
    keep = dict(group[0])
    for other in group[1:]:
        for f in FILLABLE:
            if not keep.get(f) and other.get(f):
                keep[f] = other[f]
        if keep.get("lat") is None and other.get("lat") is not None:
            keep["lat"], keep["lon"] = other["lat"], other["lon"]
        # A date-only card borrows the time from a copy that has one.
        if not _has_time(keep) and _has_time(other):
            keep["start_ts"] = other["start_ts"]
    # Only new if no source had listed it before this fetch.
    if "is_new" in keep:
        keep["is_new"] = all(r.get("is_new") for r in group)
    # Keys of the copies folded in, so a plan pinned to one still finds it.
    also = [o["key"] for o in group[1:] if o.get("key")]
    if also:
        keep["also"] = also
    return keep


def dedupe(rows: list[dict]) -> list[dict]:
    """One row per event, in the order of each event's first row."""
    by_date: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r.get("start_ts"):
            by_date.setdefault(r["start_ts"][:10], []).append(i)

    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for idx in by_date.values():
        for n, i in enumerate(idx):
            for j in idx[n + 1:]:
                if find(i) != find(j) and same_event(rows[i], rows[j]):
                    parent[find(j)] = find(i)

    groups: dict[int, list[dict]] = {}
    order: list[int] = []
    for i, r in enumerate(rows):
        root = find(i)
        if root not in groups:
            groups[root] = []
            order.append(root)
        groups[root].append(r)
    return [
        groups[root][0] if len(groups[root]) == 1 else _merge(groups[root])
        for root in order
    ]
