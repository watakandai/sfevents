from __future__ import annotations
import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from . import geocode as geocoding
from . import rank as ranking
from .db import (
    init_db, upsert_events, query_events, row_to_dict, set_scores,
    set_heuristic_scores, unscored_events,
    get_geocache, geocache_misses, put_geocache, set_coordinates,
    clear_coordinates, prune_article_events, drop_events,
)
from .dedupe import dedupe
from .fetchers.annual import AnnualEventsFetcher
from .fetchers.dothebay import DoTheBayFetcher
from .fetchers.funcheap import FuncheapFetcher
from .fetchers.nineteenhz import NineteenHzFetcher
from .fetchers.seasonal import SeasonalFetcher
from .fetchers.secretsf import SecretSFFetcher
from .fetchers.sfrecpark import SFRecParkFetcher
from .fetchers.tribe import TribeEventsFetcher

DEFAULT_DB = Path.home() / ".sfevents" / "events.db"
FETCHERS = [
    AnnualEventsFetcher(),
    FuncheapFetcher(),
    DoTheBayFetcher(),
    SFRecParkFetcher(),
    NineteenHzFetcher(),
    SeasonalFetcher(),
    # Editorial picks written as news articles; an LLM pulls the events out.
    SecretSFFetcher(),
    # Day trips: visitor-bureau calendars that all run the same WordPress
    # events plugin. South Lake Tahoe (tahoesouth.com) works too, but it's
    # ~800 listings, mostly bar nights, 3.5h+ away.
    TribeEventsFetcher("santacruz_org", "https://www.santacruz.org"),
    TribeEventsFetcher("visit_sausalito", "https://www.visitsausalito.org"),
    TribeEventsFetcher("bodega_bay", "https://www.bodegabay.com"),
]

# Sources known to be fragile (HTML scraping rather than RSS/API/local data) -
# a silent drop to zero here is the main failure mode worth watching for.
FRAGILE_SOURCES = {"dothebay", "sfrecpark", "19hz_bayarea"}


def main() -> None:
    parser = argparse.ArgumentParser(description="SF event curator")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="pull latest events from all sources into the db")

    list_parser = sub.add_parser("list", help="print stored events")
    list_parser.add_argument(
        "--sort", choices=("date", "score"), default="date",
        help="date (soonest first) or score (best match first)",
    )
    list_parser.add_argument("--limit", type=int, default=0, help="0 means no limit")

    export_parser = sub.add_parser(
        "export", help="dump all events to a JSON file (for the static-site build)"
    )
    export_parser.add_argument(
        "--out", required=True, help="output path, e.g. docs/data/events.json"
    )
    export_parser.add_argument("--sort", choices=("date", "score"), default="date")
    export_parser.add_argument(
        "--include-past", action="store_true",
        help="keep events whose start date has passed (default: drop them)",
    )
    export_parser.add_argument(
        "--full", action="store_true",
        help="export every column instead of just the fields the static page reads",
    )

    rank_parser = sub.add_parser(
        "rank", help="score events for relevance (heuristic always, LLM optionally)"
    )
    rank_parser.add_argument(
        "--llm", action="store_true",
        help="also score against profile.md with an LLM (needs the provider's API key)",
    )
    rank_parser.add_argument(
        "--provider", choices=sorted(ranking.PROVIDERS), default="anthropic",
    )
    rank_parser.add_argument(
        "--fallback", default="",
        help="comma-separated providers to try, in order, for events --provider "
             "left unscored (e.g. groq); ones without a key are skipped",
    )
    rank_parser.add_argument("--model", default=None, help="override the provider default")
    rank_parser.add_argument("--profile", default=str(ranking.DEFAULT_PROFILE))
    rank_parser.add_argument(
        "--limit", type=int, default=0,
        help="cap how many events get sent to the LLM this run (0 = all unscored)",
    )
    rank_parser.add_argument(
        "--batch-size", type=int, default=50,
        help="events per LLM request; bigger means fewer requests against a daily quota",
    )
    rank_parser.add_argument(
        "--min-interval", type=float, default=0,
        help="seconds to leave between LLM requests, to stay under a per-minute limit",
    )
    rank_parser.add_argument(
        "--rescore-all", action="store_true",
        help="re-send events already scored against the current profile",
    )

    geo_parser = sub.add_parser(
        "geocode", help="resolve venue text to coordinates for the map view"
    )
    geo_parser.add_argument(
        "--limit", type=int, default=0,
        help="max geocoder requests this run (0 = no cap); cached places are free",
    )
    geo_parser.add_argument(
        "--delay", type=float, default=1.0,
        help="seconds between requests - Nominatim asks for at least 1",
    )

    args = parser.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    init_db(args.db)

    if args.cmd == "fetch":
        _cmd_fetch(args)
    elif args.cmd == "list":
        _cmd_list(args)
    elif args.cmd == "export":
        _cmd_export(args)
    elif args.cmd == "rank":
        # Non-zero when the LLM pass left events unscored, so a scheduled run
        # goes red instead of shipping heuristic-only scores unnoticed.
        if not _cmd_rank(args):
            sys.exit(LLM_FAILED_EXIT)
    elif args.cmd == "geocode":
        _cmd_geocode(args)


def _cmd_fetch(args) -> None:
    for f in FETCHERS:
        # Sources that cache per-item work (LLM extractions) keep it here.
        if hasattr(f, "bind_db"):
            f.bind_db(args.db)
        try:
            events = f.fetch()
        except Exception as exc:
            print(f"{f.name}: FAILED ({exc})", file=sys.stderr)
            continue
        upsert_events(args.db, events)
        print(f"{f.name}: {len(events)} events fetched")
        if getattr(f, "covered_articles", None):
            pruned = prune_article_events(
                args.db, f.name, f.covered_articles, [e.source_id for e in events]
            )
            if pruned:
                print(f"  removed {pruned} stale events from re-extracted articles")
        if hasattr(f, "is_event"):
            dropped = drop_events(args.db, f.name, f.is_event)
            if dropped:
                print(f"  removed {dropped} stored listings that aren't events")
        if getattr(f, "report", ""):
            print(f"  {f.report}")
        if not events and f.name in FRAGILE_SOURCES:
            print(
                f"  warning: {f.name} returned 0 events - this source is HTML-scraped "
                "and may need its parser updated if the site changed.",
                file=sys.stderr,
            )


def _keyed(event: dict) -> dict:
    # What docs/plans.json names an event by. The id can't be used: it
    # differs between the Actions cache and a local database.
    event["key"] = f"{event['source']}:{event['source_id']}"
    return event


def _cmd_list(args) -> None:
    rows = dedupe([_keyed(row_to_dict(r)) for r in query_events(args.db, order_by=args.sort)])
    if args.limit:
        rows = rows[: args.limit]
    for row in rows:
        when = (row["start_ts"] or "?")[:16].replace("T", " ")
        free = " (free)" if row["cost"] == "0" else ""
        score = f"{row['score']:5.1f}" if row["score"] is not None else "    -"
        approx = "~" if row["date_approx"] else " "
        print(f"{score} {approx}{when:17} {row['title'][:52]:52} {row['venue']}{free}")


# Columns the static page actually reads. The rest (source_id, fetched_at,
# scored_at, profile_hash, notability) are bookkeeping that the browser never
# touches, and with ~1k events they are most of the payload - this file is
# fetched on every page load.
EXPORT_FIELDS = (
    "id", "source", "title", "start_ts", "end_ts", "venue", "address",
    "cost", "is_free", "categories", "url", "description", "images",
    "lat", "lon", "score", "score_reason", "scored_by", "date_approx", "key",
    "also",
    "is_new",
)
DESCRIPTION_LIMIT = 280


def _slim(event: dict) -> dict:
    out = {k: event[k] for k in EXPORT_FIELDS if k in event}
    desc = out.get("description") or ""
    if len(desc) > DESCRIPTION_LIMIT:
        out["description"] = desc[:DESCRIPTION_LIMIT].rstrip() + "..."
    return out


def _export_meta(events: list[dict]) -> dict:
    """When the data was refreshed and which model(s) did the latest ranking.

    "Latest" is the most recent day any LLM scored anything: a run that only
    had cached scores to reuse still credits the model that wrote them.
    """
    llm = [e for e in events if e.get("scored_at") and e.get("scored_by") not in (None, "heuristic")]
    models: list[str] = []
    if llm:
        last_day = max(e["scored_at"][:10] for e in llm)
        counts: dict[str, int] = {}
        for e in llm:
            if e["scored_at"][:10] == last_day:
                counts[e["scored_by"]] = counts.get(e["scored_by"], 0) + 1
        models = sorted(counts, key=counts.get, reverse=True)
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ranked_by": models,
        "new_count": sum(1 for e in events if e["is_new"]),
    }


def _cmd_export(args) -> None:
    today = date.today().isoformat()
    events = [_keyed(row_to_dict(row)) for row in query_events(args.db, order_by=args.sort)]
    # "New" means first seen by the most recent fetch run - compared by day,
    # since each fetcher stamps its own time within the run.
    last_fetch = max((e["fetched_at"][:10] for e in events if e.get("fetched_at")), default="")
    for e in events:
        e["is_new"] = bool(last_fetch) and (e.get("first_seen") or "")[:10] == last_fetch
    total = len(events)
    if not args.include_past:
        # Undated events are kept: "date TBD" is upcoming until proven otherwise.
        # Anything still running counts too - a festival on its second day, or
        # whale season that opened in April.
        events = [
            e for e in events
            if not e["start_ts"] or (e["end_ts"] or e["start_ts"])[:10] >= today
        ]
    # After the past-event cut, so a merged card never borrows a time or
    # key from an occurrence that's already over.
    merged = len(events)
    events = dedupe(events)
    merged -= len(events)
    meta = _export_meta(events)
    if not args.full:
        events = [_slim(e) for e in events]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # One compact event per line: valid JSON, a third smaller than indent=2,
    # and still reviewable as a diff - the weekly refresh commits this file,
    # and a single-line 500KB blob would make every change unreadable.
    lines = ",\n".join(
        "  " + json.dumps(e, separators=(",", ":"), default=str) for e in events
    )
    out_path.write_text(f"[\n{lines}\n]\n" if events else "[]\n")
    # Beside events.json rather than inside it, so the events file stays a
    # plain array and its diff isn't touched by a timestamp that always moves.
    (out_path.parent / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    dropped = total - len(events) - merged
    notes = []
    if dropped:
        notes.append(f"{dropped} past events dropped")
    if merged:
        notes.append(f"{merged} duplicates merged")
    note = f" ({', '.join(notes)})" if notes else ""
    print(f"exported {len(events)} events to {out_path}{note}")


# Distinct from argparse's 2 and a crash's 1, so the workflow can tell "the
# provider let us down" apart from a bug.
LLM_FAILED_EXIT = 3


def _cmd_rank(args) -> bool:
    """Returns False when --llm was asked for but didn't score everything."""
    rows = [row_to_dict(r) for r in query_events(args.db)]
    if not rows:
        print("no events to rank - run `fetch` first")
        return True

    scores = ranking.heuristic_scores(rows)
    written = set_heuristic_scores(args.db, scores)
    kept = len(scores) - written
    note = f" ({kept} left to the LLM)" if kept else ""
    print(f"heuristic: scored {written} events{note}")

    if not args.llm:
        print("(pass --llm to also rank against profile.md)")
        return True

    try:
        profile = ranking.load_profile(args.profile)
    except FileNotFoundError as exc:
        print(f"llm: SKIPPED ({exc})", file=sys.stderr)
        return False

    model = args.model or ranking.PROVIDERS[args.provider][1]
    phash = ranking.profile_hash(profile, model)

    if args.rescore_all:
        todo = [row_to_dict(r) for r in query_events(args.db)]
    else:
        todo = [row_to_dict(r) for r in unscored_events(args.db, phash)]
    if args.limit:
        todo = todo[: args.limit]

    if not todo:
        print(f"llm: nothing to do - all events already scored for profile {phash}")
        return True

    fallbacks = [p.strip() for p in args.fallback.split(",") if p.strip()]
    batch_no = 0

    def on_provider(name: str, used_model: str | None, pending: int, note: str) -> None:
        nonlocal batch_no
        batch_no = 0
        if used_model is None:
            print(f"llm: {name} {note}")
        elif note:
            print(f"llm: {name}/{used_model} {note}")
        else:
            print(f"llm: sending {pending} events to {name}/{used_model} "
                  f"in batches of {args.batch_size}")

    def progress(offset: int, size: int, note: str) -> None:
        nonlocal batch_no
        batch_no += 1
        print(f"  batch {batch_no} ({size} events): {note}")

    try:
        by_model = ranking.llm_scores_chain(
            todo, profile, [args.provider, *fallbacks],
            model=model,
            batch_size=args.batch_size, min_interval=args.min_interval,
            on_progress=progress, on_provider=on_provider,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"llm: SKIPPED ({exc})", file=sys.stderr)
        return False

    # The hash is the primary model's even for fallback scores: it marks the
    # event as done for this profile, so next week doesn't re-send it.
    # scored_by keeps the truth about which model gave the score.
    llm: dict = {}
    for scored_by, scores in by_model.items():
        set_scores(args.db, scores, scored_by=scored_by, profile_hash=phash)
        llm.update(scores)
    if len(by_model) > 1:
        detail = ", ".join(f"{k} {len(v)}" for k, v in by_model.items())
        print(f"llm: by model: {detail}")
    print(f"llm: scored {len(llm)}/{len(todo)} events for profile {phash}")
    if len(llm) < len(todo):
        print(
            f"llm: FAILED - {len(todo) - len(llm)} of {len(todo)} events left unscored",
            file=sys.stderr,
        )
        return False
    return True


def _cmd_geocode(args) -> None:
    rows = [row_to_dict(r) for r in query_events(args.db)]
    if not rows:
        print("no events to geocode - run `fetch` first")
        return

    # One cache entry per place, shared by every event at that place.
    place_of: dict[int, str] = {}
    for r in rows:
        key = geocoding.place_key(r["venue"], r["address"])
        if key:
            place_of[r["id"]] = key

    known = get_geocache(args.db)
    # Entries cached before the region check existed can be outside Northern
    # California; drop them so they get re-queried and then recorded as
    # unresolved rather than leaving a wrong pin on the map forever.
    bad = {k for k, (lat, lon) in known.items() if not geocoding.in_region(lat, lon)}
    for key in bad:
        del known[key]
    if bad:
        print(f"geocode: dropping {len(bad)} cached place(s) outside the region")
    misses = geocache_misses(args.db)
    wanted = list(dict.fromkeys(place_of.values()))
    todo = [p for p in wanted if p not in known and p not in misses]
    print(
        f"geocode: {len(wanted)} distinct places across {len(place_of)} events; "
        f"{len(known)} cached, {len(misses)} known-bad, {len(todo)} to look up"
    )

    def report(place, coords, note):
        mark = "ok " if coords else "-- "
        print(f"  {mark}{place[:48]:48} {note[:60]}")

    resolved, failed = geocoding.geocode_places(
        todo, known=known, skip=misses, limit=args.limit,
        delay=args.delay, on_result=report,
    )
    for place, (lat, lon) in resolved.items():
        put_geocache(args.db, place, lat, lon)
    for place in failed:
        put_geocache(args.db, place, None, None)

    # Copy coordinates onto every event at a known place, including events
    # that were already resolved on an earlier run.
    known.update(resolved)
    coords = {
        event_id: known[place]
        for event_id, place in place_of.items()
        if place in known
    }
    set_coordinates(args.db, coords)

    # Anything still carrying coordinates whose place didn't resolve this run
    # is stale - most likely stored before the region check rejected it.
    stale = [
        r["id"] for r in rows
        if r["lat"] is not None and r["id"] not in coords
    ]
    cleared = clear_coordinates(args.db, stale)
    if cleared:
        print(f"geocode: cleared {cleared} stale coordinate(s)")
    print(
        f"geocode: resolved {len(resolved)} new, {len(failed)} unresolved; "
        f"{len(coords)}/{len(rows)} events now have coordinates"
    )


if __name__ == "__main__":
    main()
