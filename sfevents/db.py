from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .models import Event

SCHEMA_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT NOT NULL,
    start_ts TEXT,
    end_ts TEXT,
    venue TEXT,
    address TEXT,
    cost TEXT,
    categories TEXT,
    url TEXT,
    description TEXT,
    fetched_at TEXT NOT NULL,
    first_seen TEXT,
    notability INTEGER NOT NULL DEFAULT 0,
    date_approx INTEGER NOT NULL DEFAULT 0,
    images TEXT,
    lat REAL,
    lon REAL,
    score REAL,
    score_reason TEXT,
    scored_by TEXT,
    scored_at TEXT,
    profile_hash TEXT,
    UNIQUE(source, source_id)
);
"""

# Indexes are applied AFTER migrations, not as part of SCHEMA: idx_events_score
# references a column that pre-ranking databases don't have yet, so creating it
# before ALTER TABLE runs fails outright and makes an existing database
# impossible to open.
# Geocode cache keyed by place rather than by event: venues repeat heavily
# (about 300 distinct places across ~1000 events) and outlive any single
# listing, so this is what keeps repeat runs from re-querying the geocoder.
GEOCACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS geocache (
    place_key TEXT PRIMARY KEY,
    lat REAL,
    lon REAL,
    display_name TEXT,
    resolved_at TEXT NOT NULL
);
"""

# What an LLM pulled out of an article, keyed by the article and its last
# edit. Editorial sources (Secret SF) publish events as prose, so turning
# one into dates and venues costs a model call - this makes that a one-time
# cost per article revision, not a weekly one. `result` is the JSON the
# fetcher parsed, kept so a run with every provider down still has events.
EXTRACTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS article_extractions (
    source TEXT NOT NULL,
    article_id TEXT NOT NULL,
    modified TEXT NOT NULL,
    result TEXT NOT NULL,
    extracted_by TEXT NOT NULL,
    extracted_at TEXT NOT NULL,
    PRIMARY KEY (source, article_id)
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_ts);
CREATE INDEX IF NOT EXISTS idx_events_score ON events(score);
"""

SCHEMA = SCHEMA_TABLE + INDEXES

# Columns added after the first release. SQLite can't add them conditionally
# in a script, so init_db diffs against the live table instead of requiring
# anyone to delete and re-fetch their database.
MIGRATIONS = {
    "images": "TEXT",
    "lat": "REAL",
    "lon": "REAL",
    "notability": "INTEGER NOT NULL DEFAULT 0",
    "date_approx": "INTEGER NOT NULL DEFAULT 0",
    "score": "REAL",
    "score_reason": "TEXT",
    "scored_by": "TEXT",
    "scored_at": "TEXT",
    "profile_hash": "TEXT",
    "first_seen": "TEXT",
}


@contextmanager
def connect(db_path: str | Path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | Path) -> None:
    """Create the table if absent, bring an older one up to date, then index.

    Strict ordering: the score index can't be created until the migration has
    added the column it covers.
    """
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_TABLE)
        conn.executescript(GEOCACHE_SCHEMA)
        conn.executescript(EXTRACTION_SCHEMA)
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(events)")}
        for column, decl in MIGRATIONS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE events ADD COLUMN {column} {decl}")
        # Rows from before first_seen existed: their last fetch is the best
        # guess, and it predates the next fetch, so they don't all show as new.
        conn.execute("UPDATE events SET first_seen = fetched_at WHERE first_seen IS NULL")
        conn.executescript(INDEXES)


def upsert_events(db_path: str | Path, events: list[Event]) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with connect(db_path) as conn:
        for e in events:
            conn.execute(
                """INSERT INTO events
                   (source, source_id, title, start_ts, end_ts, venue, address,
                    cost, categories, url, description, fetched_at, first_seen,
                    notability, date_approx, images)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(source, source_id) DO UPDATE SET
                     title=excluded.title, start_ts=excluded.start_ts, end_ts=excluded.end_ts,
                     venue=excluded.venue, address=excluded.address, cost=excluded.cost,
                     categories=excluded.categories, url=excluded.url,
                     description=excluded.description, fetched_at=excluded.fetched_at,
                     notability=excluded.notability, date_approx=excluded.date_approx,
                     images=excluded.images""",
                (
                    e.source, e.source_id, e.title,
                    e.start.isoformat() if e.start else None,
                    e.end.isoformat() if e.end else None,
                    e.venue, e.address, e.cost, ",".join(e.categories),
                    e.url, e.description, now, now,
                    e.notability, int(e.date_approx),
                    json.dumps(e.images) if e.images else None,
                ),
            )
    return len(events)


def query_events(
    db_path: str | Path,
    *,
    start_after: datetime | None = None,
    start_before: datetime | None = None,
    free_only: bool = False,
    category: str | None = None,
    order_by: str = "date",
) -> list[sqlite3.Row]:
    sql = "SELECT * FROM events WHERE 1=1"
    params: list = []
    if start_after:
        sql += " AND start_ts >= ?"
        params.append(start_after.isoformat())
    if start_before:
        sql += " AND start_ts <= ?"
        params.append(start_before.isoformat())
    if free_only:
        sql += " AND cost = '0'"
    if category:
        sql += " AND categories LIKE ?"
        params.append(f"%{category}%")
    if order_by == "score":
        # Unscored events sort last rather than first, which is what a NULL
        # would do under DESC. Date breaks ties so a scoreless run still
        # reads chronologically.
        sql += " ORDER BY score IS NULL, score DESC, start_ts ASC"
    elif order_by == "date":
        sql += " ORDER BY start_ts ASC"
    else:
        raise ValueError(f"unknown order_by: {order_by!r}")
    with connect(db_path) as conn:
        return conn.execute(sql, params).fetchall()


def delete_event(db_path: str | Path, event_id: int) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM events WHERE id = ?", (event_id,))


def get_event(db_path: str | Path, event_id: int) -> sqlite3.Row | None:
    with connect(db_path) as conn:
        return conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()


UPDATABLE_FIELDS = {
    "title", "start_ts", "end_ts", "venue", "address",
    "cost", "categories", "url", "description",
}


def update_event(db_path: str | Path, event_id: int, **fields) -> sqlite3.Row | None:
    """Partial update. Unknown keys are ignored; only UPDATABLE_FIELDS are written."""
    changes = {k: v for k, v in fields.items() if k in UPDATABLE_FIELDS}
    if not changes:
        return get_event(db_path, event_id)
    set_clause = ", ".join(f"{k} = ?" for k in changes)
    with connect(db_path) as conn:
        conn.execute(
            f"UPDATE events SET {set_clause} WHERE id = ?",
            (*changes.values(), event_id),
        )
    return get_event(db_path, event_id)


def insert_manual_event(db_path: str | Path, event: Event) -> sqlite3.Row:
    """Insert a single user-created event (e.g. an Ocean Beach bonfire) and return the stored row."""
    upsert_events(db_path, [event])
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM events WHERE source = ? AND source_id = ?",
            (event.source, event.source_id),
        ).fetchone()


def count_events(db_path: str | Path) -> int:
    with connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]


def row_to_dict(row: sqlite3.Row) -> dict:
    """Shared shape for API responses and the static-site JSON export."""
    d = dict(row)
    d["categories"] = [c for c in (d["categories"] or "").split(",") if c]
    d["is_free"] = d["cost"] == "0"
    d["date_approx"] = bool(d.get("date_approx"))
    d["images"] = json.loads(d["images"]) if d.get("images") else []
    return d


def set_scores(db_path: str | Path, scores: dict[int, tuple[float, str]],
               scored_by: str, profile_hash: str) -> int:
    """Write ranker output back onto rows, keyed by event id."""
    now = datetime.now(timezone.utc).isoformat()
    with connect(db_path) as conn:
        for event_id, (score, reason) in scores.items():
            conn.execute(
                """UPDATE events
                      SET score = ?, score_reason = ?, scored_by = ?,
                          scored_at = ?, profile_hash = ?
                    WHERE id = ?""",
                (float(score), reason, scored_by, now, profile_hash, event_id),
            )
    return len(scores)


def set_heuristic_scores(db_path: str | Path, scores: dict[int, tuple[float, str]]) -> int:
    """Write heuristic scores, but never over an LLM's verdict.

    The heuristic runs on every event on every run, while the LLM only runs
    on events it has not seen. Letting the heuristic overwrite rows the LLM
    has already scored would blank their profile_hash and make every event
    look unscored again next week - so rows carrying an LLM's provenance are
    left alone, and keep the hash that lets `unscored_events` skip them.
    """
    now = datetime.now(timezone.utc).isoformat()
    written = 0
    with connect(db_path) as conn:
        for event_id, (score, reason) in scores.items():
            cur = conn.execute(
                """UPDATE events
                      SET score = ?, score_reason = ?, scored_by = 'heuristic',
                          scored_at = ?, profile_hash = ''
                    WHERE id = ?
                      AND (scored_by IS NULL OR scored_by = 'heuristic')""",
                (float(score), reason, now, event_id),
            )
            written += cur.rowcount
    return written


def unscored_events(db_path: str | Path, profile_hash: str,
                    scored_by: str | None = None) -> list[sqlite3.Row]:
    """Rows that still need scoring for this profile.

    A row counts as needing work if it has no score, or if it was scored
    against a different profile revision, or (when scored_by is given) by a
    different ranker - so editing profile.md re-ranks everything on the next
    run without re-ranking on every unrelated run.
    """
    sql = "SELECT * FROM events WHERE score IS NULL OR profile_hash IS NOT ? "
    params: list = [profile_hash]
    if scored_by is not None:
        sql += "OR scored_by IS NOT ? "
        params.append(scored_by)
    sql += "ORDER BY start_ts ASC"
    with connect(db_path) as conn:
        return conn.execute(sql, params).fetchall()


def get_geocache(db_path: str | Path) -> dict[str, tuple[float, float]]:
    """Every resolved place, so a run can geocode only what it hasn't seen."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT place_key, lat, lon FROM geocache WHERE lat IS NOT NULL"
        ).fetchall()
    return {r["place_key"]: (r["lat"], r["lon"]) for r in rows}


def geocache_misses(db_path: str | Path) -> set[str]:
    """Places looked up before and found unresolvable.

    Recorded so a run doesn't spend its request budget re-asking about the
    same unparseable venue string every week.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT place_key FROM geocache WHERE lat IS NULL"
        ).fetchall()
    return {r["place_key"] for r in rows}


def put_geocache(db_path: str | Path, place_key: str, lat: float | None,
                 lon: float | None, display_name: str = "") -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO geocache (place_key, lat, lon, display_name, resolved_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(place_key) DO UPDATE SET
                 lat=excluded.lat, lon=excluded.lon,
                 display_name=excluded.display_name, resolved_at=excluded.resolved_at""",
            (place_key, lat, lon, display_name, now),
        )


def clear_coordinates(db_path: str | Path, event_ids: list[int]) -> int:
    """Blank coordinates on events whose place is no longer resolvable.

    Needed because set_coordinates only writes places that resolved: without
    this, a coordinate rejected or invalidated after it was first stored
    would stay on the row and keep a wrong pin on the map.
    """
    if not event_ids:
        return 0
    with connect(db_path) as conn:
        conn.executemany(
            "UPDATE events SET lat = NULL, lon = NULL WHERE id = ?",
            [(i,) for i in event_ids],
        )
    return len(event_ids)


def set_coordinates(db_path: str | Path, coords: dict[int, tuple[float, float]]) -> int:
    with connect(db_path) as conn:
        for event_id, (lat, lon) in coords.items():
            conn.execute(
                "UPDATE events SET lat = ?, lon = ? WHERE id = ?", (lat, lon, event_id)
            )
    return len(coords)


def get_extraction(db_path: str | Path, source: str, article_id: str,
                   modified: str) -> list | dict | None:
    """The cached extraction for this revision of an article, if any."""
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT result FROM article_extractions
                WHERE source = ? AND article_id = ? AND modified = ?""",
            (source, article_id, modified),
        ).fetchone()
    return json.loads(row["result"]) if row else None


def put_extraction(db_path: str | Path, source: str, article_id: str,
                   modified: str, result, extracted_by: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO article_extractions
                   (source, article_id, modified, result, extracted_by, extracted_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(source, article_id) DO UPDATE SET
                 modified=excluded.modified, result=excluded.result,
                 extracted_by=excluded.extracted_by, extracted_at=excluded.extracted_at""",
            (source, article_id, modified, json.dumps(result), extracted_by, now),
        )


def prune_article_events(db_path: str | Path, source: str, article_ids,
                         keep_source_ids) -> int:
    """Drop rows an article used to produce but no longer does.

    Upserts never delete, and an article's events can change under it - the
    site edits the article, or the extraction merges six showtimes into one
    card - leaving the old rows behind as duplicates. For every article this
    run covered (source_id "<article id>:..."), only the rows it produced
    now survive. Articles outside this run's window are left alone.
    """
    keep = set(keep_source_ids)
    removed = 0
    with connect(db_path) as conn:
        for article_id in set(article_ids):
            rows = conn.execute(
                "SELECT id, source_id FROM events WHERE source = ? AND source_id LIKE ?",
                (source, f"{article_id}:%"),
            ).fetchall()
            stale = [r["id"] for r in rows if r["source_id"] not in keep]
            for event_id in stale:
                conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
            removed += len(stale)
    return removed


def drop_events(db_path: str | Path, source: str, keep_title) -> int:
    """Delete a source's rows whose title `keep_title` now rejects.

    For listings a fetcher has learned to skip: upserts never delete, so
    without this the rows stored before the filter existed stay forever.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, title FROM events WHERE source = ?", (source,)
        ).fetchall()
        doomed = [r["id"] for r in rows if not keep_title(r["title"])]
        conn.executemany("DELETE FROM events WHERE id = ?", [(i,) for i in doomed])
    return len(doomed)
