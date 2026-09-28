from __future__ import annotations
from datetime import datetime

from sfevents.fetchers.tribe import TribeEventsFetcher


def item(id, title, start, end=None, **extra):
    base = {
        "id": id, "title": title, "start_date": start, "end_date": end or start,
        "all_day": False, "timezone": "America/Los_Angeles", "status": "publish",
        "url": f"https://example.test/e/{id}", "cost": "", "categories": [],
        "description": "", "venue": {}, "image": False,
    }
    return {**base, **extra}


def parse(items):
    return TribeEventsFetcher("santacruz_org", "https://example.test").parse(items)


def test_fields_map_and_html_is_decoded():
    [e] = parse([item(
        7, "Kids &amp; Families: Tide Pools", "2026-10-03 09:00:00", "2026-10-03 11:00:00",
        cost="Free",
        categories=[{"name": "Food &amp; Drink"}, {"name": "Nature"}],
        description="<p>Meet at the <b>stairs</b>.</p><p>Bring boots.</p>",
        venue={"venue": "Natural Bridges", "address": "2531 W Cliff Dr", "city": "Santa Cruz"},
        image={"url": "https://example.test/i.jpg"},
    )])
    assert e.source == "santacruz_org" and e.source_id == "7"
    assert e.title == "Kids & Families: Tide Pools"
    assert e.categories == ["Food & Drink", "Nature"]
    assert e.description == "Meet at the stairs. Bring boots."
    assert e.venue == "Natural Bridges"
    assert e.address == "2531 W Cliff Dr, Santa Cruz"
    assert e.is_free
    assert e.images == ["https://example.test/i.jpg"]
    assert e.start.isoformat() == "2026-10-03T09:00:00-07:00"


def test_all_day_events_stay_on_their_local_date():
    [e] = parse([item(1, "Triathlon", "2026-09-27 00:00:00", "2026-09-27 23:59:59", all_day=True)])
    assert e.start == datetime(2026, 9, 27)


def test_a_repeating_listing_keeps_only_its_next_date():
    events = parse([
        item(3, "Steam Train", "2026-10-10 11:00:00"),
        item(1, "Steam Train", "2026-09-20 11:00:00"),
        item(2, "Steam train", "2026-09-27 00:00:00", all_day=True),
        item(4, "Harvest Fair", "2026-10-04 10:00:00"),
    ])
    assert [e.title for e in events] == ["Steam Train", "Harvest Fair"]
    train = events[0]
    assert train.source_id == "1"
    assert train.description.startswith("Repeats: 2 more dates through Oct 10.")


def test_hidden_and_unpublished_listings_are_skipped():
    events = parse([
        item(1, "Hidden", "2026-10-01 10:00:00", hide_from_listings=True),
        item(2, "Draft", "2026-10-01 10:00:00", status="draft"),
        item(3, "Real", "2026-10-01 10:00:00"),
    ])
    assert [e.title for e in events] == ["Real"]


def test_a_day_trip_calendar_keeps_only_weekend_events():
    f = TribeEventsFetcher("santacruz_org", "https://example.test", weekends_only=True)
    events = f.parse([
        item(1, "Wednesday Sailboat Races", "2026-09-30 18:00:00"),
        item(2, "Chili Cook-Off", "2026-10-24 11:00:00"),  # Saturday
        item(3, "Plein Air Festival", "2026-10-30 09:00:00", "2026-11-02 17:00:00"),  # Fri-Mon
        item(4, "Exhibit", "2026-10-05 10:00:00", "2026-10-09 17:00:00"),  # Mon-Fri
    ])
    assert [e.title for e in events] == ["Chili Cook-Off", "Plein Air Festival"]
