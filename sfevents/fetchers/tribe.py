from __future__ import annotations
import html
import json
import re
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9
    ZoneInfo = None

from ..models import Event

API_PATH = "/wp-json/tribe/events/v1/events"
PER_PAGE = 50
MAX_PAGES = 20


class TribeEventsFetcher:
    """Fetches a WordPress site's "The Events Calendar" (Tribe) REST feed.

    Many visitor bureaus and towns outside SF run this one plugin, and it
    exposes the same JSON everywhere - so one fetcher, pointed at several
    sites, reaches the coast, the wine country and the mountains without a
    scraper per site. Sites are listed in cli.FETCHERS.

    Tourism calendars repeat themselves: Santa Cruz lists the Roaring Camp
    steam train 100 times in three months. Only the next date of each title
    is kept, with the rest summarized in its description, so a daily tour
    doesn't bury every other event (and cost 100 LLM scores). Next week's
    fetch picks up the following date.

    A day-trip calendar also lists every weeknight contra dance and sailboat
    race in its town - hundreds of listings that would outnumber SF's own.
    `weekends_only` keeps what could make a day trip (profile.md: weeknights
    are for SF); the export's quota (cli.SOURCE_GROUPS) then keeps the best
    few per weekend.
    """

    def __init__(self, name: str, site: str, days: int = 60, timeout: int = 30,
                 today: date | None = None, weekends_only: bool = False):
        self.name = name
        self.site = site.rstrip("/")
        self.days = days
        self.timeout = timeout
        self._today = today  # injectable for tests
        self.weekends_only = weekends_only

    def fetch(self) -> list[Event]:
        today = self._today or date.today()
        until = today + timedelta(days=self.days)
        items: list[dict] = []
        for page in range(1, MAX_PAGES + 1):
            query = urllib.parse.urlencode({
                "per_page": PER_PAGE,
                "page": page,
                "start_date": today.isoformat(),
                "end_date": until.isoformat(),
            })
            req = urllib.request.Request(
                f"{self.site}{API_PATH}?{query}",
                headers={"User-Agent": "sfevents/0.1"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items.extend(data.get("events") or [])
            if page >= int(data.get("total_pages") or 1):
                break
        return self.parse(items)

    def parse(self, items: list[dict]) -> list[Event]:
        events = [e for e in (self._event(item) for item in items) if e]
        if self.weekends_only:
            events = [e for e in events if touches_weekend(e.start, e.end)]
        return collapse_repeats(events)

    def _event(self, item: dict) -> Event | None:
        if item.get("hide_from_listings") or item.get("status") not in (None, "publish"):
            return None
        title = _clean(item.get("title"))
        if not title or not item.get("start_date"):
            return None
        tz = _zone(item.get("timezone"))
        venue = item.get("venue") if isinstance(item.get("venue"), dict) else {}
        address = ", ".join(
            p for p in (_clean(venue.get("address")), _clean(venue.get("city"))) if p
        )
        categories = [_clean(c.get("name")) for c in item.get("categories") or []]
        image = item.get("image") if isinstance(item.get("image"), dict) else {}
        return Event(
            source=self.name,
            source_id=str(item.get("id") or item.get("url")),
            title=title,
            start=_dt(item.get("start_date"), tz, item.get("all_day")),
            end=_dt(item.get("end_date"), tz, item.get("all_day")),
            venue=_clean(venue.get("venue")),
            address=address,
            cost=_cost(item.get("cost"), categories),
            categories=[c for c in categories if c],
            url=item.get("url") or "",
            description=_text(item.get("description"))[:600],
            images=[image["url"]] if image.get("url") else [],
        )


def touches_weekend(start: datetime | None, end: datetime | None) -> bool:
    """Whether an event is on, or runs through, a Saturday or Sunday.

    Undated counts as yes: there's no telling it isn't.
    """
    if start is None:
        return True
    first = start.date()
    last = max(end.date(), first) if end is not None else first
    return (last - first).days >= 6 or any(
        (first + timedelta(days=i)).weekday() >= 5 for i in range((last - first).days + 1)
    )


def collapse_repeats(events: list[Event]) -> list[Event]:
    groups: dict[str, list[Event]] = {}
    # All-day dates are naive and timed ones aware; both are the site's local
    # time, so compare them as wall-clock times.
    for e in sorted(events, key=lambda e: (e.start or datetime.max).replace(tzinfo=None)):
        groups.setdefault(e.title.casefold(), []).append(e)
    out = []
    for group in groups.values():
        first = group[0]
        if len(group) > 1:
            last = group[-1].start
            note = f"Repeats: {len(group) - 1} more date{'s' if len(group) > 2 else ''}"
            if last:
                note += f" through {last:%b} {last.day}"
            first.description = f"{note}. {first.description}".strip()
        out.append(first)
    return out


def _clean(value) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def _text(fragment) -> str:
    """HTML description to plain text."""
    text = re.sub(r"<(br|/p|/li|/h\d)[^>]*>", " ", str(fragment or ""), flags=re.I)
    return _clean(re.sub(r"<[^>]+>", "", text))


def _zone(name):
    if not name or ZoneInfo is None:
        return None
    try:
        return ZoneInfo(name)
    except Exception:
        return None


def _dt(value, tz, all_day) -> datetime | None:
    try:
        dt = datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None
    # An all-day event has no clock time worth converting; keep it a plain
    # date so it doesn't slide into the previous day in UTC.
    if all_day:
        return datetime(dt.year, dt.month, dt.day)
    return dt.replace(tzinfo=tz) if tz else dt


def _cost(raw, categories: list[str]) -> str:
    cost = _clean(raw)
    if cost.lower() in ("free", "$0", "0", "$0.00") or (
        not cost and any(c.lower() in ("free events", "free") for c in categories)
    ):
        return "0"
    return cost
