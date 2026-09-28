from __future__ import annotations
import io
import json
from datetime import datetime
from pathlib import Path

import pytest

from sfevents import rank
from sfevents.db import get_extraction, init_db
from sfevents.fetchers import secretsf
from sfevents.fetchers.secretsf import SecretSFFetcher
from sfevents.llm import FallbackLLM, NoProviderAvailable

POSTS = json.loads((Path(__file__).parent / "fixtures" / "secretsf_posts.json").read_text())
SUNSET, SAUSALITO, SUNDOWN, GUIDE = POSTS

# What a model would say about each fixture article.
REPLIES = {
    "Sunset District": {"events": [{
        "title": "Sunset Night Market", "start": "2026-09-25T17:00",
        "end": "2026-09-25T22:00", "venue": "Irving Street",
        "address": "Irving St, San Francisco", "cost": "Free", "kind": "market",
        "summary": "Five blocks of vendors, a circus zone and karaoke.",
    }]},
    "Italian Riviera": {"events": [{
        "title": "Sausalito Boat Show", "start": "2026-09-25", "end": "2026-09-27",
        "venue": "Clipper Yacht Harbor", "address": "Sausalito", "cost": "$25",
        "kind": "festival", "summary": "Yachts, electric boats and seminars.",
    }]},
    "Sundown Cinema": {"events": [
        {"title": "Sundown Cinema: School of Rock", "start": "2026-09-25",
         "venue": "Ferry Building", "cost": "0", "kind": "film"},
        {"title": "Sundown Cinema: Beetlejuice", "start": "2026-10-30",
         "venue": "Crane Cove Park", "cost": "0", "kind": "film"},
    ]},
}


class StubLLM:
    """Stands in for FallbackLLM: answers from REPLIES by article title."""

    def __init__(self, replies=REPLIES, down=False):
        self.replies = replies
        self.down = down
        self.prompts: list[str] = []
        self.providers = ["stub"]
        self.unconfigured: list[str] = []

    available = True

    def complete(self, prompt: str, parse=None):
        self.prompts.append(prompt)
        if self.down:
            raise NoProviderAvailable("stub: HTTP 503: high demand")
        title = prompt.split("ARTICLE TITLE\n", 1)[1].split("\n", 1)[0]
        reply = '{"events": []}'
        for needle, answer in self.replies.items():
            if needle in title:
                reply = json.dumps(answer)
                break
        return (parse(reply) if parse else reply), "stub:model"


def fetcher(tmp_path=None, llm=None):
    f = SecretSFFetcher(llm=llm or StubLLM())
    if tmp_path is not None:
        db_path = tmp_path / "events.db"
        init_db(db_path)
        f.bind_db(db_path)
    return f


# ------------------------------ extraction ------------------------------

def test_an_article_becomes_a_structured_event():
    events = fetcher().parse([SUNSET])
    [e] = events
    assert e.source == "secretsf"
    assert e.source_id == "52503:2026-09-25:sunset-night-market"
    assert e.title == "Sunset Night Market"
    assert e.start.isoformat() == "2026-09-25T17:00:00-07:00"
    assert e.end.isoformat() == "2026-09-25T22:00:00-07:00"
    assert e.venue == "Irving Street"
    assert e.is_free  # "Free" is normalised to the codebase's "0"
    assert e.categories == ["market"]
    assert e.url == SUNSET["link"]


def test_a_date_only_event_is_all_day_like_other_sources():
    [e] = fetcher().parse([SAUSALITO])
    assert e.start == datetime(2026, 9, 25)
    assert e.end == datetime(2026, 9, 27)
    assert e.cost == "$25"
    assert e.images and e.images[0].startswith("https://")


def test_one_article_can_list_several_dates():
    events = fetcher().parse([SUNDOWN])
    assert [e.venue for e in events] == ["Ferry Building", "Crane Cove Park"]
    assert len({e.source_id for e in events}) == 2


def test_an_article_that_is_not_an_event_yields_nothing():
    assert fetcher().parse([GUIDE]) == []


def test_the_prompt_carries_the_update_date_and_clean_article_text():
    llm = StubLLM()
    fetcher(llm=llm).parse([SUNSET])
    [prompt] = llm.prompts
    assert "last updated 2026-09-21" in prompt
    assert "Sunset Night Market" in prompt
    assert "<p>" not in prompt and "<strong>" not in prompt
    assert "See also" not in prompt


def test_article_text_drops_embed_chrome_and_the_related_links_footer():
    text = secretsf.article_text(
        "<p>Market is <b>Friday</b>.</p><blockquote>View this post on Instagram</blockquote>"
        "<p>Bring cash.</p><p>See also: <a>another story</a></p><p>More links</p>"
    )
    assert text == "Market is Friday.\nBring cash."


# ------------------------------ validation ------------------------------

def _one(item, post=SUNSET):
    return secretsf.to_events(post, [item], "secretsf")


def test_entries_without_a_title_or_date_are_dropped():
    assert _one({"title": "", "start": "2026-09-25"}) == []
    assert _one({"title": "A", "start": "next Friday"}) == []
    assert _one({"title": "A"}) == []


def test_a_date_before_the_article_or_far_after_it_is_a_misread_year():
    assert _one({"title": "A", "start": "2025-09-25"}) == []
    assert _one({"title": "A", "start": "2028-09-25"}) == []
    assert len(_one({"title": "A", "start": "2026-12-11"})) == 1


def test_an_end_before_the_start_is_discarded_not_trusted():
    [e] = _one({"title": "A", "start": "2026-09-25", "end": "2026-09-20"})
    assert e.end is None


def test_an_unknown_kind_is_left_uncategorised():
    [e] = _one({"title": "A", "start": "2026-09-25", "kind": "vibes"})
    assert e.categories == []


def test_two_events_on_one_date_get_distinct_ids():
    events = secretsf.to_events(SUNSET, [
        {"title": "A", "start": "2026-09-25"}, {"title": "B", "start": "2026-09-25"},
    ], "secretsf")
    assert [e.source_id for e in events] == ["52503:2026-09-25:a", "52503:2026-09-25:b"]


def test_ids_survive_a_round_up_reordering_its_items():
    a, b = {"title": "Aura", "start": "2026-09-25"}, {"title": "Opera", "start": "2026-09-25"}
    first = {e.title: e.source_id for e in secretsf.to_events(SUNSET, [a, b], "secretsf")}
    again = {e.title: e.source_id for e in secretsf.to_events(SUNSET, [b, a], "secretsf")}
    assert first == again
    assert len(set(first.values())) == 2


# ------------------------------- round-ups ------------------------------

# An evergreen post: first published in 2022, rewritten every week.
ROUNDUP = {**SUNSET, "id": 5623, "date": "2022-03-04T10:48:32",
           "modified": "2026-09-23T10:12:25",
           "title": {"rendered": "17 fun things to do this weekend: September 25-27"}}


def test_a_rewritten_round_up_is_dated_by_its_last_update():
    [e] = _one({"title": "Aura", "start": "2026-09-25", "end": "2026-09-27",
                "date_from_title": True}, post=ROUNDUP)
    assert e.date_approx
    llm = StubLLM()
    fetcher(llm=llm).parse([ROUNDUP])
    assert "last updated 2026-09-23" in llm.prompts[0]


def test_a_running_exhibit_that_opened_before_the_update_is_kept():
    assert _one({"title": "Old", "start": "2026-06-01"}, post=ROUNDUP) == []
    assert len(_one({"title": "Exhibit", "start": "2026-06-01", "end": "2026-12-01"},
                    post=ROUNDUP)) == 1


def test_a_long_round_up_is_extracted_a_chunk_at_a_time():
    section = "<h2>Event {n}</h2><p>" + "words " * 400 + "</p>"
    html = "<p>Intro.</p>" + "".join(section.format(n=n) for n in range(6))
    chunks = secretsf.article_chunks(html)
    assert len(chunks) == 3 and all(len(c) <= secretsf.MAX_ARTICLE_CHARS for c in chunks)
    # Headings never land apart from their text.
    assert all(c.startswith(("Intro.", "Event")) for c in chunks)
    llm = StubLLM()
    fetcher(llm=llm).parse([{**ROUNDUP, "content": {"rendered": html}}])
    assert len(llm.prompts) == 3 and "part 2 of 3" in llm.prompts[1]


def test_parse_reply_tolerates_code_fences_and_rejects_non_json():
    assert secretsf.parse_reply('```json\n{"events": [{"title": "A"}]}\n```') == [{"title": "A"}]
    with pytest.raises(ValueError):
        secretsf.parse_reply("Sorry, I can't help with that.")
    with pytest.raises(ValueError):
        secretsf.parse_reply('{"items": []}')


# -------------------------------- caching -------------------------------

def test_each_article_is_sent_to_the_llm_only_once(tmp_path):
    llm = StubLLM()
    f = fetcher(tmp_path, llm)
    first = f.parse(POSTS)
    second = f.parse(POSTS)

    assert len(llm.prompts) == 4, "second run re-sent articles"
    assert [e.source_id for e in first] == [e.source_id for e in second]
    assert f.report == "4 articles: 4 cached, 0 extracted"


def test_non_events_are_cached_too(tmp_path):
    """Otherwise every travel guide is re-sent to the model every week."""
    f = fetcher(tmp_path)
    f.parse([GUIDE])
    assert get_extraction(f.db_path, "secretsf", str(GUIDE["id"]), GUIDE["modified"]) == []


def test_an_edited_article_is_extracted_again(tmp_path):
    llm = StubLLM()
    f = fetcher(tmp_path, llm)
    f.parse([SUNSET])
    f.parse([{**SUNSET, "modified": "2026-09-23T10:00:00"}])
    assert len(llm.prompts) == 2


def test_a_provider_outage_is_retried_next_run_not_cached(tmp_path):
    f = fetcher(tmp_path, StubLLM(down=True))
    assert f.parse([SUNSET]) == []
    assert "1 not extracted (retried next run)" in f.report
    assert "503" in f.report

    f.llm = StubLLM()
    assert len(f.parse([SUNSET])) == 1


def test_an_unparseable_reply_is_not_cached(tmp_path, monkeypatch):
    _providers(monkeypatch, garbled=lambda *a: '{"events": [{"title": "A" "start"}]}')
    f = fetcher(tmp_path, FallbackLLM(["garbled"], sleep=lambda s: None))
    f.parse([SUNSET])
    assert get_extraction(f.db_path, "secretsf", str(SUNSET["id"]), SUNSET["modified"]) is None
    assert "unparseable reply" in f.report


def test_without_any_llm_key_the_report_says_so(monkeypatch):
    for name in ("groq", "gemini", "ollama"):
        monkeypatch.delenv(rank.PROVIDERS[name][0], raising=False)
    f = SecretSFFetcher(providers=("groq", "gemini", "ollama"))
    assert f.parse([SUNSET]) == []
    assert "no LLM key set for groq, gemini, ollama" in f.report


# ---------------------------------- fetch -------------------------------

def test_fetch_asks_for_recent_non_sponsored_things_to_do(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return io.BytesIO(json.dumps([SUNSET]).encode())

    monkeypatch.setattr(secretsf.urllib.request, "urlopen", fake_urlopen)
    events = SecretSFFetcher(llm=StubLLM()).fetch()

    assert len(events) == 1
    assert "categories=13%2C2967%2C3" in seen["url"]
    assert "categories_exclude=12" in seen["url"]
    assert "after=" in seen["url"]
    assert "wp%3Afeaturedmedia" in seen["url"]


def test_fetch_follows_every_page(monkeypatch):
    urls = []

    class Page(io.BytesIO):
        headers = {"X-WP-TotalPages": "2"}

    def fake_urlopen(req, timeout=None):
        urls.append(req.full_url)
        post = SUNSET if len(urls) == 1 else SUNDOWN
        return Page(json.dumps([post]).encode())

    monkeypatch.setattr(secretsf.urllib.request, "urlopen", fake_urlopen)
    events = SecretSFFetcher(llm=StubLLM()).fetch()

    assert ["page=1" in urls[0], "page=2" in urls[1], len(urls)] == [True, True, 2]
    assert {e.title for e in events} >= {"Sunset Night Market", "Sundown Cinema: School of Rock"}


# ----------------------------- FallbackLLM ------------------------------

def _providers(monkeypatch, **fns):
    for name, fn in fns.items():
        monkeypatch.setitem(rank.PROVIDERS, name, (f"{name.upper()}_KEY", f"{name}-m", fn))
        monkeypatch.setenv(f"{name.upper()}_KEY", "x")
    monkeypatch.setattr(rank, "RETRY_WAITS", (0, 0, 0))


def _down(prompt, model, key, timeout):
    raise rank.ProviderError("HTTP 503: high demand", status=503)


def test_fallback_llm_moves_on_when_a_provider_is_down(monkeypatch):
    _providers(monkeypatch, first=_down, second=lambda *a: "hello")
    llm = FallbackLLM(["first", "second"], sleep=lambda s: None)
    assert llm.complete("hi") == ("hello", "second:second-m")


def test_fallback_llm_stops_asking_a_provider_whose_quota_is_gone(monkeypatch):
    calls = {"first": 0}

    def quota(prompt, model, key, timeout):
        calls["first"] += 1
        raise rank.ProviderError("HTTP 429: quota", status=429, daily=True)

    _providers(monkeypatch, first=quota, second=lambda *a: "ok")
    llm = FallbackLLM(["first", "second"], sleep=lambda s: None)
    llm.complete("a")
    llm.complete("b")
    assert calls["first"] == 1


def test_fallback_llm_paces_each_provider(monkeypatch):
    _providers(monkeypatch, first=lambda *a: "ok")
    waits: list[float] = []
    ticks = iter([0, 5, 5])  # first call at t=0, second asks at t=5
    llm = FallbackLLM(["first"], min_interval={"first": 15}, sleep=waits.append,
                      clock=lambda: next(ticks))
    llm.complete("a")
    llm.complete("b")
    assert waits == [10]


def test_fallback_llm_skips_unconfigured_providers(monkeypatch):
    _providers(monkeypatch, second=lambda *a: "ok")
    monkeypatch.delenv("FIRST_KEY", raising=False)
    monkeypatch.setitem(rank.PROVIDERS, "first", ("FIRST_KEY", "m", _down))
    llm = FallbackLLM(["first", "second"])
    assert llm.unconfigured == ["first"]
    assert llm.complete("a")[1] == "second:second-m"


def test_fallback_llm_with_nothing_configured_says_so(monkeypatch):
    monkeypatch.setitem(rank.PROVIDERS, "first", ("FIRST_KEY", "m", _down))
    monkeypatch.delenv("FIRST_KEY", raising=False)
    with pytest.raises(NoProviderAvailable, match="no provider configured"):
        FallbackLLM(["first"]).complete("a")


def test_a_malformed_reply_gets_one_more_try(monkeypatch):
    """Seen live: gpt-oss dropped a comma on 4 of 25 articles."""
    replies = iter(['{"events": [{"title": "A" "start": "2026-09-25"}]}',
                    '{"events": [{"title": "A", "start": "2026-09-25"}]}'])
    _providers(monkeypatch, first=lambda *a: next(replies))
    llm = FallbackLLM(["first"], sleep=lambda s: None)
    result, by = llm.complete("x", parse=secretsf.parse_reply)
    assert result == [{"title": "A", "start": "2026-09-25"}]


def test_repeated_showtimes_become_one_card():
    """Seen live: one bar's six showtimes came back as six events."""
    items = [{"title": "The Nightmare Bar", "start": f"2026-10-{d:02d}T19:00"} for d in (8, 9, 12)]
    events = SecretSFFetcher(llm=StubLLM({"Sunset District": {"events": items}})).parse([SUNSET])
    [e] = events
    assert e.start.day == 8
    assert e.description.startswith("Repeats: 2 more dates through Oct 12.")


def test_an_articles_old_events_are_replaced_not_left_as_duplicates(tmp_path):
    """Seen live: six per-showtime rows survived the switch to one card."""
    from sfevents.db import prune_article_events, query_events, upsert_events
    f = fetcher(tmp_path)
    shows = [{"title": "Show", "start": f"2026-10-{d:02d}"} for d in (8, 9)]
    old = secretsf.to_events(SUNSET, shows, "secretsf")
    unrelated = secretsf.to_events({**SAUSALITO, "id": 1}, [{"title": "B", "start": "2026-09-25"}],
                                   "secretsf")
    upsert_events(f.db_path, old + unrelated)

    new = f.parse([SUNSET])
    upsert_events(f.db_path, new)
    removed = prune_article_events(f.db_path, "secretsf", f.covered_articles,
                                   [e.source_id for e in new])

    assert removed == 2  # the two Show dates -> only the market now
    ids = sorted(r["source_id"] for r in query_events(f.db_path))
    assert ids == ["1:2026-09-25:b", "52503:2026-09-25:sunset-night-market"]


def test_an_article_that_failed_to_extract_keeps_its_rows(tmp_path):
    f = fetcher(tmp_path, StubLLM(down=True))
    f.parse([SUNSET])
    assert f.covered_articles == []


def test_two_articles_about_one_show_make_one_card():
    """Seen live: a preview and a "tickets on sale" post for the same concert."""
    show = {"title": "Candlelight: The Lord of the Rings", "start": "2026-12-04T18:30"}
    other_date = {**show, "start": "2027-01-16T18:30"}
    llm = StubLLM({"Sunset District": {"events": [show]},
                   "Italian Riviera": {"events": [{**show, "title": "Candlelight - The Lord of the Rings"},
                                                  other_date]}})
    events = SecretSFFetcher(llm=llm).parse([SUNSET, SAUSALITO])
    assert [(e.source_id.split(":")[0], e.start.date().isoformat()) for e in events] == [
        ("52503", "2026-12-04"), ("52469", "2027-01-16"),
    ]
