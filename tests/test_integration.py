from __future__ import annotations
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from sfevents.db import init_db, upsert_events, query_events, count_events
from sfevents.fetchers.funcheap import FuncheapFetcher
from sfevents.fetchers.dothebay import DoTheBayFetcher
from sfevents import cli as cli_module

FIXTURE = Path(__file__).parent / "fixtures" / "funcheap_sample.xml"
DOTHEBAY_FIXTURE = Path(__file__).parent / "fixtures" / "dothebay_sample.html"


class FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._data


@pytest.fixture
def fake_network(monkeypatch):
    """Point every fetcher's urllib calls at captured/hand-built fixtures instead
    of the live network, so CLI-level tests stay deterministic and offline.

    Both fetcher modules do `import urllib.request`, so they share the exact
    same global module object - patching urlopen via two separate
    `monkeypatch.setattr("module.urllib.request.urlopen", ...)` calls would
    just overwrite the same global symbol twice, silently breaking whichever
    fetcher was patched first. One URL-dispatching fake avoids that.
    """
    funcheap_bytes = FIXTURE.read_bytes()
    dothebay_bytes = DOTHEBAY_FIXTURE.read_bytes()

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        if "dothebay.com" in url:
            return FakeResponse(dothebay_bytes)
        return FakeResponse(funcheap_bytes)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


@pytest.fixture
def only_fixture_fetchers(monkeypatch):
    """Restrict the CLI to the two sources that have fixtures.

    These tests exercise CLI orchestration - fetch/list/export wiring - not
    the source registry, so they pin the fetcher list rather than asserting
    against whatever FETCHERS currently holds. Without this, adding a source
    breaks them, and AnnualEventsFetcher in particular reads a local data
    file, so `fake_network` cannot stub it out.
    """
    monkeypatch.setattr(
        cli_module, "FETCHERS", [FuncheapFetcher(), DoTheBayFetcher()]
    )


def test_pipeline_parse_store_and_filter(tmp_path):
    db_path = tmp_path / "events.db"
    init_db(db_path)

    events = FuncheapFetcher().parse(FIXTURE.read_bytes())
    upsert_events(db_path, events)

    assert count_events(db_path) == 3

    free_events = query_events(db_path, free_only=True)
    assert len(free_events) == 3  # all fixture events are free

    outdoors = query_events(db_path, category="Outdoors")
    assert len(outdoors) == 1
    assert "Union Square" in outdoors[0]["title"]


def test_pipeline_is_idempotent_on_repeated_fetch(tmp_path):
    """Running the pipeline twice (as a scheduled job would) must not duplicate rows."""
    db_path = tmp_path / "events.db"
    init_db(db_path)
    events = FuncheapFetcher().parse(FIXTURE.read_bytes())

    upsert_events(db_path, events)
    upsert_events(db_path, events)  # simulate a second scheduled run

    assert count_events(db_path) == 3


def run_cli(args: list[str]) -> str:
    old_argv = sys.argv
    sys.argv = ["sfevents", *args]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            cli_module.main()
    finally:
        sys.argv = old_argv
    return buf.getvalue()


def test_cli_fetch_then_list_end_to_end(tmp_path, fake_network, only_fixture_fetchers):
    db_path = tmp_path / "events.db"

    fetch_output = run_cli(["--db", str(db_path), "fetch"])
    assert "funcheap_sf: 3 events fetched" in fetch_output
    assert "dothebay: 4 events fetched" in fetch_output

    list_output = run_cli(["--db", str(db_path), "list"])
    lines = [l for l in list_output.splitlines() if l.strip()]
    assert len(lines) == 7  # 3 funcheap + 4 dothebay
    # soonest event (Oct) should appear before latest (Dec)
    assert list_output.index("Union Square") < list_output.index("Beer Ride")


def test_cli_list_on_empty_db_prints_nothing(tmp_path):
    db_path = tmp_path / "events.db"
    output = run_cli(["--db", str(db_path), "list"])
    assert output.strip() == ""


def test_cli_creates_db_file_and_parent_dirs(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "events.db"
    run_cli(["--db", str(db_path), "list"])
    assert db_path.exists()


def test_cli_second_fetch_does_not_duplicate_rows(tmp_path, fake_network, only_fixture_fetchers):
    db_path = tmp_path / "events.db"
    run_cli(["--db", str(db_path), "fetch"])
    run_cli(["--db", str(db_path), "fetch"])
    assert count_events(db_path) == 7


def run_cli_capture_stderr(args: list[str]) -> tuple[str, str]:
    """Like run_cli, but also captures stderr (used for warnings/failures)."""
    import contextlib
    old_argv = sys.argv
    sys.argv = ["sfevents", *args]
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), contextlib.redirect_stderr(err):
            cli_module.main()
    finally:
        sys.argv = old_argv
    return out.getvalue(), err.getvalue()


def test_a_failing_fetcher_does_not_block_the_others(tmp_path, monkeypatch, only_fixture_fetchers):
    """If DoTheBay's scraper breaks (e.g. site redesign), Funcheap must still work."""
    funcheap_bytes = FIXTURE.read_bytes()

    def dispatch_urlopen(req, timeout=None):
        if "dothebay.com" in req.full_url:
            raise RuntimeError("simulated DoTheBay outage/redesign")
        return FakeResponse(funcheap_bytes)

    monkeypatch.setattr("urllib.request.urlopen", dispatch_urlopen)

    db_path = tmp_path / "events.db"
    out, err = run_cli_capture_stderr(["--db", str(db_path), "fetch"])

    assert "funcheap_sf: 3 events fetched" in out
    assert "dothebay: FAILED" in err
    assert count_events(db_path) == 3  # funcheap's events still made it in


def test_zero_results_from_a_fragile_source_prints_a_warning(tmp_path, monkeypatch, only_fixture_fetchers):
    funcheap_bytes = FIXTURE.read_bytes()
    empty_html = b"<html><body>no events here</body></html>"

    def dispatch_urlopen(req, timeout=None):
        if "dothebay.com" in req.full_url:
            return FakeResponse(empty_html)
        return FakeResponse(funcheap_bytes)

    monkeypatch.setattr("urllib.request.urlopen", dispatch_urlopen)

    db_path = tmp_path / "events.db"
    out, err = run_cli_capture_stderr(["--db", str(db_path), "fetch"])

    assert "dothebay: 0 events fetched" in out
    assert "warning: dothebay returned 0 events" in err


def test_export_writes_valid_json_with_expected_shape(tmp_path, fake_network, only_fixture_fetchers):
    import json

    db_path = tmp_path / "events.db"
    out_path = tmp_path / "docs" / "data" / "events.json"

    run_cli(["--db", str(db_path), "fetch"])
    output = run_cli(
        ["--db", str(db_path), "export", "--out", str(out_path), "--include-past", "--full"]
    )

    assert "exported 7 events" in output
    assert out_path.exists()

    data = json.loads(out_path.read_text())
    assert len(data) == 7
    assert all("id" in e and "title" in e and "start_ts" in e for e in data)
    # categories must be split back into a list, not left as the DB's comma-joined string
    assert all(isinstance(e["categories"], list) for e in data)
    assert all("is_free" in e for e in data)
    funcheap_events = [e for e in data if e["source"] == "funcheap_sf"]
    assert all(e["is_free"] for e in funcheap_events)  # all fixture events are free


def test_export_creates_parent_directories(tmp_path):
    db_path = tmp_path / "events.db"
    out_path = tmp_path / "deeply" / "nested" / "dir" / "events.json"
    run_cli(["--db", str(db_path), "export", "--out", str(out_path)])
    assert out_path.exists()


def test_export_on_empty_db_writes_empty_array(tmp_path):
    import json

    db_path = tmp_path / "events.db"
    out_path = tmp_path / "events.json"
    output = run_cli(["--db", str(db_path), "export", "--out", str(out_path)])
    assert "exported 0 events" in output
    assert json.loads(out_path.read_text()) == []


def test_export_drops_past_events_by_default(tmp_path, fake_network, only_fixture_fetchers):
    """The static page only ever shows upcoming events, so past ones are dead weight."""
    import json
    from datetime import date

    db_path = tmp_path / "events.db"
    out_path = tmp_path / "events.json"
    run_cli(["--db", str(db_path), "fetch"])
    run_cli(["--db", str(db_path), "export", "--out", str(out_path)])

    today = date.today().isoformat()
    data = json.loads(out_path.read_text())
    assert all((e["end_ts"] or e["start_ts"])[:10] >= today for e in data if e["start_ts"])


def test_export_keeps_events_still_running(tmp_path):
    """Whale season opened in April; it's still something you can go to."""
    import json
    from datetime import date, datetime, timedelta

    from sfevents.models import Event

    db_path = tmp_path / "events.db"
    out_path = tmp_path / "events.json"
    init_db(db_path)
    today = date.today()
    started = datetime(today.year, today.month, today.day) - timedelta(days=30)
    upsert_events(db_path, [
        Event("seasonal_bay_area", "running", "Running season", started, started + timedelta(days=60)),
        Event("seasonal_bay_area", "over", "Finished season", started, started + timedelta(days=10)),
    ])
    run_cli(["--db", str(db_path), "export", "--out", str(out_path)])

    titles = [e["title"] for e in json.loads(out_path.read_text())]
    assert titles == ["Running season"]


def test_export_slims_fields_unless_full(tmp_path, fake_network, only_fixture_fetchers):
    """Bookkeeping columns the browser never reads stay out of the payload."""
    import json

    db_path = tmp_path / "events.db"
    slim_path = tmp_path / "slim.json"
    full_path = tmp_path / "full.json"
    run_cli(["--db", str(db_path), "fetch"])
    run_cli(["--db", str(db_path), "export", "--out", str(slim_path), "--include-past"])
    run_cli(["--db", str(db_path), "export", "--out", str(full_path), "--include-past", "--full"])

    slim = json.loads(slim_path.read_text())[0]
    full = json.loads(full_path.read_text())[0]
    assert "fetched_at" in full and "source_id" in full
    assert "fetched_at" not in slim and "profile_hash" not in slim
    assert {"id", "title", "start_ts", "score", "date_approx"} <= set(slim)
    assert slim_path.stat().st_size < full_path.stat().st_size


def test_export_is_one_event_per_line(tmp_path, fake_network, only_fixture_fetchers):
    """Compact per-line JSON keeps the committed refresh diff reviewable."""
    db_path = tmp_path / "events.db"
    out_path = tmp_path / "events.json"
    run_cli(["--db", str(db_path), "fetch"])
    run_cli(["--db", str(db_path), "export", "--out", str(out_path), "--include-past"])

    lines = out_path.read_text().splitlines()
    assert lines[0] == "[" and lines[-1] == "]"
    assert len(lines) == 7 + 2  # one line per event, plus the brackets


def test_cli_rank_then_list_by_score(tmp_path, fake_network, only_fixture_fetchers):
    """`rank` with no --llm scores everything heuristically and list --sort score uses it."""
    db_path = tmp_path / "events.db"
    run_cli(["--db", str(db_path), "fetch"])
    rank_output = run_cli(["--db", str(db_path), "rank"])
    assert "heuristic: scored 7 events" in rank_output

    rows = query_events(db_path, order_by="score")
    scores = [r["score"] for r in rows]
    assert all(s is not None for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_cli_rank_without_llm_does_not_need_a_key(tmp_path, fake_network, only_fixture_fetchers):
    """No API key, no network to a provider: the heuristic path must still work."""
    db_path = tmp_path / "events.db"
    run_cli(["--db", str(db_path), "fetch"])
    output = run_cli(["--db", str(db_path), "rank"])
    assert "pass --llm" in output


def test_cli_rank_on_empty_db_is_graceful(tmp_path):
    db_path = tmp_path / "events.db"
    output = run_cli(["--db", str(db_path), "rank"])
    assert "no events to rank" in output


@pytest.fixture
def stub_provider(monkeypatch):
    """A provider that records every event it is asked to score.

    Returns the list of batches sent, so a test can assert what the weekly
    run would actually have paid for.
    """
    from sfevents import rank as ranking

    sent: list[list[str]] = []

    def fake_call(prompt: str, model: str, key: str, timeout: int) -> str:
        batch = re.findall(r"^\s*(\d+)\.\s", prompt, re.MULTILINE)
        sent.append(batch)
        return json.dumps(
            [{"i": int(i), "score": 90, "reason": "stub"} for i in batch]
        )

    monkeypatch.setitem(ranking.PROVIDERS, "stub", ("STUB_KEY", "stub-model", fake_call))
    monkeypatch.setenv("STUB_KEY", "present")
    return sent


def test_llm_rescores_only_events_it_has_not_seen(tmp_path, fake_network,
                                                  only_fixture_fetchers, stub_provider):
    """The weekly run must not re-pay for events already scored.

    Regression test: the heuristic pass used to overwrite every row's
    profile_hash, so the LLM re-scored the entire database every week.
    """
    db_path = tmp_path / "events.db"
    profile = tmp_path / "profile.md"
    profile.write_text("I like free outdoor festivals.\n")
    argv = ["--db", str(db_path), "rank", "--llm",
            "--provider", "stub", "--profile", str(profile)]

    run_cli(["--db", str(db_path), "fetch"])
    first = run_cli(argv)
    assert "llm: scored 7/7" in first

    second = run_cli(argv)
    assert "nothing to do" in second, second
    assert sum(len(b) for b in stub_provider) == 7, "second run sent events again"


def test_editing_the_profile_re_ranks_everything(tmp_path, fake_network,
                                                 only_fixture_fetchers, stub_provider):
    """A changed profile means the old scores no longer reflect the user."""
    db_path = tmp_path / "events.db"
    profile = tmp_path / "profile.md"
    profile.write_text("I like free outdoor festivals.\n")
    argv = ["--db", str(db_path), "rank", "--llm",
            "--provider", "stub", "--profile", str(profile)]

    run_cli(["--db", str(db_path), "fetch"])
    run_cli(argv)

    profile.write_text("Actually I only want museum lectures.\n")
    again = run_cli(argv)
    assert "llm: scored 7/7" in again


def test_heuristic_pass_leaves_llm_scores_alone(tmp_path, fake_network,
                                                only_fixture_fetchers, stub_provider):
    """`rank` (no --llm) must not blank the provenance the cache depends on."""
    db_path = tmp_path / "events.db"
    profile = tmp_path / "profile.md"
    profile.write_text("I like free outdoor festivals.\n")

    run_cli(["--db", str(db_path), "fetch"])
    run_cli(["--db", str(db_path), "rank", "--llm",
             "--provider", "stub", "--profile", str(profile)])

    output = run_cli(["--db", str(db_path), "rank"])
    assert "7 left to the LLM" in output, output

    rows = query_events(db_path)
    assert {r["scored_by"] for r in rows} == {"stub:stub-model"}
    assert all(r["score"] == 90 for r in rows)


def test_llm_failure_makes_rank_exit_nonzero(tmp_path, fake_network,
                                             only_fixture_fetchers, monkeypatch):
    """The alert: a run where the provider scored nothing must not look green.

    Regression: Gemini 503'd every batch (0/200 scored) and the workflow
    still reported success.
    """
    from sfevents import cli as cli_mod
    from sfevents import rank as ranking

    def down(prompt, model, key, timeout):
        raise ranking.ProviderError("HTTP 503: high demand", status=503)

    monkeypatch.setitem(ranking.PROVIDERS, "stub", ("STUB_KEY", "stub-model", down))
    monkeypatch.setenv("STUB_KEY", "present")
    monkeypatch.setattr(ranking, "RETRY_WAITS", (0, 0, 0))
    db_path = tmp_path / "events.db"
    profile = tmp_path / "profile.md"
    profile.write_text("I like free outdoor festivals.\n")

    run_cli(["--db", str(db_path), "fetch"])
    with pytest.raises(SystemExit) as info:
        run_cli_capture_stderr(["--db", str(db_path), "rank", "--llm",
                                "--provider", "stub", "--profile", str(profile)])
    assert info.value.code == cli_mod.LLM_FAILED_EXIT


def test_a_fully_scored_llm_run_exits_cleanly(tmp_path, fake_network,
                                               only_fixture_fetchers, stub_provider):
    db_path = tmp_path / "events.db"
    profile = tmp_path / "profile.md"
    profile.write_text("I like free outdoor festivals.\n")
    run_cli(["--db", str(db_path), "fetch"])
    # No SystemExit raised means exit status 0.
    run_cli(["--db", str(db_path), "rank", "--llm",
             "--provider", "stub", "--profile", str(profile)])


def test_fallback_scores_are_kept_and_credited_to_their_model(
        tmp_path, fake_network, only_fixture_fetchers, stub_provider, monkeypatch):
    """Gemini down, Groq up: the run succeeds and each row says who scored it."""
    from sfevents import rank as ranking

    def down(prompt, model, key, timeout):
        raise ranking.ProviderError("HTTP 503: high demand", status=503)

    monkeypatch.setitem(ranking.PROVIDERS, "down", ("DOWN_KEY", "down-model", down))
    monkeypatch.setenv("DOWN_KEY", "present")
    monkeypatch.setitem(ranking.FALLBACK_PACING, "stub", (50, 0))
    monkeypatch.setattr(ranking, "RETRY_WAITS", (0, 0, 0))
    db_path = tmp_path / "events.db"
    profile = tmp_path / "profile.md"
    profile.write_text("I like free outdoor festivals.\n")
    argv = ["--db", str(db_path), "rank", "--llm", "--provider", "down",
            "--fallback", "stub", "--profile", str(profile)]

    run_cli(["--db", str(db_path), "fetch"])
    output = run_cli(argv)  # no SystemExit: everything got scored

    assert "llm: scored 7/7" in output, output
    assert {r["scored_by"] for r in query_events(db_path)} == {"stub:stub-model"}
    # Marked done for this profile, so next week doesn't re-send them.
    assert "nothing to do" in run_cli(argv)


def test_export_gives_each_event_a_stable_key(tmp_path):
    """docs/plans.json pitches an event by source:source_id, not the db id."""
    import json
    from datetime import date, datetime, timedelta

    from sfevents.models import Event

    db_path = tmp_path / "events.db"
    out_path = tmp_path / "events.json"
    init_db(db_path)
    day = datetime.combine(date.today() + timedelta(days=5), datetime.min.time())
    upsert_events(db_path, [Event("funcheap_sf", "abc", "Pumpkin Festival", day, day)])
    run_cli(["--db", str(db_path), "export", "--out", str(out_path)])

    [event] = json.loads(out_path.read_text())
    assert event["key"] == "funcheap_sf:abc"


def test_export_flags_events_new_in_the_latest_fetch(tmp_path):
    """Only events first seen by the latest fetch are "new"; meta.json names the model."""
    import json
    from datetime import date, datetime, timedelta

    from sfevents.db import connect, set_scores
    from sfevents.models import Event

    db_path = tmp_path / "events.db"
    out_path = tmp_path / "events.json"
    init_db(db_path)
    day = datetime.combine(date.today() + timedelta(days=5), datetime.min.time())
    upsert_events(db_path, [Event("funcheap_sf", "old", "Old Fair", day, day)])
    # Pretend that one was first seen last week; re-fetching mustn't reset it.
    with connect(db_path) as conn:
        conn.execute("UPDATE events SET first_seen = '2020-01-01T00:00:00+00:00'")
    upsert_events(db_path, [Event("funcheap_sf", "old", "Old Fair", day, day),
                            Event("funcheap_sf", "new", "New Fair", day, day)])
    ids = {r["source_id"]: r["id"] for r in query_events(db_path)}
    set_scores(db_path, {ids["new"]: (80, "good")}, "ollama:qwen3.5:4b", "h")

    run_cli(["--db", str(db_path), "export", "--out", str(out_path)])

    events = {e["title"]: e for e in json.loads(out_path.read_text())}
    assert events["New Fair"]["is_new"] and not events["Old Fair"]["is_new"]
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["ranked_by"] == ["ollama:qwen3.5:4b"]
    assert meta["new_count"] == 1
    assert meta["updated_at"]


def test_export_holds_each_source_group_to_its_weekly_quota(tmp_path, monkeypatch):
    import json
    from datetime import datetime

    from sfevents.db import set_scores
    from sfevents.fetchers.tribe import TribeEventsFetcher
    from sfevents.models import Event

    monkeypatch.setattr(cli_module, "FETCHERS", [
        TribeEventsFetcher("santacruz_org", "https://x.test", weekends_only=True),
    ])
    monkeypatch.setattr(cli_module, "SOURCE_GROUPS", {
        "Day trips": (("santacruz_org",), 2), "SF listings": (("funcheap_sf",), None),
    })
    db_path = tmp_path / "events.db"
    out_path = tmp_path / "events.json"
    init_db(db_path)
    sat, sun, wed = datetime(2099, 10, 24, 11), datetime(2099, 10, 25, 11), datetime(2099, 10, 21, 18)
    upsert_events(db_path, [
        Event("santacruz_org", "1", "Low", sat, sat), Event("santacruz_org", "2", "High", sun, sun),
        Event("santacruz_org", "3", "Mid", sat, sat), Event("santacruz_org", "4", "Weeknight", wed, wed),
        Event("funcheap_sf", "5", "SF thing", wed, wed),
    ])
    ids = {r["title"]: r["id"] for r in query_events(db_path)}
    set_scores(db_path, {ids["Low"]: (40, ""), ids["High"]: (90, ""), ids["Mid"]: (60, ""),
                         ids["Weeknight"]: (99, "")}, "stub:model", "h")

    output = run_cli(["--db", str(db_path), "export", "--out", str(out_path)])

    titles = {e["title"] for e in json.loads(out_path.read_text())}
    assert titles == {"High", "Mid", "SF thing"}
    assert "2 over their group's quota" in output


def test_sources_in_one_group_share_its_quota(monkeypatch):
    monkeypatch.setattr(cli_module, "FETCHERS", [])
    monkeypatch.setattr(cli_module, "SOURCE_GROUPS", {"SF": (("funcheap_sf", "dothebay"), 2)})
    events = [
        {"source": "funcheap_sf", "title": "a", "start_ts": "2099-10-20T10:00:00", "score": 50},
        {"source": "dothebay", "title": "b", "start_ts": "2099-10-21T10:00:00", "score": 90},
        {"source": "dothebay", "title": "c", "start_ts": "2099-10-22T10:00:00", "score": 70},
        {"source": "dothebay", "title": "d", "start_ts": "2099-10-27T10:00:00", "score": 10},  # next week
        {"source": "dothebay", "title": "e", "start_ts": None, "score": 0},  # undated: never capped
    ]
    assert [e["title"] for e in cli_module._apply_quotas(events)] == ["b", "c", "d", "e"]
