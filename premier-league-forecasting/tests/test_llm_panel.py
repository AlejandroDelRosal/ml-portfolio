import json

import numpy as np
import pandas as pd
import pytest

from src import llm_panel
from src.llm_panel import (
    PanelError,
    ask_model,
    build_prompt,
    extract_json,
    free_models,
    load_records,
    parse_response,
    record,
    run_panel,
)
from src.models import OUTCOME_COLUMNS

FIXTURES = pd.DataFrame({
    "Datetime": [pd.Timestamp("2026-09-19 14:00"), pd.Timestamp("2026-09-19 16:30")],
    "HomeTeam": ["Arsenal", "Hull"],
    "AwayTeam": ["Chelsea", "Man United"],
})


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, content="", models=None):
        self.content = content
        self.models = models or []
        self.posts = []

    def get(self, url, **kwargs):
        return FakeResponse({"data": self.models})

    def post(self, url, **kwargs):
        self.posts.append(kwargs)
        return FakeResponse({"choices": [{"message": {"content": self.content}}]})


def reply(rows):
    return json.dumps(rows)


def test_prompt_lists_every_fixture():
    prompt = build_prompt(FIXTURES)
    assert "Arsenal vs Chelsea" in prompt
    assert "Hull vs Man United" in prompt


def test_json_is_extracted_from_a_chatty_reply():
    text = 'Sure! Here are my picks:\n```json\n[{"home": "A", "away": "B"}]\n```\nHope it helps.'
    assert extract_json(text) == [{"home": "A", "away": "B"}]


def test_a_reply_without_json_is_rejected():
    with pytest.raises(PanelError):
        extract_json("I cannot predict football matches.")


def test_probabilities_are_renormalised_when_they_nearly_sum_to_one():
    text = reply([{"home": "Arsenal", "away": "Chelsea", "p_home": 0.5, "p_draw": 0.3, "p_away": 0.3}])
    row = parse_response(text, FIXTURES.head(1))
    assert row[OUTCOME_COLUMNS].sum(axis=1).item() == pytest.approx(1.0)


def test_fixtures_the_model_skipped_come_back_empty():
    text = reply([{"home": "Arsenal", "away": "Chelsea", "p_home": 0.5, "p_draw": 0.25, "p_away": 0.25}])
    frame = parse_response(text, FIXTURES)
    assert frame.iloc[0].notna().sum() == 3
    assert frame.iloc[1][OUTCOME_COLUMNS].isna().all()


def test_team_names_match_despite_punctuation_case_and_club_suffix():
    text = reply([{"home": "arsenal fc", "away": "CHELSEA", "p_home": 0.5, "p_draw": 0.25, "p_away": 0.25}])
    assert parse_response(text, FIXTURES.head(1)).iloc[0]["p_home"] == pytest.approx(0.5)


def test_a_complete_reply_in_order_is_matched_by_position():
    text = reply([
        {"home": "Arsenal", "away": "Chelsea", "p_home": 0.5, "p_draw": 0.25, "p_away": 0.25},
        {"home": "Hull City", "away": "Manchester United", "p_home": 0.2, "p_draw": 0.3, "p_away": 0.5},
    ])
    frame = parse_response(text, FIXTURES)
    assert frame.iloc[1]["p_away"] == pytest.approx(0.5)


def test_positional_matching_is_not_used_on_a_partial_reply():
    text = reply([{"home": "Sunderland", "away": "Everton", "p_home": 0.4, "p_draw": 0.3, "p_away": 0.3}])
    frame = parse_response(text, FIXTURES)
    assert frame[OUTCOME_COLUMNS].isna().all(axis=None)


@pytest.mark.parametrize("bad", [
    {"home": "Arsenal", "away": "Chelsea", "p_home": -0.5, "p_draw": 0.75, "p_away": 0.75},
    {"home": "Arsenal", "away": "Chelsea", "p_home": 0.0, "p_draw": 0.0, "p_away": 0.0},
    {"home": "Arsenal", "away": "Chelsea", "p_home": "mucho", "p_draw": 0.3, "p_away": 0.2},
    {"home": "Arsenal", "away": "Chelsea"},
])
def test_unusable_probabilities_are_discarded(bad):
    assert parse_response(reply([bad]), FIXTURES.head(1)).iloc[0].isna().all()


def test_the_panel_sends_exactly_one_request_per_model():
    session = FakeSession(content=reply([
        {"home": "Arsenal", "away": "Chelsea", "p_home": 0.5, "p_draw": 0.25, "p_away": 0.25},
        {"home": "Hull", "away": "Man United", "p_home": 0.3, "p_draw": 0.3, "p_away": 0.4},
    ]))
    panel = run_panel(FIXTURES, models=["a:free", "b:free"], key="test", session=session)
    assert len(session.posts) == 2
    assert set(panel) == {"a:free", "b:free"}
    assert panel["a:free"][OUTCOME_COLUMNS].notna().all(axis=None)


def test_a_failing_model_does_not_sink_the_panel():
    class BrokenSession(FakeSession):
        def post(self, url, **kwargs):
            raise TimeoutError("upstream is down")

    assert run_panel(FIXTURES, models=["a:free"], key="test", session=BrokenSession()) == {}


def test_free_models_are_read_live_and_ranked_by_context():
    session = FakeSession(models=[
        {"id": "small:free", "context_length": 8000},
        {"id": "paid", "context_length": 999999},
        {"id": "big:free", "context_length": 200000},
    ])
    assert free_models(limit=2, session=session) == ["big:free", "small:free"]


def test_a_missing_key_is_reported_clearly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(PanelError):
        run_panel(FIXTURES, models=["a:free"], session=FakeSession())


def test_recording_after_kick_off_is_refused(tmp_path):
    panel = {"a:free": parse_response(reply([{"home": "Arsenal", "away": "Chelsea", "p_home": 0.5, "p_draw": 0.25, "p_away": 0.25}]), FIXTURES.head(1))}
    with pytest.raises(PanelError):
        record(panel, FIXTURES.head(1), asked_at=pd.Timestamp("2026-09-19 15:00"), directory=tmp_path)


def test_a_recorded_panel_round_trips(tmp_path):
    text = reply([{"home": "Arsenal", "away": "Chelsea", "p_home": 0.5, "p_draw": 0.25, "p_away": 0.25}])
    panel = {"a:free": parse_response(text, FIXTURES.head(1))}
    asked_at = pd.Timestamp("2026-09-18 09:00")
    record(panel, FIXTURES.head(1), asked_at=asked_at, directory=tmp_path)
    stored = load_records(tmp_path)
    assert len(stored) == 1
    assert stored["model"].item() == "a:free"
    assert stored["asked_at"].item() == asked_at


def test_an_empty_archive_still_has_the_right_shape(tmp_path):
    assert list(load_records(tmp_path).columns)[:2] == ["model", "asked_at"]


def store(directory, asked_at, fixtures=None):
    fixtures = FIXTURES.head(1) if fixtures is None else fixtures
    text = reply([{"home": "Arsenal", "away": "Chelsea", "p_home": 0.5, "p_draw": 0.25, "p_away": 0.25}])
    record({"a:free": parse_response(text, fixtures)}, fixtures, asked_at=asked_at, directory=directory)


def test_every_fixture_is_asked_when_nothing_has_been_recorded(tmp_path):
    assert len(llm_panel.unasked(FIXTURES, load_records(tmp_path))) == len(FIXTURES)


def test_a_fixture_already_recorded_is_not_asked_again(tmp_path):
    store(tmp_path, pd.Timestamp("2026-09-18 09:00"))
    assert list(llm_panel.unasked(FIXTURES, load_records(tmp_path))["HomeTeam"]) == ["Hull"]


def test_the_same_teams_at_a_different_kick_off_are_a_new_fixture(tmp_path):
    store(tmp_path, pd.Timestamp("2026-09-18 09:00"))
    rescheduled = FIXTURES.head(1).assign(Datetime=[pd.Timestamp("2026-10-01 19:45")])
    assert len(llm_panel.unasked(rescheduled, load_records(tmp_path))) == 1


def test_the_same_fixture_recorded_twice_is_kept_once_at_the_earliest_asking(tmp_path):
    store(tmp_path, pd.Timestamp("2026-09-18 09:00"))
    store(tmp_path, pd.Timestamp("2026-09-17 09:00"))
    stored = load_records(tmp_path)
    assert len(stored) == 1
    assert stored["asked_at"].item() == pd.Timestamp("2026-09-17 09:00")


def two_model_panel():
    first = reply([
        {"home": "Arsenal", "away": "Chelsea", "p_home": 0.5, "p_draw": 0.25, "p_away": 0.25},
        {"home": "Hull", "away": "Man United", "p_home": 0.2, "p_draw": 0.3, "p_away": 0.5},
    ])
    second = reply([{"home": "Arsenal", "away": "Chelsea", "p_home": 0.4, "p_draw": 0.3, "p_away": 0.3}])
    return {"a:free": parse_response(first, FIXTURES), "b:free": parse_response(second, FIXTURES)}


def test_a_snapshot_carries_each_model_answer_per_fixture():
    snapshot = llm_panel.snapshot(two_model_panel(), FIXTURES, asked_at=pd.Timestamp("2026-09-18 09:00"))
    assert snapshot["models"] == ["a:free", "b:free"]
    assert [row["home"] for row in snapshot["fixtures"]] == ["Arsenal", "Hull"]
    assert snapshot["fixtures"][0]["forecasts"]["a:free"] == pytest.approx([0.5, 0.25, 0.25])


def test_a_snapshot_omits_a_fixture_a_model_did_not_price():
    snapshot = llm_panel.snapshot(two_model_panel(), FIXTURES, asked_at=pd.Timestamp("2026-09-18 09:00"))
    assert "b:free" not in snapshot["fixtures"][1]["forecasts"]


def test_a_snapshot_is_json_serialisable():
    snapshot = llm_panel.snapshot(two_model_panel(), FIXTURES, asked_at=pd.Timestamp("2026-09-18 09:00"))
    assert json.loads(json.dumps(snapshot))["asked_at"].startswith("2026-09-18")
