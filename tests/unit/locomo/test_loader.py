"""
tests/unit/locomo/test_loader.py
================================
Unit tests for the LOCOMO dataset loader — parsed against a synthetic
two-session fixture mirroring the real ``locomo10.json`` shape, so we
never need the (large) dataset present to verify parsing logic.
"""
from __future__ import annotations

import json

import pytest

from evals.locomo.loader import category_name, load_locomo

pytestmark = pytest.mark.unit


_FIXTURE = [
    {
        "sample_id": "S1",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1:00 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "I adopted a dog named Rex."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "Nice!"},
            ],
            "session_2_date_time": "9:00 am on 20 May, 2023",
            "session_2": [
                {"speaker": "Alice", "dia_id": "D2:1", "text": "Rex is a golden retriever."},
                {"speaker": "Bob", "dia_id": "D2:2", "text": "", "img_url": "x"},  # empty → skipped
            ],
        },
        "qa": [
            {"question": "What breed is Rex?", "answer": "golden retriever",
             "evidence": ["D2:1"], "category": 4},
            {"question": "How many days between adopting Rex and learning his breed?",
             "answer": "12 days", "evidence": ["D1:1", "D2:1"], "category": 2},
            {"question": "What is Alice's cat's name?", "answer": "Not mentioned",
             "evidence": [], "category": 5},
        ],
    }
]


@pytest.fixture
def locomo_file(tmp_path):
    p = tmp_path / "locomo10.json"
    p.write_text(json.dumps(_FIXTURE))
    return p


def test_load_returns_one_pair_per_sample(locomo_file):
    data = load_locomo(locomo_file)
    assert len(data) == 1
    conv, questions = data[0]
    assert conv.sample_id == "S1"
    assert conv.speaker_a == "Alice"
    assert len(questions) == 3


def test_turns_flattened_in_session_order_with_metadata(locomo_file):
    conv, _ = load_locomo(locomo_file)[0]
    # Empty-text turn (D2:2) is dropped → 3 turns.
    assert [t.dia_id for t in conv.turns] == ["D1:1", "D1:2", "D2:1"]
    assert conv.turns[0].session_id == "session_1"
    assert conv.turns[0].session_date == "1:00 pm on 8 May, 2023"
    assert conv.turns[2].session_id == "session_2"
    assert conv.turns[2].speaker == "Alice"


def test_questions_carry_evidence_and_category(locomo_file):
    _, questions = load_locomo(locomo_file)[0]
    q0 = questions[0]
    assert q0.answer == "golden retriever"
    assert q0.evidence == ["D2:1"]
    assert q0.category == 4
    assert q0.category_label == "single-hop"
    # temporal
    assert questions[1].category_label == "temporal"
    # adversarial (no evidence)
    assert questions[2].category == 5
    assert questions[2].category_label == "adversarial"
    assert questions[2].evidence == []


def test_category_name_maps_known_and_unknown():
    assert category_name(1) == "multi-hop"
    assert category_name(2) == "temporal"
    assert category_name(5) == "adversarial"
    assert category_name(99) == "category-99"
    assert category_name(None) == "category-None"


def test_missing_file_raises_with_download_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="locomo10.json"):
        load_locomo(tmp_path / "nope.json")


def test_numeric_answer_coerced_to_str(tmp_path):
    fixture = [{
        "sample_id": "S2",
        "conversation": {"speaker_a": "A", "speaker_b": "B",
                         "session_1_date_time": "d",
                         "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hi"}]},
        "qa": [{"question": "how many?", "answer": 3, "evidence": ["D1:1"], "category": 1}],
    }]
    p = tmp_path / "locomo10.json"
    p.write_text(json.dumps(fixture))
    _, questions = load_locomo(p)[0]
    assert questions[0].answer == "3"
    assert isinstance(questions[0].answer, str)
