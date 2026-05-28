"""
tests/unit/evals/test_evidence_packet.py
========================================
Tests for :mod:`evals.longmemeval.evidence_packet`.

Acceptance scenarios from the spec:

1. The Spotify-playlist question must not see Apple finance numbers —
   unrelated candidates that failed the verifier never enter the
   final answer prompt.
2. The packet-rendered prompt is **strictly smaller** than the raw
   wiki window text (the spec's "final prompt token count drops").
3. Telemetry surfaces ``selected_evidence_count`` and
   ``excluded_noise_count`` for every packet built.

Plus per-shape unit coverage and the abstain-on-empty contract.
"""
from __future__ import annotations

from evals.longmemeval.candidates import Candidate
from evals.longmemeval.evidence_packet import (
    PacketClaim,
    build_evidence_packet,
    render_evidence_packet_prompt,
)


def _verified(
    *, value: str, subject: str, relation: str = "is",
    answer_type: str = "value", session_id: str = "s1",
    confidence: float = 0.85, claim: str = "",
    source_span: str = "",
) -> Candidate:
    return Candidate(
        value=value,
        normalized_value=value.lower(),
        candidate_type="value",
        answer_type=answer_type,
        subject=subject,
        relation=relation,
        object_=value,
        source_session_id=session_id,
        confidence=confidence,
        claim=claim or f"{subject} {relation} {value}.",
        source_span=source_span or value,
    )


# ─── ACCEPTANCE 1 — Spotify playlist question never sees Apple finance ────


def test_spotify_question_does_not_see_apple_finance_numbers():
    """
    The verifier already filtered the Apple finance candidates out
    (they're either wrong subject, wrong answer_type, or both). The
    packet builder must take ONLY the verifier's PASS list — never
    silently reintroduce excluded candidates.
    """
    apple_finance = [
        _verified(
            value="$2 trillion", subject="Apple market cap",
            relation="is_worth", answer_type="value",
            session_id="s_apple_news",
            claim="Apple market cap is around $2 trillion.",
            source_span="around $2 trillion",
            confidence=0.6,
        ),
        _verified(
            value="$394B", subject="Apple revenue",
            relation="is_worth", answer_type="value",
            session_id="s_apple_news",
            claim="Apple FY24 revenue is $394B.",
            source_span="$394B",
            confidence=0.7,
        ),
    ]
    spotify_playlist = [
        _verified(
            value="Summer Vibes", subject="Spotify playlist",
            relation="named", answer_type="entity",
            session_id="s_spotify",
            claim="My Spotify playlist is called \"Summer Vibes\".",
            source_span="called \"Summer Vibes\"",
            confidence=0.95,
        ),
    ]
    # The full candidate pool — what extraction emitted, before
    # verification. The verifier passed only the Spotify claim.
    all_candidates = apple_finance + spotify_playlist
    verified = spotify_playlist  # the verifier's output

    packet = build_evidence_packet(
        "What is my Spotify playlist called?",
        verified,
        all_candidates=all_candidates,
    )

    # The packet carries ONLY the verified Spotify claim.
    assert packet.selected_evidence_count == 1
    assert packet.claims[0].object_ == "Summer Vibes"
    # Apple finance facts excluded — count = 2.
    assert packet.excluded_noise_count == 2

    # Render the packet and confirm none of the Apple text appears.
    prompt = render_evidence_packet_prompt(packet)
    assert "Spotify" in prompt
    assert "Summer Vibes" in prompt
    # The Apple terms — values, source spans, claim strings — must
    # NOT be findable anywhere in the answerer's prompt.
    forbidden = ("Apple", "trillion", "$394B", "revenue", "market cap")
    for term in forbidden:
        assert term not in prompt, (
            f"Apple finance leakage: {term!r} appears in packet prompt"
        )


# ─── ACCEPTANCE 2 — Token count drops ─────────────────────────────────────


def test_packet_prompt_token_count_drops_vs_raw_window():
    """
    The packet should be strictly shorter than passing the raw wiki
    window through. We approximate token count by whitespace-splitting
    so the test stays hermetic — the exact ratio doesn't matter, only
    the direction.
    """
    # Simulate a realistic 4-item wiki window. Each item's content is
    # ~80 words of mostly distractor prose plus the answer-bearing
    # sentence at the end.
    raw_window = "\n\n".join(
        "## Matched Turn\n"
        "- (user): Long preamble about my day — went to the office, "
        "met with the team, lunch was unremarkable, came home and "
        "watched a documentary on Apple's market cap, glanced at "
        "the Spotify charts, and then I asked my friend about music. "
        "Anyway, my Spotify playlist is called \"Summer Vibes\".\n"
        "## Nearby Context\n"
        "- (user): I love Apple stock — up 12% last quarter.\n"
        "- (assistant): That's a great gain.\n"
        "- (user): Tomorrow I'll listen to my playlist on the commute.\n"
        for _ in range(4)
    )

    verified = [
        _verified(
            value="Summer Vibes", subject="Spotify playlist",
            relation="named", session_id="s_spotify",
            claim="My Spotify playlist is called \"Summer Vibes\".",
            source_span="called \"Summer Vibes\"",
            confidence=0.95,
        ),
    ]
    packet = build_evidence_packet(
        "What is my Spotify playlist called?",
        verified,
    )
    prompt = render_evidence_packet_prompt(packet)

    raw_tokens = len(raw_window.split())
    packet_tokens = len(prompt.split())
    assert packet_tokens < raw_tokens, (
        f"packet ({packet_tokens} tokens) is NOT smaller than "
        f"raw window ({raw_tokens} tokens)"
    )
    # And the compression should be meaningful — at least 3× smaller.
    assert raw_tokens / max(1, packet_tokens) >= 3.0, (
        f"compression ratio {raw_tokens/packet_tokens:.2f}x below 3x"
    )


# ─── ACCEPTANCE 3 — Telemetry counts surfaced ─────────────────────────────


def test_telemetry_reports_selected_and_excluded_counts():
    verified = [
        _verified(value="A", subject="x", confidence=0.9),
        _verified(value="B", subject="y", confidence=0.8),
    ]
    failed = [
        _verified(value="C", subject="distractor", confidence=0.5),
        _verified(value="D", subject="distractor", confidence=0.4),
        _verified(value="E", subject="distractor", confidence=0.3),
    ]
    packet = build_evidence_packet(
        "Test question",
        verified,
        all_candidates=verified + failed,
    )
    d = packet.to_dict()
    assert d["selected_evidence_count"] == 2
    assert d["excluded_noise_count"] == 3
    # Telemetry dict shape covers every required field.
    assert "claims" in d and len(d["claims"]) == 2
    assert d["question"] == "Test question"


# ─── Abstain on empty packet ──────────────────────────────────────────────


def test_empty_packet_forces_abstain():
    """No verified evidence → packet renders an explicit abstain prompt."""
    packet = build_evidence_packet(
        "What is the answer?",
        [],  # verifier rejected everything
        all_candidates=[
            _verified(value="X", subject="distractor", confidence=0.3),
            _verified(value="Y", subject="distractor", confidence=0.2),
        ],
    )
    assert packet.is_empty
    assert packet.selected_evidence_count == 0
    assert packet.excluded_noise_count == 2

    prompt = render_evidence_packet_prompt(packet)
    # Hard abstain phrasing — model must say "I don't know"
    assert "I don't know" in prompt
    assert "do not infer" in prompt.lower() or "do not guess" in prompt.lower()
    # And the question is still present so trace logs can locate it.
    assert "What is the answer?" in prompt


def test_empty_packet_is_compact():
    """Empty-packet prompt is the smallest path — well under 100 tokens."""
    packet = build_evidence_packet("X?", [])
    prompt = render_evidence_packet_prompt(packet)
    assert len(prompt.split()) < 100


# ─── PacketClaim shape ────────────────────────────────────────────────────


def test_packet_claim_preserves_all_grounding_fields():
    c = _verified(
        value="UCLA",
        subject="Bachelor's degree",
        relation="got_at",
        answer_type="location",
        session_id="s_edu",
        claim="I got my Bachelor's at UCLA in 2018.",
        source_span="at UCLA",
        confidence=0.92,
    )
    packet = build_evidence_packet(
        "Where did I get my Bachelor's?", [c],
    )
    pc = packet.claims[0]
    assert pc.claim == "I got my Bachelor's at UCLA in 2018."
    assert pc.source_span == "at UCLA"
    assert pc.source_session_id == "s_edu"
    assert pc.subject == "Bachelor's degree"
    assert pc.relation == "got_at"
    assert pc.object_ == "UCLA"
    assert pc.answer_type == "location"
    assert abs(pc.confidence - 0.92) < 1e-9


def test_packet_claim_to_dict_uses_object_not_underscored():
    pc = PacketClaim(
        claim="x", source_span="x", source_session_id="s",
        confidence=0.8, subject="subj", relation="is",
        object_="UCLA", answer_type="location",
    )
    d = pc.to_dict()
    assert d["object"] == "UCLA"
    assert "object_" not in d


def test_packet_serializes_cleanly_for_json():
    """Round-trippable to JSON via the to_dict() outputs."""
    import json
    packet = build_evidence_packet(
        "q?",
        [_verified(value="A", subject="x", confidence=0.9)],
        all_candidates=[_verified(value="A", subject="x", confidence=0.9)],
    )
    serialized = json.dumps(packet.to_dict())
    # No exception → serializable. Confirm key fields survived.
    parsed = json.loads(serialized)
    assert parsed["selected_evidence_count"] == 1
    assert parsed["claims"][0]["object"] == "A"


# ─── Ranking + truncation ─────────────────────────────────────────────────


def test_claims_sorted_by_confidence_descending():
    """Top-confidence claim renders first so the answerer sees it as [1]."""
    verified = [
        _verified(value="low", subject="x", confidence=0.55),
        _verified(value="high", subject="x", confidence=0.95),
        _verified(value="mid", subject="x", confidence=0.75),
    ]
    packet = build_evidence_packet("q?", verified)
    objects = [c.object_ for c in packet.claims]
    assert objects == ["high", "mid", "low"]


def test_max_claims_truncates_packet():
    """Cap at ``max_claims`` after sorting by confidence."""
    verified = [
        _verified(value=f"v{i}", subject="x", confidence=1.0 - i * 0.1)
        for i in range(10)
    ]
    packet = build_evidence_packet("q?", verified, max_claims=3)
    assert packet.selected_evidence_count == 3
    assert [c.object_ for c in packet.claims] == ["v0", "v1", "v2"]


# ─── Immutability — answer-time consumer can't tamper ─────────────────────


def test_evidence_packet_is_frozen():
    import dataclasses

    import pytest
    packet = build_evidence_packet("q?", [])
    with pytest.raises(dataclasses.FrozenInstanceError):
        packet.question = "different"  # type: ignore[misc]


def test_packet_claim_is_frozen():
    import dataclasses

    import pytest
    packet = build_evidence_packet(
        "q?",
        [_verified(value="A", subject="x", confidence=0.9)],
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        packet.claims[0].confidence = 0.0  # type: ignore[misc]


# ─── Prompt formatting smoke ──────────────────────────────────────────────


def test_render_packet_includes_question_and_strict_instruction():
    packet = build_evidence_packet(
        "What is my Spotify playlist called?",
        [_verified(
            value="Summer Vibes", subject="playlist",
            session_id="s1",
            claim="My playlist is \"Summer Vibes\".",
            source_span="\"Summer Vibes\"",
            confidence=0.95,
        )],
    )
    prompt = render_evidence_packet_prompt(packet)
    # Question echoed back
    assert "What is my Spotify playlist called?" in prompt
    # The verified claim's exact substring is in the prompt
    assert "Summer Vibes" in prompt
    # The session id is attributed
    assert "s1" in prompt
    # The confidence is shown to 2 decimals
    assert "0.95" in prompt
    # The strict "use only this evidence" instruction is present
    assert "USING ONLY" in prompt or "use only" in prompt.lower()
    # The "abstain when nothing matches" instruction is present
    assert "I don't know" in prompt
