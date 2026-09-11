"""Approval tokens: consent the model cannot manufacture.

Pins the four properties the gate depends on. Each maps to a way the
2026-09-08 incident could recur if the property were absent:

* prose never mints          -> "I offered to wait, and read the next turn as a yes"
* single-use                 -> one approval becoming a standing permission
* subject-bound              -> an approval for A sending B
* session-bound / expiring   -> yesterday's yes acting today
"""

import time

import pytest

from agent.approval_tokens import (
    KIND_DRAFT,
    KIND_MEMORY_REMOVE,
    ApprovalRegistry,
    memory_removal_subject,
)

SESSION = "sess_abc"


@pytest.fixture
def reg():
    return ApprovalRegistry(ttl_seconds=600)


# --- minting ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "yes send it",
        "yes please, go ahead",
        "approved",
        "approve draft_123",          # no slash
        "looks good — send",
        "/approve",                    # no subject
        "sure",
        "ok do it",
        "/approve draft_1 and also draft_2",  # ambiguous, two subjects
    ],
)
def test_prose_never_mints_a_token(reg, text):
    """Only an exact '/approve <id>' mints. Interpreting prose is the bug."""
    assert reg.mint_from_user_text(text, kind=KIND_DRAFT, session_id=SESSION) is None


def test_exact_approve_command_mints(reg):
    tok = reg.mint_from_user_text(
        "/approve draft_20260911_a1b2", kind=KIND_DRAFT, session_id=SESSION
    )
    assert tok is not None
    assert tok.subject_id == "draft_20260911_a1b2"
    assert tok.kind == KIND_DRAFT


def test_approve_command_tolerates_surrounding_whitespace(reg):
    tok = reg.mint_from_user_text(
        "  /APPROVE draft_x  ", kind=KIND_DRAFT, session_id=SESSION
    )
    assert tok is not None and tok.subject_id == "draft_x"


# --- spending --------------------------------------------------------------


def test_no_token_is_refused_with_an_actionable_reason(reg):
    ok, reason = reg.consume("", KIND_DRAFT, "draft_1", SESSION)
    assert ok is False
    assert "/approve draft_1" in reason
    # The refusal must name the real remedy, not invite a retry.
    assert "not an approval" in reason


def test_token_authorises_exactly_one_send(reg):
    tok = reg.mint(KIND_DRAFT, "draft_1", SESSION)
    assert reg.consume(tok.token, KIND_DRAFT, "draft_1", SESSION)[0] is True
    ok, reason = reg.consume(tok.token, KIND_DRAFT, "draft_1", SESSION)
    assert ok is False
    assert "already spent" in reason


def test_token_cannot_send_a_different_draft(reg):
    tok = reg.mint(KIND_DRAFT, "draft_1", SESSION)
    ok, reason = reg.consume(tok.token, KIND_DRAFT, "draft_2", SESSION)
    assert ok is False
    assert "authorises 'draft_1'" in reason


def test_mismatched_token_is_burned_not_left_for_a_second_try(reg):
    """A wrong-subject presentation must not leave the token usable.

    Otherwise the model can present one token against each candidate id until
    one fits, which is approval-by-search.
    """
    tok = reg.mint(KIND_DRAFT, "draft_1", SESSION)
    reg.consume(tok.token, KIND_DRAFT, "draft_2", SESSION)
    ok, _ = reg.consume(tok.token, KIND_DRAFT, "draft_1", SESSION)
    assert ok is False


def test_draft_token_cannot_authorise_a_memory_removal(reg):
    tok = reg.mint(KIND_DRAFT, "draft_1", SESSION)
    ok, reason = reg.consume(tok.token, KIND_MEMORY_REMOVE, "draft_1", SESSION)
    assert ok is False
    assert "issued for 'draft'" in reason


def test_token_does_not_cross_sessions(reg):
    tok = reg.mint(KIND_DRAFT, "draft_1", SESSION)
    ok, reason = reg.consume(tok.token, KIND_DRAFT, "draft_1", "other_session")
    assert ok is False
    assert "different session" in reason


def test_expired_token_is_not_consent():
    reg = ApprovalRegistry(ttl_seconds=1)
    tok = reg.mint(KIND_DRAFT, "draft_1", SESSION)
    time.sleep(1.1)
    ok, reason = reg.consume(tok.token, KIND_DRAFT, "draft_1", SESSION)
    assert ok is False
    assert "expired" in reason


def test_unknown_token_is_refused(reg):
    ok, reason = reg.consume("apv_fabricated", KIND_DRAFT, "draft_1", SESSION)
    assert ok is False
    assert "unknown" in reason


# --- subjects --------------------------------------------------------------


def test_memory_subject_is_stable_and_content_bound():
    entry = "vision_analyze: ONE call/image, full Q up front."
    assert memory_removal_subject(entry) == memory_removal_subject(entry)
    assert memory_removal_subject(entry) != memory_removal_subject(entry + "!")


def test_memory_subject_survives_multiline_japanese_entries():
    """Real entries are long, multi-line and non-ASCII; the id must be typable."""
    entry = "彼女の数字指摘は正しい—検算する。\n§\nJP仕入先は直送;一括配送NG。"
    subject = memory_removal_subject(entry)
    assert subject.startswith("mem_")
    assert len(subject) <= 20
    assert subject.isascii()


def test_unknown_kind_cannot_be_minted(reg):
    with pytest.raises(ValueError):
        reg.mint("arbitrary_action", "x", SESSION)
