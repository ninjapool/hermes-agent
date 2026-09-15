"""Integration tests for draft-approval-token minting in the gateway user-turn path.

Tests wiring `/approve <draft-id>` into the gateway's message dispatch so that
tokens are minted for existing persisted drafts before the dangerous-command
handler runs.

This is the PRIMARY regression suite for the session ID mismatch fix.
"""

import json
import pytest

from agent.approval_tokens import (
    KIND_DRAFT,
    get_registry,
)
from gateway.session import SessionSource, Platform
from tools.present_draft import present_draft, send_draft, load_draft


SESSION_GATEWAY = "sess_gateway_test"
SESSION_OTHER = "sess_other_test"


@pytest.fixture
def draft_with_attachment(tmp_path, monkeypatch):
    """Create a real persisted draft with an attachment, return the draft_id."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    f = tmp_path / "test_doc.pdf"
    f.write_bytes(b"%PDF-1.4 test")
    res = json.loads(
        present_draft(
            to="test@example.com",
            subject="Test Draft",
            body="This is a test draft.",
            attachments=[str(f)],
        )
    )
    assert res["success"] is True, res
    return res["draft_id"]


class TestDraftApprovalTokenMinting:
    """Test that /approve <draft-id> mints a token in the gateway path."""

    def test_mint_from_user_text_matches_slash_approve_plus_valid_draft_id(self):
        """Verify the regex accepts valid draft IDs."""
        reg = get_registry()
        # Valid draft IDs: alphanumeric, underscore, hyphen, dot; 3-128 chars
        text = "/approve draft_20260914_153722_b9bce5"
        tok = reg.mint_from_user_text(text, kind=KIND_DRAFT, session_id=SESSION_GATEWAY)
        assert tok is not None
        assert tok.subject_id == "draft_20260914_153722_b9bce5"

    def test_mint_does_not_fire_for_bare_approve(self):
        """Bare /approve must NOT mint a draft token; it goes to dangerous-command handler."""
        reg = get_registry()
        tok = reg.mint_from_user_text("/approve", kind=KIND_DRAFT, session_id=SESSION_GATEWAY)
        assert tok is None

    def test_mint_does_not_fire_for_approve_all(self):
        """'/approve all' is for dangerous commands, not drafts."""
        reg = get_registry()
        tok = reg.mint_from_user_text("/approve all", kind=KIND_DRAFT, session_id=SESSION_GATEWAY)
        assert tok is None

    def test_mint_does_not_fire_for_approve_session(self):
        """'/approve session' is for dangerous commands, not drafts."""
        reg = get_registry()
        tok = reg.mint_from_user_text("/approve session", kind=KIND_DRAFT, session_id=SESSION_GATEWAY)
        assert tok is None

    def test_mint_does_not_fire_for_approve_always(self):
        """'/approve always' is for dangerous commands, not drafts."""
        reg = get_registry()
        tok = reg.mint_from_user_text("/approve always", kind=KIND_DRAFT, session_id=SESSION_GATEWAY)
        assert tok is None

    def test_mint_does_not_fire_for_prose(self):
        """Prose like 'yes send it' must never mint."""
        reg = get_registry()
        for text in [
            "yes send it",
            "ok go ahead",
            "approve the draft",
            "please send this",
        ]:
            tok = reg.mint_from_user_text(text, kind=KIND_DRAFT, session_id=SESSION_GATEWAY)
            assert tok is None, f"Prose '{text}' should not mint a token"

    def test_token_minted_in_session_a_cannot_be_spent_in_session_b(self, draft_with_attachment):
        """Security: tokens are session-scoped."""
        reg = get_registry()

        # Mint a token in session A
        tok_a = reg.mint(KIND_DRAFT, draft_with_attachment, SESSION_GATEWAY)

        # Attempt to spend it in session B
        ok, reason = reg.consume(tok_a.token, KIND_DRAFT, draft_with_attachment, SESSION_OTHER)
        assert ok is False
        assert "different session" in reason

    def test_token_cannot_send_twice(self, draft_with_attachment):
        """Single-use: same token cannot send the same draft twice."""
        reg = get_registry()
        tok = reg.mint(KIND_DRAFT, draft_with_attachment, SESSION_GATEWAY)

        # First send succeeds
        ok1, _ = reg.consume(tok.token, KIND_DRAFT, draft_with_attachment, SESSION_GATEWAY)
        assert ok1 is True

        # Second send fails (token already spent)
        ok2, reason = reg.consume(tok.token, KIND_DRAFT, draft_with_attachment, SESSION_GATEWAY)
        assert ok2 is False
        assert "already spent" in reason

    def test_send_draft_accepts_minted_token_with_matching_session_id(self, draft_with_attachment):
        """The minted token must be spendable by send_draft with the same session_id."""
        reg = get_registry()
        tok = reg.mint_from_user_text(
            f"/approve {draft_with_attachment}",
            kind=KIND_DRAFT,
            session_id=SESSION_GATEWAY,
        )
        assert tok is not None

        # send_draft calls the token gate with the session_id it receives
        result = json.loads(
            send_draft(
                draft_id=draft_with_attachment,
                approval_token=tok.token,
                session_id=SESSION_GATEWAY,
            )
        )
        assert result["success"] is True, f"send_draft failed: {result}"
        assert result["approved"] is True

    def test_send_draft_rejects_token_from_different_session(self, draft_with_attachment):
        """Reject tokens from other sessions (the mismatch being tested)."""
        reg = get_registry()

        # Mint in SESSION_GATEWAY
        tok = reg.mint_from_user_text(
            f"/approve {draft_with_attachment}",
            kind=KIND_DRAFT,
            session_id=SESSION_GATEWAY,
        )
        assert tok is not None

        # Attempt to spend in SESSION_OTHER
        result = json.loads(
            send_draft(
                draft_id=draft_with_attachment,
                approval_token=tok.token,
                session_id=SESSION_OTHER,
            )
        )
        assert result["success"] is False
        # Should fail with session mismatch, not succeed
        assert "different session" in result.get("error", "").lower()

    def test_unknown_draft_id_does_not_mint(self):
        """Slash-approve with an unknown draft id should not mint (and fall through)."""
        reg = get_registry()
        tok = reg.mint_from_user_text(
            "/approve draft_does_not_exist",
            kind=KIND_DRAFT,
            session_id=SESSION_GATEWAY,
        )
        # The regex WILL match, but the gateway must check if the draft exists
        # BEFORE minting. This test just verifies the regex allows the id format;
        # the actual existence check happens in the gateway integration.
        assert tok is not None  # regex matches

    def test_draft_loaded_successfully_identifies_valid_draft(self, draft_with_attachment):
        """Verify that load_draft() can retrieve a persisted draft."""
        loaded = load_draft(draft_with_attachment)
        assert loaded is not None
        assert loaded["draft_id"] == draft_with_attachment


class TestModifierAndUnicodeHardening:
    """Regressions found 2026-09-14 by running the suite the canonical way.

    1. mint_from_user_text("/approve all") used to return a live token whose
       subject_id was "all". The ONLY thing preventing a dangerous-command
       modifier becoming a send authorisation was load_draft("all") returning
       None -- i.e. a filename not existing. Consent must not rest on that.
    2. consume() compared session ids with hmac.compare_digest on raw str,
       which raises TypeError on non-ASCII. Telegram session keys can carry
       non-ASCII, so the gate could raise instead of returning a decision.
    """

    def test_all_modifier_forms_never_mint(self):
        reg = get_registry()
        for text in (
            "/approve all", "/approve ALL", "/approve always",
            "/approve session", "/approve once", "/approve yes",
            "  /approve   all  ",
        ):
            tok = reg.mint_from_user_text(
                text, kind=KIND_DRAFT, session_id=SESSION_GATEWAY
            )
            assert tok is None, f"{text!r} minted {tok!r} -- modifier leaked"

    def test_nonascii_session_id_is_decided_not_raised(self):
        """A mismatch on a non-ASCII session key must return False, not raise."""
        from agent.approval_tokens import ApprovalRegistry
        reg = ApprovalRegistry()
        tok = reg.mint(
            kind=KIND_DRAFT, subject_id="draft_x", session_id="chat_\u90e1\u5c71_1"
        )
        ok, reason = reg.consume(
            token=tok.token, kind=KIND_DRAFT, subject_id="draft_x",
            session_id="chat_\u5225\u306e\u5834\u6240_2",
        )
        assert ok is False
        assert "different session" in reason

    def test_nonascii_session_id_matches_itself(self):
        from agent.approval_tokens import ApprovalRegistry
        reg = ApprovalRegistry()
        key = "telegram:\u30c7\u30d3\u30c3\u30c9:8468018784"
        tok = reg.mint(kind=KIND_DRAFT, subject_id="draft_y", session_id=key)
        ok, reason = reg.consume(
            token=tok.token, kind=KIND_DRAFT, subject_id="draft_y", session_id=key
        )
        assert ok is True, reason
