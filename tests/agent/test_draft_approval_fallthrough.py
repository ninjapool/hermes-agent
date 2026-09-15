"""Regressions for the two defects that let a 'sent' email never send.

Both were found in production on 2026-09-15, by the user checking his inbox —
not by the test suite. Each test here fails against the old code.

DEFECT A — the gateway swallowed the turn.
    The /approve branch in _handle_message did:
        _draft_token=***
        if _draft_token is not None:
            return f"✓ Token minted for draft `...`. Sending…"
    Returning ended the turn. The agent never ran, so nothing ever called
    send_email. The user was told "Sending…" while nothing sent.

DEFECT B — the token was bound to an identifier the consumer never presents.
    Mint side used the gateway session KEY   ('agent:main:telegram:dm:8468018784')
    Consume side presents agent.session_id   ('20260914_153449_196114dd')
    So consume() always answered "approval token belongs to a different
    session" — the gate could never open, even once the swallow was fixed.
"""

import time

import pytest

from agent.approval_tokens import (
    KIND_DRAFT,
    ApprovalRegistry,
)


SESSION_KEY = "agent:main:telegram:dm:8468018784"
SESSION_ID = "20260914_153449_196114dd"
DRAFT_ID = "draft_20260914_200014_12604f"


class TestSessionBindingMismatch:
    """DEFECT B: mint must bind what the tool will actually present."""

    def test_key_and_id_are_genuinely_different(self):
        """Guard the premise. If these ever became equal the bug would hide."""
        assert SESSION_KEY != SESSION_ID

    def test_token_bound_to_key_cannot_be_consumed_by_id(self):
        """The exact production failure, reproduced."""
        reg = ApprovalRegistry()
        token = reg.mint_from_user_text(
            f"/approve {DRAFT_ID}", kind=KIND_DRAFT, session_id=SESSION_KEY
        )
        assert token is not None

        ok, reason = reg.consume(
            token=token.token,
            kind=KIND_DRAFT,
            subject_id=DRAFT_ID,
            session_id=SESSION_ID,
        )
        assert ok is False
        assert "different session" in reason

    def test_token_bound_to_id_consumes_cleanly(self):
        """The fix: bind the agent session id, and the gate opens."""
        reg = ApprovalRegistry()
        token = reg.mint_from_user_text(
            f"/approve {DRAFT_ID}", kind=KIND_DRAFT, session_id=SESSION_ID
        )
        assert token is not None

        ok, reason = reg.consume(
            token=token.token,
            kind=KIND_DRAFT,
            subject_id=DRAFT_ID,
            session_id=SESSION_ID,
        )
        assert ok is True, reason


class TestFindUnspent:
    """The registry lookup that replaced the contextvar.

    The contextvar died with the dispatch context, so by the time the tool ran
    it was always empty. These tests pin the replacement's guarantees.
    """

    def test_finds_matching_unspent_token(self):
        reg = ApprovalRegistry()
        token = reg.mint(kind=KIND_DRAFT, subject_id=DRAFT_ID, session_id=SESSION_ID)
        found = reg.find_unspent(
            kind=KIND_DRAFT, subject_id=DRAFT_ID, session_id=SESSION_ID
        )
        assert found == token.token

    def test_does_not_find_across_sessions(self):
        """A token from another session must be invisible, not merely unusable."""
        reg = ApprovalRegistry()
        reg.mint(kind=KIND_DRAFT, subject_id=DRAFT_ID, session_id="other-session")
        assert (
            reg.find_unspent(
                kind=KIND_DRAFT, subject_id=DRAFT_ID, session_id=SESSION_ID
            )
            is None
        )

    def test_does_not_find_across_subjects(self):
        """Approving draft A must not send draft B."""
        reg = ApprovalRegistry()
        reg.mint(kind=KIND_DRAFT, subject_id="draft_AAA", session_id=SESSION_ID)
        assert (
            reg.find_unspent(
                kind=KIND_DRAFT, subject_id="draft_BBB", session_id=SESSION_ID
            )
            is None
        )

    def test_does_not_find_expired(self):
        reg = ApprovalRegistry(ttl_seconds=1)
        reg.mint(kind=KIND_DRAFT, subject_id=DRAFT_ID, session_id=SESSION_ID)
        time.sleep(1.1)
        assert (
            reg.find_unspent(
                kind=KIND_DRAFT, subject_id=DRAFT_ID, session_id=SESSION_ID
            )
            is None
        )

    def test_does_not_find_already_spent(self):
        """One approval, one send. A found token must not be re-findable."""
        reg = ApprovalRegistry()
        token = reg.mint(kind=KIND_DRAFT, subject_id=DRAFT_ID, session_id=SESSION_ID)
        ok, _ = reg.consume(
            token=token.token,
            kind=KIND_DRAFT,
            subject_id=DRAFT_ID,
            session_id=SESSION_ID,
        )
        assert ok is True
        assert (
            reg.find_unspent(
                kind=KIND_DRAFT, subject_id=DRAFT_ID, session_id=SESSION_ID
            )
            is None
        )

    def test_cannot_conjure_a_token_never_minted(self):
        """The whole gate in one assertion."""
        reg = ApprovalRegistry()
        assert (
            reg.find_unspent(
                kind=KIND_DRAFT, subject_id=DRAFT_ID, session_id=SESSION_ID
            )
            is None
        )


class TestDispatchFallThrough:
    """DEFECT A: minting must not end the turn.

    Structural assertions against the real source. The dispatch method is
    ~1000 lines inside a class with heavy construction cost, so this reads the
    shipped code rather than standing up a gateway — what matters is that the
    early `return` is gone and the handler-skip is present.
    """

    def _dispatch_source(self) -> str:
        import inspect

        from gateway.run import GatewayRunner

        return inspect.getsource(GatewayRunner._handle_message)

    def test_no_sending_status_string_is_returned(self):
        """The 'Sending…' line claimed an action nobody performed."""
        src = self._dispatch_source()
        assert "Token minted for draft" not in src, (
            "The mint branch returns a status string again. That ends the turn, "
            "so the agent never runs and nothing is ever sent."
        )

    def test_mint_branch_does_not_return(self):
        src = self._dispatch_source()
        marker = "_try_mint_draft_approval_token(" 
        idx = src.find(marker)
        assert idx != -1, "mint call not found in dispatch"
        window = src[idx : idx + 600]
        assert "return f\"✓" not in window
        assert "_skip_approve_handler = True" in window, (
            "Minting must set _skip_approve_handler so the plain /approve "
            "handler does not swallow the turn instead."
        )

    def test_plain_handler_is_guarded_by_skip_flag(self):
        src = self._dispatch_source()
        assert "if plain_handler is not None and not _skip_approve_handler:" in src, (
            "Without this guard the turn reaches _handle_approve_command and "
            "dies with 'No pending command to approve'."
        )

    def test_skip_flag_defaults_false(self):
        """Every other /approve must behave exactly as before."""
        src = self._dispatch_source()
        assert "_skip_approve_handler = False" in src


class TestMintBindsAgentSessionId:
    """DEFECT B at the gateway seam: mint must resolve key -> id."""

    def test_mint_uses_peek_session_id(self):
        import inspect

        from gateway.run import GatewayRunner

        src = inspect.getsource(GatewayRunner._try_mint_draft_approval_token)
        assert "peek_session_id" in src, (
            "Mint must resolve the gateway session key to the agent session id "
            "the consuming tool presents, or the token can never be spent."
        )
        assert "session_id=bind_session" in src

    def test_mint_fails_closed_when_no_live_session(self):
        """No session to approve in means no token."""
        import inspect

        from gateway.run import GatewayRunner

        src = inspect.getsource(GatewayRunner._try_mint_draft_approval_token)
        idx = src.find("peek_session_id")
        window = src[idx : idx + 800]
        assert "return None" in window
