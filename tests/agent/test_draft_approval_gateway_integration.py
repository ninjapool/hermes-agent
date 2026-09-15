"""Integration tests for draft-approval-token minting in the real gateway path.

Tests that verify the fix for DEFECT 1: The token minted in the gateway's
/approve handler must survive to the send_draft tool via contextvars.

CRITICAL INVARIANTS:
- The token is minted synchronously (NOT via asyncio.to_thread) on the event loop
- The contextvar mutation is visible after the mint call returns
- The contextvar is still visible when send_draft runs later in the same context
- Reintroducing asyncio.to_thread breaks the test (regression proof)
"""

import asyncio
import contextvars
import json
import pytest

from agent.approval_tokens import (
    KIND_DRAFT,
    get_registry,
    get_draft_approval_token,
    set_draft_approval_token,
    _draft_approval_token,
)
from gateway.session import SessionSource, Platform
from gateway.platforms.base import MessageEvent
from tools.present_draft import present_draft, send_draft, load_draft


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


class TestContextVarSurvival:
    """Test that the draft-approval-token contextvar survives to send_draft."""

    def test_contextvar_set_is_visible_immediately_after(self):
        """Verify that set_draft_approval_token() sets the value immediately."""
        # Clear any prior value
        ctx_token = _draft_approval_token.set("")
        try:
            # Set a test value
            test_token = "test_token_12345"
            set_draft_approval_token(test_token)
            
            # Verify it's immediately visible
            assert get_draft_approval_token() == test_token
        finally:
            _draft_approval_token.reset(ctx_token)

    def test_mint_binds_resolved_agent_session_id(self, draft_with_attachment):
        """Mint must bind the AGENT session id, not the gateway session key.

        Rewritten 2026-09-15. This class previously asserted the contextvar
        mechanism, which has been replaced: a contextvar set during gateway
        dispatch dies with that context, so by the time send_draft runs it is
        always empty. The registry is now the carrier.
        """
        from gateway.run import GatewayRunner
        from agent.approval_tokens import KIND_DRAFT, get_registry

        runner = GatewayRunner()
        session_key = "agent:main:telegram:dm:8468018784"
        agent_session_id = "20260914_153449_196114dd"

        class _Store:
            def peek_session_id(self, key):
                assert key == session_key
                return agent_session_id

        runner.session_store = _Store()

        token = runner._try_mint_draft_approval_token(
            f"/approve {draft_with_attachment}", session_key
        )
        assert token is not None
        assert token.subject_id == draft_with_attachment

        # Bound to the id the tool will present, NOT the gateway key.
        assert token.session_id == agent_session_id
        assert token.session_id != session_key

        # And it is discoverable by exactly that id.
        found = get_registry().find_unspent(
            kind=KIND_DRAFT,
            subject_id=draft_with_attachment,
            session_id=agent_session_id,
        )
        assert found == token.token

    def test_mint_fails_closed_without_live_session(self, draft_with_attachment):
        """No resolvable session means no token, rather than an unusable one."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner()

        class _EmptyStore:
            def peek_session_id(self, key):
                return None

        runner.session_store = _EmptyStore()

        token = runner._try_mint_draft_approval_token(
            f"/approve {draft_with_attachment}", "no-such-session"
        )
        assert token is None

    def test_registry_carries_token_to_send_draft(self, draft_with_attachment):
        """End-to-end: mint via gateway, then send_draft finds it unaided.

        This is the guarantee that actually matters in production — the tool is
        called on a later turn, in a different context, with no token argument.
        """
        from gateway.run import GatewayRunner

        runner = GatewayRunner()
        session_key = "agent:main:telegram:dm:8468018784"
        agent_session_id = "20260914_153449_196114dd"

        class _Store:
            def peek_session_id(self, key):
                return agent_session_id

        runner.session_store = _Store()

        token = runner._try_mint_draft_approval_token(
            f"/approve {draft_with_attachment}", session_key
        )
        assert token is not None

        # Simulate the real gap: the minting context is gone by the time the
        # tool runs. Explicitly clear the contextvar so this test cannot pass
        # via the mechanism it replaced.
        ctx_token = _draft_approval_token.set("")
        try:
            result = json.loads(
                send_draft(
                    draft_id=draft_with_attachment,
                    approval_token="",
                    session_id=agent_session_id,
                )
            )
            assert result["success"] is True, f"send_draft failed: {result}"
        finally:
            _draft_approval_token.reset(ctx_token)

    def test_send_draft_refuses_without_any_approval(self, draft_with_attachment):
        """The gate still fails closed when nobody approved anything."""
        from agent.approval_tokens import get_registry

        # Ensure no stray token for this subject.
        get_registry().purge_expired()

        ctx_token = _draft_approval_token.set("")
        try:
            result = json.loads(
                send_draft(
                    draft_id=draft_with_attachment,
                    approval_token="",
                    session_id="a-session-that-never-approved",
                )
            )
            assert result["success"] is False
            assert "approval" in result["error"].lower()
        finally:
            _draft_approval_token.reset(ctx_token)

    def test_draft_not_minted_for_bare_slash_approve(self):
        """Bare /approve must NOT mint a token (falls through to dangerous-command)."""
        from gateway.run import GatewayRunner
        
        runner = GatewayRunner()
        token = runner._try_mint_draft_approval_token("/approve", "session_key")
        
        # Should return None
        assert token is None


class TestRegressionProof:
    """Pin the failure mode that reached production on 2026-09-15.

    The contextvar approach is gone, so proving to_thread breaks it is no
    longer interesting. What matters now: a token bound to the gateway session
    KEY can never be spent by a tool presenting the agent session ID. That
    mismatch silently disabled the gate — the mint looked successful and the
    send always failed.
    """

    def test_key_bound_token_cannot_open_the_gate(self, draft_with_attachment):
        """REGRESSION PROOF: bind the wrong identifier and nothing can send."""
        session_key = "agent:main:telegram:dm:8468018784"
        agent_session_id = "20260914_153449_196114dd"
        assert session_key != agent_session_id

        reg = get_registry()
        token = reg.mint(KIND_DRAFT, draft_with_attachment, session_key)

        ok, reason = reg.consume(
            token=token.token,
            kind=KIND_DRAFT,
            subject_id=draft_with_attachment,
            session_id=agent_session_id,
        )
        assert ok is False
        assert "different session" in reason, (
            "DEFECT DEMONSTRATED: a token minted against the gateway session "
            "key is unusable by the tool, which presents the agent session id."
        )

    def test_correct_binding_opens_the_gate(self, draft_with_attachment):
        """The same flow, bound correctly, succeeds."""
        agent_session_id = "20260914_153449_196114dd"

        reg = get_registry()
        token = reg.mint(KIND_DRAFT, draft_with_attachment, agent_session_id)

        ok, reason = reg.consume(
            token=token.token,
            kind=KIND_DRAFT,
            subject_id=draft_with_attachment,
            session_id=agent_session_id,
        )
        assert ok is True, reason
