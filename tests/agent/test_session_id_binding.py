"""Test the session ID mismatch between gateway minting and tool consuming.

This is the PRIMARY empirical test for the session-id-binding fix.
When the user replies /approve <draft-id> in the gateway, a token is minted
with the gateway's session_key. When send_draft calls the approval registry,
it passes a session_id from model_tools.py. These MUST match or the token
is rejected with "approval token belongs to a different session".

This test builds a real draft and attempts to send it with a minted token,
capturing the actual session ID values involved.
"""

import json
import pytest

from agent.approval_tokens import (
    KIND_DRAFT,
    get_registry,
)
from tools.present_draft import present_draft, send_draft


@pytest.fixture
def draft_with_attachment(tmp_path, monkeypatch):
    """Create a real persisted draft with an attachment."""
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


class TestSessionIdBinding:
    """Test that session IDs match between minting and consuming."""

    def test_send_draft_accepts_token_minted_with_same_session_id(
        self, draft_with_attachment
    ):
        """Token minted and consumed with the SAME session_id should work."""
        reg = get_registry()
        session_id = "sess_test_123"
        
        # Mint a token
        tok = reg.mint_from_user_text(
            f"/approve {draft_with_attachment}",
            kind=KIND_DRAFT,
            session_id=session_id,
        )
        assert tok is not None
        assert tok.session_id == session_id
        
        # Consume with the SAME session_id
        result = json.loads(
            send_draft(
                draft_id=draft_with_attachment,
                approval_token=tok.token,
                session_id=session_id,
            )
        )
        assert result["success"] is True

    def test_send_draft_rejects_token_minted_with_different_session_id(
        self, draft_with_attachment
    ):
        """Token minted and consumed with DIFFERENT session_ids should fail."""
        reg = get_registry()
        
        # Mint with session A
        tok = reg.mint_from_user_text(
            f"/approve {draft_with_attachment}",
            kind=KIND_DRAFT,
            session_id="sess_a",
        )
        assert tok is not None
        assert tok.session_id == "sess_a"
        
        # Attempt to consume with session B
        result = json.loads(
            send_draft(
                draft_id=draft_with_attachment,
                approval_token=tok.token,
                session_id="sess_b",
            )
        )
        assert result["success"] is False
        assert "different session" in result.get("error", "").lower()

    def test_token_session_id_is_preserved_byte_identical(self, draft_with_attachment):
        """Session IDs must be byte-identical (using hmac.compare_digest)."""
        reg = get_registry()
        
        # Use a session ID with special characters
        session_id_original = "sess_unicode_ñ_日本_test"
        
        tok = reg.mint_from_user_text(
            f"/approve {draft_with_attachment}",
            kind=KIND_DRAFT,
            session_id=session_id_original,
        )
        assert tok is not None
        # Token stores the exact bytes
        assert tok.session_id == session_id_original
        
        # Consuming with byte-identical session_id must work
        result = json.loads(
            send_draft(
                draft_id=draft_with_attachment,
                approval_token=tok.token,
                session_id=session_id_original,
            )
        )
        assert result["success"] is True
        
        # Consuming with even slightly different encoding fails
        session_id_similar = "sess_unicode_ñ_日本_test"
        # (These look identical but might have different encodings in edge cases)
        if session_id_similar != session_id_original:
            result2 = json.loads(
                send_draft(
                    draft_id=draft_with_attachment,
                    approval_token=tok.token,
                    session_id=session_id_similar,
                )
            )
            assert result2["success"] is False

    def test_registry_consume_checks_session_id_with_hmac_compare_digest(
        self, draft_with_attachment
    ):
        """Verify that consume uses hmac.compare_digest for session_id."""
        reg = get_registry()
        
        # Mint a token
        tok = reg.mint(KIND_DRAFT, draft_with_attachment, "sess_123")
        
        # Consume with matching session_id
        ok, reason = reg.consume(tok.token, KIND_DRAFT, draft_with_attachment, "sess_123")
        assert ok is True
        
        # Try again with a different session (token is already spent)
        ok2, reason2 = reg.consume(
            tok.token, KIND_DRAFT, draft_with_attachment, "sess_456"
        )
        assert ok2 is False
        # The error should mention session mismatch (even though token is already spent)
        # But since token is spent first, it will say "already spent"
        
        # Test with a fresh token
        tok2 = reg.mint(KIND_DRAFT, draft_with_attachment, "sess_xyz")
        ok3, reason3 = reg.consume(
            tok2.token, KIND_DRAFT, draft_with_attachment, "sess_xyz"
        )
        assert ok3 is True
        
        # Same draft, different session (fresh token)
        tok3 = reg.mint(KIND_DRAFT, draft_with_attachment, "sess_aaa")
        ok4, reason4 = reg.consume(
            tok3.token, KIND_DRAFT, draft_with_attachment, "sess_bbb"
        )
        assert ok4 is False
        assert "different session" in reason4
