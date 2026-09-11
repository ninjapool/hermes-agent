"""The two token gates, end to end: send_email and memory removals.

These exercise the real tool entry points, not the registry in isolation —
the gate only counts if it fires where the model actually calls.
"""

import json

import pytest

from agent.approval_tokens import (
    KIND_DRAFT,
    KIND_MEMORY_REMOVE,
    get_registry,
    memory_removal_subject,
)
from tools.memory_tool import MemoryStore, memory_tool
from tools.present_draft import present_draft, send_draft

SESSION = "sess_gate"


@pytest.fixture
def draft(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    f = tmp_path / "4063_minutes_20260818.pdf"
    f.write_bytes(b"%PDF-1.4 minutes")
    res = json.loads(
        present_draft(
            to="hk@kotoholdings.com",
            subject="Chita 4063 — handover",
            body="Attached are the minutes.",
            attachments=[str(f)],
        )
    )
    assert res["success"] is True, res
    return res


# --- send_email ------------------------------------------------------------


def test_draft_id_alone_does_not_send(draft):
    """The id says WHICH draft; only a token says to send it."""
    res = json.loads(send_draft(draft_id=draft["draft_id"], session_id=SESSION))
    assert res["success"] is False
    assert f"/approve {draft['draft_id']}" == res["needs"]


def test_send_succeeds_with_a_user_minted_token(draft):
    tok = get_registry().mint_from_user_text(
        f"/approve {draft['draft_id']}", kind=KIND_DRAFT, session_id=SESSION
    )
    assert tok is not None
    res = json.loads(
        send_draft(
            draft_id=draft["draft_id"],
            approval_token=tok.token,
            session_id=SESSION,
        )
    )
    assert res["success"] is True
    assert res["approved"] is True
    assert res["attachment_count"] == 1


def test_approval_does_not_authorise_a_second_send(draft):
    tok = get_registry().mint(KIND_DRAFT, draft["draft_id"], SESSION)
    first = json.loads(
        send_draft(draft["draft_id"], tok.token, SESSION)
    )
    assert first["success"] is True
    second = json.loads(send_draft(draft["draft_id"], tok.token, SESSION))
    assert second["success"] is False


def test_token_for_one_draft_cannot_send_another(draft, tmp_path):
    other = tmp_path / "other.pdf"
    other.write_bytes(b"%PDF-1.4 other")
    second = json.loads(
        present_draft(
            to="a@b.com", subject="other", body="see other.pdf",
            attachments=[str(other)],
        )
    )
    tok = get_registry().mint(KIND_DRAFT, draft["draft_id"], SESSION)
    res = json.loads(send_draft(second["draft_id"], tok.token, SESSION))
    assert res["success"] is False


def test_unknown_draft_is_refused_before_any_token_work():
    res = json.loads(send_draft("draft_does_not_exist", "", SESSION))
    assert res["success"] is False
    assert "no draft" in res["error"]


def test_attachment_deleted_after_approval_blocks_the_send(draft, tmp_path):
    """The reviewer approved openable files; sending unopenable ones is not it."""
    (tmp_path / "4063_minutes_20260818.pdf").unlink()
    tok = get_registry().mint(KIND_DRAFT, draft["draft_id"], SESSION)
    res = json.loads(send_draft(draft["draft_id"], tok.token, SESSION))
    assert res["success"] is False
    assert "no longer exist" in res["error"]


# --- memory removals -------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    s = MemoryStore()
    s.add("memory", "ENTRY_ONE: keep this")
    s.add("memory", "ENTRY_TWO: keep this too")
    return s


def test_removal_without_a_token_is_refused_and_shows_the_entry(store):
    res = json.loads(
        memory_tool(
            action="remove",
            old_text="ENTRY_ONE: keep this",
            store=store,
            session_id=SESSION,
        )
    )
    assert res["success"] is False
    shown = res["entries_requiring_approval"]
    assert shown[0]["entry"] == "ENTRY_ONE: keep this"
    assert shown[0]["approve_with"].startswith("/approve mem_")
    # And nothing was deleted.
    assert "ENTRY_ONE" in "\n".join(store._entries_for("memory"))


def test_removal_proceeds_with_a_matching_token(store):
    entry = "ENTRY_ONE: keep this"
    tok = get_registry().mint(
        KIND_MEMORY_REMOVE, memory_removal_subject(entry), SESSION
    )
    res = json.loads(
        memory_tool(
            action="remove",
            old_text=entry,
            store=store,
            approval_token=tok.token,
            session_id=SESSION,
        )
    )
    assert res["success"] is True
    assert "ENTRY_ONE" not in "\n".join(store._entries_for("memory"))


def test_token_for_one_entry_cannot_remove_another(store):
    tok = get_registry().mint(
        KIND_MEMORY_REMOVE,
        memory_removal_subject("ENTRY_ONE: keep this"),
        SESSION,
    )
    res = json.loads(
        memory_tool(
            action="remove",
            old_text="ENTRY_TWO: keep this too",
            store=store,
            approval_token=tok.token,
            session_id=SESSION,
        )
    )
    assert res["success"] is False
    assert "ENTRY_TWO" in "\n".join(store._entries_for("memory"))


def test_batch_containing_an_unapproved_removal_is_refused_whole(store):
    """The 2026-09-08 shape exactly: adds/replaces carrying removals along."""
    res = json.loads(
        memory_tool(
            operations=[
                {"action": "add", "content": "NEW: fine on its own"},
                {"action": "remove", "old_text": "ENTRY_ONE: keep this"},
                {"action": "remove", "old_text": "ENTRY_TWO: keep this too"},
            ],
            store=store,
            session_id=SESSION,
        )
    )
    assert res["success"] is False
    assert len(res["entries_requiring_approval"]) == 2
    body = "\n".join(store._entries_for("memory"))
    assert "ENTRY_ONE" in body and "ENTRY_TWO" in body
    # The unrelated add must not have landed either — the batch is one unit.
    assert "NEW: fine on its own" not in body


def test_adds_and_replaces_are_not_gated(store):
    """Only removals need a token; recoverable writes stay frictionless."""
    res = json.loads(
        memory_tool(
            action="add", content="ADDED: no token needed",
            store=store, session_id=SESSION,
        )
    )
    assert res["success"] is True
    res = json.loads(
        memory_tool(
            action="replace",
            old_text="ENTRY_ONE: keep this",
            content="ENTRY_ONE: revised",
            store=store,
            session_id=SESSION,
        )
    )
    assert res["success"] is True
