"""Contract tests for the draft ``from`` field and clean attachment filenames.

Two defects, both found by sending a real test message through the live path
on 2026-09-16:

1. ``present_draft`` had no ``from`` field at all. The sending account was
   decided later, out of band, by whoever invoked the transport — so the
   approval card described an envelope whose sender the reviewer never saw.
   On a desktop session the field is now REQUIRED and never inferred: a
   default-account config, an env var, or "the account we used last time"
   are all guesses, and a guessed sender on a client email is exactly the
   failure this tool exists to prevent.

2. Attachments went out named ``d34d294f_invoice.pdf`` — the cypress staging
   hash prefix leaked into the recipient-visible filename. The prefix existed
   to keep two same-named files from colliding in one flat staging directory;
   the fix keeps that collision safety by putting the hash in a DIRECTORY
   component and leaving the filename clean.

Session state is established through ``set_session_vars`` — the same function
production uses (``tui_gateway/server.py:4511``) — never by writing the env
var directly. A predicate test that sets the variable itself proves only that
the variable exists, which is how the platform-vs-source defect passed a
green suite.
"""

from __future__ import annotations

import json

import pytest

from agent.approval_tokens import ApprovalRegistry
import agent.approval_tokens as approval_tokens
import tools.present_draft as pd


DESKTOP_SESSION = "20260916_120000_desktopaa"
FROM_ADDR = "ds@solanasystems.com"


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch, tmp_path):
    registry = ApprovalRegistry()
    monkeypatch.setattr(approval_tokens, "_REGISTRY", registry)
    draft_dir = tmp_path / "drafts"
    draft_dir.mkdir()
    monkeypatch.setattr(pd, "_DRAFT_DIR", draft_dir)
    monkeypatch.setattr(pd, "_publish_attachment_to_cypress", lambda p: (None, None))
    yield registry


@pytest.fixture
def desktop_session():
    """Bind a real desktop session the way the desktop gateway does."""
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(
        session_id=DESKTOP_SESSION,
        source="desktop",
    )
    try:
        yield
    finally:
        clear_session_vars(tokens)


@pytest.fixture
def telegram_session():
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(
        session_id="20260916_120000_tg",
        platform="telegram",
    )
    try:
        yield
    finally:
        clear_session_vars(tokens)


def _present(**kw):
    payload = {
        "to": "someone@example.com",
        "subject": "contract test",
        "body": "Body text.",
        "session_id": DESKTOP_SESSION,
    }
    payload.update(kw)
    return json.loads(pd.present_draft(**payload))


# --- from: required on desktop -------------------------------------------


def test_desktop_draft_without_from_is_refused(desktop_session):
    """No sender, no draft. The card cannot describe an envelope it lacks."""
    result = _present()
    assert result["success"] is False, result
    assert "from" in result["error"].lower()


def test_refusal_renders_no_draft_file(desktop_session, tmp_path):
    _present()
    assert list((tmp_path / "drafts").glob("*.json")) == []


def test_desktop_draft_with_from_succeeds_and_persists_it(desktop_session):
    result = _present(**{"from": FROM_ADDR})
    assert result["success"] is True, result
    record = pd.load_draft(result["draft_id"])
    assert record["from"] == FROM_ADDR


def test_from_appears_in_the_rendered_draft(desktop_session):
    result = _present(**{"from": FROM_ADDR})
    assert f"**From:** {FROM_ADDR}" in result["rendered"]


def test_from_appears_on_the_approval_card(desktop_session):
    result = _present(**{"from": FROM_ADDR})
    record = pd.load_draft(result["draft_id"])
    assert f"From: {FROM_ADDR}" in pd._draft_card_description(record)


# --- from: never inferred -------------------------------------------------


def test_default_account_config_does_not_satisfy_from(desktop_session, monkeypatch):
    """A configured default sender is not consent to send AS that sender."""
    monkeypatch.setenv("HERMES_DEFAULT_MAIL_FROM", FROM_ADDR)
    result = _present()
    assert result["success"] is False, result


def test_blank_from_is_not_a_from(desktop_session):
    result = _present(**{"from": "   "})
    assert result["success"] is False, result


# --- from: other surfaces keep the old contract ---------------------------


def test_non_desktop_session_does_not_require_from(telegram_session):
    """Messaging surfaces are unchanged; this is a desktop-scoped requirement."""
    result = _present(session_id="20260916_120000_tg")
    assert result["success"] is True, result


# --- attachment filenames -------------------------------------------------


def test_staged_attachment_keeps_its_clean_filename(tmp_path):
    """The recipient sees invoice.pdf, not <hash>_invoice.pdf."""
    src = tmp_path / "invoice.pdf"
    src.write_bytes(b"%PDF-1.4 fixture\n")
    remote_dir, remote_name = pd._staged_attachment_location(src, "a" * 64)
    assert remote_name == "invoice.pdf"


def test_hash_moves_into_the_directory_component(tmp_path):
    """Collision safety is preserved — it just stops renaming the file."""
    src = tmp_path / "invoice.pdf"
    src.write_bytes(b"x")
    remote_dir, _ = pd._staged_attachment_location(src, "a" * 64)
    assert remote_dir.endswith("/" + "a" * 8)


def test_same_name_different_content_does_not_collide(tmp_path):
    """Two files both called invoice.pdf must land on distinct remote paths."""
    one = tmp_path / "one" / "invoice.pdf"
    two = tmp_path / "two" / "invoice.pdf"
    for p in (one, two):
        p.parent.mkdir(parents=True)
    one.write_bytes(b"first")
    two.write_bytes(b"second")
    d1, n1 = pd._staged_attachment_location(one, "1" * 64)
    d2, n2 = pd._staged_attachment_location(two, "2" * 64)
    assert n1 == n2 == "invoice.pdf"
    assert f"{d1}/{n1}" != f"{d2}/{n2}"


def test_no_hash_prefix_survives_anywhere_in_the_filename(tmp_path):
    src = tmp_path / "report.pdf"
    src.write_bytes(b"y")
    _, remote_name = pd._staged_attachment_location(src, "deadbeef" * 8)
    assert "deadbeef" not in remote_name
