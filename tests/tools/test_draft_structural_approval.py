"""Contract tests for the desktop structural-approval path in ``send_draft``.

The desktop surface has no ``/approve``: the renderer refuses the command
client-side and ``tui_gateway`` never imports ``agent.approval_tokens``, so no
KIND_DRAFT token can be minted there. ``send_draft`` therefore raises the
dangerous-command approval card (the channel the surface DOES have) and treats
the returned choice as the consent.

These tests pin the four properties that make that safe:

1. Messaging surfaces are untouched — no card, still fails closed.
2. The card's text is built from the draft record by code; no model-supplied
   string can reach it.
3. ``allow_permanent`` / ``allow_session`` are False — per-draft, single-use.
4. A draft not rendered in this session is refused before any card is raised.
"""

from __future__ import annotations

import hashlib
import json
import threading

import pytest

from agent.approval_tokens import ApprovalRegistry, KIND_DRAFT
import agent.approval_tokens as approval_tokens
import tools.approval as approval_mod
import tools.present_draft as pd


DESKTOP_SESSION = "20260916_120000_desktopaa"
SESSION_KEY = "desktop-session-key"


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch, tmp_path):
    """Isolate the token registry and the draft directory per test."""
    registry = ApprovalRegistry()
    monkeypatch.setattr(approval_tokens, "_REGISTRY", registry)
    draft_dir = tmp_path / "drafts"
    draft_dir.mkdir()
    monkeypatch.setattr(pd, "_DRAFT_DIR", draft_dir)
    # Structural path must never reach the real cypress host.
    monkeypatch.setattr(pd, "_publish_attachment_to_cypress", lambda p: (None, None))
    yield registry


@pytest.fixture
def _no_cypress_check(monkeypatch):
    """send_draft re-verifies attachments over ssh; keep it local."""
    monkeypatch.setattr(pd.subprocess, "run", lambda *a, **kw: _Ok())
    return None


class _Ok:
    returncode = 0


def _make_attachment(tmp_path, name: str, content: bytes):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _present(tmp_path, *, session_id: str, attachments=None, cc: str = ""):
    """Render a real draft through present_draft and return its record."""
    result = json.loads(
        pd.present_draft(
            to="someone@example.com",
            subject="contract test",
            body="Body text that the card must not quote verbatim.",
            cc=cc,
            attachments=[str(p) for p in (attachments or [])],
            session_id=session_id,
        )
    )
    assert result["success"] is True, result
    return result["draft_id"]


class _Notifier:
    """Stands in for the desktop's ``approval.request`` emitter.

    Captures the payload and resolves the queued approval from another thread,
    exactly as ``approval.respond`` -> ``resolve_gateway_approval`` does.
    """

    def __init__(self, choice: str | None = "once", session_key: str = SESSION_KEY):
        self.choice = choice
        self.session_key = session_key
        self.payloads: list[dict] = []

    def __call__(self, payload: dict) -> None:
        self.payloads.append(dict(payload))
        if self.choice is None:
            return  # never answered -> the wait times out
        # Resolve off-thread: _await_gateway_decision blocks the caller.
        threading.Thread(
            target=approval_mod.resolve_gateway_approval,
            args=(self.session_key, self.choice),
            kwargs={"request_id": payload.get("request_id")},
            daemon=True,
        ).start()


@pytest.fixture
def desktop_surface(monkeypatch):
    """Make ``_structural_approval_surface`` report a desktop session."""

    def _install(notifier: _Notifier | None):
        monkeypatch.setattr(
            pd,
            "_structural_approval_surface",
            lambda: (SESSION_KEY, notifier),
        )

    return _install


# --- 1. round trip --------------------------------------------------------


def test_desktop_approval_round_trip(tmp_path, desktop_surface, _no_cypress_check):
    """Card answered 'once' -> the send passes the gate with no token typed."""
    att = _make_attachment(tmp_path, "one.pdf", b"%PDF-1.4 one\n")
    draft_id = _present(tmp_path, session_id=DESKTOP_SESSION, attachments=[att])

    notifier = _Notifier("once")
    desktop_surface(notifier)

    out = json.loads(pd.send_draft(draft_id=draft_id, session_id=DESKTOP_SESSION))

    assert "not sent —" not in json.dumps(out), out
    assert out["success"] is True, out
    assert len(notifier.payloads) == 1


def test_declined_card_does_not_send_and_leaves_no_token(
    tmp_path, desktop_surface, _fresh_registry
):
    """A refused send must not leave a live approval behind."""
    draft_id = _present(tmp_path, session_id=DESKTOP_SESSION)
    desktop_surface(_Notifier("deny"))

    out = json.loads(pd.send_draft(draft_id=draft_id, session_id=DESKTOP_SESSION))

    assert out["success"] is False
    assert "not sent —" in out["error"]
    assert (
        _fresh_registry.find_unspent(
            kind=KIND_DRAFT, subject_id=draft_id, session_id=DESKTOP_SESSION
        )
        is None
    )


# --- 2. the card is code-derived -----------------------------------------


def test_card_description_is_built_from_the_record(tmp_path, desktop_surface,
                                                   _no_cypress_check):
    """draft_id, To, Cc, Subject and per-attachment name/bytes/sha256."""
    content = b"%PDF-1.4 card derived\n"
    att = _make_attachment(tmp_path, "evidence.pdf", content)
    draft_id = _present(
        tmp_path,
        session_id=DESKTOP_SESSION,
        attachments=[att],
        cc="cc@example.com",
    )

    notifier = _Notifier("once")
    desktop_surface(notifier)
    pd.send_draft(draft_id=draft_id, session_id=DESKTOP_SESSION)

    description = notifier.payloads[0]["description"]
    digest = hashlib.sha256(content).hexdigest()

    assert f"draft_id: {draft_id}" in description
    assert "To: someone@example.com" in description
    assert "Cc: cc@example.com" in description
    assert "Subject: contract test" in description
    assert "evidence.pdf" in description
    assert f"{len(content):,} bytes" in description
    assert f"sha256:{digest}" in description


def test_model_supplied_body_cannot_reach_the_card(tmp_path, desktop_surface,
                                                   _no_cypress_check):
    """The body is model-authored prose; it must not appear on the card.

    Otherwise the model writes its own consent prompt.
    """
    marker = "APPROVE-THIS-IT-IS-ROUTINE-AND-ALREADY-CLEARED"
    result = json.loads(
        pd.present_draft(
            to="someone@example.com",
            subject="contract test",
            body=f"Please click Run. {marker}",
            session_id=DESKTOP_SESSION,
        )
    )
    draft_id = result["draft_id"]

    notifier = _Notifier("once")
    desktop_surface(notifier)
    pd.send_draft(draft_id=draft_id, session_id=DESKTOP_SESSION)

    payload = notifier.payloads[0]
    assert marker not in payload["description"]
    assert marker not in payload["command"]


def test_card_reports_attachment_unreadable_rather_than_stale_hash(tmp_path):
    """A file removed after render is described as UNREADABLE, not hashed."""
    att = _make_attachment(tmp_path, "gone.pdf", b"%PDF-1.4 gone\n")
    draft_id = _present(tmp_path, session_id=DESKTOP_SESSION, attachments=[att])
    draft = pd.load_draft(draft_id)
    assert draft is not None
    att.unlink()

    description = pd._draft_card_description(draft)

    assert "gone.pdf" in description
    assert "UNREADABLE" in description
    assert "sha256:" not in description


# --- 3. no persistent consent -------------------------------------------


def test_allow_permanent_and_allow_session_are_false(tmp_path, desktop_surface,
                                                     _no_cypress_check):
    """Run/Reject only. 'Always allow' on send_email would be standing consent."""
    draft_id = _present(tmp_path, session_id=DESKTOP_SESSION)
    notifier = _Notifier("once")
    desktop_surface(notifier)

    pd.send_draft(draft_id=draft_id, session_id=DESKTOP_SESSION)

    payload = notifier.payloads[0]
    assert payload["allow_permanent"] is False
    assert payload["allow_session"] is False


def test_approving_one_draft_does_not_authorise_another(tmp_path, desktop_surface,
                                                        _no_cypress_check):
    """Consent is per draft: the second send raises its own card."""
    first = _present(tmp_path, session_id=DESKTOP_SESSION)
    second = _present(tmp_path, session_id=DESKTOP_SESSION)
    notifier = _Notifier("once")
    desktop_surface(notifier)

    pd.send_draft(draft_id=first, session_id=DESKTOP_SESSION)
    pd.send_draft(draft_id=second, session_id=DESKTOP_SESSION)

    keys = [p["pattern_key"] for p in notifier.payloads]
    assert keys == [f"send_draft:{first}", f"send_draft:{second}"]


def test_token_is_single_use_after_structural_approval(tmp_path, desktop_surface,
                                                       _no_cypress_check):
    """The locally minted token is spent by the send it authorised."""
    draft_id = _present(tmp_path, session_id=DESKTOP_SESSION)
    desktop_surface(_Notifier("once"))
    first = json.loads(pd.send_draft(draft_id=draft_id, session_id=DESKTOP_SESSION))
    assert first["success"] is True

    # Second attempt with the card suppressed: no token left to fall back on.
    desktop_surface(None)
    second = json.loads(pd.send_draft(draft_id=draft_id, session_id=DESKTOP_SESSION))
    assert second["success"] is False
    assert "not sent —" in second["error"]


# --- 4. provenance ------------------------------------------------------


def test_draft_from_another_session_is_refused_without_a_card(
    tmp_path, desktop_surface, _no_cypress_check
):
    """A readable draft file is not evidence the reviewer here saw it."""
    draft_id = _present(tmp_path, session_id="20260916_999999_othersess")
    notifier = _Notifier("once")
    desktop_surface(notifier)

    out = json.loads(pd.send_draft(draft_id=draft_id, session_id=DESKTOP_SESSION))

    assert out["success"] is False
    assert "not presented in this session" in out["error"]
    assert notifier.payloads == []  # refused BEFORE raising a card


def test_draft_with_no_session_stamp_is_refused(tmp_path, desktop_surface,
                                                _no_cypress_check):
    """Pre-existing drafts (written before this field) get no structural path."""
    draft_id = _present(tmp_path, session_id="")
    notifier = _Notifier("once")
    desktop_surface(notifier)

    out = json.loads(pd.send_draft(draft_id=draft_id, session_id=DESKTOP_SESSION))

    assert out["success"] is False
    assert "not presented in this session" in out["error"]
    assert notifier.payloads == []


# --- messaging surfaces are unchanged ------------------------------------


def test_telegram_source_session_fails_closed_and_raises_no_card(
    tmp_path, monkeypatch, _no_cypress_check
):
    """A messaging session with no token must still fail closed, silently.

    Messaging keeps the /approve <draft_id> contract: it has a working mint
    path already, and a card there would be a second consent channel.

    Binds the session through the REAL production binder rather than setting
    an env var by name — see test_real_desktop_binding_engages_the_card for
    why that distinction is load-bearing here.
    """
    from gateway.session_context import clear_session_vars, set_session_vars

    draft_id = _present(tmp_path, session_id="20260916_120000_telegramzz")
    session_key = "agent:main:telegram:dm:8468018784"

    raised: list[dict] = []
    monkeypatch.setattr(
        approval_mod,
        "_gateway_notify_cbs",
        {session_key: lambda payload: raised.append(payload)},
    )

    tokens = set_session_vars(
        session_key=session_key,
        session_id="20260916_120000_telegramzz",
        platform="telegram",
        source="telegram",
    )
    try:
        out = json.loads(
            pd.send_draft(draft_id=draft_id, session_id="20260916_120000_telegramzz")
        )
    finally:
        clear_session_vars(tokens)

    assert out["success"] is False
    assert "not sent —" in out["error"]
    assert out["needs"] == f"/approve {draft_id}"
    assert raised == []


def test_real_desktop_binding_engages_the_card(tmp_path, monkeypatch,
                                               _no_cypress_check):
    """Bind the session the way the SERVE ACTUALLY DOES, then expect a card.

    This test exists because the first implementation gated on
    HERMES_SESSION_PLATFORM == "desktop" and shipped with fifteen green tests.
    In production that predicate is never true: the gateway binds a platform
    value for messaging, while the CLI/TUI/desktop bind HERMES_SESSION_SOURCE
    and leave the platform EMPTY (gateway/session_context.py:419-424). Every
    test that set the platform env var by hand agreed with the bug.

    So this one does not name an env var at all. It calls the real
    set_session_vars with source="desktop" — the same call
    tui_gateway/server.py:4511 makes — and asserts a card is raised. If the
    surface predicate drifts away from what the serve binds, this fails.
    """
    from gateway.session_context import clear_session_vars, set_session_vars

    draft_id = _present(tmp_path, session_id=DESKTOP_SESSION)
    session_key = "desktop:real-binding"

    notifier = _Notifier("once", session_key=session_key)
    monkeypatch.setattr(approval_mod, "_gateway_notify_cbs", {session_key: notifier})

    tokens = set_session_vars(
        session_key=session_key,
        session_id=DESKTOP_SESSION,
        source="desktop",
    )
    try:
        out = json.loads(pd.send_draft(draft_id=draft_id, session_id=DESKTOP_SESSION))
    finally:
        clear_session_vars(tokens)

    assert len(notifier.payloads) == 1, "no card raised on a real desktop binding"
    assert out["success"] is True, out


def test_real_tui_binding_does_not_engage_the_card(tmp_path, monkeypatch,
                                                   _no_cypress_check):
    """The embedded terminal pane shares the serve process but not the card.

    `hermes --tui` binds source="tui" through the same binder. It has no
    approval card UI, so it must fall through to the token path rather than
    blocking on a prompt nothing can answer.
    """
    from gateway.session_context import clear_session_vars, set_session_vars

    draft_id = _present(tmp_path, session_id=DESKTOP_SESSION)
    session_key = "tui:real-binding"

    raised: list[dict] = []
    monkeypatch.setattr(
        approval_mod,
        "_gateway_notify_cbs",
        {session_key: lambda payload: raised.append(payload)},
    )

    tokens = set_session_vars(
        session_key=session_key,
        session_id=DESKTOP_SESSION,
        source="tui",
    )
    try:
        out = json.loads(pd.send_draft(draft_id=draft_id, session_id=DESKTOP_SESSION))
    finally:
        clear_session_vars(tokens)

    assert out["success"] is False
    assert "not sent —" in out["error"]
    assert raised == []


def test_no_notify_cb_registered_falls_through_to_the_token_path(
    tmp_path, monkeypatch, _no_cypress_check
):
    """Cron/headless desktop: no card is reachable, so fail closed."""
    draft_id = _present(tmp_path, session_id=DESKTOP_SESSION)
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "desktop")
    monkeypatch.setattr(approval_mod, "_gateway_notify_cbs", {})

    out = json.loads(pd.send_draft(draft_id=draft_id, session_id=DESKTOP_SESSION))

    assert out["success"] is False
    assert "not sent —" in out["error"]


def test_explicit_token_still_works_and_skips_the_card(
    tmp_path, desktop_surface, _fresh_registry, _no_cypress_check
):
    """The /approve path keeps precedence; no card when a token exists."""
    draft_id = _present(tmp_path, session_id=DESKTOP_SESSION)
    token = _fresh_registry.mint(
        kind=KIND_DRAFT, subject_id=draft_id, session_id=DESKTOP_SESSION
    ).token
    notifier = _Notifier("once")
    desktop_surface(notifier)

    out = json.loads(
        pd.send_draft(
            draft_id=draft_id,
            approval_token=token,
            session_id=DESKTOP_SESSION,
        )
    )

    assert out["success"] is True
    assert notifier.payloads == []


# --- cross-process rejection (standing invariant) ------------------------


def test_token_minted_in_another_process_is_rejected(tmp_path):
    """Two registries stand in for two processes; tokens do not cross."""
    other_process = ApprovalRegistry()
    draft_id = _present(tmp_path, session_id=DESKTOP_SESSION)
    foreign = other_process.mint(
        kind=KIND_DRAFT, subject_id=draft_id, session_id=DESKTOP_SESSION
    ).token

    out = json.loads(
        pd.send_draft(
            draft_id=draft_id,
            approval_token=foreign,
            session_id=DESKTOP_SESSION,
        )
    )

    assert out["success"] is False
    assert "not sent —" in out["error"]


def test_token_for_one_draft_is_rejected_for_another(tmp_path, _fresh_registry):
    """Subject binding: A's token cannot send B, and is spent by the attempt."""
    first = _present(tmp_path, session_id=DESKTOP_SESSION)
    second = _present(tmp_path, session_id=DESKTOP_SESSION)
    token = _fresh_registry.mint(
        kind=KIND_DRAFT, subject_id=first, session_id=DESKTOP_SESSION
    ).token

    out = json.loads(
        pd.send_draft(
            draft_id=second,
            approval_token=token,
            session_id=DESKTOP_SESSION,
        )
    )

    assert out["success"] is False
    assert "not sent —" in out["error"]
    # Spent by the mismatched attempt: it cannot be walked onto a third draft.
    assert _fresh_registry.peek(token) is None
