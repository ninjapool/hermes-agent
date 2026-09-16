"""``present_draft`` — the only sanctioned way to show an email draft for review.

Why this is a tool and not a rule
---------------------------------
On 2026-09-08 a seven-attachment client handover was presented with all seven
filenames written in the body text, wrapped in backticks, and zero of them
actually attached to the chat message. The reviewer could read the names but
could not open a single file, so the draft could not be checked. The standing
instruction ("name the attachment with a backticked path") had been followed
to the letter — and naming a file is not showing it.

A rule the model must remember is exactly the wrong shape of fix for a failure
whose mechanism is the model believing it has complied. So the count lives
here instead: ``present_draft`` renders the attachment lines itself, from the
``attachments`` list, and refuses to render at all if a path does not resolve.
The model cannot produce a draft whose attachment lines disagree with the
files that would actually be sent, because it no longer writes those lines.

``send_email`` accepts only a ``draft_id`` minted here, so the reviewed object
and the sent object are the same object.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import display_hermes_home, get_hermes_home
from agent.approval_tokens import KIND_DRAFT, get_registry, get_draft_approval_token
from tools.registry import registry

logger = logging.getLogger("tools.present_draft")

# Drafts live under HERMES_HOME so each profile keeps its own (never ~/.hermes).
_DRAFT_DIR = get_hermes_home() / "drafts"

# Extensions we treat as review deliverables. A filename mentioned in body text
# with one of these extensions is a document the reviewer will expect to open.
_DOC_EXT = (
    "pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt", "csv",
    "zip", "png", "jpg", "jpeg", "gif", "webp", "dwg", "dxf", "txt", "md",
)

# A filename-shaped token in prose. Spaces are deliberately NOT allowed: a
# permissive class swallows the preceding prose word ("and 4063_flowchart.xlsx"),
# which then never matches the real basename. CEJ filenames are underscore- and
# hyphen-separated, so this is the right shape; a genuinely space-containing
# filename in prose is rarer than the false match it would cost.
_FILENAME_IN_TEXT = re.compile(
    r"[\w][\w.\-()（）]{0,120}\.(?:" + "|".join(_DOC_EXT) + r")\b",
    re.IGNORECASE,
)


def _draft_path(draft_id: str) -> Path:
    return _DRAFT_DIR / f"{draft_id}.json"


def filenames_mentioned_in_text(body: str) -> List[str]:
    """Return document filenames that appear in prose.

    Used by both the tool (to catch a body that references a document while
    ``attachments`` is empty) and the outbound lint. Deliberately matches on a
    document EXTENSION rather than on backticks: the 2026-09-08 failure was a
    body full of correctly-backticked names, so backticks cannot be the
    signal. Any mention of a document the reviewer might expect to open counts.
    """
    if not body:
        return []
    seen: Dict[str, None] = {}
    for match in _FILENAME_IN_TEXT.finditer(body):
        name = match.group(0).strip().strip("`").lstrip("/")
        seen.setdefault(os.path.basename(name), None)
    return list(seen)


def _staged_attachment_location(local_path: Path, sha256: str) -> tuple[str, str]:
    """Return ``(remote_dir, remote_name)`` for an attachment staged on cypress.

    The filename is left CLEAN. The old scheme prefixed it with the first 8
    hex of the digest (``d34d294f_invoice.pdf``) to keep two same-named files
    from colliding in one flat staging directory — and that prefix rode all
    the way out to the recipient, because the transport passes the staged
    basename to the MIME part. The reviewer approved ``invoice.pdf`` and the
    client received ``d34d294f_invoice.pdf``.

    The collision safety was real, so it is kept: the hash moves into a
    DIRECTORY component. Two files named ``invoice.pdf`` with different
    content stage to different directories and keep their own names; the same
    file staged twice lands on the same path, which is idempotent and correct.
    """
    return f"~/.hermes-attachments/{sha256[:8]}", local_path.name


def _publish_attachment_to_cypress(local_path: str) -> tuple[Optional[str], Optional[str]]:
    """Publish an attachment to the cypress remote host for send-time retrieval.
    
    The hermes-send transport reads attachments from cypress's local disk via
    /run/hermes-mailsend/sock. This function copies the file there and verifies
    it by content hash.
    
    Attachments are staged to ~/.hermes-attachments on cypress (the home directory
    of the account running the ssh connection, typically the deploy user ds).
    This directory is created with mode 0700 to ensure client invoices
    (containing customer commercial figures) are not world-readable.
    
    NOTE (2026-09-14): While the hermes-send transport runs as user mailsend
    (uid=976), we create the staging directory under the SSH user's home (~ds)
    because we lack passwordless sudo to create it as mailsend. The transport
    validates that it can read from the paths hermes-send provides, so file
    ownership is not a blocker. The security property (0700 mode, not world-readable)
    is preserved.
    
    Returns (cypress_path, sha256_hex) on success, (None, None) on failure.
    """
    import hashlib
    import subprocess

    # Staging is a network round-trip (ssh + scp + remote hash). Timeouts must
    # be generous enough for a real WAN hop under load: the original 5s ssh
    # timeout made present_draft fail intermittently whenever the host was
    # busy, and a *timed-out* publish is indistinguishable from a refused one,
    # so a slow link silently became "no draft". Env-overridable for slow links.
    _ssh_timeout = int(os.environ.get("HERMES_DRAFT_SSH_TIMEOUT", "30"))
    _scp_timeout = int(os.environ.get("HERMES_DRAFT_SCP_TIMEOUT", "120"))

    local_path_obj = Path(local_path)
    if not local_path_obj.is_file():
        return None, None
    
    # Read and hash the source
    try:
        content = local_path_obj.read_bytes()
        sha256 = hashlib.sha256(content).hexdigest()
    except OSError as e:
        logger.warning("Failed to read source file %s: %s", local_path, e)
        return None, None
    
    # Publish to cypress via scp
    # Use the secure staging directory under SSH user's home
    remote_dir, remote_name = _staged_attachment_location(local_path_obj, sha256)
    remote_path = f"{remote_dir}/{remote_name}"
    
    try:
        # Create remote directory with secure mode (0700). Both levels: the
        # per-hash subdir AND its parent, or the parent lands on the default
        # umask and the 0700 guarantee is only true of the leaf.
        parent_dir = remote_dir.rsplit("/", 1)[0]
        subprocess.run(
            [
                "ssh",
                "cypress",
                f"mkdir -p -m 0700 {parent_dir} && mkdir -p -m 0700 {remote_dir}",
            ],
            check=True,
            capture_output=True,
            timeout=_ssh_timeout,
        )
        
        # Copy file
        subprocess.run(
            ["scp", local_path, f"cypress:{remote_path}"],
            check=True,
            capture_output=True,
            timeout=_scp_timeout,
        )
        
        # Verify by hash on remote
        result = subprocess.run(
            ["ssh", "cypress", f"sha256sum {shlex.quote(remote_path)}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=_ssh_timeout,
        )
        remote_sha256 = result.stdout.split()[0]
        
        if remote_sha256 != sha256:
            logger.error(
                "Hash mismatch for %s: local=%s remote=%s",
                local_path, sha256, remote_sha256
            )
            return None, None
        
        # Expand the remote path for logging and storage (convert ~)
        result_path = subprocess.run(
            ["ssh", "cypress", f"echo {remote_path}"],
            capture_output=True,
            text=True,
            timeout=_ssh_timeout,
            check=True,
        )
        expanded_path = result_path.stdout.strip()
        
        logger.info(
            "Published attachment %s -> %s (sha256=%s)",
            local_path, expanded_path, sha256
        )
        return expanded_path, sha256
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        logger.warning(
            "Failed to publish %s to cypress: %s",
            local_path, e
        )
        return None, None


def _resolve_attachments(
    attachments: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Resolve every path to a real file, or report why it failed.

    An unresolvable path is a hard failure, not a warning. A draft that lists a
    file the reviewer cannot open is the exact defect this tool exists to
    prevent, so there is no partial-success mode: either every attachment
    resolves or nothing is rendered.
    
    As of 2026-09-14, this ALSO publishes each attachment to cypress and
    stores both the local review path and the remote cypress path in the record.
    """
    resolved: List[Dict[str, Any]] = []
    errors: List[str] = []
    for raw in attachments:
        if not isinstance(raw, str) or not raw.strip():
            errors.append(f"empty or non-string attachment path: {raw!r}")
            continue
        path = Path(os.path.expanduser(raw.strip()))
        if not path.is_absolute():
            errors.append(
                f"{raw!r}: not an absolute path. Attachment paths must be "
                "absolute — a relative path resolves differently for the "
                "gateway than for you."
            )
            continue
        if not path.exists():
            errors.append(f"{path}: does not exist")
            continue
        if not path.is_file():
            errors.append(f"{path}: not a regular file")
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            errors.append(f"{path}: cannot stat ({exc})")
            continue
        if size == 0:
            errors.append(f"{path}: zero bytes — nothing to review")
            continue
        
        # Stage to the host the transport actually reads from.
        #
        # HERMES_DRAFT_SKIP_CYPRESS=1 renders with local paths only and no
        # cypress_path — send_draft then treats the record exactly like an
        # old-format one. This is a TEST affordance and is opt-IN by env:
        # a network failure can never quietly take this branch. Making it
        # automatic-on-error would let a draft render while its attachments
        # sat nowhere the transport could reach — the exact failure this
        # staging exists to prevent. On any real publish failure we still
        # refuse the whole draft.
        entry = {
            "path": str(path),           # Review path (local to gateway's client)
            "name": path.name,
            "bytes": size,
        }
        if os.environ.get("HERMES_DRAFT_SKIP_CYPRESS") != "1":
            cypress_path, sha256 = _publish_attachment_to_cypress(str(path))
            if cypress_path is None:
                errors.append(
                    f"{path}: failed to publish to cypress "
                    "(attachment will not be sendable)"
                )
                continue
            entry["cypress_path"] = cypress_path  # where hermes-send reads it
            entry["sha256"] = sha256              # verification hash

        resolved.append(entry)
    return resolved, errors


def _session_source() -> str:
    """The current session's SOURCE, read the way production binds it.

    Same rule as ``_structural_approval_surface``: read
    ``HERMES_SESSION_SOURCE`` through ``get_session_env``, never the process
    env directly and never ``HERMES_DESKTOP`` — one serve process answers many
    sessions, and the platform var is empty on desktop/CLI/TUI
    (gateway/session_context.py:419-424).
    """
    try:
        from gateway.session_context import get_session_env

        return (get_session_env("HERMES_SESSION_SOURCE", "") or "").strip().lower()
    except Exception:
        return (os.environ.get("HERMES_SESSION_SOURCE", "") or "").strip().lower()


def _render(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str],
    resolved: List[Dict[str, Any]],
    from_addr: str = "",
) -> str:
    """Render the reviewable draft, attachment lines included.

    The model does not write these lines and cannot omit one: they are
    generated from the same list that will be sent.
    """
    lines = []
    if from_addr:
        lines.append(f"**From:** {from_addr}")
    lines.append(f"**To:** {to}")
    if cc:
        lines.append(f"**Cc:** {cc}")
    lines.append(f"**Subject:** {subject}")
    for i, att in enumerate(resolved, 1):
        lines.append(
            f"**Attachment {i}/{len(resolved)}:** `{att['path']}` "
            f"· {att['bytes']:,} bytes"
        )
    lines += ["", "---", body.strip(), "---"]
    # The MEDIA: lines are what make the files openable in chat. They are the
    # payload of this whole tool; everything above is context for them.
    lines.append("")
    for att in resolved:
        lines.append(f"MEDIA:{att['path']}")
    return "\n".join(lines)


def present_draft(
    to: str = "",
    subject: str = "",
    body: str = "",
    cc: str = "",
    attachments: Optional[List[str]] = None,
    session_id: str = "",
    task_id: Optional[str] = None,
    **kwargs: Any,
) -> str:
    attachments = attachments or []
    # ``from`` is a Python keyword, so it can only arrive via **kwargs.
    from_addr = str(kwargs.get("from", "") or "").strip()

    missing = [n for n, v in (("to", to), ("subject", subject), ("body", body)) if not v]
    if missing:
        return json.dumps(
            {"success": False, "error": f"missing required field(s): {', '.join(missing)}"},
            ensure_ascii=False,
        )

    # The sending identity is REQUIRED on desktop sessions and is never
    # inferred. A default-account setting, an env var, or "the account we used
    # last time" are all guesses, and the reviewer cannot check a guess they
    # were never shown: the From address decides which thread the reply lands
    # in and whether the recipient recognises the sender. Sending from the
    # wrong identity cannot be undone, so the tool refuses rather than picks.
    #
    # Scoped to desktop because that is the surface with the approval card —
    # the card is where the address gets reviewed. Messaging surfaces keep the
    # prior contract unchanged.
    if _session_source() == "desktop" and not from_addr:
        return json.dumps(
            {
                "success": False,
                "error": (
                    "missing required field: from. The sending identity is "
                    "never inferred — pass the full From address explicitly."
                ),
                "hint": (
                    "Pass from='<full address>'. Verify it from a real Sent "
                    "header rather than from an account nickname; if the "
                    "correct mailbox is ambiguous, ask before drafting."
                ),
            },
            ensure_ascii=False,
        )

    resolved, errors = _resolve_attachments(attachments)
    if errors:
        return json.dumps(
            {
                "success": False,
                "error": "attachment paths did not resolve; no draft was rendered",
                "failures": errors,
                "hint": (
                    "Publish the files to the review folder first, then pass "
                    "their absolute paths. A draft is never rendered with an "
                    "attachment the reviewer could not open."
                ),
            },
            ensure_ascii=False,
        )

    # The 2026-09-08 defect, caught structurally: the body talks about a
    # document but no file is attached, so the reviewer gets a name and no
    # openable object.
    if not resolved:
        mentioned = filenames_mentioned_in_text(body)
        if mentioned:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "the body references document(s) "
                        f"({', '.join(mentioned[:5])}) but attachments is empty. "
                        "Naming a file is not showing it."
                    ),
                    "hint": "Pass the absolute path of each referenced file in attachments.",
                },
                ensure_ascii=False,
            )

    draft_id = f"draft_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    record = {
        "draft_id": draft_id,
        "created_at": time.time(),
        # Provenance. The structural-approval path refuses to raise a card for
        # a draft this session did not render: a draft file is visible to every
        # process sharing HERMES_HOME (the Telegram gateway can read a draft
        # this serve wrote), so the file's existence is not evidence that the
        # reviewer on THIS surface ever saw it.
        "session_id": session_id or "",
        "from": from_addr,
        "to": to,
        "cc": cc,
        "subject": subject,
        "body": body,
        "attachments": resolved,
    }
    try:
        _DRAFT_DIR.mkdir(parents=True, exist_ok=True)
        _draft_path(draft_id).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        return json.dumps(
            {"success": False, "error": f"could not persist draft: {exc}"},
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "success": True,
            "draft_id": draft_id,
            "attachment_count": len(resolved),
            "rendered": _render(to, subject, body, cc, resolved, from_addr),
            "note": (
                "Post 'rendered' VERBATIM as your reply — the MEDIA: lines are "
                "what make the files openable. Do not add, remove or reword an "
                f"attachment line. Send only with draft_id={draft_id} after "
                "explicit approval."
            ),
            "draft_file": str(_draft_path(draft_id)).replace(
                str(get_hermes_home()), display_hermes_home()
            ),
        },
        ensure_ascii=False,
    )


def load_draft(draft_id: str) -> Optional[Dict[str, Any]]:
    """Load a persisted draft. ``send_email`` uses this to refuse free-form sends."""
    if not draft_id or not re.fullmatch(r"draft_[\w]+", draft_id):
        return None
    path = _draft_path(draft_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- Outbound lint --------------------------------------------------------


def lint_outbound_reply(text: str) -> Optional[str]:
    """Return an error string if a reply names documents but attaches none.

    The backstop for the case where the model bypasses ``present_draft``
    entirely and hand-writes a draft. Checks the *rendered reply*, which is
    the only place the reviewer's actual experience is visible.

    Returns ``None`` when the reply is fine. Kept as a pure function so the
    rule is testable without a gateway, and so both the gateway and the CLI
    can share one definition of the failure.
    """
    if not text or not text.strip():
        return None

    # Only lint replies that are presenting something for review. A reply that
    # merely discusses a filename ("I read 4063_minutes.pdf") is not a draft
    # and must not be blocked — the failure class is specifically a DRAFT that
    # cannot be checked.
    lowered = text.lower()
    draft_markers = ("**to:**", "**subject:**", "\nto:", "\nsubject:")
    if not any(m in lowered for m in draft_markers):
        return None

    mentioned = filenames_mentioned_in_text(text)
    if not mentioned:
        return None

    # MEDIA: lines are the openable objects. Count them, not the names.
    media_count = len(re.findall(r"^\s*MEDIA:\S+", text, re.MULTILINE))
    if media_count >= 1:
        return None

    return (
        f"This reply presents a draft that names {len(mentioned)} document(s) "
        f"({', '.join(mentioned[:5])}) with zero attached files. Naming a file "
        "is not showing it — the reviewer cannot open any of them. Rebuild the "
        "draft with present_draft(to, subject, body, attachments=[...]) and "
        "post its 'rendered' output verbatim."
    )


# --- Sending (token-gated) ------------------------------------------------


def _get_minted_draft_token() -> str:
    """Retrieve the draft-approval token from the current turn's context, if minted.
    
    The gateway mints tokens for /approve <draft-id> messages in the user-turn path
    and stores them in a contextvar (not in the prompt) to preserve cache stability.
    This helper lets send_draft auto-populate the token without user involvement.
    """
    return get_draft_approval_token()


def cleanup_cypress_attachments(draft_id: str) -> dict:
    """Remove a draft's staged attachment copies from cypress.

    Call this ONLY after the transport has confirmed a successful send (a 250
    from the mail helper). Staged files are client invoices; they should not
    outlive the send that needed them.

    Deliberately NOT called from send_draft(): send_draft verifies approval and
    returns ``approved: True``, it does not itself put mail on the wire. The
    actual send is the hermes-send/helper call that happens after it. Deleting
    at approval time would remove the files out from under the transport and
    guarantee a failed send — so the caller that owns the 250 owns the cleanup.

    Returns a per-file report. Never raises: a failed cleanup must not be
    mistaken for a failed send, and leftover staging is a hygiene problem, not
    a correctness one.
    """
    import subprocess

    draft = load_draft(draft_id)
    if draft is None:
        return {"draft_id": draft_id, "error": "no such draft", "removed": []}

    removed: list[str] = []
    failed: list[dict] = []
    for a in draft.get("attachments", []):
        cypress_path = a.get("cypress_path")
        if not cypress_path:
            continue  # old-format record, nothing was staged
        try:
            result = subprocess.run(
                ["ssh", "cypress", "rm", "-f", "--", cypress_path],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                removed.append(cypress_path)
            else:
                failed.append(
                    {"path": cypress_path, "error": result.stderr.strip()}
                )
        except Exception as e:  # noqa: BLE001 - cleanup must never raise
            failed.append({"path": cypress_path, "error": str(e)})

    return {"draft_id": draft_id, "removed": removed, "failed": failed}


def _draft_card_description(draft: Dict[str, Any]) -> str:
    """Build the approval card's text from the draft RECORD alone.

    Every character comes from persisted fields or from os.stat/hashlib; no
    model-supplied string reaches the card except the recipient/subject values
    that ARE the thing being approved (and which the reviewer already read in
    the rendered draft). The body is deliberately NOT included: it is long,
    model-authored prose, and the card must describe the ENVELOPE — who
    receives what — which is the part a misdirected send gets wrong.

    Attachment bytes and SHA-256 are recomputed from disk here rather than read
    back from the record. A stale hash would describe the file as it was at
    render time, and the whole point of the card is to describe what is about
    to leave the machine.
    """
    import hashlib

    lines = [
        f"draft_id: {draft.get('draft_id', '')}",
    ]
    # The sending identity leads the card: it is the field a misdirected send
    # gets wrong in the way that cannot be retracted.
    if draft.get("from"):
        lines.append(f"From: {draft.get('from')}")
    lines.append(f"To: {draft.get('to', '')}")
    if draft.get("cc"):
        lines.append(f"Cc: {draft.get('cc')}")
    lines.append(f"Subject: {draft.get('subject', '')}")

    attachments = draft.get("attachments") or []
    if not attachments:
        lines.append("Attachments: none")
    for i, att in enumerate(attachments, 1):
        path = Path(str(att.get("path", "")))
        name = path.name or "?"
        try:
            content = path.read_bytes()
            size = len(content)
            digest = hashlib.sha256(content).hexdigest()
        except OSError:
            # Unreadable now: say so on the card. The send-time re-verification
            # below will refuse anyway, but the reviewer should not be asked to
            # approve a file the machine cannot read.
            lines.append(f"Attachment {i}/{len(attachments)}: {name} — UNREADABLE")
            continue
        lines.append(
            f"Attachment {i}/{len(attachments)}: {name} — {size:,} bytes — "
            f"sha256:{digest}"
        )
    return "\n".join(lines)


def _structural_approval_surface() -> tuple[str, Any]:
    """Return ``(session_key, notify_cb)`` when this surface can raise a card.

    Scoped to desktop sessions. Messaging platforms keep the ``/approve
    <draft_id>`` contract unchanged: they already have a working mint path
    (gateway/run.py:8181), the token they mint is bound to the same process
    that will spend it, and a card there would be a second, redundant consent
    channel on a surface that does not need one.

    Returns ``("", None)`` for every other surface, which leaves send_draft on
    the token path and therefore fails closed.
    """
    try:
        from tools.approval import (
            _gateway_notify_cbs,
            _lock,
            get_current_session_key,
        )
    except Exception:  # pragma: no cover - approval module always importable
        logger.debug("approval module unavailable for structural path", exc_info=True)
        return "", None

    # Read the SOURCE, not the platform. gateway/session_context.py:419-424:
    # the gateway binds a platform value ("telegram") to
    # HERMES_SESSION_PLATFORM, while the CLI, TUI and desktop bind
    # HERMES_SESSION_SOURCE ("cli"/"tui"/"desktop") and leave the platform
    # EMPTY. A desktop check against HERMES_SESSION_PLATFORM therefore never
    # fires in production, however green its unit tests are.
    #
    # Not HERMES_DESKTOP either: that marks a backend SPAWNED by the app,
    # which is not the claim we need. The same serve process also answers
    # `hermes --tui` in the embedded terminal pane, and a tui session must not
    # inherit the desktop's card.
    try:
        from gateway.session_context import get_session_env

        source = (get_session_env("HERMES_SESSION_SOURCE", "") or "").lower()
        platform = (get_session_env("HERMES_SESSION_PLATFORM", "") or "").lower()
    except Exception:
        source = (os.environ.get("HERMES_SESSION_SOURCE", "") or "").lower()
        platform = (os.environ.get("HERMES_SESSION_PLATFORM", "") or "").lower()

    if source.strip() != "desktop":
        return "", None
    # Defensive: if a messaging platform is somehow bound alongside a desktop
    # source, that session has a chat channel behind it and keeps /approve.
    if platform.strip() not in ("", "desktop", "local"):
        return "", None

    session_key = get_current_session_key("") or ""
    if not session_key:
        return "", None
    with _lock:
        notify_cb = _gateway_notify_cbs.get(session_key)
    return session_key, notify_cb


def send_draft(
    draft_id: str = "",
    approval_token: str = "",
    session_id: str = "",
    task_id: Optional[str] = None,
) -> str:
    """Send a previously-presented draft. Requires a user-minted token.

    A draft id alone is NOT authority to send. The id proves *which* object
    would go out; the token proves a human said to send it. Keeping them
    separate is the whole point: on 2026-09-08 the agent offered to wait for
    approval, received none across eight user turns, and acted as though the
    wait had happened. An id it minted itself would have let that recur.

    The token is spent whatever the send's outcome — one approval is one
    send attempt, never a standing permission.
    """
    draft = load_draft(draft_id)
    if draft is None:
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"no draft {draft_id!r}. Present it with present_draft "
                    "first; drafts are not sendable until rendered."
                ),
            }
        )

    # Fall back to the token the gateway minted for this turn.
    #
    # This lookup lives HERE, not in the tool-registry handler, because
    # send_draft is the actual gate. When the fallback sat in the handler
    # lambda, any caller reaching send_draft directly — the test suite, the
    # CLI, a future adapter — bypassed it and saw an empty token, so the
    # mechanism the design depends on was never exercised end to end.
    # The gate and the thing that feeds the gate belong in one place.
    #
    # This is not the model approving itself: the registry only holds tokens
    # the gateway minted after parsing a real user /approve message, and
    # consume() still enforces kind, subject, session and expiry.
    #
    # Resolution order:
    #   1. An explicitly passed token (tests, CLI, future adapters).
    #   2. The contextvar, if we happen to still be inside the minting context.
    #   3. The registry, keyed by (kind, subject, session).
    #
    # (3) is what makes this work in production. The contextvar is set during
    # gateway dispatch and dies when that context ends, long before the agent
    # calls this tool, so it is effectively always empty here. The registry
    # survives because it is a process-wide store, and the token is still
    # bound to this exact session and this exact draft.
    effective_session = session_id or draft.get("session_id", "")
    effective_token = approval_token or _get_minted_draft_token()
    if not effective_token:
        effective_token = get_registry().find_unspent(
            kind=KIND_DRAFT,
            subject_id=draft_id,
            session_id=effective_session,
        ) or ""

    # ── Structural approval (desktop only) ────────────────────────────────
    # The desktop has no /approve: the renderer refuses the command
    # client-side (desktop-slash-commands.ts:303) and tui_gateway never
    # imports agent.approval_tokens, so no KIND_DRAFT token can exist on this
    # surface. Use the consent channel the surface DOES have — the same queue
    # and card the dangerous-command guard uses (tools/approval.py:3838).
    #
    # Raised from send_draft, never present_draft: this BLOCKS the agent
    # thread, so the draft must already be rendered and read before the card
    # can appear. Asking in present_draft would freeze the turn before the
    # reviewer had anything to review.
    #
    # Consent here is a RETURN VALUE on the thread that will do the send, not
    # a token left lying in a registry — so the cross-process gap that makes
    # a Telegram /approve unable to authorise a serve-side draft cannot arise.
    # The mint+consume pair below is a local formality that keeps consume()
    # the single enforcement point rather than growing a second bypass branch.
    if not effective_token:
        session_key, notify_cb = _structural_approval_surface()
        if notify_cb is not None:
            # Provenance: refuse to raise a card for a draft this session did
            # not render. Without this, any draft file readable under
            # HERMES_HOME — including one written by another process, or by a
            # session the current reviewer never saw — could be escalated into
            # a card here.
            drafted_in = str(draft.get("session_id") or "")
            if not drafted_in or drafted_in != effective_session:
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            "not sent — this draft was not presented in this "
                            "session. Re-present it with present_draft here, "
                            "then approve the draft this session rendered."
                        ),
                        "draft_id": draft_id,
                        "note": (
                            "Do not retry. A draft file being readable is not "
                            "evidence that the reviewer saw it."
                        ),
                    }
                )

            from tools.approval import _await_gateway_decision

            decision = _await_gateway_decision(
                session_key,
                notify_cb,
                {
                    "command": f"send_email {draft_id}",
                    "description": _draft_card_description(draft),
                    "pattern_key": f"send_draft:{draft_id}",
                    "pattern_keys": [f"send_draft:{draft_id}"],
                    # Consent is PER DRAFT and single-use. A persisted pattern
                    # ("always allow send_email") would be a standing
                    # permission to mail anyone, which is precisely what the
                    # token design refuses to offer. The card computes its
                    # button set from these two flags, so False/False renders
                    # Run/Reject only — no "always allow", no "this session".
                    "allow_permanent": False,
                    "allow_session": False,
                },
                surface="draft",
            )
            if decision.get("choice") != "once":
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            "not sent — approval was declined, timed out, or "
                            "the prompt could not be delivered"
                        ),
                        "draft_id": draft_id,
                        "note": (
                            "Do not re-raise the prompt. Ask the user and stop."
                        ),
                    }
                )
            effective_token = get_registry().mint(
                kind=KIND_DRAFT,
                subject_id=draft_id,
                session_id=effective_session,
            ).token

    ok, reason = get_registry().consume(
        token=effective_token,
        kind=KIND_DRAFT,
        subject_id=draft_id,
        session_id=effective_session,
    )
    if not ok:
        return json.dumps(
            {
                "success": False,
                "error": f"not sent — {reason}",
                "draft_id": draft_id,
                "needs": f"/approve {draft_id}",
                "note": (
                    "Do not retry with a different token or re-present the "
                    "draft to obtain one. Ask the user and stop."
                ),
            }
        )

    # Re-verify the attachments at send time. The draft may have been rendered
    # minutes ago and files may have moved or been deleted on either the local
    # machine or cypress.  The reviewer approved openable files, so sending
    # unopenable ones would break the contract.
    #
    # NEW (2026-09-14): Verify BOTH the local review path and the cypress
    # transport path. The local path might exist but cypress's copy may have
    # been deleted. We check both to ensure hermes-send will succeed.
    missing_local = []
    missing_cypress = []
    
    for a in draft.get("attachments", []):
        # Check local (review) path
        if not Path(a["path"]).is_file():
            missing_local.append(a["path"])
        
        # Check cypress (transport) path if it exists in the record
        cypress_path = a.get("cypress_path")
        if cypress_path:
            try:
                result = subprocess.run(
                    ["ssh", "cypress", f"test -f {cypress_path}"],
                    check=False,
                    capture_output=True,
                    # Generous: a TIMEOUT here would be read as "attachment
                    # missing" and burn an already-consumed approval on a
                    # send that never happened. Slow != gone.
                    timeout=int(os.environ.get("HERMES_DRAFT_SSH_TIMEOUT", "30")),
                )
                if result.returncode != 0:
                    missing_cypress.append((a["path"], cypress_path))
            except Exception as e:
                logger.warning(
                    "Failed to check cypress path %s: %s",
                    cypress_path, e
                )
                missing_cypress.append((a["path"], cypress_path))
    
    if missing_local or missing_cypress:
        error_parts = []
        if missing_local:
            error_parts.append(
                f"{len(missing_local)} local file(s) no longer exist: "
                f"{', '.join(missing_local)}"
            )
        if missing_cypress:
            cypress_list = [f"{local} -> {cypress}" for local, cypress in missing_cypress]
            error_parts.append(
                f"{len(missing_cypress)} cypress copy/copies no longer exist: "
                f"{'; '.join(cypress_list)}"
            )
        
        return json.dumps(
            {
                "success": False,
                "error": (
                    "approval consumed but NOT sent: "
                    + " and ".join(error_parts)
                    + ". Re-present the draft and ask for approval again."
                ),
                "draft_id": draft_id,
            }
        )

    return json.dumps(
        {
            "success": True,
            "draft_id": draft_id,
            "to": draft.get("to"),
            "cc": draft.get("cc"),
            "subject": draft.get("subject"),
            "attachment_count": len(draft.get("attachments", [])),
            "approved": True,
            "note": (
                "Approval verified and spent. Hand this draft to the configured "
                "send transport, then verify against the Sent folder — the "
                "transport's success line is a claim, the Sent message is the "
                "fact."
            ),
        }
    )


registry.register(
    name="send_email",
    toolset="email_draft",
    schema={
        "name": "send_email",
        "description": (
            "Send a draft previously rendered by present_draft. Requires BOTH "
            "the draft_id AND an approval_token that the user minted by "
            "replying '/approve <draft_id>'. There is no other way to send: "
            "your own reading of the conversation, however clear, is not an "
            "approval. If you have no token, ask for one and stop."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "draft_id": {
                    "type": "string",
                    "description": "The id returned by present_draft.",
                },
                "approval_token": {
                    "type": "string",
                    "description": (
                        "The token issued when the user replied "
                        "'/approve <draft_id>'. Never invent or guess this."
                    ),
                },
            },
            "required": ["draft_id", "approval_token"],
        },
    },
    handler=lambda args, **kw: send_draft(
        draft_id=args.get("draft_id", ""),
        # No contextvar fallback here — send_draft does it. One gate, one
        # place. Duplicating it meant direct callers silently skipped it.
        approval_token=args.get("approval_token", ""),
        session_id=kw.get("session_id", "") or "",
        task_id=kw.get("task_id"),
    ),
)


registry.register(
    name="present_draft",
    toolset="email_draft",
    schema={
        "name": "present_draft",
        "description": (
            "Render an email draft for human review with its attachments made "
            "openable. REQUIRED for any draft that has attachments — it writes "
            "the attachment lines itself from the files you pass, so the draft "
            "the reviewer sees always matches what would be sent. Returns a "
            "draft_id; post the returned 'rendered' text verbatim as your reply. "
            "Fails if any path does not resolve, and fails if the body "
            "references a document while attachments is empty."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from": {
                    "type": "string",
                    "description": (
                        "Full sending address, e.g. 'you@example.com'. REQUIRED "
                        "on desktop sessions and never inferred: state the "
                        "identity explicitly, and if the correct mailbox is "
                        "ambiguous ask the user instead of guessing."
                    ),
                },
                "to": {"type": "string", "description": "Full recipient address."},
                "subject": {"type": "string"},
                "body": {"type": "string", "description": "Body text only, no headers."},
                "cc": {
                    "type": "string",
                    "description": "Comma-joined full addresses. Optional.",
                },
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Absolute path of every file the email will send. Each "
                        "becomes a rendered, openable attachment line."
                    ),
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
    handler=lambda args, **kw: present_draft(
        to=args.get("to", ""),
        subject=args.get("subject", ""),
        body=args.get("body", ""),
        cc=args.get("cc", ""),
        attachments=args.get("attachments") or [],
        # Stamped into the record so send_draft's structural path can refuse a
        # draft this session did not render. Not a model-supplied field.
        session_id=kw.get("session_id", "") or "",
        task_id=kw.get("task_id"),
        **{"from": args.get("from", "")},
    ),
)
