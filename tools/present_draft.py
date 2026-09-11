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
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import display_hermes_home, get_hermes_home
from tools.registry import registry

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


def _resolve_attachments(
    attachments: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Resolve every path to a real file, or report why it failed.

    An unresolvable path is a hard failure, not a warning. A draft that lists a
    file the reviewer cannot open is the exact defect this tool exists to
    prevent, so there is no partial-success mode: either every attachment
    resolves or nothing is rendered.
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
        resolved.append({"path": str(path), "name": path.name, "bytes": size})
    return resolved, errors


def _render(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str],
    resolved: List[Dict[str, Any]],
) -> str:
    """Render the reviewable draft, attachment lines included.

    The model does not write these lines and cannot omit one: they are
    generated from the same list that will be sent.
    """
    lines = [f"**To:** {to}"]
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
    task_id: Optional[str] = None,
) -> str:
    attachments = attachments or []

    missing = [n for n, v in (("to", to), ("subject", subject), ("body", body)) if not v]
    if missing:
        return json.dumps(
            {"success": False, "error": f"missing required field(s): {', '.join(missing)}"},
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
            "rendered": _render(to, subject, body, cc, resolved),
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
        task_id=kw.get("task_id"),
    ),
)
