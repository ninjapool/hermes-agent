"""Approval tokens: consent that comes from a user turn, not from a reading.

Why this exists
---------------
Two incidents, one mechanism. On 2026-09-08 the agent offered *"say the word
and I'll prune"*, received no answer across eight user turns, and then removed
three memory entries anyway. Earlier the same day it presented a draft and
treated its own reading of the conversation as authority to proceed. In both
cases the model's belief that consent had been given WAS the consent — there
was nothing else to check against.

So consent is made a thing the model cannot manufacture. A token is minted only
by the harness, only from a user turn carrying an explicit ``/approve <id>``,
and is:

* **single-use** — an approval authorises one action, not a policy;
* **bound to one subject id** — a token for a draft cannot send a different
  draft, and a token naming one memory entry cannot remove another;
* **session-scoped** — tokens do not leak across sessions;
* **expiring** — stale consent is not consent (a draft approved yesterday
  should not send itself today).

The model can read tokens but never create them: ``mint`` is called from the
user-turn path in the gateway/CLI, never from a tool handler.
"""

from __future__ import annotations

import contextvars
import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# How long an approval stays valid. Long enough to survive a slow tool chain in
# the same sitting; short enough that yesterday's "yes" cannot act today.
DEFAULT_TTL_SECONDS = 30 * 60

# ``/approve <subject-id>`` — the only phrase that mints. Deliberately not a
# natural-language match: "yes please send it" must NOT mint a token, because
# interpreting prose is the failure being fixed.
_APPROVE_RE = re.compile(
    r"^\s*/approve\s+([A-Za-z0-9_\-.]{3,128})\s*$", re.IGNORECASE
)

# Words that are MODIFIERS of the dangerous-command ``/approve``, never subject
# ids. ``/approve all`` means "approve the blocked shell command for every
# future call in this session" — it belongs to the dangerous-command handler
# and must never be read as "approve the thing named 'all'".
#
# Without this guard the only thing standing between `/approve all` and a
# minted draft token is ``load_draft("all")`` happening to return None. A file
# named ``all.json`` in the drafts dir would turn a dangerous-command modifier
# into a send authorisation. Consent must not depend on a filename not
# existing, so the exclusion lives here, at the single minting door, rather
# than in each caller.
_APPROVE_MODIFIERS = frozenset({"all", "always", "session", "once", "yes", "y"})

# Memory-removal approvals arrive differently from draft approvals. A draft is
# approved one at a time, on a line of its own. A memory prune is a LIST: the
# tool refuses N removals at once and issues N ``/approve mem_…`` lines, which
# the user pastes back as a block — often with the entry text still trailing
# each line ("/approve mem_ab12  [16] Client docs…"), because that is how the
# lines were shown to them.
#
# So this pattern is deliberately looser than _APPROVE_RE in two ways, and
# strict in the way that matters:
#   * it SCANS (finditer) rather than matching the whole message, so many
#     approvals in one message all mint, and trailing text is ignored;
#   * it anchors each hit to a line start, so a ``mem_`` id quoted mid-sentence
#     ("the tool said /approve mem_x") inside prose cannot mint;
#   * the subject must be a literal ``mem_<hex>`` id — the exact shape
#     ``memory_removal_subject`` produces. Nothing else mints a memory token,
#     so ``/approve all`` and friends are structurally excluded without needing
#     the modifier list.
_APPROVE_MEMORY_RE = re.compile(
    r"^[ \t]*/approve[ \t]+(mem_[0-9a-f]{6,64})\b",
    re.IGNORECASE | re.MULTILINE,
)


def _digest_for_compare(value: str) -> bytes:
    """Fixed-width digest of a string, for constant-time comparison.

    ``hmac.compare_digest`` raises TypeError on str inputs containing
    non-ASCII characters. Session keys are derived from chat/thread identity
    and can carry non-ASCII (a Telegram chat titled 「郡山」, an emoji in a
    display name), so comparing the raw strings turns a routine mismatch into
    an exception inside the consent check.

    Hashing first makes the comparison total: any two strings become equal-
    length digests, UTF-8 encoded, so the gate always returns a decision
    instead of raising. SHA-256 is not used here for secrecy — the inputs are
    not secrets — only to give compare_digest a uniform, ASCII-safe input
    while preserving its constant-time property.
    """
    return hashlib.sha256(value.encode("utf-8")).digest()

# Subjects a token can authorise. Kept explicit so a new gated action has to be
# named here rather than silently reusing another action's approvals.
KIND_DRAFT = "draft"
KIND_MEMORY_REMOVE = "memory_remove"
_KINDS = frozenset({KIND_DRAFT, KIND_MEMORY_REMOVE})


@dataclass(frozen=True)
class ApprovalToken:
    token: str
    kind: str
    subject_id: str
    session_id: str
    issued_at: float
    expires_at: float


class ApprovalRegistry:
    """In-process store of unspent approval tokens.

    Deliberately NOT persisted to disk. An approval that survives a restart is
    an approval nobody is watching — and the whole point is that a human was
    present when it was minted.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self._ttl = max(1, int(ttl_seconds))
        self._tokens: Dict[str, ApprovalToken] = {}
        self._lock = threading.Lock()

    # --- minting (harness only) ------------------------------------------

    def mint(self, kind: str, subject_id: str, session_id: str) -> ApprovalToken:
        """Mint a token. Callable ONLY from a user-turn handler.

        No tool handler may reach this. If a future caller needs a token, that
        caller is by definition asking to approve its own action, which is the
        thing this module exists to prevent.
        """
        if kind not in _KINDS:
            raise ValueError(f"unknown approval kind: {kind!r}")
        if not subject_id or not session_id:
            raise ValueError("approval requires both a subject id and a session id")
        now = time.time()
        token = ApprovalToken(
            token=f"apv_{secrets.token_urlsafe(24)}",
            kind=kind,
            subject_id=subject_id,
            session_id=session_id,
            issued_at=now,
            expires_at=now + self._ttl,
        )
        with self._lock:
            self._tokens[token.token] = token
        return token

    def mint_from_user_text(
        self, text: str, kind: str, session_id: str
    ) -> Optional[ApprovalToken]:
        """Mint from a user turn if it carries an exact ``/approve <id>``.

        Returns ``None`` for anything else — including prose that clearly means
        yes. That is intentional: an approval has to be an unambiguous act, not
        an interpretation.
        """
        if not text:
            return None
        match = _APPROVE_RE.match(text)
        if not match:
            return None
        subject = match.group(1)
        # Modifier keywords belong to the dangerous-command handler. Compared
        # casefolded because _APPROVE_RE is IGNORECASE, so "/approve ALL" must
        # be excluded too.
        if subject.casefold() in _APPROVE_MODIFIERS:
            return None
        return self.mint(kind=kind, subject_id=subject, session_id=session_id)

    def mint_memory_approvals_from_user_text(
        self, text: str, session_id: str
    ) -> list["ApprovalToken"]:
        """Mint one KIND_MEMORY_REMOVE token per ``/approve mem_…`` line.

        Returns ``[]`` when the text carries none, so the caller falls through
        to the dangerous-command handler unchanged.

        Duplicate ids in one paste mint once: a second token for the same
        subject would let a single approval authorise two removals if the tool
        were called twice.
        """
        if not text:
            return []
        minted: list[ApprovalToken] = []
        seen: set[str] = set()
        for match in _APPROVE_MEMORY_RE.finditer(text):
            subject = match.group(1)
            key = subject.casefold()
            if key in seen:
                continue
            seen.add(key)
            minted.append(
                self.mint(
                    kind=KIND_MEMORY_REMOVE,
                    subject_id=subject,
                    session_id=session_id,
                )
            )
        return minted

    # --- spending (tool side) --------------------------------------------

    def consume(
        self, token: str, kind: str, subject_id: str, session_id: str
    ) -> Tuple[bool, str]:
        """Spend a token for exactly one action.

        Returns ``(ok, reason)``. The token is removed on ANY outcome that
        proves it was presented for the wrong action, so a mismatched token
        cannot be re-tried against a different subject until it happens to fit.
        """
        if not token:
            return False, (
                "no approval token. This action needs an explicit approval from "
                f"the user: ask them to reply '/approve {subject_id}'. Your "
                "reading of the conversation is not an approval."
            )
        with self._lock:
            record = self._tokens.pop(token, None)
        if record is None:
            return False, "approval token is unknown or already spent"
        if record.kind != kind:
            return False, (
                f"approval token was issued for {record.kind!r}, not {kind!r}"
            )
        if not hmac.compare_digest(
            _digest_for_compare(record.subject_id), _digest_for_compare(subject_id)
        ):
            return False, (
                f"approval token authorises {record.subject_id!r}, "
                f"not {subject_id!r}"
            )
        if not hmac.compare_digest(
            _digest_for_compare(record.session_id), _digest_for_compare(session_id)
        ):
            return False, "approval token belongs to a different session"
        if time.time() > record.expires_at:
            return False, (
                "approval token has expired; ask the user to approve again"
            )
        return True, "ok"

    def find_unspent(
        self, kind: str, subject_id: str, session_id: str
    ) -> Optional[str]:
        """Return the token string for an unspent, unexpired match, else None.

        This is the lookup that lets a tool find the approval a human minted
        earlier in the SAME session for the SAME subject, without the caller
        having to carry the token string across a process boundary.

        Why this exists: the token was previously handed to the tool through a
        contextvar set during gateway dispatch. A contextvar dies with the
        context that set it, so the moment dispatch returned, the approval was
        gone. The registry already holds the binding (kind + subject + session),
        so ask it directly.

        This is NOT a weakening of the gate. It still requires that a real
        ``/approve <subject>`` was typed by a human, in this session, for this
        exact subject, within the TTL. It cannot conjure a token that was never
        minted, and it does not spend anything — ``consume()`` still does the
        one-shot spend and all four checks.
        """
        if not kind or not subject_id or not session_id:
            return None
        now = time.time()
        subject_d = _digest_for_compare(subject_id)
        session_d = _digest_for_compare(session_id)
        with self._lock:
            for tok, rec in self._tokens.items():
                if rec.kind != kind:
                    continue
                if not hmac.compare_digest(
                    _digest_for_compare(rec.subject_id), subject_d
                ):
                    continue
                if not hmac.compare_digest(
                    _digest_for_compare(rec.session_id), session_d
                ):
                    continue
                if now > rec.expires_at:
                    continue
                return tok
        return None

    def peek(self, token: str) -> Optional[ApprovalToken]:
        """Non-consuming read, for display/debug only."""
        with self._lock:
            return self._tokens.get(token)

    def purge_expired(self) -> int:
        now = time.time()
        with self._lock:
            dead = [t for t, rec in self._tokens.items() if rec.expires_at < now]
            for t in dead:
                del self._tokens[t]
        return len(dead)


# Process-wide registry. One per process is correct: tokens are session-bound
# internally, so sessions cannot borrow each other's approvals.
_REGISTRY = ApprovalRegistry()

# Per-turn context for a minted draft-approval token from a user's /approve message.
# The gateway mints this in the user-turn path (before agent execution), and the
# send_draft tool reads it to auto-populate the approval_token parameter. This
# keeps the token out of the prompt (preserving cache stability) and avoids a
# synthetic user message that would break role alternation.
#
# Scoped to the turn's context so concurrent sessions don't leak tokens.
_draft_approval_token: contextvars.ContextVar[str] = contextvars.ContextVar(
    "draft_approval_token",
    default="",
)


def get_registry() -> ApprovalRegistry:
    return _REGISTRY


def get_draft_approval_token() -> str:
    """Read the minted draft-approval token for the current turn's context, if set."""
    return _draft_approval_token.get()


def set_draft_approval_token(token: str) -> None:
    """Store a draft-approval token in the current turn's context.
    
    Called by the gateway when /approve <draft-id> is detected and a token
    is successfully minted. The agent's send_draft tool handler reads this
    to auto-populate the approval_token parameter without user involvement.
    """
    _draft_approval_token.set(token)


def memory_removal_subject(entry_text: str) -> str:
    """Stable subject id for a memory entry the user is asked to approve.

    Hashed rather than the raw text: the id travels through ``/approve`` in a
    chat message, and memory entries are long, multi-line and full of Japanese.
    Truncated to keep it typable.
    """
    digest = hashlib.sha256(entry_text.strip().encode("utf-8")).hexdigest()
    return f"mem_{digest[:12]}"
