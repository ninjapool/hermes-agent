"""Contract tests for the draft record hash and the address-book flag.

Two hardenings, both aimed at the same gap: the reviewer approves an
*envelope* — who receives what — and until now nothing tied the envelope they
read to the envelope that leaves the machine.

1. **Record hash.** ``present_draft`` digests the canonical record (from, to,
   cc, subject, body, and each attachment's SHA-256) and stores it. The
   rendered draft and the approval card both carry its first 8 characters.
   ``send_draft`` recomputes the digest from the file on disk and refuses if it
   moved. A draft file is world-readable to every process sharing HERMES_HOME;
   the hash is what makes tampering between render and send detectable.

2. **Address-book flag.** Any To/Cc address that is not in the verified address
   book is marked ``[NEW ADDRESS]`` on the render and on the card, so a
   plausible-looking typo or an unfamiliar recipient is visible rather than
   buried in a list of five.
"""

from __future__ import annotations

import json

import pytest

from agent.approval_tokens import ApprovalRegistry
import agent.approval_tokens as approval_tokens
import tools.present_draft as pd


DESKTOP_SESSION = "20260920_120000_desktopbb"
SESSION_KEY = "desktop-session-key"

KNOWN = "hk@kotoholdings.com"
UNKNOWN = "stranger@example.net"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    registry = ApprovalRegistry()
    monkeypatch.setattr(approval_tokens, "_REGISTRY", registry)
    draft_dir = tmp_path / "drafts"
    draft_dir.mkdir()
    monkeypatch.setattr(pd, "_DRAFT_DIR", draft_dir)
    monkeypatch.setattr(pd, "_publish_attachment_to_cypress", lambda p: (None, None))
    monkeypatch.setattr(
        pd, "_verified_addresses", lambda: frozenset({KNOWN.lower()})
    )
    yield registry


def _present(*, to=KNOWN, cc="", subject="contract test", body="Body text.",
             attachments=None, session_id=DESKTOP_SESSION):
    out = json.loads(
        pd.present_draft(
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            attachments=[str(p) for p in (attachments or [])],
            session_id=session_id,
        )
    )
    assert out["success"] is True, out
    return out


# --- 1. record hash -------------------------------------------------------


def test_present_draft_stores_and_shows_the_record_hash(tmp_path):
    """The record carries a digest and the render shows its first 8 chars."""
    out = _present()
    record = json.loads(
        (pd._DRAFT_DIR / f"{out['draft_id']}.json").read_text(encoding="utf-8")
    )

    digest = record["record_hash"]
    assert len(digest) == 64
    assert digest == pd.canonical_record_digest(record)
    assert f"**Record:** {digest[:8]}" in out["rendered"]


def test_send_refuses_a_draft_modified_after_it_was_presented(tmp_path):
    """Change the recipient on disk after the render -> the send refuses."""
    out = _present()
    draft_id = out["draft_id"]
    path = pd._DRAFT_DIR / f"{draft_id}.json"

    record = json.loads(path.read_text(encoding="utf-8"))
    record["to"] = "attacker@example.org"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    result = json.loads(pd.send_draft(draft_id=draft_id, session_id=DESKTOP_SESSION))

    assert result["success"] is False
    assert "record hash" in result["error"].lower()
    assert "attacker@example.org" not in json.dumps(result)


def test_send_refuses_when_an_attachment_changed_on_disk(tmp_path):
    """The digest covers attachment content, not just the envelope fields."""
    att = tmp_path / "one.pdf"
    att.write_bytes(b"%PDF-1.4 original\n")
    out = _present(attachments=[att])

    att.write_bytes(b"%PDF-1.4 SWAPPED\n")

    result = json.loads(
        pd.send_draft(draft_id=out["draft_id"], session_id=DESKTOP_SESSION)
    )
    assert result["success"] is False
    assert "record hash" in result["error"].lower()


def test_unmodified_draft_passes_the_hash_check(tmp_path):
    """The check must not refuse an untouched draft (it fails later, on consent)."""
    out = _present()
    result = json.loads(
        pd.send_draft(draft_id=out["draft_id"], session_id=DESKTOP_SESSION)
    )
    assert result["success"] is False
    assert "record hash" not in result["error"].lower()


def test_card_description_carries_the_record_prefix(tmp_path):
    """The reviewer can match the card against the draft they read."""
    out = _present()
    record = json.loads(
        (pd._DRAFT_DIR / f"{out['draft_id']}.json").read_text(encoding="utf-8")
    )

    description = pd._draft_card_description(record)

    assert f"Record: {record['record_hash'][:8]}" in description


# --- 2. address-book flag -------------------------------------------------


def test_unknown_address_is_flagged_and_known_address_is_not():
    """One known To, one unknown Cc -> exactly one flag, on the unknown."""
    out = _present(to=KNOWN, cc=UNKNOWN)
    rendered = out["rendered"]

    to_line = next(ln for ln in rendered.splitlines() if ln.startswith("**To:**"))
    cc_line = next(ln for ln in rendered.splitlines() if ln.startswith("**Cc:**"))

    assert "[NEW ADDRESS]" not in to_line
    assert "[NEW ADDRESS]" in cc_line
    assert rendered.count("[NEW ADDRESS]") == 1


def test_address_flag_appears_on_the_card_too():
    out = _present(to=KNOWN, cc=UNKNOWN)
    record = json.loads(
        (pd._DRAFT_DIR / f"{out['draft_id']}.json").read_text(encoding="utf-8")
    )

    description = pd._draft_card_description(record)

    assert "[NEW ADDRESS]" in description
    to_line = next(ln for ln in description.splitlines() if ln.startswith("To:"))
    assert "[NEW ADDRESS]" not in to_line


def test_flag_is_per_address_within_one_cc_list():
    """A multi-address Cc flags only the addresses that are actually unknown."""
    out = _present(to=KNOWN, cc=f"{KNOWN}, {UNKNOWN}")
    cc_line = next(
        ln for ln in out["rendered"].splitlines() if ln.startswith("**Cc:**")
    )

    assert cc_line.count("[NEW ADDRESS]") == 1
    assert cc_line.index(UNKNOWN) < cc_line.index("[NEW ADDRESS]")


def test_address_matching_is_case_insensitive():
    out = _present(to=KNOWN.upper())
    to_line = next(
        ln for ln in out["rendered"].splitlines() if ln.startswith("**To:**")
    )
    assert "[NEW ADDRESS]" not in to_line


def test_display_name_form_is_matched_on_the_address_not_the_label():
    """``Koto <hk@kotoholdings.com>`` is the known address, not a new one."""
    out = _present(to=f"Koto-san <{KNOWN}>")
    to_line = next(
        ln for ln in out["rendered"].splitlines() if ln.startswith("**To:**")
    )
    assert "[NEW ADDRESS]" not in to_line


def test_hash_does_not_cover_the_flag_text():
    """The flag is presentation; it must not perturb the digest."""
    out = _present(to=KNOWN, cc=UNKNOWN)
    record = json.loads(
        (pd._DRAFT_DIR / f"{out['draft_id']}.json").read_text(encoding="utf-8")
    )
    assert "[NEW ADDRESS]" not in record["to"]
    assert "[NEW ADDRESS]" not in record["cc"]
    assert record["record_hash"] == pd.canonical_record_digest(record)
