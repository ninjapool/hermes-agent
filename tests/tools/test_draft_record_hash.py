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

import copy
import hashlib
import json
import pathlib

import pytest

from agent.approval_tokens import ApprovalRegistry, KIND_DRAFT
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
    monkeypatch.setattr(
        pd, "_publish_attachment_to_cypress", lambda p, **kw: (None, None)
    )
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


def test_remote_bytes_swapped_in_place_is_refused(tmp_path, monkeypatch, _fresh):
    """The transport reads REMOTE bytes; verify those, not just the path.

    Hashing cypress_path catches a REPOINTED attachment. It does not catch an
    attacker overwriting the file in place at the same path on the staging
    host — the local file, the local digest, and the whole record are
    untouched, so every local check agrees while different bytes ride out.
    Found by an adversarial review of the repoint fix: the same bug class one
    level further out. The reviewer consents to CONTENT, not to a path.
    """
    local = tmp_path / "invoice.pdf"
    local.write_bytes(b"ORIGINAL REVIEWED INVOICE - 500")

    remote_fs = {}

    def fake_publish(path, **kw):
        p = pathlib.Path(path)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        remote = f"/srv/hermes-staging/{digest[:8]}/{p.name}"
        remote_fs[remote] = p.read_bytes()
        return remote, digest

    monkeypatch.setattr(pd, "_publish_attachment_to_cypress", fake_publish)
    monkeypatch.setattr(
        pd,
        "_remote_sha256",
        lambda rp, timeout=30: (
            hashlib.sha256(remote_fs[rp]).hexdigest() if rp in remote_fs else None
        ),
    )
    # Existence is checked by an inline `ssh cypress test -f` in send_draft;
    # stub subprocess so it answers from our fake remote filesystem.
    def fake_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        r = R()
        if cmd[:2] == ["ssh", "cypress"] and "test -f" in cmd[2]:
            target = cmd[2].split("test -f ", 1)[1].strip().strip("'\"")
            r.returncode = 0 if target in remote_fs else 1
        return r

    monkeypatch.setattr(pd.subprocess, "run", fake_run)
    monkeypatch.delenv("HERMES_DRAFT_SKIP_CYPRESS", raising=False)

    out = _present(attachments=[local])
    draft_id = out["draft_id"]
    record = pd.load_draft(draft_id)
    remote_path = record["attachments"][0]["cypress_path"]

    # The attacker never touches the local file or the record: only the
    # already-staged remote copy, at the very same path.
    remote_fs[remote_path] = b"TAMPERED INVOICE - PAY ATTACKER 999999"
    assert pathlib.Path(record["attachments"][0]["path"]).read_bytes().startswith(
        b"ORIGINAL"
    )
    assert pd.canonical_record_digest(record) == record["record_hash"]

    out = json.loads(
        pd.send_draft(
            draft_id=draft_id,
            approval_token=_fresh.mint(
                kind=KIND_DRAFT, subject_id=draft_id, session_id=DESKTOP_SESSION
            ).token,
            session_id=DESKTOP_SESSION,
        )
    )
    assert out.get("success") is not True, (
        "swapped remote bytes were accepted: the reviewer approved content "
        "they never saw"
    )
    assert "content" in (out.get("error", "") + out.get("note", "")).lower()


# --- 1. record hash -------------------------------------------------------


def test_repointing_the_cypress_path_moves_the_digest(tmp_path):
    """The transport path is hashed, not just the local one.

    ``cypress_path`` is where hermes-send actually reads the bytes from
    (``_resolve_attachments``). send_draft only checks that path still
    EXISTS -- never what is in it. If the digest ignored it, repointing it at
    another file on cypress would pass the existence check AND the hash check
    and attach something the reviewer never saw. The local ``content_sha256``
    does not cover this: it digests the local copy, not the remote one.
    """
    record = {
        "from": "f@x.com",
        "to": KNOWN,
        "cc": "",
        "subject": "s",
        "body": "b",
        "attachments": [
            {
                "path": "/local/report.pdf",
                "name": "report.pdf",
                "bytes": 100,
                "content_sha256": "aa" * 32,
                "cypress_path": "/home/ds/.hermes-attachments/report.pdf",
            }
        ],
    }
    before = pd.canonical_record_digest(record)

    repointed = copy.deepcopy(record)
    repointed["attachments"][0]["cypress_path"] = (
        "/home/ds/.hermes-attachments/other.pdf"
    )
    assert pd.canonical_record_digest(repointed) != before

    renamed = copy.deepcopy(record)
    renamed["attachments"][0]["name"] = "invoice.pdf"
    assert pd.canonical_record_digest(renamed) != before


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
