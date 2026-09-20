"""Freeze-and-compare: the draft file is sealed, not summarised.

The record-hash design this replaces digested a *reconstruction* of the
record, and recomputed some fields from disk while doing it. Three
separate holes came out of that one idea, each in the same place: a value
the verifier recomputed silently shadowed the stored value it was meant to
be checking, so rewriting the stored one changed what was sent without
moving the digest.

Freeze-and-compare has no reconstruction step. The seal is the SHA-256 of
the record file's exact bytes, taken once when the file is written and
never recomputed from fields. Every check compares something freshly
measured against something stored, and never substitutes one for the
other:

    (a) record file bytes   vs  the stored seal        (before consent)
    (b) local file bytes    vs  stored content_sha256  (after consent)
    (c) remote staged bytes vs  stored content_sha256  (after consent)

A stored digest that is missing, empty, or malformed refuses. It never
means "nothing to check".
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from agent.approval_tokens import ApprovalRegistry, KIND_DRAFT
import agent.approval_tokens as approval_tokens
import tools.present_draft as pd


DESKTOP_SESSION = "20260920_120000_desktopbb"
KNOWN = "hk@kotoholdings.com"
UNKNOWN = "someone-new@example.org"


@pytest.fixture
def remote_fs():
    """A stand-in for the cypress staging host, keyed by remote path."""
    return {}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path, remote_fs):
    registry = ApprovalRegistry()
    monkeypatch.setattr(approval_tokens, "_REGISTRY", registry)
    draft_dir = tmp_path / "drafts"
    draft_dir.mkdir()
    monkeypatch.setattr(pd, "_DRAFT_DIR", draft_dir)
    monkeypatch.setattr(
        pd, "_verified_addresses", lambda: frozenset({KNOWN.lower()})
    )

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
    yield registry


def _present(*, to=KNOWN, cc="", subject="contract test", body="Body text.",
             attachments=None, session_id=DESKTOP_SESSION, allow_refusal=False):
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
    if not allow_refusal:
        assert out["success"] is True, out
    return out


def _send(draft_id, registry, session_id=DESKTOP_SESSION):
    token = registry.mint(
        kind=KIND_DRAFT, subject_id=draft_id, session_id=session_id
    ).token
    return json.loads(
        pd.send_draft(
            draft_id=draft_id, approval_token=token, session_id=session_id
        )
    )


def _attachment(tmp_path, name="invoice.pdf", content=b"ORIGINAL INVOICE - 500"):
    p = tmp_path / name
    p.write_bytes(content)
    return p


# --- the seal is the file, not a summary of it ----------------------------


def test_seal_is_the_sha256_of_the_record_files_exact_bytes(tmp_path, _fresh):
    out = _present(attachments=[_attachment(tmp_path)])
    draft_id = out["draft_id"]
    raw = pd._draft_path(draft_id).read_bytes()
    assert pd._stored_seal(draft_id) == hashlib.sha256(raw).hexdigest()


def test_the_record_file_is_never_rewritten_after_present(tmp_path, _fresh):
    """No second write: a file written twice has a window between writes."""
    out = _present(attachments=[_attachment(tmp_path)])
    draft_id = out["draft_id"]
    path = pd._draft_path(draft_id)
    before = path.read_bytes()
    mtime = path.stat().st_mtime_ns

    assert _send(draft_id, _fresh).get("success") is True
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == mtime


def test_render_and_card_show_the_same_seal_prefix(tmp_path, _fresh):
    out = _present(attachments=[_attachment(tmp_path)])
    seal = pd._stored_seal(out["draft_id"])
    assert f"**Record:** {seal[:8]}" in out["rendered"]
    card = pd._draft_card_description(pd.load_draft(out["draft_id"]))
    assert seal[:8] in card.splitlines()[0], card


# --- untampered drafts still send -----------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"cc": ""},
        {"subject": "契約書の件", "body": "お世話になっております。\n添付をご確認ください。"},
        {"to": KNOWN, "cc": KNOWN},
    ],
    ids=["plain", "empty-cc", "japanese", "cc-present"],
)
def test_untampered_draft_sends(tmp_path, _fresh, kwargs):
    out = _present(attachments=[_attachment(tmp_path)], **kwargs)
    result = _send(out["draft_id"], _fresh)
    assert result.get("success") is True, result


def test_untampered_draft_with_no_attachments_sends(_fresh):
    result = _send(_present()["draft_id"], _fresh)
    assert result.get("success") is True, result


def test_untampered_draft_with_several_attachments_sends(tmp_path, _fresh):
    atts = [
        _attachment(tmp_path, "a.pdf", b"AAA"),
        _attachment(tmp_path, "b.pdf", b"BBB"),
        _attachment(tmp_path, "c.pdf", b"CCC"),
    ]
    result = _send(_present(attachments=atts)["draft_id"], _fresh)
    assert result.get("success") is True, result


def test_skip_cypress_path_still_sends_an_untampered_draft(
    tmp_path, monkeypatch, _fresh
):
    """Item 5: no cypress_path must not trip the refuse-on-missing branch."""
    monkeypatch.setenv("HERMES_DRAFT_SKIP_CYPRESS", "1")
    monkeypatch.setattr(
        pd, "_publish_attachment_to_cypress", lambda p, **kw: (None, None)
    )
    out = _present(attachments=[_attachment(tmp_path)])
    record = pd.load_draft(out["draft_id"])
    assert not record["attachments"][0].get("cypress_path")
    result = _send(out["draft_id"], _fresh)
    assert result.get("success") is True, result


# --- (a) the record file ---------------------------------------------------


def _reason(result):
    return (result.get("error", "") + " " + result.get("note", "")).lower()


def test_one_byte_edited_in_the_record_file_refuses(tmp_path, _fresh):
    out = _present(attachments=[_attachment(tmp_path)])
    draft_id = out["draft_id"]
    path = pd._draft_path(draft_id)
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw.replace("contract test", "contract tesT"), encoding="utf-8")

    result = _send(draft_id, _fresh)
    assert result.get("success") is not True, result
    assert "record" in _reason(result)


def test_recipient_rewritten_on_disk_refuses(tmp_path, _fresh):
    out = _present(attachments=[_attachment(tmp_path)])
    draft_id = out["draft_id"]
    record = pd.load_draft(draft_id)
    record["to"] = "attacker@example.net"
    pd._draft_path(draft_id).write_text(json.dumps(record), encoding="utf-8")

    result = _send(draft_id, _fresh)
    assert result.get("success") is not True, result
    assert "record" in _reason(result)


def test_attachment_entry_deleted_refuses(tmp_path, _fresh):
    out = _present(
        attachments=[
            _attachment(tmp_path, "a.pdf", b"AAA"),
            _attachment(tmp_path, "b.pdf", b"BBB"),
        ]
    )
    draft_id = out["draft_id"]
    record = pd.load_draft(draft_id)
    del record["attachments"][1]
    pd._draft_path(draft_id).write_text(json.dumps(record), encoding="utf-8")

    result = _send(draft_id, _fresh)
    assert result.get("success") is not True, result
    assert "record" in _reason(result)


def test_attachment_entry_added_refuses(tmp_path, _fresh):
    out = _present(attachments=[_attachment(tmp_path, "a.pdf", b"AAA")])
    draft_id = out["draft_id"]
    record = pd.load_draft(draft_id)
    smuggled = _attachment(tmp_path, "secret.pdf", b"NOT REVIEWED")
    record["attachments"].append(
        {
            "path": str(smuggled),
            "name": "secret.pdf",
            "bytes": smuggled.stat().st_size,
            "content_sha256": hashlib.sha256(smuggled.read_bytes()).hexdigest(),
        }
    )
    pd._draft_path(draft_id).write_text(json.dumps(record), encoding="utf-8")

    result = _send(draft_id, _fresh)
    assert result.get("success") is not True, result
    assert "record" in _reason(result)


def test_a_missing_seal_refuses(tmp_path, _fresh):
    out = _present(attachments=[_attachment(tmp_path)])
    draft_id = out["draft_id"]
    pd._seal_path(draft_id).unlink()

    result = _send(draft_id, _fresh)
    assert result.get("success") is not True, result
    assert "seal" in _reason(result)


@pytest.mark.parametrize("junk", ["", "   ", "not-a-hash", "abc123", "z" * 64])
def test_a_malformed_seal_refuses(tmp_path, _fresh, junk):
    out = _present(attachments=[_attachment(tmp_path)])
    draft_id = out["draft_id"]
    pd._seal_path(draft_id).write_text(junk, encoding="utf-8")

    result = _send(draft_id, _fresh)
    assert result.get("success") is not True, result
    assert "seal" in _reason(result)


def test_reseal_after_tamper_is_still_refused_by_the_rendered_prefix(
    tmp_path, _fresh
):
    """Rewriting BOTH file and seal changes the prefix the human was shown.

    An attacker with HERMES_HOME write access can reseal a tampered record.
    The defence is not that this is impossible, but that the seal on the
    render and card no longer matches the one on disk — so the value the
    human holds is the check of last resort. Pin that it actually moves.
    """
    out = _present(attachments=[_attachment(tmp_path)])
    draft_id = out["draft_id"]
    shown = pd._stored_seal(draft_id)

    record = pd.load_draft(draft_id)
    record["to"] = "attacker@example.net"
    raw = json.dumps(record).encode("utf-8")
    pd._draft_path(draft_id).write_bytes(raw)
    pd._seal_path(draft_id).write_text(
        hashlib.sha256(raw).hexdigest(), encoding="utf-8"
    )

    assert pd._stored_seal(draft_id) != shown
    assert shown[:8] in out["rendered"]


# --- (b) the local attachment bytes ---------------------------------------


def test_local_attachment_bytes_swapped_refuses(tmp_path, _fresh):
    local = _attachment(tmp_path)
    out = _present(attachments=[local])
    local.write_bytes(b"TAMPERED INVOICE - PAY ATTACKER 999999")

    result = _send(out["draft_id"], _fresh)
    assert result.get("success") is not True, result
    assert "content" in _reason(result)


def test_local_attachment_deleted_refuses(tmp_path, _fresh):
    local = _attachment(tmp_path)
    out = _present(attachments=[local])
    local.unlink()

    result = _send(out["draft_id"], _fresh)
    assert result.get("success") is not True, result
    assert "exist" in _reason(result) or "missing" in _reason(result)


# --- (c) the remote staged bytes ------------------------------------------


def test_remote_bytes_swapped_in_place_refuses(tmp_path, _fresh, remote_fs):
    """Round 2's exploit: overwrite the staged file at the unchanged path."""
    out = _present(attachments=[_attachment(tmp_path)])
    draft_id = out["draft_id"]
    remote_path = pd.load_draft(draft_id)["attachments"][0]["cypress_path"]
    remote_fs[remote_path] = b"TAMPERED INVOICE - PAY ATTACKER 999999"

    result = _send(draft_id, _fresh)
    assert result.get("success") is not True, result
    assert "content" in _reason(result)


def test_repointed_cypress_path_refuses(tmp_path, _fresh, remote_fs):
    """Round 1's exploit: aim cypress_path at another staged file."""
    out = _present(attachments=[_attachment(tmp_path)])
    draft_id = out["draft_id"]
    other = b"A DIFFERENT DOCUMENT ENTIRELY"
    decoy = f"/srv/hermes-staging/{hashlib.sha256(other).hexdigest()[:8]}/invoice.pdf"
    remote_fs[decoy] = other

    record = pd.load_draft(draft_id)
    record["attachments"][0]["cypress_path"] = decoy
    pd._draft_path(draft_id).write_text(json.dumps(record), encoding="utf-8")

    result = _send(draft_id, _fresh)
    assert result.get("success") is not True, result
    assert "record" in _reason(result)


def test_remote_file_vanished_refuses(tmp_path, _fresh, remote_fs):
    out = _present(attachments=[_attachment(tmp_path)])
    draft_id = out["draft_id"]
    remote_fs.pop(pd.load_draft(draft_id)["attachments"][0]["cypress_path"])

    result = _send(draft_id, _fresh)
    assert result.get("success") is not True, result


def test_remote_digest_unreadable_refuses(tmp_path, monkeypatch, _fresh):
    """_remote_sha256 returning None must fail closed, not wave through."""
    out = _present(attachments=[_attachment(tmp_path)])
    monkeypatch.setattr(pd, "_remote_sha256", lambda rp, timeout=30: None)

    result = _send(out["draft_id"], _fresh)
    assert result.get("success") is not True, result


def test_remote_digest_raising_refuses(tmp_path, monkeypatch, _fresh):
    def boom(rp, timeout=30):
        raise OSError("ssh exploded")

    out = _present(attachments=[_attachment(tmp_path)])
    monkeypatch.setattr(pd, "_remote_sha256", boom)

    result = _send(out["draft_id"], _fresh)
    assert result.get("success") is not True, result


# --- stored digests: missing/empty/malformed refuse, never skip -----------


@pytest.mark.parametrize(
    "value", ["", None, "   ", "deadbeef", "z" * 64, 12345, ["a"] * 64]
)
def test_a_malformed_stored_attachment_digest_refuses(tmp_path, _fresh, value):
    """Round 3's exploit generalised: no stored digest means unverifiable."""
    out = _present(attachments=[_attachment(tmp_path)])
    draft_id = out["draft_id"]
    record = pd.load_draft(draft_id)
    record["attachments"][0]["content_sha256"] = value
    pd._draft_path(draft_id).write_text(json.dumps(record), encoding="utf-8")

    result = _send(draft_id, _fresh)
    assert result.get("success") is not True, result


def test_stored_digest_replaced_with_a_valid_hash_of_another_file_refuses(
    tmp_path, _fresh
):
    """A well-formed digest of the WRONG document is still a tamper.

    The subtle case: the attacker swaps both the local file and the stored
    digest so those two agree with each other. Only the seal, and the remote
    copy, disagree.
    """
    local = _attachment(tmp_path)
    out = _present(attachments=[local])
    draft_id = out["draft_id"]

    forged = b"TAMPERED INVOICE - PAY ATTACKER 999999"
    local.write_bytes(forged)
    record = pd.load_draft(draft_id)
    record["attachments"][0]["content_sha256"] = hashlib.sha256(forged).hexdigest()
    pd._draft_path(draft_id).write_text(json.dumps(record), encoding="utf-8")

    result = _send(draft_id, _fresh)
    assert result.get("success") is not True, result
    assert "record" in _reason(result)


# --- the design itself -----------------------------------------------------


MALFORMED_SHAPES = {
    "attachments is a string": "not-a-list",
    "attachments is a dict": {"path": "/etc/passwd"},
    "attachments is null": None,
    "entry is a string": ["/etc/passwd"],
    "entry is null": [None],
    "entry is a list": [[]],
    "entry missing every field": [{}],
    "digest is an int": [{"path": "/tmp/x", "content_sha256": 12345}],
    "digest is a list": [{"path": "/tmp/x", "content_sha256": ["a" * 64]}],
    "digest is uppercase": [{"path": "/tmp/x", "content_sha256": "A" * 64}],
    "digest is whitespace": [{"path": "/tmp/x", "content_sha256": "   "}],
    "digest wrong length": [{"path": "/tmp/x", "content_sha256": "abc"}],
    "digest non-hex": [{"path": "/tmp/x", "content_sha256": "z" * 64}],
    "cypress_path empty": [
        {"path": "/tmp/x", "content_sha256": "a" * 64, "cypress_path": ""}
    ],
}


def test_a_racing_writer_cannot_get_its_own_bytes_sealed(tmp_path, _fresh, monkeypatch):
    """The seal must bind the bytes we MEANT to write, not whatever landed.

    present_draft wrote the file, then re-read it to seal it. A process that
    overwrote the same path in that window got ITS bytes sealed, while the
    returned render — built from in-memory values, never re-read — still
    described the original envelope. The human read one draft; all three
    checks then certified another, consistently, with a matching displayed
    prefix. Approved to a known address, sent to the attacker's.
    """
    real_write_bytes = pathlib.Path.write_bytes
    swapped = {}
    observed = {}

    def racing_write(self, data):
        # Land the intended bytes, then let the "other process" win the race
        # before present_draft can read the file back.
        result = real_write_bytes(self, data)
        if self.suffix == ".json" and not swapped:
            record = json.loads(data.decode("utf-8"))
            record["to"] = "attacker@evil.example"
            swapped["record"] = record
            real_write_bytes(
                self, json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
            )
        return result

    monkeypatch.setattr(pathlib.Path, "write_bytes", racing_write)
    out = _present(to=KNOWN, attachments=[_attachment(tmp_path)], allow_refusal=True)
    # Read the sealed state while the draft still exists, before any fixture
    # teardown removes it.
    if out.get("success"):
        did = out["draft_id"]
        p = pd._draft_path(did)
        if p.is_file():
            observed["on_disk"] = json.loads(p.read_text(encoding="utf-8"))
            observed["stored_seal"] = pd._stored_seal(did)
            observed["live_seal"] = pd.seal_bytes(p.read_bytes())
    monkeypatch.undo()

    assert swapped, "the race never fired; the test proves nothing"

    # The invariant, stated directly: whatever got sealed must be what the
    # human was shown. Asserting on the eventual send would hide this behind
    # the address book, which refuses the attacker's address for an unrelated
    # reason and would let the bug through on any address the book knows.
    if not out.get("success"):
        return  # refusing outright is a fine answer

    assert observed, "draft reported success but no record file was found"
    rendered = out.get("rendered", "")

    assert observed["on_disk"]["to"] == KNOWN, (
        "the racing writer's envelope is what got sealed: "
        f"{observed['on_disk']['to']!r} while the human was shown {KNOWN!r}"
    )
    assert "attacker@evil.example" not in rendered
    assert observed["stored_seal"] == observed["live_seal"]


def test_the_seal_covers_the_bytes_we_intended_to_write(tmp_path, _fresh, monkeypatch):
    """Directly: a read-back that disagrees with the intended bytes refuses.

    Independent of timing — if the file that comes back is not the file that
    went out, present_draft must not proceed to seal it.
    """
    real_write_bytes = pathlib.Path.write_bytes

    def write_then_corrupt(self, data):
        result = real_write_bytes(self, data)
        if self.suffix == ".json":
            real_write_bytes(self, data + b"\n")
        return result

    monkeypatch.setattr(pathlib.Path, "write_bytes", write_then_corrupt)
    out = _present(attachments=[_attachment(tmp_path)], allow_refusal=True)
    monkeypatch.undo()

    if out.get("success"):
        # If it was allowed through, the seal must still reject at send time.
        result = _send(out["draft_id"], _fresh)
        assert result.get("success") is not True, (
            "a record whose read-back differed from the intended bytes was "
            "sealed and sent"
        )


@pytest.mark.parametrize("label", sorted(MALFORMED_SHAPES))
def test_malformed_attachments_refuse_without_raising(
    tmp_path, _fresh, label
):
    """A refusal must survive any record shape, not just plausible ones.

    Round 4 noted that a malformed entry crashed out of the old digest with
    an uncaught exception instead of returning refusal JSON. Nothing was
    sent, so it was never a bypass — but a caller that inspects return
    values more carefully than it catches exceptions could read a crash as
    something other than "refused", and the card path builds its description
    from an unverified record. Every shape returns clean refusal JSON.
    """
    out = _present(attachments=[_attachment(tmp_path)])
    draft_id = out["draft_id"]

    record = pd.load_draft(draft_id)
    record["attachments"] = MALFORMED_SHAPES[label]
    pd._draft_path(draft_id).write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )

    # Must not raise, and must not send.
    result = _send(draft_id, _fresh)
    assert result.get("success") is not True, (label, result)
    assert result.get("error"), (label, result)

    # The card is built before any seal check, from whatever is on disk.
    pd._draft_card_description(pd.load_draft(draft_id))


def test_a_bytes_only_edit_moves_the_seal(tmp_path, _fresh):
    """Round 4's suggestion, closed by construction rather than by a fold.

    The old design recomputed `bytes` from disk and overwrote the stored
    value — the same asymmetry as bug #3, harmless then only because nothing
    read the stored number. Under freeze-and-compare every byte of the record
    file is covered, so there is no field left to shadow: editing `bytes`
    alone, touching nothing else, must move the seal.
    """
    out = _present(attachments=[_attachment(tmp_path)])
    draft_id = out["draft_id"]
    stored = pd._stored_seal(draft_id)

    record = pd.load_draft(draft_id)
    record["attachments"][0]["bytes"] = 999999
    pd._draft_path(draft_id).write_text(json.dumps(record), encoding="utf-8")

    assert pd._live_seal(draft_id) != stored
    result = _send(draft_id, _fresh)
    assert result.get("success") is not True, result


def test_no_recomputed_field_digest_survives_in_the_module():
    """Item 4: the shadowing bug class is gone because the code is gone."""
    assert not hasattr(pd, "_live_record_digest")
    assert not hasattr(pd, "canonical_record_digest")


def test_each_tamper_class_refuses_with_a_distinct_reason(tmp_path, _fresh, remote_fs):
    """A refusal has to tell the human WHICH thing moved."""
    reasons = {}

    out = _present(attachments=[_attachment(tmp_path, "r1.pdf", b"R1")])
    rec = pd.load_draft(out["draft_id"])
    rec["subject"] = "changed"
    pd._draft_path(out["draft_id"]).write_text(json.dumps(rec), encoding="utf-8")
    reasons["record"] = _send(out["draft_id"], _fresh).get("error", "")

    local = _attachment(tmp_path, "r2.pdf", b"R2")
    out = _present(attachments=[local])
    local.write_bytes(b"R2-TAMPERED")
    reasons["local"] = _send(out["draft_id"], _fresh).get("error", "")

    out = _present(attachments=[_attachment(tmp_path, "r3.pdf", b"R3")])
    rp = pd.load_draft(out["draft_id"])["attachments"][0]["cypress_path"]
    remote_fs[rp] = b"R3-TAMPERED"
    reasons["remote"] = _send(out["draft_id"], _fresh).get("error", "")

    assert all(reasons.values()), reasons
    assert len(set(reasons.values())) == 3, reasons


# --- address-book flag ------------------------------------------------------
#
# These six came from the deleted test_draft_record_hash.py. The design they
# were written against is gone, but the behaviour they pin is not: the flag is
# presentation, computed from an address book that never learns from its own
# sends. Deleting the old file took them with it; a reviewer caught that the
# coverage had silently vanished.


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


def test_the_seal_does_not_cover_the_flag_text():
    """The flag is presentation; it must not reach the sealed record.

    The old version of this test asserted the flag stayed out of the
    recomputed digest. Under freeze-and-compare there is no digest to
    perturb, so the equivalent statement is that the flag never enters the
    file that gets sealed, and that a flagged draft still verifies clean.
    """
    out = _present(to=KNOWN, cc=UNKNOWN)
    draft_id = out["draft_id"]
    record = json.loads(
        (pd._DRAFT_DIR / f"{draft_id}.json").read_text(encoding="utf-8")
    )

    assert "[NEW ADDRESS]" not in record["to"]
    assert "[NEW ADDRESS]" not in record["cc"]
    assert "[NEW ADDRESS]" in out["rendered"]

    assert pd._stored_seal(draft_id) == pd.seal_bytes(
        pd._draft_path(draft_id).read_bytes()
    )


def test_the_flag_the_human_saw_survives_to_the_consent_card(tmp_path, _fresh, monkeypatch):
    """The [NEW ADDRESS] warning must not evaporate between render and consent.

    Bug #5, same shape as bug #4: the flag was recomputed at card time from
    verified_addresses.txt, a file outside the seal. Anything that added the
    recipient to that book between render and consent removed the warning from
    the card the human actually approves against -- while the Record prefix on
    both surfaces stayed byte-identical, because the seal never covered it.

    The flag is a decision made once, when the human was warned. The card
    reports that decision; it does not re-derive it.
    """
    stranger = "someone-brand-new@example.org"
    out = _present(to=stranger)
    draft_id = out["draft_id"]

    assert "[NEW ADDRESS]" in out["rendered"], out["rendered"]

    # The address book learns the address AFTER the human read the render.
    monkeypatch.setattr(
        pd, "_verified_addresses", lambda: frozenset({KNOWN, stranger})
    )

    record = json.loads(
        pd._draft_path(draft_id).read_text(encoding="utf-8")
    )
    description = pd._draft_card_description(record)

    assert "[NEW ADDRESS]" in description, (
        "the warning the human saw at render time vanished from the consent "
        "card: " + description
    )


def test_the_flag_decision_is_inside_the_seal(tmp_path, _fresh):
    """Whatever drives the card's flag must be sealed, or it can be swapped."""
    out = _present(to="another-stranger@example.org")
    draft_id = out["draft_id"]
    raw = pd._draft_path(draft_id).read_bytes()

    record = json.loads(raw.decode("utf-8"))
    assert "new_addresses" in record, (
        "the flag decision is not persisted, so the card must re-derive it "
        "from a file outside the seal"
    )
    assert "another-stranger@example.org" in record["new_addresses"]

    # And it is covered: perturbing it moves the seal.
    tampered = dict(record)
    tampered["new_addresses"] = []
    pd._draft_path(draft_id).write_bytes(
        json.dumps(tampered, ensure_ascii=False, indent=2).encode("utf-8")
    )
    assert pd._stored_seal(draft_id) != pd.seal_bytes(
        pd._draft_path(draft_id).read_bytes()
    )


def test_the_draft_that_is_sent_is_the_one_the_seal_approved(
    tmp_path, _fresh, monkeypatch
):
    """One read. The bytes checked and the bytes used must be the same bytes.

    Bug #7: send_draft called load_draft() to get `draft`, and only afterwards
    read the file AGAIN for the seal comparison. Two independent reads of a
    mutable file. A writer that served forged bytes to the first read and
    restored the pristine bytes before the second got a passing seal check --
    identical prefix -- while the card, the attachment checks and the response
    all ran on its envelope.
    """
    approved = KNOWN
    out = _present(to=approved)
    draft_id = out["draft_id"]
    path = pd._draft_path(draft_id)
    pristine = path.read_bytes()

    forged = json.loads(pristine.decode("utf-8"))
    forged["to"] = "attacker@evil.example"
    forged_bytes = json.dumps(forged, ensure_ascii=False, indent=2).encode("utf-8")

    real_read_bytes = pathlib.Path.read_bytes
    real_read_text = pathlib.Path.read_text
    state = {"served_forged": False}

    def racing_read_bytes(self, *a, **kw):
        # Serve the forgery to the FIRST read of the record, then restore the
        # pristine bytes -- the shape bug #7 exploited. If the code reads the
        # record twice, the second read (the seal check) sees pristine bytes
        # and passes while the forgery is the one in hand.
        if self == path and not state["served_forged"]:
            state["served_forged"] = True
            return forged_bytes
        return real_read_bytes(self, *a, **kw)

    def racing_read_text(self, *a, **kw):
        if self == path and not state["served_forged"]:
            state["served_forged"] = True
            return forged_bytes.decode("utf-8")
        return real_read_text(self, *a, **kw)

    token = _fresh.mint(
        kind=KIND_DRAFT, subject_id=draft_id, session_id=DESKTOP_SESSION
    ).token

    monkeypatch.setattr(pathlib.Path, "read_text", racing_read_text)
    monkeypatch.setattr(pathlib.Path, "read_bytes", racing_read_bytes)
    result = json.loads(
        pd.send_draft(
            draft_id=draft_id,
            approval_token=token,
            session_id=DESKTOP_SESSION,
        )
    )

    assert state["served_forged"], "the harness never served the forged read"
    assert result.get("to") != "attacker@evil.example", (
        "send_draft operated on bytes the seal never approved: " + repr(result)
    )


def test_the_render_and_the_seal_read_the_book_once_between_them(_fresh, monkeypatch):
    """One read of the address book per present_draft, not two.

    Bug #8: _new_addresses() read verified_addresses.txt to seal the
    decision, and _render() called annotate_addresses() which read it
    AGAIN to build the text the human is shown. On every surface whose
    consent IS the render -- messaging, CLI, /approve -- a writer that
    taught the book the address between those two reads erased the
    warning from the only thing the human ever sees, while the sealed
    record still, correctly, called it new. Same prefix on both.
    """
    stranger = "stranger@example.org"
    reads = {"n": 0}

    def racing_book():
        # First read (the seal) does not know the address; by the second
        # (the render) the attacker has taught the book.
        reads["n"] += 1
        return frozenset() if reads["n"] == 1 else frozenset({stranger})

    monkeypatch.setattr(pd, "_verified_addresses", racing_book)
    out = _present(to=stranger)

    record = json.loads(pd._draft_path(out["draft_id"]).read_bytes())
    assert stranger in record["new_addresses"], record

    assert reads["n"] <= 1, (
        f"the address book was read {reads['n']} times in one present_draft; "
        "the render and the seal can disagree under a racing writer"
    )
    assert "[NEW ADDRESS]" in out["rendered"], (
        "the seal flagged the address as new but the render the human "
        "reads omitted the warning: " + out["rendered"]
    )


@pytest.mark.parametrize(
    "bad",
    [
        "good@x.com\r\nBcc: evil@attacker.example",
        "good@x.com\nBcc: evil@attacker.example",
        "good@x.com\r",
        "good@x.com\n",
    ],
)
def test_a_carriage_return_in_a_header_is_refused_outright(_fresh, bad):
    """CR and LF do not belong in a header value, at any position.

    A newline in To/Cc/Subject/From is header injection: downstream
    transports that fold on it see a header the human never read. The
    splitter happens to surface the smuggled line as its own flagged
    entry, but 'visibly flagged' is the wrong answer for a character
    that has no legitimate use here. Refuse, do not render.
    """
    out = _present(to=bad, allow_refusal=True)
    assert out["success"] is False, out
    assert "line break" in out["error"].lower(), out["error"]


def test_a_carriage_return_in_the_subject_is_refused_too(_fresh):
    out = _present(subject="Invoice\r\nBcc: evil@attacker.example", allow_refusal=True)
    assert out["success"] is False, out
    assert "line break" in out["error"].lower(), out["error"]


def test_an_ordinary_multiline_body_is_still_fine(_fresh):
    """The body is not a header: newlines there are just prose."""
    out = _present(body="Dear Koto,\n\nPlease find attached.\n\nRegards,")
    assert out["success"] is True, out


# --- multi-address entries --------------------------------------------------


# A newline is deliberately NOT here: CR/LF in a header value is refused
# outright by present_draft (header injection), so it never reaches the
# splitter. That stronger behaviour is pinned by
# test_a_carriage_return_in_a_header_is_refused_outright.
MULTI_SEPARATORS = {
    "space": "{a} {b}",
    "tab": "{a}\t{b}",
    "space padded comma": "{a} , {b}",
    "semicolon": "{a};{b}",
}


@pytest.mark.parametrize("label", sorted(MULTI_SEPARATORS))
def test_every_address_in_a_header_is_flagged_whatever_separates_them(
    label, tmp_path, _fresh
):
    """A second recipient must never ride along inside another's entry.

    Bug #6: _split_addresses split only on [,;], so 'good@x.com evil@e.test'
    parsed as ONE entry and _bare_address took only the first address. The
    second was absent from new_addresses and unflagged on BOTH surfaces --
    no filesystem tampering needed, just a space instead of a comma.
    """
    stranger = "rider@attacker.example"
    value = MULTI_SEPARATORS[label].format(a=KNOWN, b=stranger)

    out = _present(to=value)
    draft_id = out["draft_id"]
    record = json.loads(pd._draft_path(draft_id).read_text(encoding="utf-8"))
    card = pd._draft_card_description(record)

    assert stranger in record.get("new_addresses", []), (
        f"[{label}] the second address never entered the sealed decision: "
        f"{record.get('new_addresses')!r}"
    )
    assert "[NEW ADDRESS]" in out["rendered"], f"[{label}] render: {out['rendered']}"
    assert "[NEW ADDRESS]" in card, f"[{label}] card: {card}"


def test_a_display_name_is_not_split_into_two_entries(tmp_path, _fresh):
    """The fix must not treat the spaces inside 'Name <addr>' as separators."""
    out = _present(to=f"Koto Holdings KK <{KNOWN}>")

    assert "[NEW ADDRESS]" not in out["rendered"], out["rendered"]

    record = json.loads(
        pd._draft_path(out["draft_id"]).read_text(encoding="utf-8")
    )
    assert record["new_addresses"] == [], record["new_addresses"]
    assert "[NEW ADDRESS]" not in pd._draft_card_description(record)


def test_split_addresses_separates_on_whitespace_between_addresses():
    """Unit-level: the splitter itself must see two entries, not one."""
    assert pd._split_addresses("a@x.test b@y.test") == ["a@x.test", "b@y.test"]
    assert pd._split_addresses("a@x.test\tb@y.test") == ["a@x.test", "b@y.test"]
    assert pd._split_addresses("a@x.test\nb@y.test") == ["a@x.test", "b@y.test"]
    # ...while a display name keeps its spaces:
    assert pd._split_addresses("Koto KK <a@x.test>") == ["Koto KK <a@x.test>"]
    assert pd._split_addresses("Koto KK <a@x.test>, B <b@y.test>") == [
        "Koto KK <a@x.test>",
        "B <b@y.test>",
    ]
