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
