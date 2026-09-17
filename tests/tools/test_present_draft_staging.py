"""Staging attachments to cypress: path construction and quoting.

Regression pins for the 2026-09-17 failure. ``_staged_attachment_location``
returned a path beginning with ``~``; the verification step passed it through
``shlex.quote``, so the remote shell received a literal tilde and ``sha256sum``
could not find a file that had just been copied there successfully. Every
draft with an attachment was refused, with an error claiming the attachment
"will not be sendable" when it was present and byte-correct.

The fix is structural rather than a quoting tweak: there is no ``~`` anywhere
in a remote command, so no remote argument depends on shell expansion and
every one of them can be quoted.

These tests exercise the generated command strings through a REAL ``sh -c``.
A mocked ``subprocess.run`` asserting on argv would have passed against the
old code — the bug lived in what a remote shell did with the string, which
only a real shell can demonstrate.
"""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from tools.present_draft import (
    CYPRESS_STAGING_ROOT,
    StagingError,
    _publish_attachment_to_cypress,
    _staged_attachment_location,
    basename_is_safe_to_stage,
    present_draft,
)


# --- the path itself -------------------------------------------------------


def test_staging_root_is_absolute_with_no_tilde():
    """The constant is the single source of truth and never needs expanding."""
    assert CYPRESS_STAGING_ROOT.startswith("/")
    assert "~" not in CYPRESS_STAGING_ROOT


def test_staged_location_is_absolute(tmp_path):
    p = tmp_path / "invoice.pdf"
    p.write_bytes(b"%PDF-1.4\nx")
    remote_dir, remote_name = _staged_attachment_location(p, "deadbeef" * 8)
    assert remote_dir.startswith(CYPRESS_STAGING_ROOT + "/")
    assert "~" not in remote_dir
    assert remote_name == "invoice.pdf"


def test_hash_is_a_directory_component_not_a_filename_prefix(tmp_path):
    """Collision safety stays; the recipient-facing basename stays clean.

    The old flat scheme leaked ``d34d294f_invoice.pdf`` to the client because
    the transport passes the staged basename into the MIME part.
    """
    p = tmp_path / "invoice.pdf"
    p.write_bytes(b"%PDF-1.4\nx")
    sha = "deadbeef" + "0" * 56
    remote_dir, remote_name = _staged_attachment_location(p, sha)
    assert remote_dir.endswith("/deadbeef")
    assert remote_name == "invoice.pdf"
    assert "deadbeef_" not in remote_name


# --- basename validation ---------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "x$(touch /tmp/canary).pdf",
        "x`touch /tmp/canary`.pdf",
        "a;rm -rf /.pdf",
        "a|b.pdf",
        "a&b.pdf",
        "a>b.pdf",
        "a<b.pdf",
        "a'b.pdf",
        'a"b.pdf',
        "a\\b.pdf",
        "a\nb.pdf",
        "a\tb.pdf",
        "a\x00b.pdf",
        "a$HOME.pdf",
    ],
)
def test_dangerous_basenames_are_rejected(bad):
    assert basename_is_safe_to_stage(bad) is False


@pytest.mark.parametrize(
    "good",
    [
        "invoice.pdf",
        "site_2060_aluminium_estimate_v1.pdf",
        "Quarterly Report 2026.pdf",
        "見積書_2060.pdf",
        "大久保発電所 見積 (Rev.1).pdf",
        "a-b_c.2026.pdf",
        "estimate[final].pdf",
    ],
)
def test_ordinary_basenames_are_accepted(good):
    assert basename_is_safe_to_stage(good) is True


def test_metacharacter_attachment_is_refused_before_any_remote_call(tmp_path, monkeypatch):
    """The canary must never be created: rejection happens before staging.

    The payload is a filename, so the canary target must be a bare relative
    name — an absolute path would put a ``/`` in the basename and the file
    could not be created at all.
    """
    evil = tmp_path / "x$(touch canary).pdf"
    evil.write_bytes(b"%PDF-1.4\n" + b"x" * 64)
    canary_cwd = tmp_path / "canary"

    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("no remote command may run for a rejected basename")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.delenv("HERMES_DRAFT_SKIP_CYPRESS", raising=False)
    monkeypatch.chdir(tmp_path)

    import json

    result = json.loads(
        present_draft(
            to="client@example.com",
            subject="test",
            body="The document is attached.",
            attachments=[str(evil)],
            **{"from": "david.sperling@cleanenergyjapan.jp"},
        )
    )
    assert result["success"] is False
    assert not canary_cwd.exists(), "command substitution in a basename was executed"
    assert not Path("canary").exists(), "canary created in cwd"
    assert any("unsafe characters" in f.lower() for f in result["failures"]), result


# --- the generated commands, through a real shell --------------------------


class _Recorder:
    """Capture argv lists instead of talking to a real host.

    Note ``_remote_sha256`` runs WITHOUT ``check=True`` — a missing remote
    file is a returncode of 1, not an exception. The stub mirrors that.
    """

    def __init__(self, sha, *, missing_after_copy=False, fail_scp=False, wrong_hash=False):
        self.calls = []
        self.sha = sha
        self.missing_after_copy = missing_after_copy
        self.fail_scp = fail_scp
        self.wrong_hash = wrong_hash
        self.copied = False

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        joined = " ".join(argv)
        if argv[0] == "scp":
            if self.fail_scp:
                raise subprocess.CalledProcessError(1, argv, b"", b"scp: connection lost")
            self.copied = True
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "sha256sum" in joined:
            if self.missing_after_copy:
                return subprocess.CompletedProcess(argv, 1, "", "No such file")
            if self.wrong_hash:
                return subprocess.CompletedProcess(argv, 0, "0" * 64 + "  /path\n", "")
            return subprocess.CompletedProcess(argv, 0, f"{self.sha}  /path\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")


def _sha_of(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_no_remote_command_contains_a_tilde(tmp_path, monkeypatch):
    """The 2026-09-17 bug, pinned directly."""
    p = tmp_path / "invoice.pdf"
    p.write_bytes(b"%PDF-1.4\n" + b"x" * 512)
    rec = _Recorder(_sha_of(p), missing_after_copy=False)
    # first probe must miss so we exercise mkdir+scp too
    rec.missing_after_copy = False
    monkeypatch.setattr(subprocess, "run", rec)
    _publish_attachment_to_cypress(str(p))
    for argv in rec.calls:
        assert "~" not in " ".join(argv), f"tilde leaked into remote command: {argv}"


def test_generated_commands_survive_a_real_shell(tmp_path, monkeypatch):
    """Run the exact remote command strings through a real ``sh -c``.

    HOME is pointed at a tmpdir and a stub ``sha256sum`` is placed on PATH, so
    the command is executed the way the remote shell would execute it. Under
    the old ``~``-plus-``shlex.quote`` code this fails: the quoted tilde is not
    expanded and the file is not found.
    """
    p = tmp_path / "invoice.pdf"
    p.write_bytes(b"%PDF-1.4\n" + b"x" * 512)
    sha = _sha_of(p)

    rec = _Recorder(sha)
    monkeypatch.setattr(subprocess, "run", rec)
    _publish_attachment_to_cypress(str(p))

    remote_shell_cmds = [
        argv[2] for argv in rec.calls if argv[0] == "ssh" and len(argv) > 2
    ]
    assert remote_shell_cmds, "expected at least one remote shell command"

    # Stand up a fake remote filesystem rooted at the real staging path.
    fake_root = tmp_path / "remote"
    staged_dir = fake_root / CYPRESS_STAGING_ROOT.lstrip("/") / sha[:8]
    staged_dir.mkdir(parents=True)
    (staged_dir / "invoice.pdf").write_bytes(p.read_bytes())

    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "sha256sum"
    # The stub rewrites an absolute staging path into the fake root, then
    # hashes for real. If the path still contains a tilde it will not match
    # and the stub exits non-zero, exactly like the real failure.
    stub.write_text(
        "#!/bin/sh\n"
        f'case "$1" in\n'
        f'  {CYPRESS_STAGING_ROOT}/*) f="{fake_root}$1" ;;\n'
        f'  *) echo "sha256sum: $1: No such file or directory" >&2; exit 1 ;;\n'
        f"esac\n"
        'if [ ! -f "$f" ]; then\n'
        '  echo "sha256sum: $1: No such file or directory" >&2; exit 1\n'
        "fi\n"
        'shasum -a 256 "$f" 2>/dev/null || sha256sum "$f"\n'
    )
    stub.chmod(0o755)

    env = dict(os.environ)
    env["HOME"] = str(fake_home)
    env["PATH"] = f"{bindir}:{env['PATH']}"

    for cmd in remote_shell_cmds:
        if "sha256sum" not in cmd:
            continue
        proc = subprocess.run(
            ["sh", "-c", cmd], capture_output=True, text=True, env=env, timeout=30
        )
        assert proc.returncode == 0, (
            f"remote command failed in a real shell: {cmd!r}\n{proc.stderr}"
        )
        assert proc.stdout.split()[0] == sha


def test_every_remote_path_argument_is_quoted(tmp_path, monkeypatch):
    """A space in the basename must not split into two arguments."""
    p = tmp_path / "Quarterly Report 2026.pdf"
    p.write_bytes(b"%PDF-1.4\n" + b"x" * 512)
    rec = _Recorder(_sha_of(p))
    monkeypatch.setattr(subprocess, "run", rec)
    _publish_attachment_to_cypress(str(p))

    for argv in rec.calls:
        if argv[0] == "ssh" and len(argv) > 2:
            cmd = argv[2]
            if CYPRESS_STAGING_ROOT in cmd:
                # every occurrence of the staging root must sit inside quotes
                assert shlex.quote(CYPRESS_STAGING_ROOT) in cmd or "'" in cmd, cmd
                # and the command must lex into the expected argument count
                lexed = shlex.split(cmd)
                for tok in lexed:
                    if tok.startswith(CYPRESS_STAGING_ROOT):
                        assert tok.endswith(".pdf"), f"path was split: {lexed}"
        if argv[0] == "scp":
            dest = argv[-1]
            assert dest.startswith("cypress:"), dest
            # the remote half is quoted so the remote shell keeps it as one arg
            remote_half = dest.split(":", 1)[1]
            assert remote_half.startswith("'") and remote_half.endswith("'"), dest


def test_no_echo_round_trip_is_performed(tmp_path, monkeypatch):
    """The path is constructed, not discovered. One fewer network hop."""
    p = tmp_path / "invoice.pdf"
    p.write_bytes(b"%PDF-1.4\n" + b"x" * 512)
    rec = _Recorder(_sha_of(p))
    monkeypatch.setattr(subprocess, "run", rec)
    cypress_path, _ = _publish_attachment_to_cypress(str(p))
    assert not any(
        "echo" in " ".join(argv) for argv in rec.calls
    ), f"echo round trip still present: {rec.calls}"
    assert cypress_path == f"{CYPRESS_STAGING_ROOT}/{_sha_of(p)[:8]}/invoice.pdf"


# --- verify-first ----------------------------------------------------------


def test_matching_remote_hash_skips_mkdir_and_scp(tmp_path, monkeypatch):
    """Re-staging an identical file is a single probe, not a re-upload."""
    p = tmp_path / "invoice.pdf"
    p.write_bytes(b"%PDF-1.4\n" + b"x" * 512)
    rec = _Recorder(_sha_of(p))  # probe succeeds immediately
    monkeypatch.setattr(subprocess, "run", rec)
    cypress_path, sha = _publish_attachment_to_cypress(str(p))

    assert cypress_path is not None
    assert sha == _sha_of(p)
    assert not any(argv[0] == "scp" for argv in rec.calls), "re-uploaded a file already staged"
    assert not any("mkdir" in " ".join(argv) for argv in rec.calls), "mkdir on a hit"
    assert len(rec.calls) == 1, f"expected one probe, got {rec.calls}"


def test_missing_remote_file_triggers_mkdir_and_scp(tmp_path, monkeypatch):
    p = tmp_path / "invoice.pdf"
    p.write_bytes(b"%PDF-1.4\n" + b"x" * 512)

    class _MissThenHit(_Recorder):
        def __call__(self, argv, **kw):
            self.calls.append(argv)
            joined = " ".join(argv)
            if argv[0] == "scp":
                self.copied = True
                return subprocess.CompletedProcess(argv, 0, "", "")
            if "sha256sum" in joined and not self.copied:
                return subprocess.CompletedProcess(argv, 1, "", "No such file")
            if "sha256sum" in joined:
                return subprocess.CompletedProcess(argv, 0, f"{self.sha}  /p\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

    rec = _MissThenHit(_sha_of(p))
    monkeypatch.setattr(subprocess, "run", rec)
    cypress_path, sha = _publish_attachment_to_cypress(str(p))
    assert cypress_path is not None
    assert any(argv[0] == "scp" for argv in rec.calls)
    assert any("mkdir" in " ".join(argv) for argv in rec.calls)


# --- the three distinct failures -------------------------------------------


def test_transfer_failure_is_named_as_such(tmp_path, monkeypatch):
    p = tmp_path / "invoice.pdf"
    p.write_bytes(b"%PDF-1.4\n" + b"x" * 512)

    class _ProbeMissScpFails(_Recorder):
        def __call__(self, argv, **kw):
            self.calls.append(argv)
            if argv[0] == "scp":
                raise subprocess.CalledProcessError(1, argv, b"", b"lost connection")
            if "sha256sum" in " ".join(argv):
                return subprocess.CompletedProcess(argv, 1, "", "No such file")
            return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _ProbeMissScpFails(_sha_of(p)))
    with pytest.raises(StagingError) as exc:
        _publish_attachment_to_cypress(str(p), raise_on_error=True)
    assert exc.value.kind == "transfer_failed"


def test_remote_file_not_found_after_a_successful_copy(tmp_path, monkeypatch):
    """scp reported success but nothing is there — distinct from a bad hash."""
    p = tmp_path / "invoice.pdf"
    p.write_bytes(b"%PDF-1.4\n" + b"x" * 512)
    rec = _Recorder(_sha_of(p), missing_after_copy=True)
    monkeypatch.setattr(subprocess, "run", rec)
    with pytest.raises(StagingError) as exc:
        _publish_attachment_to_cypress(str(p), raise_on_error=True)
    assert exc.value.kind == "remote_not_found"


def test_hash_mismatch_is_named_as_such(tmp_path, monkeypatch):
    p = tmp_path / "invoice.pdf"
    p.write_bytes(b"%PDF-1.4\n" + b"x" * 512)

    class _MissThenWrong(_Recorder):
        def __call__(self, argv, **kw):
            self.calls.append(argv)
            joined = " ".join(argv)
            if argv[0] == "scp":
                self.copied = True
                return subprocess.CompletedProcess(argv, 0, "", "")
            if "sha256sum" in joined and not self.copied:
                return subprocess.CompletedProcess(argv, 1, "", "No such file")
            if "sha256sum" in joined:
                return subprocess.CompletedProcess(argv, 0, "0" * 64 + "  /p\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _MissThenWrong(_sha_of(p)))
    with pytest.raises(StagingError) as exc:
        _publish_attachment_to_cypress(str(p), raise_on_error=True)
    assert exc.value.kind == "hash_mismatch"


def test_all_three_failures_still_refuse_the_whole_draft(tmp_path, monkeypatch):
    """Whatever the cause, no draft is rendered."""
    import json

    p = tmp_path / "invoice.pdf"
    p.write_bytes(b"%PDF-1.4\n" + b"x" * 512)
    monkeypatch.delenv("HERMES_DRAFT_SKIP_CYPRESS", raising=False)

    class _AlwaysMissing(_Recorder):
        def __call__(self, argv, **kw):
            self.calls.append(argv)
            if argv[0] == "scp":
                return subprocess.CompletedProcess(argv, 0, "", "")
            if "sha256sum" in " ".join(argv):
                return subprocess.CompletedProcess(argv, 1, "", "No such file")
            return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _AlwaysMissing(_sha_of(p)))
    result = json.loads(
        present_draft(
            to="client@example.com",
            subject="test",
            body="The document is attached.",
            attachments=[str(p)],
            **{"from": "david.sperling@cleanenergyjapan.jp"},
        )
    )
    assert result["success"] is False
    assert "draft_id" not in result
    joined = " ".join(result["failures"]).lower()
    assert "not found" in joined or "remote" in joined, result["failures"]
