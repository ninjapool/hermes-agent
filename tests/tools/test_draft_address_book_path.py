"""The verified address book is the human's, not a profile's.

``verified_addresses.txt`` is curated by hand (root-owned on the reference
deployment) and records which recipients the HUMAN has confirmed. It lived at
``get_hermes_home() / "verified_addresses.txt"``, which under a named profile
resolves to ``<root>/profiles/<name>/verified_addresses.txt`` -- a file that
does not exist. The missing-book path deliberately flags everything, so every
draft rendered from a non-default profile marked every recipient
``[NEW ADDRESS]``, including addresses the book lists. A warning that fires on
every address trains the reviewer to ignore it, which is the failure the flag
exists to prevent.

These tests drive the real resolution chain (Path.home + HERMES_HOME), no
monkeypatch of ``_verified_addresses``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tools.present_draft as pd


KNOWN = "sperling.david@gmail.com"
STRANGER = "stranger@example.org"


@pytest.fixture
def root(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "verified_addresses.txt").write_text(
        "# curated by hand\n" + KNOWN + "\n", encoding="utf-8"
    )
    return home


def _enter_profile(root: Path, monkeypatch, name: str = "exec") -> Path:
    profile = root / "profiles" / name
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    return profile


def test_default_profile_reads_root_book(root):
    assert KNOWN in pd._verified_addresses()
    assert pd.annotate_addresses(KNOWN) == KNOWN


def test_named_profile_reads_the_same_root_book(root, monkeypatch):
    _enter_profile(root, monkeypatch)

    book = pd._verified_addresses()
    assert KNOWN in book
    assert pd.annotate_addresses(KNOWN) == KNOWN
    assert pd._new_addresses(KNOWN, STRANGER) == [STRANGER]


def test_named_profile_still_flags_unknown_addresses(root, monkeypatch):
    _enter_profile(root, monkeypatch)
    assert pd.annotate_addresses(STRANGER) == f"{STRANGER} [NEW ADDRESS]"


def test_file_inside_profile_does_not_certify_addresses(root, monkeypatch):
    """A profile dir is writable by the agent; a book dropped there must not
    silence the flag. The human's root book is the only source."""
    profile = _enter_profile(root, monkeypatch)
    (profile / "verified_addresses.txt").write_text(STRANGER + "\n", encoding="utf-8")

    assert STRANGER not in pd._verified_addresses()
    assert pd.annotate_addresses(STRANGER) == f"{STRANGER} [NEW ADDRESS]"


def test_missing_root_book_still_flags_everything(root, monkeypatch):
    (root / "verified_addresses.txt").unlink()
    _enter_profile(root, monkeypatch)
    assert pd._verified_addresses() == frozenset()
    assert pd.annotate_addresses(KNOWN) == f"{KNOWN} [NEW ADDRESS]"
