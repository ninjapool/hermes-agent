"""Session-scoped cap on at-capacity memory consolidation failures.

Regression pin for the 2026-09-08 incident. The per-turn cap (#42405) stops a
runaway loop inside ONE turn, but it resets at every turn boundary, so a store
that is genuinely full re-issues the same doomed batch turn after turn. The
recorded sequence: four failures ending in the terminal "stop retrying"
verdict, then — 44 minutes and eight user turns later — four MORE calls, with
the per-turn counter blind to the earlier ones.

The session budget spans turn boundaries and LATCHES. Only a successful write
or explicit user authorisation clears it; never the model's own judgement.
"""

import json

import pytest

from tools.memory_tool import MemoryStore, memory_tool


@pytest.fixture
def full_store(tmp_path):
    """A store whose budget is effectively spent: tiny cap, one entry."""
    store = MemoryStore(memory_char_limit=120, user_char_limit=120)
    store.memory_entries = ["an existing entry that already fills the store"]
    store.user_entries = []
    return store


def _fill_attempt(store):
    """One at-capacity write attempt, as the model would issue it."""
    return json.loads(
        memory_tool(
            action="add",
            target="memory",
            content="x" * 200,  # cannot fit under any consolidation
            store=store,
        )
    )


def test_per_turn_cap_still_terminates_within_one_turn(full_store):
    """The original #42405 behaviour must survive."""
    results = [_fill_attempt(full_store) for _ in range(4)]
    assert all(r["success"] is False for r in results)
    assert results[-1].get("done") is True


def test_session_budget_survives_turn_boundaries(full_store):
    """The recorded incident: failures, a turn boundary, then more calls."""
    for _ in range(4):
        _fill_attempt(full_store)

    # New turn. This is exactly what reset the counter on 2026-09-08.
    full_store.reset_consolidation_failures()
    assert full_store._consolidation_failures == 0, "per-turn counter resets"
    assert full_store._session_consolidation_failures >= 4, (
        "session counter must NOT reset at a turn boundary"
    )

    # Post-boundary calls keep counting against the session budget and latch.
    result = None
    for _ in range(4):
        result = _fill_attempt(full_store)

    assert full_store._session_memory_latched is True
    assert result is not None
    assert result["success"] is False
    assert result.get("done") is True


def test_latched_session_refuses_at_the_entry_point(full_store):
    """Once latched, further calls are refused without doing write work."""
    for _ in range(8):
        _fill_attempt(full_store)
    assert full_store._session_memory_latched is True

    refusal = _fill_attempt(full_store)
    assert refusal.get("latched") is True
    assert "disabled for this session" in refusal["error"]


def test_latched_refusal_does_not_invite_another_retry(full_store):
    """The refusal must route to the user, not hand back a retry recipe."""
    for _ in range(8):
        _fill_attempt(full_store)
    refusal = _fill_attempt(full_store)

    err = refusal["error"].lower()
    assert "retry" not in err, "a retry recipe restarts the loop it just stopped"
    assert "consolidate" not in err
    assert "ask" in err and "prune" in err
    # The 33333-33340 failure: entries removed with no user authorisation.
    assert "do not remove entries on your own judgement" in err


def test_latch_survives_many_turn_boundaries(full_store):
    """A turn boundary must never be a way to launder the latch away."""
    for _ in range(8):
        _fill_attempt(full_store)
    assert full_store._session_memory_latched is True

    for _ in range(5):
        full_store.reset_consolidation_failures()
    assert full_store._session_memory_latched is True
    assert _fill_attempt(full_store).get("latched") is True


def test_explicit_authorisation_clears_the_latch(full_store):
    """The single sanctioned escape hatch: a user turn asking for memory work."""
    for _ in range(8):
        _fill_attempt(full_store)
    assert full_store._session_memory_latched is True

    full_store.authorize_memory_retries()
    assert full_store._session_memory_latched is False
    assert full_store._session_consolidation_failures == 0

    # Writes are attempted again (this one still fails on size — but it is
    # evaluated, not refused at the gate).
    assert _fill_attempt(full_store).get("latched") is not True


def test_successful_write_clears_the_session_budget(full_store):
    """Progress proves the store is writable; the budget resets."""
    for _ in range(2):
        _fill_attempt(full_store)
    assert full_store._session_consolidation_failures >= 1

    # Make room, then write for real.
    full_store.memory_entries = ["short"]
    result = json.loads(
        memory_tool(action="add", target="memory", content="a durable fact",
                    store=full_store)
    )
    assert result["success"] is True
    assert full_store._session_consolidation_failures == 0
    assert full_store._session_memory_latched is False


def test_healthy_store_is_never_latched(tmp_path):
    """The cap must not interfere with ordinary memory use."""
    store = MemoryStore(memory_char_limit=2000, user_char_limit=2000)
    store.memory_entries = []
    for i in range(10):
        result = json.loads(
            memory_tool(action="add", target="memory", content=f"fact {i}", store=store)
        )
        assert result["success"] is True
    assert store._session_memory_latched is False
    assert store._session_consolidation_failures == 0
