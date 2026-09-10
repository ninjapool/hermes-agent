"""Memory tool-result summaries must carry the write's OUTCOME.

Regression pin for a real incident (2026-09-08): the memory branch of
``_summarize_tool_result`` rendered only the CALL ARGUMENTS, so a rejected
write and a saved write both compacted to the byte-identical line
``[memory] ? on memory``.  Reconstructed history therefore asserted that
writes had landed when the tool had explicitly refused them, and the agent
reported a persisted rule that was never persisted.

The ``?`` came from ``dict.get(..., "?")`` defaults: batch calls put
``action`` inside ``operations`` entries, leaving no top-level key.

These tests use the real payloads recorded in the session store for that
incident.  They assert the CONTRACT (outcome is recoverable from the
summary), not the exact rendering.
"""

import json

from agent.context_compressor import _summarize_tool_result

# Verbatim from the session store, rows 33172 / 33190 / 33340.
_REJECTED = json.dumps(
    {
        "success": False,
        "error": (
            "After applying all 1 operations, memory would be at 2,333/2,200 "
            "chars -- over the limit. Remove or shorten more entries in the "
            "same batch (see current_entries below), then retry."
        ),
        "current_entries": ["gateway restart/launchctl BLOCKED inside gateway"],
    }
)
_STOP = json.dumps(
    {
        "success": False,
        "done": True,
        "error": (
            "Memory consolidation failed 4 times this turn. Stop retrying "
            "memory calls — leave memory unchanged for now and continue with "
            "your reply to the user."
        ),
    }
)
_SAVED = json.dumps(
    {
        "success": True,
        "done": True,
        "target": "memory",
        "usage": "95% — 2,110/2,200 chars",
        "entry_count": 20,
        "message": "Applied 5 operation(s).",
        "note": "Write saved. This update is complete — do not repeat it.",
    }
)

_BATCH_ARGS = json.dumps(
    {
        "target": "memory",
        "operations": [
            {"action": "replace", "old_text": "EMAIL DRAFTS", "content": "..."},
            {"action": "remove", "old_text": "vision_analyze: ONE call/image"},
        ],
    }
)
_SINGLE_ARGS = json.dumps(
    {"target": "memory", "action": "add", "content": "some durable fact"}
)


def test_failed_write_is_not_rendered_as_a_bare_call():
    """A refused write must be distinguishable from a successful one."""
    summary = _summarize_tool_result("memory", _BATCH_ARGS, _REJECTED)
    assert "success" not in summary.lower() or "FAILED" in summary
    assert "FAILED" in summary


def test_successful_write_reports_success_and_usage():
    summary = _summarize_tool_result("memory", _BATCH_ARGS, _SAVED)
    assert "success" in summary.lower()
    assert "2,110/2,200" in summary, "tool-reported usage must survive"


def test_saved_and_rejected_summaries_are_never_identical():
    """The core invariant: outcome must be recoverable from the summary.

    This is the exact bug — both calls produced ``[memory] ? on memory``.
    """
    saved = _summarize_tool_result("memory", _BATCH_ARGS, _SAVED)
    rejected = _summarize_tool_result("memory", _BATCH_ARGS, _REJECTED)
    assert saved != rejected


def test_stop_retrying_instruction_survives_summarization():
    """The tool's explicit stop must not vanish from reconstructed history."""
    summary = _summarize_tool_result("memory", _BATCH_ARGS, _STOP)
    assert "FAILED" in summary
    assert "Stop retrying" in summary


def test_batch_call_does_not_degrade_to_question_marks():
    """Batch calls have no top-level 'action' — must not render as '?'."""
    summary = _summarize_tool_result("memory", _BATCH_ARGS, _SAVED)
    assert "? on ?" not in summary
    assert "2 ops" in summary


def test_single_op_call_still_names_its_action():
    summary = _summarize_tool_result("memory", _SINGLE_ARGS, _SAVED)
    assert "add" in summary
    assert "memory" in summary


def test_unparseable_result_does_not_crash_or_claim_success():
    """Compression must never crash, and must not invent an outcome."""
    summary = _summarize_tool_result("memory", _BATCH_ARGS, "not json at all")
    assert isinstance(summary, str) and summary
    assert "success" not in summary.lower()
    assert "FAILED" not in summary


def test_empty_result_is_not_reported_as_success():
    summary = _summarize_tool_result("memory", _BATCH_ARGS, "")
    assert "success" not in summary.lower()
