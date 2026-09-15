"""Memory-removal approvals must survive the trip from a real user message.

The bug this covers: `/approve mem_…` reached the dangerous-command handler,
which answered "No pending command to approve" and swallowed the turn — the
same failure the draft path had before it was given a fall-through. These tests
exercise the whole chain with real imports: mint from raw user text, resolve
from the registry inside the tool, spend once.
"""

import json

import pytest

from agent.approval_tokens import (
    ApprovalRegistry,
    KIND_DRAFT,
    KIND_MEMORY_REMOVE,
    memory_removal_subject,
)


# --------------------------------------------------------------------------
# Minting from raw user text
# --------------------------------------------------------------------------

def test_single_approve_line_mints():
    reg = ApprovalRegistry()
    toks = reg.mint_memory_approvals_from_user_text(
        "/approve mem_a183dbcef7ff", session_id="s1"
    )
    assert [t.subject_id for t in toks] == ["mem_a183dbcef7ff"]
    assert toks[0].kind == KIND_MEMORY_REMOVE


def test_ten_lines_in_one_message_mint_ten_tokens():
    """The real shape of the paste: ten lines, each with trailing entry text."""
    ids = [
        "mem_a183dbcef7ff", "mem_e9d2609df987", "mem_0f037c290011",
        "mem_e78d9b1e2cb6", "mem_be7bf603750f", "mem_11766f168c4e",
        "mem_6017b0345667", "mem_ab099be26a92", "mem_aea61a68e766",
        "mem_255e9521a725",
    ]
    text = "\n".join(
        f"/approve {i}   [{n:02d}] some entry text 日本語 included"
        for n, i in enumerate(ids)
    )
    reg = ApprovalRegistry()
    toks = reg.mint_memory_approvals_from_user_text(text, session_id="s1")
    assert [t.subject_id for t in toks] == ids


def test_trailing_text_is_ignored():
    reg = ApprovalRegistry()
    toks = reg.mint_memory_approvals_from_user_text(
        "/approve mem_a183dbcef7ff  [00] gw restart BLOCKED in-gateway…",
        session_id="s1",
    )
    assert [t.subject_id for t in toks] == ["mem_a183dbcef7ff"]


def test_duplicate_ids_mint_once():
    """Two tokens for one subject would authorise two removals from one yes."""
    reg = ApprovalRegistry()
    toks = reg.mint_memory_approvals_from_user_text(
        "/approve mem_a183dbcef7ff\n/approve mem_a183dbcef7ff", session_id="s1"
    )
    assert len(toks) == 1


@pytest.mark.parametrize(
    "text",
    [
        "/approve",
        "/approve all",
        "/approve always",
        "/approve session",
        "/approve draft_20260914_153722_b9bce5",
        "the tool told me to type /approve mem_a183dbcef7ff",  # mid-prose
        "",
    ],
)
def test_non_memory_approvals_mint_nothing(text):
    """Everything else must fall through to the dangerous-command handler."""
    reg = ApprovalRegistry()
    assert reg.mint_memory_approvals_from_user_text(text, session_id="s1") == []


def test_draft_path_still_rejects_memory_ids_and_vice_versa():
    """The two doors stay separate: a memory token cannot send a draft."""
    reg = ApprovalRegistry()
    toks = reg.mint_memory_approvals_from_user_text(
        "/approve mem_a183dbcef7ff", session_id="s1"
    )
    ok, reason = reg.consume(
        token=toks[0].token,
        kind=KIND_DRAFT,
        subject_id="mem_a183dbcef7ff",
        session_id="s1",
    )
    assert not ok and "issued for" in reason


# --------------------------------------------------------------------------
# End-to-end through the real memory tool
# --------------------------------------------------------------------------

class _FakeStore:
    """Minimal MemoryStore stand-in: records what got applied.

    Implements the real interface (``target_enabled``) rather than relying on
    monkeypatching module internals — patching ``_memory_target_error`` passed
    in isolation but not under the full suite, where the module object the test
    patched was not always the one the tool resolved.
    """

    def __init__(self):
        self.applied = None
        self._session_memory_latched = False
        self.memory_enabled = True
        self.user_profile_enabled = True

    def target_enabled(self, target: str) -> bool:
        return self.user_profile_enabled if target == "user" else self.memory_enabled

    def apply_batch(self, target, operations):
        self.applied = (target, operations)
        return {"success": True, "message": f"Applied {len(operations)} operation(s)."}


def _gate_off(monkeypatch):
    """Neutralise the unrelated write-approval gate so only the token gate runs."""
    import tools.memory_tool as mt
    monkeypatch.setattr(mt, "_apply_batch_write_gate", lambda *a, **k: None)


def test_batch_removal_refused_without_approval(monkeypatch):
    from tools.memory_tool import memory_tool
    _gate_off(monkeypatch)
    store = _FakeStore()
    out = json.loads(memory_tool(
        operations=[{"action": "remove", "old_text": "entry A"}],
        store=store, session_id="s1",
    ))
    assert out["success"] is False
    assert "no approval token" in out["error"]
    assert out["entries_requiring_approval"][0]["approve_with"] == (
        f"/approve {memory_removal_subject('entry A')}"
    )
    assert store.applied is None


def test_ten_pasted_approvals_authorise_a_ten_op_batch(monkeypatch):
    """The end-to-end case: one pasted message, one batch, all ten removed.

    This is the case a single `approval_token` parameter cannot express — the
    tool takes one token string but the batch removes ten entries. It only
    works because the gate resolves each subject from the registry.
    """
    import agent.approval_tokens as at
    from tools.memory_tool import memory_tool
    _gate_off(monkeypatch)

    entries = [f"entry {i} 日本語" for i in range(10)]
    reg = ApprovalRegistry()
    monkeypatch.setattr(at, "_REGISTRY", reg)

    text = "\n".join(
        f"/approve {memory_removal_subject(e)}   {e[:20]}" for e in entries
    )
    minted = reg.mint_memory_approvals_from_user_text(text, session_id="s1")
    assert len(minted) == 10

    store = _FakeStore()
    out = json.loads(memory_tool(
        operations=[{"action": "remove", "old_text": e} for e in entries],
        store=store, session_id="s1",
    ))
    assert out["success"] is True, out
    assert store.applied is not None
    assert len(store.applied[1]) == 10


def test_approval_is_single_use(monkeypatch):
    import agent.approval_tokens as at
    from tools.memory_tool import memory_tool
    _gate_off(monkeypatch)

    reg = ApprovalRegistry()
    monkeypatch.setattr(at, "_REGISTRY", reg)
    reg.mint_memory_approvals_from_user_text(
        f"/approve {memory_removal_subject('entry A')}", session_id="s1"
    )
    ops = [{"action": "remove", "old_text": "entry A"}]

    first = json.loads(memory_tool(operations=ops, store=_FakeStore(), session_id="s1"))
    assert first["success"] is True
    second = json.loads(memory_tool(operations=ops, store=_FakeStore(), session_id="s1"))
    assert second["success"] is False


def test_approval_does_not_cross_sessions(monkeypatch):
    import agent.approval_tokens as at
    from tools.memory_tool import memory_tool
    _gate_off(monkeypatch)

    reg = ApprovalRegistry()
    monkeypatch.setattr(at, "_REGISTRY", reg)
    reg.mint_memory_approvals_from_user_text(
        f"/approve {memory_removal_subject('entry A')}", session_id="s1"
    )
    out = json.loads(memory_tool(
        operations=[{"action": "remove", "old_text": "entry A"}],
        store=_FakeStore(), session_id="OTHER",
    ))
    assert out["success"] is False


def test_approval_for_one_entry_does_not_remove_another(monkeypatch):
    import agent.approval_tokens as at
    from tools.memory_tool import memory_tool
    _gate_off(monkeypatch)

    reg = ApprovalRegistry()
    monkeypatch.setattr(at, "_REGISTRY", reg)
    reg.mint_memory_approvals_from_user_text(
        f"/approve {memory_removal_subject('entry A')}", session_id="s1"
    )
    store = _FakeStore()
    out = json.loads(memory_tool(
        operations=[{"action": "remove", "old_text": "entry B"}],
        store=store, session_id="s1",
    ))
    assert out["success"] is False
    assert store.applied is None


def test_partial_approval_does_not_burn_the_approved_tokens(monkeypatch):
    """A batch where 1 of 2 is approved must refuse WITHOUT spending the good one.

    Otherwise the user re-approves entries they already approved, because the
    refused batch silently consumed their consent.
    """
    import agent.approval_tokens as at
    from tools.memory_tool import memory_tool
    _gate_off(monkeypatch)

    reg = ApprovalRegistry()
    monkeypatch.setattr(at, "_REGISTRY", reg)
    reg.mint_memory_approvals_from_user_text(
        f"/approve {memory_removal_subject('entry A')}", session_id="s1"
    )

    mixed = json.loads(memory_tool(
        operations=[
            {"action": "remove", "old_text": "entry A"},
            {"action": "remove", "old_text": "entry B"},
        ],
        store=_FakeStore(), session_id="s1",
    ))
    assert mixed["success"] is False
    assert [e["entry"] for e in mixed["entries_requiring_approval"]] == ["entry B"]

    # entry A's approval must still be live.
    retry = json.loads(memory_tool(
        operations=[{"action": "remove", "old_text": "entry A"}],
        store=_FakeStore(), session_id="s1",
    ))
    assert retry["success"] is True, "approval for entry A was burned by the refusal"


def test_adds_and_replaces_are_not_gated(monkeypatch):
    from tools.memory_tool import memory_tool
    _gate_off(monkeypatch)
    store = _FakeStore()
    out = json.loads(memory_tool(
        operations=[
            {"action": "add", "content": "new fact"},
            {"action": "replace", "old_text": "old", "content": "new"},
        ],
        store=store, session_id="s1",
    ))
    assert out["success"] is True
    assert store.applied is not None


# ==========================================================================
# Gateway seam — the part that was actually broken in production
# ==========================================================================

class TestGatewaySeam:
    """`/approve mem_…` must mint at the gateway and NOT reach the
    dangerous-command handler, which answers "No pending command to approve"
    and swallows the turn.
    """

    def _runner(self, agent_session_id="20260915_070503_99ea4222"):
        from gateway.run import GatewayRunner

        runner = GatewayRunner()

        class _Store:
            def peek_session_id(self, key):
                return agent_session_id

        runner.session_store = _Store()
        return runner, agent_session_id

    def test_mints_ten_from_one_pasted_message(self):
        runner, sid = self._runner()
        ids = [f"mem_{i:012x}" for i in range(10)]
        text = "\n".join(f"/approve {i}   [{n:02d}] entry text" for n, i in enumerate(ids))
        toks = runner._try_mint_memory_approval_tokens(text, "agent:main:telegram:dm:8468018784")
        assert [t.subject_id for t in toks] == ids
        assert all(t.session_id == sid for t in toks)

    def test_binds_agent_session_id_not_gateway_key(self):
        """The defect that made the draft gate unopenable, guarded here too."""
        runner, sid = self._runner()
        key = "agent:main:telegram:dm:8468018784"
        toks = runner._try_mint_memory_approval_tokens("/approve mem_a183dbcef7ff", key)
        assert len(toks) == 1
        assert toks[0].session_id == sid != key

    def test_fails_closed_without_live_session(self):
        from gateway.run import GatewayRunner

        runner = GatewayRunner()

        class _Empty:
            def peek_session_id(self, key):
                return None

        runner.session_store = _Empty()
        assert runner._try_mint_memory_approval_tokens("/approve mem_a183dbcef7ff", "k") == []

    def test_bare_and_modifier_approvals_mint_nothing(self):
        runner, _ = self._runner()
        for text in ("/approve", "/approve all", "/approve session"):
            assert runner._try_mint_memory_approval_tokens(text, "k") == []


class TestDispatchOrdering:
    """Structural: the memory branch sits before the dangerous-command handler
    and sets the skip flag, or the turn dies exactly as it did before.
    """

    def _dispatch_source(self) -> str:
        import inspect

        from gateway.run import GatewayRunner

        return inspect.getsource(GatewayRunner._handle_message)

    def test_memory_mint_is_called_in_dispatch(self):
        assert "_try_mint_memory_approval_tokens(" in self._dispatch_source()

    def test_memory_mint_sets_skip_flag_and_does_not_return(self):
        src = self._dispatch_source()
        idx = src.find("_try_mint_memory_approval_tokens(")
        window = src[idx: idx + 700]
        assert "_skip_approve_handler = True" in window
        assert "return " not in window, (
            "Returning here swallows the turn: the agent never runs, so the "
            "approved removals never happen."
        )

    def test_memory_branch_precedes_the_plain_handler(self):
        src = self._dispatch_source()
        assert src.find("_try_mint_memory_approval_tokens(") < src.find(
            "plain_handler = self._gateway_plain_command_handlers()"
        ), "Minting after the handler dispatch is minting after the turn died."

    def test_draft_path_is_tried_first_and_still_works(self):
        src = self._dispatch_source()
        assert src.find("_try_mint_draft_approval_token(") < src.find(
            "_try_mint_memory_approval_tokens("
        )


class TestAllCallSitesPassSessionId:
    """Every dispatch path into memory_tool must supply session_id.

    The gate resolves the user's approval by (kind, subject, session). A call
    site that omits the session id refuses removals the user actually approved
    — and which path runs depends on the surface, so one fixed site is not a
    fixed bug.
    """

    def _src(self, fn):
        import inspect

        return inspect.getsource(fn)

    def test_tool_executor_passes_session_id(self):
        import agent.tool_executor as te

        src = self._src(te)
        idx = src.find("from tools.memory_tool import memory_tool as _memory_tool")
        assert idx != -1
        assert "session_id=" in src[idx: idx + 1200]

    def test_runtime_helpers_passes_session_id(self):
        import agent.agent_runtime_helpers as rh

        src = self._src(rh)
        idx = src.find("from tools.memory_tool import memory_tool as _memory_tool")
        assert idx != -1
        assert "session_id=" in src[idx: idx + 1200]

    def test_registry_handler_passes_session_id_from_kwargs_not_args(self):
        import tools.memory_tool as mt

        src = self._src(mt)
        idx = src.find('registry.register(\n    name="memory"')
        assert idx != -1
        window = src[idx: idx + 1200]
        assert 'session_id=kw.get("session_id"' in window, (
            "Registry path must pass session_id."
        )
        assert 'session_id=args.get' not in window, (
            "session_id must come from the runtime, never from model-supplied "
            "args — otherwise the model names the session its own approval is "
            "checked against."
        )

    def test_model_cannot_supply_session_id_or_token_via_schema(self):
        """Neither gate parameter may appear in the tool schema."""
        from tools.memory_tool import MEMORY_SCHEMA

        props = (
            MEMORY_SCHEMA.get("function", {})
            .get("parameters", {})
            .get("properties", {})
        )
        assert "session_id" not in props
        assert "approval_token" not in props
