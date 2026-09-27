"""The halt message must describe WHY the turn stopped.

Incident 2026-09-27: vision_analyze hit loop_vision_cap after 6 DISTINCT screenshots (a budget,
spent on purpose) and the user was told "after 6 repeated non-progressing attempts" -- the wording of
the loop detector. A budget cap is not a loop: the message must say which it was.
"""

from agent.tool_guardrails import (
    LoopCapConfig,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolGuardrailDecision,
    toolguard_halt_response,
)


def _hit_vision_cap(cap=3):
    c = ToolCallGuardrailController(ToolCallGuardrailConfig(loop_caps=LoopCapConfig(max_vision_calls=cap)))
    for i in range(cap):
        assert c.before_call("vision_analyze", {"image_url": f"/tmp/shot{i}.png", "question": f"q{i}"}).action == "allow"
    return c.before_call("vision_analyze", {"image_url": "/tmp/shot_last.png", "question": "q"})


def test_budget_cap_halt_is_not_called_a_loop():
    msg = toolguard_halt_response(_hit_vision_cap(3))
    assert "non-progressing" not in msg
    assert "repeated" not in msg


def test_budget_cap_halt_states_budget_and_how_to_raise_it():
    msg = toolguard_halt_response(_hit_vision_cap(3))
    assert "3" in msg and "budget" in msg
    assert "tool_loop_guardrails.loop_caps.max_vision_calls" in msg


def test_budget_cap_halt_names_the_images_it_spent():
    d = _hit_vision_cap(3)
    assert d.evidence == ("/tmp/shot0.png", "/tmp/shot1.png", "/tmp/shot2.png")
    msg = toolguard_halt_response(d)
    for i in range(3):
        assert f"shot{i}.png" in msg


def test_other_budget_caps_get_budget_wording_too():
    for tool, key, code in (("web_search", "max_web_searches", "loop_web_search_cap"),
                            ("delegate_task", "max_subagents", "loop_subagent_cap")):
        c = ToolCallGuardrailController(ToolCallGuardrailConfig(loop_caps=LoopCapConfig(**{key: 1})))
        args = {"query": "a"} if tool == "web_search" else {"tasks": [{"goal": "g"}]}
        c.before_call(tool, args)
        d = c.before_call(tool, args)
        assert d.code == code, d
        msg = toolguard_halt_response(d)
        assert "non-progressing" not in msg and "budget" in msg
        assert f"tool_loop_guardrails.loop_caps.{key}" in msg


def test_real_loop_halt_keeps_loop_wording():
    d = ToolGuardrailDecision(action="halt", code="exact_failure", tool_name="terminal", count=5)
    msg = toolguard_halt_response(d)
    assert "5 repeated non-progressing attempts" in msg


def test_evidence_resets_between_turns():
    c = ToolCallGuardrailController(ToolCallGuardrailConfig(loop_caps=LoopCapConfig(max_vision_calls=1)))
    c.before_call("vision_analyze", {"image_url": "/tmp/old.png"})
    c.reset_for_turn()
    c.before_call("vision_analyze", {"image_url": "/tmp/new.png"})
    d = c.before_call("vision_analyze", {"image_url": "/tmp/x.png"})
    assert d.evidence == ("/tmp/new.png",)
