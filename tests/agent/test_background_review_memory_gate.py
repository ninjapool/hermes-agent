"""Memory-write admission gate on the background-review prompts.

The review fork's memory prompt used to be two questions plus "if something
stands out, save it" — no duplicate check, no fact/procedure split, and no
cost test, while the skill prompt directly below it carried a detailed
"do NOT capture" list. A production review took that literally and spent ~9%
of a 2,200-char capped store on an entry whose second half was already
verbatim in a skill and whose first half was a procedure (an approval
workflow), which memory then replayed into every subsequent turn as a
standing instruction.

``_MEMORY_ADMISSION_RULES`` is the fix: one shared gate injected into every
prompt that authorizes a memory write.

These are contract tests, not snapshots. They assert the properties that must
hold for the gate to work at all — it reaches every memory-writing path, the
paths cannot drift apart, and it carries all three parts of the store's test.
They deliberately do NOT freeze the prompt wording, which is expected to be
edited.
"""

from __future__ import annotations

import pytest

from agent import background_review


# Every prompt that can authorize a memory write. A new memory-writing prompt
# must be added here — that is the point of the list.
MEMORY_WRITING_PROMPTS = (
    "_MEMORY_REVIEW_PROMPT",
    "_COMBINED_REVIEW_PROMPT",
)


@pytest.mark.parametrize("prompt_name", MEMORY_WRITING_PROMPTS)
def test_memory_writing_prompts_embed_the_shared_gate(prompt_name):
    """The gate reaches every memory-writing path, by identity not by copy.

    Substring-of-the-shared-constant is the anti-drift contract: editing
    ``_MEMORY_ADMISSION_RULES`` updates every path at once. A path that
    paraphrased the gate locally would pass a keyword check but fail here,
    which is the regression this guards.
    """
    prompt = getattr(background_review, prompt_name)
    assert background_review._MEMORY_ADMISSION_RULES in prompt, (
        f"{prompt_name} does not embed _MEMORY_ADMISSION_RULES verbatim; "
        "the memory-write paths have drifted apart"
    )


def test_gate_requires_checking_skills_and_topic_files_first():
    """Rule 1: look for an existing home before composing an entry.

    The duplicated half of the incident entry was already in a skill. Without
    a mandatory pre-write lookup the fork cannot know that.
    """
    rules = background_review._MEMORY_ADMISSION_RULES.lower()
    assert "skills_list" in rules and "skill_view" in rules, (
        "the gate must name the tools used to check for an existing home"
    )
    assert "pointer" in rules, (
        "a fact that already lives in a skill must degrade to a pointer, "
        "not a second copy"
    )


def test_gate_routes_procedures_to_skills():
    """Rule 2: procedures are skill material, never memory material.

    A procedure in memory is re-read every turn as a standing instruction —
    the failure mode that made the incident entry expensive rather than
    merely redundant.
    """
    rules = background_review._MEMORY_ADMISSION_RULES
    assert "PROCEDURES GO TO SKILLS" in rules
    assert "skill_manage" in rules, (
        "the gate must name the tool that puts the procedure in its right home"
    )


def test_gate_carries_all_three_parts_of_the_store_test():
    """Rule 3: the same three-part test the store itself applies.

    All three must be present; two out of three silently readmits a whole
    class of entry (e.g. dropping 'stable for months' readmits in-flight
    task state).
    """
    rules = background_review._MEMORY_ADMISSION_RULES
    for part in (
        "NEEDED BEFORE KNOWING",
        "NOT RETRIEVABLE ON DEMAND",
        "STABLE FOR MONTHS",
    ):
        assert part in rules, f"gate is missing the '{part}' criterion"


def test_gate_is_conjunctive_and_blocking():
    """The gate must read as a hard filter, not as advice.

    The prompt it replaced framed a no-op review as a missed opportunity,
    which is what pushed the fork to write something rather than nothing.
    A gate the model reads as a suggestion reproduces that bias.
    """
    rules = background_review._MEMORY_ADMISSION_RULES
    assert "ALL THREE" in rules, "the three criteria must be conjunctive"
    assert "must not be" in rules or "do not write it" in rules.lower(), (
        "the gate must state the blocking consequence of failing it"
    )
    assert "SUCCESSFUL review" in rules, (
        "writing nothing must be framed as success, or the fork keeps "
        "grasping for something to save"
    )


def test_skill_only_prompt_does_not_carry_the_memory_gate():
    """The skill-only path writes no memory, so the gate would be noise.

    Guards against a well-meaning 'apply it everywhere' edit that would pay
    for ~2.7k chars of memory rules on a path that cannot write memory.
    """
    assert (
        background_review._MEMORY_ADMISSION_RULES
        not in background_review._SKILL_REVIEW_PROMPT
    )
