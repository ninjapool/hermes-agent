"""Tests for the cost-visibility local patch (agent/cost_visibility.py).

Covers the three behaviours the feature is accountable for:

1. Footer math — ctx percentage and the turn/session dollar split.
2. Warnings — each threshold fires exactly once per crossing, and /new
   (``reset_session_state``) re-arms them.
3. Handoff — the note is written at /new, survives as a file, is consumed
   once, and respects the configured word cap.

These are behaviour contracts, not snapshots: nothing here asserts a
specific price-table entry or a config version literal.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent import cost_visibility as cv


class _FakeCompressor:
    def __init__(self, last_prompt_tokens=0, context_length=0):
        self.last_prompt_tokens = last_prompt_tokens
        self.context_length = context_length


class _FakeAgent:
    """Minimal stand-in for AIAgent exposing only what the module reads."""

    def __init__(
        self,
        session_id="sess-1",
        cost=0.0,
        used=0,
        window=0,
        platform="telegram",
        messages=None,
        model="",
        provider="",
    ):
        self.session_id = session_id
        self.session_estimated_cost_usd = cost
        self.context_compressor = _FakeCompressor(used, window)
        self.platform = platform
        self.messages = messages or []
        self.model = model
        self.provider = provider


class CostVisibilityTestBase(unittest.TestCase):
    """Anchors HERMES_HOME at a temp dir so no test touches the real ledger."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.dict(os.environ, {"HERMES_HOME": self._tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def cfg(**kwargs):
        return cv.CostVisibilityConfig(**kwargs)


class TestFooterMath(CostVisibilityTestBase):
    def test_format_matches_spec(self):
        line = cv.format_footer_line(42.0, 0.31, 4.10)
        self.assertEqual(line, "ctx 42% · turn $0.31 · session $4.10")

    def test_ctx_percentage_is_used_over_window(self):
        agent = _FakeAgent(used=80_000, window=200_000)
        pct = cv.context_pct(agent)
        assert pct is not None
        self.assertAlmostEqual(pct, 40.0)

    def test_ctx_unknown_when_window_missing(self):
        self.assertIsNone(cv.context_pct(_FakeAgent(used=100, window=0)))
        self.assertIn("ctx —", cv.format_footer_line(None, 0.0, 0.0))

    def test_ctx_unknown_after_compression_sentinel(self):
        """last_prompt_tokens parks at -1 right after a compaction."""
        agent = _FakeAgent(used=-1, window=200_000)
        self.assertIsNone(cv.context_pct(agent))

    def test_turn_cost_is_delta_and_session_accumulates(self):
        cfg = self.cfg()
        agent = _FakeAgent(session_id="s-delta", cost=1.00, used=10, window=100)

        first = cv.render_footer(agent, "s-delta", cfg)
        self.assertEqual(first, "ctx 10% · turn $1.00 · session $1.00")

        # Agent's cumulative counter advances; the turn figure is the delta.
        agent.session_estimated_cost_usd = 1.75
        second = cv.render_footer(agent, "s-delta", cfg)
        self.assertEqual(second, "ctx 10% · turn $0.75 · session $1.75")

    def test_session_total_survives_agent_rebuild(self):
        """A rebuilt agent restarts its counter; the session total must not."""
        cfg = self.cfg()
        agent = _FakeAgent(session_id="s-rebuild", cost=4.00, used=10, window=100)
        cv.render_footer(agent, "s-rebuild", cfg)

        # Gateway evicts the agent and builds a fresh one for the SAME session.
        rebuilt = _FakeAgent(session_id="s-rebuild", cost=0.50, used=10, window=100)
        line = cv.render_footer(rebuilt, "s-rebuild", cfg)

        # Session keeps accumulating rather than collapsing back to $0.50.
        self.assertEqual(line, "ctx 10% · turn $0.50 · session $4.50")

    def test_subcent_turn_is_not_rendered_as_zero(self):
        line = cv.format_footer_line(5.0, 0.0004, 0.0004)
        self.assertIn("$0.0004", line)
        self.assertNotIn("$0.00 ", line)

    def test_footer_disabled_by_config(self):
        agent = _FakeAgent(session_id="s-off", cost=1.0, used=10, window=100)
        self.assertEqual(cv.render_footer(agent, "s-off", self.cfg(footer=False)), "")
        self.assertEqual(cv.render_footer(agent, "s-off", self.cfg(enabled=False)), "")

    def test_cli_surface_excluded_unless_opted_in(self):
        cli_agent = _FakeAgent(platform="cli")
        self.assertFalse(cv.surface_enabled(cli_agent, self.cfg()))
        self.assertTrue(cv.surface_enabled(cli_agent, self.cfg(include_cli=True)))
        self.assertTrue(cv.surface_enabled(_FakeAgent(platform="telegram"), self.cfg()))


class TestWarnings(CostVisibilityTestBase):
    def test_cost_warning_fires_once_per_crossing(self):
        cfg = self.cfg(cost_warn_usd=25.0, ctx_warn_pct=80)
        agent = _FakeAgent(session_id="s-cost", cost=26.0, used=61, window=100)

        cv.render_footer(agent, "s-cost", cfg)
        first = cv.check_warnings(agent, "s-cost", cfg)
        self.assertEqual(len(first), 1)
        self.assertIn("Session at $26.00", first[0])
        self.assertIn("Context 61%", first[0])
        self.assertIn("/new", first[0])

        # Still over the threshold on later turns — must stay silent.
        for extra in (27.0, 40.0):
            agent.session_estimated_cost_usd = extra
            cv.render_footer(agent, "s-cost", cfg)
            self.assertEqual(cv.check_warnings(agent, "s-cost", cfg), [])

    def test_no_cost_warning_below_threshold(self):
        cfg = self.cfg(cost_warn_usd=25.0)
        agent = _FakeAgent(session_id="s-under", cost=24.99, used=10, window=100)
        cv.render_footer(agent, "s-under", cfg)
        self.assertEqual(cv.check_warnings(agent, "s-under", cfg), [])

    def test_context_warning_fires_once_per_crossing(self):
        cfg = self.cfg(cost_warn_usd=1000.0, ctx_warn_pct=80)
        agent = _FakeAgent(session_id="s-ctx", cost=0.5, used=80, window=100)

        cv.render_footer(agent, "s-ctx", cfg)
        first = cv.check_warnings(agent, "s-ctx", cfg)
        self.assertEqual(len(first), 1)
        self.assertIn("Context at 80%", first[0])
        self.assertIn("compaction will start soon", first[0])

        agent.context_compressor.last_prompt_tokens = 92
        cv.render_footer(agent, "s-ctx", cfg)
        self.assertEqual(cv.check_warnings(agent, "s-ctx", cfg), [])

    def test_thresholds_come_from_config_not_code(self):
        cfg = self.cfg(cost_warn_usd=2.0, ctx_warn_pct=10)
        agent = _FakeAgent(session_id="s-cfg", cost=2.5, used=15, window=100)
        cv.render_footer(agent, "s-cfg", cfg)
        warnings = cv.check_warnings(agent, "s-cfg", cfg)
        self.assertEqual(len(warnings), 2)

    def test_reset_rearms_warnings(self):
        """/new clears the latches so a new session can warn again."""
        cfg = self.cfg(cost_warn_usd=25.0)
        agent = _FakeAgent(session_id="s-reset", cost=30.0, used=10, window=100)

        cv.render_footer(agent, "s-reset", cfg)
        self.assertEqual(len(cv.check_warnings(agent, "s-reset", cfg)), 1)
        self.assertEqual(cv.check_warnings(agent, "s-reset", cfg), [])

        cv.reset_session_state("s-reset")

        # Fresh session: totals restart, and the threshold can fire again.
        agent2 = _FakeAgent(session_id="s-reset", cost=26.0, used=10, window=100)
        cv.render_footer(agent2, "s-reset", cfg)
        self.assertEqual(len(cv.check_warnings(agent2, "s-reset", cfg)), 1)

    def test_warnings_disabled_by_config(self):
        cfg = self.cfg(warnings=False, cost_warn_usd=1.0)
        agent = _FakeAgent(session_id="s-nowarn", cost=99.0, used=99, window=100)
        cv.render_footer(agent, "s-nowarn", cfg)
        self.assertEqual(cv.check_warnings(agent, "s-nowarn", cfg), [])

    def test_sessions_latch_independently(self):
        cfg = self.cfg(cost_warn_usd=10.0)
        a = _FakeAgent(session_id="s-a", cost=11.0, used=10, window=100)
        b = _FakeAgent(session_id="s-b", cost=11.0, used=10, window=100)
        cv.render_footer(a, "s-a", cfg)
        cv.render_footer(b, "s-b", cfg)
        self.assertEqual(len(cv.check_warnings(a, "s-a", cfg)), 1)
        self.assertEqual(len(cv.check_warnings(b, "s-b", cfg)), 1)


class TestHandoff(CostVisibilityTestBase):
    @staticmethod
    def _conversation():
        return [
            {"role": "user", "content": "Analyse the inverter photos and log faults."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/tmp/site_a.csv"}',
                        }
                    },
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path": "/tmp/report.md"}',
                        }
                    },
                ],
            },
            {"role": "user", "content": "Now summarise the faults by site."},
        ]

    def test_note_captures_session_shape(self):
        agent = _FakeAgent(session_id="s-h", messages=self._conversation())
        note = cv.build_handoff_note(agent, config=self.cfg())

        self.assertIn("handoff from previous session", note)
        self.assertIn("Now summarise the faults by site.", note)
        self.assertIn("/tmp/report.md", note)
        self.assertIn("read_file", note)

    def test_note_respects_word_cap(self):
        long_msg = " ".join(f"word{i}" for i in range(2000))
        agent = _FakeAgent(
            session_id="s-long", messages=[{"role": "user", "content": long_msg}]
        )
        note = cv.build_handoff_note(agent, config=self.cfg(handoff_max_words=300))
        self.assertLessEqual(len(note.split()), 300)

    def test_store_then_consume_is_one_shot(self):
        note = "[handoff from previous session]\nDid a thing."
        self.assertTrue(cv.store_handoff("telegram:123", note))

        self.assertEqual(cv.consume_handoff("telegram:123"), note)
        # Second read returns nothing — the note is not re-injected forever.
        self.assertEqual(cv.consume_handoff("telegram:123"), "")

    def test_handoff_survives_a_restart(self):
        """The note is on disk, so a gateway restart between /new and the
        next message does not lose it. Simulated by clearing all in-memory
        state and re-importing the module."""
        import importlib

        cv.store_handoff("telegram:restart", "[handoff from previous session]\nX")

        reloaded = importlib.reload(cv)
        self.assertIn("handoff", reloaded.consume_handoff("telegram:restart"))

    def test_consume_missing_key_is_empty(self):
        self.assertEqual(cv.consume_handoff("telegram:never-written"), "")
        self.assertEqual(cv.consume_handoff(""), "")

    def test_session_key_is_sanitised_into_a_filename(self):
        weird = "telegram:-100123/thread 7"
        self.assertTrue(cv.store_handoff(weird, "note body"))
        self.assertEqual(cv.consume_handoff(weird), "note body")


class TestSelfCheckAndConfig(CostVisibilityTestBase):
    def test_selfcheck_reports_live_values(self):
        line = cv.selfcheck_line(self.cfg(cost_warn_usd=25.0, ctx_warn_pct=80))
        self.assertIn("cost_visibility loaded", line)
        self.assertIn("cost_warn_usd=25.0", line)
        self.assertIn("ctx_warn_pct=80", line)

    def test_shipped_defaults_match_the_documented_contract(self):
        """The defaults the feature promises: on, $25, 80%, 300 words."""
        cfg = cv.load_cost_visibility_config({})
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.footer)
        self.assertEqual(cfg.cost_warn_usd, 25.0)
        self.assertEqual(cfg.ctx_warn_pct, 80.0)
        self.assertEqual(cfg.handoff_max_words, 300)

    def test_user_config_overrides_defaults(self):
        cfg = cv.load_cost_visibility_config(
            {"cost_visibility": {"cost_warn_usd": 5, "ctx_warn_pct": 50, "footer": False}}
        )
        self.assertEqual(cfg.cost_warn_usd, 5.0)
        self.assertEqual(cfg.ctx_warn_pct, 50.0)
        self.assertFalse(cfg.footer)

    def test_malformed_config_falls_back_to_defaults(self):
        cfg = cv.load_cost_visibility_config(
            {"cost_visibility": {"cost_warn_usd": "not-a-number", "ctx_warn_pct": None}}
        )
        self.assertEqual(cfg.cost_warn_usd, 25.0)
        self.assertEqual(cfg.ctx_warn_pct, 80.0)

    def test_config_section_is_present_in_shipped_defaults(self):
        """The section must exist in DEFAULT_CONFIG or an upgrade's deep-merge
        will not create it for existing users."""
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        self.assertIn(cv.CONFIG_SECTION, DEFAULT_CONFIG)
        section = DEFAULT_CONFIG[cv.CONFIG_SECTION]
        for key in ("enabled", "cost_warn_usd", "ctx_warn_pct", "handoff_max_words"):
            self.assertIn(key, section)


class UnknownCostRendersQuestionMarkTests(CostVisibilityTestBase):
    """A model with no pricing must never render as $0.00.

    This is the bug that hid the $191 session: an unpriced model reports
    0.0 for every turn, and "$0.00" is indistinguishable from a free turn.
    """

    def test_money_renders_none_as_question_mark(self):
        self.assertEqual(cv._money(None), "$?")

    def test_money_still_renders_a_real_zero_as_zero(self):
        # A genuinely-free turn on a PRICED model must keep saying $0.00 —
        # "$?" is reserved for "we could not determine this".
        self.assertEqual(cv._money(0.0), "$0.00")

    def test_footer_line_renders_question_marks_when_cost_unknown(self):
        line = cv.format_footer_line(42.0, None, None)
        self.assertIn("turn $?", line)
        self.assertIn("session $?", line)
        self.assertNotIn("$0.00", line)
        # Context is measured independently of pricing, so it still shows.
        self.assertIn("ctx 42%", line)

    def test_render_footer_uses_question_mark_for_unpriced_model(self):
        agent = _FakeAgent(
            session_id="unpriced-1",
            cost=0.0,
            used=1000,
            window=10000,
            model="totally-made-up-model-xyz",
            provider="anthropic",
        )
        with patch.object(cv, "cost_is_known", return_value=False):
            footer = cv.render_footer(agent, "unpriced-1")
        self.assertIn("$?", footer)
        self.assertNotIn("$0.00", footer)

    def test_render_footer_uses_dollars_for_priced_model(self):
        agent = _FakeAgent(
            session_id="priced-1",
            cost=0.25,
            used=1000,
            window=10000,
            model="claude-opus-5",
            provider="anthropic",
        )
        with patch.object(cv, "cost_is_known", return_value=True):
            footer = cv.render_footer(agent, "priced-1")
        self.assertIn("$0.25", footer)
        self.assertNotIn("$?", footer)

    def test_cost_is_known_is_true_when_model_unset(self):
        # No pinned model on the agent: don't cry wolf.
        self.assertTrue(cv.cost_is_known(_FakeAgent()))


class ActiveModelPricingWarningTests(CostVisibilityTestBase):
    """Startup must shout when the configured model cannot be priced."""

    def _write_config(self, model, provider="anthropic"):
        import pathlib

        cfg = pathlib.Path(self._tmp.name) / "config.yaml"
        cfg.write_text(
            f"model:\n  provider: {provider}\n  model: {model}\n", encoding="utf-8"
        )
        return cfg

    def test_warns_for_unpriced_model(self):
        self._write_config("totally-made-up-model-xyz")
        msg = cv.active_model_pricing_warning()
        self.assertIn("NO PRICING", msg)
        self.assertIn("totally-made-up-model-xyz", msg)

    def test_silent_for_priced_model(self):
        self._write_config("claude-opus-5")
        self.assertEqual(cv.active_model_pricing_warning(), "")

    def test_silent_when_no_model_pinned(self):
        import pathlib

        pathlib.Path(self._tmp.name, "config.yaml").write_text(
            "model:\n  provider: anthropic\n", encoding="utf-8"
        )
        self.assertEqual(cv.active_model_pricing_warning(), "")


class PricingTableContractTests(unittest.TestCase):
    """Invariants about the pricing table — not a snapshot of its contents."""

    def test_current_anthropic_models_are_priced(self):
        """Models Hermes actually routes to must resolve to a price.

        Deliberately not asserting the rate: that changes. Asserting only
        that the lookup succeeds, which is what prevents a silent $0.00.
        """
        from agent.usage_pricing import has_known_pricing

        for model in ("claude-opus-5", "claude-fable-5", "claude-sonnet-5"):
            with self.subTest(model=model):
                self.assertTrue(
                    has_known_pricing(model, provider="anthropic"),
                    f"{model} has no pricing entry — cost will report $0.00",
                )

    def test_anthropic_1h_cache_write_is_double_base_input(self):
        """Anthropic's published rule: 1h writes bill at 2x base input.

        A relationship, not a frozen number — it stays true across price
        changes and catches a mistyped rate.
        """
        from decimal import Decimal

        from agent.usage_pricing import _OFFICIAL_DOCS_PRICING

        checked = 0
        for (provider, model), entry in _OFFICIAL_DOCS_PRICING.items():
            if provider != "anthropic":
                continue
            if entry.cache_write_1h_cost_per_million is None:
                continue
            if entry.input_cost_per_million is None:
                continue
            checked += 1
            self.assertEqual(
                entry.cache_write_1h_cost_per_million,
                entry.input_cost_per_million * Decimal("2"),
                f"{model}: 1h cache-write rate is not 2x base input",
            )
        self.assertGreater(checked, 0, "no Anthropic 1h rates found to check")

    def test_1h_cache_write_tokens_bill_higher_than_5m(self):
        """The TTL split must actually reach the arithmetic."""
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

        all_5m = CanonicalUsage(cache_write_tokens=1_000_000)
        all_1h = CanonicalUsage(
            cache_write_tokens=1_000_000, cache_write_1h_tokens=1_000_000
        )
        cost_5m = estimate_usage_cost("claude-opus-5", all_5m, provider="anthropic")
        cost_1h = estimate_usage_cost("claude-opus-5", all_1h, provider="anthropic")
        assert cost_5m.amount_usd is not None
        assert cost_1h.amount_usd is not None
        self.assertGreater(
            cost_1h.amount_usd,
            cost_5m.amount_usd,
            "1h cache writes must cost more than 5m writes",
        )

    def test_config_override_prices_an_unknown_model(self):
        """An operator can price a brand-new model without a code patch."""
        import pathlib
        import tempfile

        from agent import usage_pricing as up

        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "config.yaml").write_text(
                "pricing:\n"
                "  overrides:\n"
                '    "anthropic/some-unreleased-model":\n'
                "      input: 7.00\n"
                "      output: 21.00\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"HERMES_HOME": tmp}):
                up._PRICING_OVERRIDE_CACHE = None
                up._PRICING_OVERRIDE_MTIME = -1.0
                try:
                    self.assertTrue(
                        up.has_known_pricing(
                            "some-unreleased-model", provider="anthropic"
                        )
                    )
                    usage = up.CanonicalUsage(
                        input_tokens=1_000_000, output_tokens=1_000_000
                    )
                    result = up.estimate_usage_cost(
                        "some-unreleased-model", usage, provider="anthropic"
                    )
                    assert result.amount_usd is not None
                    self.assertEqual(int(result.amount_usd), 28)
                finally:
                    up._PRICING_OVERRIDE_CACHE = None
                    up._PRICING_OVERRIDE_MTIME = -1.0


if __name__ == "__main__":
    unittest.main()
