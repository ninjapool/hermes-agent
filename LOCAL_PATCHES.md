# Local patches

Changes carried on our pinned branch that are **not** upstream. Each one has to
be re-applied (cherry-picked) when we move to a new pinned release, or
consciously dropped if upstream has landed an equivalent.

Check this file before and after every pin bump.

## TODO — unfinished, carried across sessions

**Push the upstream port of the image-eviction fix and open the PR.**
Ported and committed but NOT pushed as of 2026-09-02.

- **Local branch:** `fix/cache-stable-image-eviction-upstream`, commit `69a34575c`
- **Blocked on:** the `gh` token lacks the `workflow` scope. The branch carries
  upstream commits that touch `.github/workflows/case-collision-check.yml`
  (our own commit touches no workflow file), so both `git push` and
  `gh repo sync` are rejected. Fix with `gh auth refresh -s workflow`, or sync
  the fork's `main` from the GitHub web UI, then:

  ```bash
  git push fork fix/cache-stable-image-eviction-upstream
  gh pr create --repo NousResearch/hermes-agent --base main \
    --head ninjapool:fix/cache-stable-image-eviction-upstream
  ```

- **Then:** add the PR link to the eviction entry below (it currently cites
  only fork PR #1).
- **Note for whoever picks this up:** upstream `main` has moved past
  `pinned-0.20.6` and the function now lives in
  `agent/anthropic_message_convert.py`, not `agent/anthropic_adapter.py`, which
  re-exports it. The port already accounts for this — the tests patch the
  **defining** module, because patching the adapter's re-export silently does
  not affect the caller. Do not "simplify" those patch targets back.

## How to carry these forward

```bash
# after creating the next pinned branch, e.g. pinned-0.21.x
git cherry-pick <sha>          # per entry below, oldest first
scripts/run_tests.sh tests/agent/test_image_eviction_cache_stability.py
```

If upstream has since fixed the same thing, prefer **their** version: drop our
commit, verify the behaviour with our test, and delete the entry here.

---

## Open — needs cherry-pick onto the next pinned branch

### `feat(cost-visibility): per-reply cost footer, spend warnings, /new handoff`

- **Commit:** `7279dd2b6` (branch `feat/cost-visibility`)
- **Base:** `fix/cache-stable-image-eviction` — like the entries below, the
  pinned tag is not an ancestor, so cherry-pick onto the next pin.
- **Files:** `agent/cost_visibility.py` (new — the whole feature),
  `agent/turn_finalizer.py` (+29), `gateway/run.py` (+53),
  `gateway/slash_commands.py` (+33), `hermes_cli/config_defaults.py` (+16),
  `tests/agent/test_cost_visibility.py` (new)
- **Why:** the $191.61 session on 2026-09-01 was invisible until the bill
  arrived. The eviction fix below removes that specific cause; this makes the
  cost of *any* session visible while it is still running.
- **What it does:** three surfaces, all config-gated —
  1. a status footer on every messaging reply:
     `ctx 42% · turn $0.31 · session $4.10`;
  2. warnings that fire **once per threshold crossing** (session spend over
     `cost_warn_usd`, context over `ctx_warn_pct`) and re-arm on `/new`;
  3. a ≤300 word handoff note written at `/new` and injected once as a prefix
     on the first user message of the next session.
- **Config keys read:** `cost_visibility.enabled`, `.footer`, `.warnings`,
  `.handoff`, `.cost_warn_usd` (25.0), `.ctx_warn_pct` (80),
  `.handoff_max_words` (300), `.include_cli` (false). Shipped in
  `DEFAULT_CONFIG` **and** written into `~/.hermes/config.yaml` so a
  code-replacing upgrade cannot reset them.
- **Touch points (keep these small — that is the point):** the four edited
  files contain only a guarded call into `agent/cost_visibility.py`, each
  wrapped in try/except so cost telemetry can never break a reply. All logic
  lives in the module.
- **Design constraints that must survive a re-port:**
  - The footer is appended to `final_response` **inside `finalize_turn`**, so
    it rides the `finish(final_text)` payload. Do not move it into an adapter
    or apply it after the stream seals — that re-opens the duplicate-final bug
    class documented in `AGENTS.md`.
  - The handoff is prefixed onto the next **user** message, never the system
    prompt; a system-prompt edit would break per-conversation prompt caching.
  - Cost accumulates in a durable on-disk ledger
    (`$HERMES_HOME/cost_visibility/ledger.json`), **not** in memory: the
    gateway evicts and rebuilds agent objects, which resets
    `session_estimated_cost_usd` mid-conversation. A counter that goes
    backwards is treated as a rebuild, not a negative turn. An in-memory
    counter silently zeroes the session figure — the exact blind spot this
    feature exists to close.
  - Pricing is **not** reimplemented: `agent/usage_pricing.py` already owns
    the price tables. Do not add a second one; it will drift from billing.
- **Self-check:** the gateway logs `cost_visibility loaded — …` with its live
  config at startup, and a loud `WARNING` if the module fails to import after
  an upgrade, rather than silently running without cost visibility.
- **Upstream status (2026-09-02):** PR **#100877** opened —
  <https://github.com/NousResearch/hermes-agent/pull/100877>
  (branch `feat/cost-visibility-pr`, based on `fork/main` so the PR diff is
  exactly the six files above). If it lands, drop our commit and delete this
  entry. Note the upstream branch had to be cut from `fork/main` rather than
  `origin/main`: the `gh` token lacks the `workflow` scope, and a branch
  carrying upstream commits that touch `.github/workflows/` is rejected on
  push (same blocker as the TODO at the top of this file).
- **Guard test:** `tests/agent/test_cost_visibility.py` (27 tests) — footer
  math including the agent-rebuild case, once-per-crossing latching and `/new`
  re-arming, and the handoff write/consume round-trip across a simulated
  restart.

### `fix(anthropic): make image eviction cache-stable (batch, not rolling)`

- **Commit:** `6d1b29e77` (branch `fix/cache-stable-image-eviction`, PR #1)
- **Base:** `fix/vision-analyze-loop-cap` — NOT `pinned-0.20.6`. The pinned tag
  is not an ancestor of that branch, so this must be cherry-picked, not merged,
  onto the next pin.
- **Files:** `agent/anthropic_adapter.py`, `hermes_cli/config_defaults.py`,
  `tests/agent/test_image_eviction_cache_stability.py`
- **Why:** `_evict_old_screenshots` used a rolling keep-newest-N window running
  in the payload builder on every call, rewriting a `tool_result` before the
  cache breakpoint each time. The cached prefix was invalidated every call
  (`cache_write ≈ cache_read`, w/r ≈ 1.0). ~90% of a $191.61 / 129-call image
  session on 2026-09-01; same shape in three other sessions that week.
- **What it does:** eviction becomes a step function of image count — fires only
  when the count crosses `evict_at_images`, batch-evicts down to `keep_images`,
  then leaves the prefix untouched until the next crossing. Also replaces the
  `[screenshot removed to save context]` placeholder, which models read as a
  FAILED call and retried ~50 times over 5 files.
- **Config:** `compression.image_eviction` — shipped defaults
  `evict_at_images: 32`, `keep_images: 8`, `mode: count`.
- **Upstream status (checked 2026-09-02):** no fix on `main`; the function is
  byte-identical there. Two open PRs **replicate** the bug on other transports
  rather than fixing it — **#97889** (`evict_stale_outbound_tool_images`) and
  **#52675** (chat_completions, claims "prompt caching preserved" while
  rewriting per call). Related but distinct: **#87513** (user-message images are
  never evicted). Re-check these before the next pin bump — if one is reworked
  into a real fix, prefer it over ours.
- **Guard test:** `tests/agent/test_image_eviction_cache_stability.py` —
  A/B's the old algorithm against the new one through the real payload builder,
  so it fails if a rolling window is reintroduced.

### `fix(guardrails): cap vision_analyze calls per turn`

- **Commit:** `3c899b090` (branch `fix/vision-analyze-loop-cap`)
- **Why:** bounds the retry loop from the same 2026-09-01 incident.
- **Config:** `tool_loop_guardrails.loop_caps.max_vision_calls: 6`
- **Upstream status:** not upstream as of 2026-09-02.
