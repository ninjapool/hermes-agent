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

### `fix(pricing): add Claude 5-series rates and model 1h cache writes`

- **Commit:** `36b9cc8ce` (branch `feat/cost-visibility`; PR copy `cb184164a`
  on `feat/pricing-5series-pr`, cut from `fork/main`)
- **Files:** `agent/usage_pricing.py` (+257), `tests/agent/test_usage_pricing.py`
  (+105). The local commit also carries the `$?` change in
  `agent/cost_visibility.py` / its tests, which belongs to the entry below —
  the upstream PR branch deliberately excludes it so the PR diff is
  pricing-only.
- **Why:** `claude-opus-5` and `claude-fable-5` — the only two models actually
  serving traffic on this box (82 and 141 sessions) — were absent from
  `_OFFICIAL_DOCS_PRICING`, whose Anthropic block stopped at
  `claude-opus-4-8`. `has_known_pricing()` returned False, so EVERY session
  recorded `estimated_cost_usd = 0.0` / `cost_status = 'unknown'`. This is the
  real reason the $191.61 session was invisible: Hermes never priced the
  traffic at all, so the footer patch below had nothing true to display.
  Separately, 1h cache writes were never modelled — `PricingEntry` carried
  only the 5m rate, and `prompt_caching.cache_ttl: '1h'` (this box's setting)
  bills writes at 2x base input, not 1.25x, so even priced models were
  undercounted ~60% on every write.
- **What it adds:** entries for `claude-opus-5`, `claude-opus-5-fast`,
  `claude-fable-5`, `claude-fable-5-1`, `claude-mythos-5`, `claude-mythos-5-1`;
  `PricingEntry.cache_write_1h_cost_per_million` +
  `CanonicalUsage.cache_write_1h_tokens`, read from Anthropic's
  `cache_creation.ephemeral_1h_input_tokens` and billed as a separate tier;
  and a `pricing.overrides` config hook (`provider/model` → rates, checked
  BEFORE the shipped table, cached on config mtime) so the next model bump
  needs no code patch.
- **Rates:** all from <https://docs.claude.com/en/docs/about-claude/pricing>,
  cited line-by-line in the PR body. Two footnotes matter: Fable 5.1 /
  Mythos 5.1 price cache hits at **0.025x** base input where every other model
  uses 0.1x (so they need their own entries — aliasing to 5.0 overcharges
  cache reads 4x), and Sonnet 5's $2/$10 introductory rate **became
  permanent** — the scheduled increase to $3/$15 was cancelled. The old
  in-tree comment told the next reader to raise it; corrected with a
  do-not-restore note.
- **Do NOT add a "nearest model" fallback for an unpriced model.**
  `get_pricing_entry` returns `None`, and a test pins that. Anthropic's legacy
  Opus 4.1 card is exactly **3x** Opus 5's in every token class
  ($15/$75/$18.75/$1.50 vs $5/$25/$6.25/$0.50), so reconciling a missing entry
  against the newest older card yields a figure that is wrong by a constant 3x
  and looks entirely plausible. That is how four sessions came to be reported
  at $459.66/$351.06/$322.40/$191.61 when the true Opus 5 cost was
  $153.23/$117.02/$107.47/$64.99 (ratio 2.9998–3.0000 across completely
  different token mixes, which is what finally gave it away). `None` +
  `cost_status='unknown'` is the contract; a guessed number gets believed.
- **Guard test:** `tests/agent/test_usage_pricing.py` (52 tests) — written as
  relationships, not snapshots, so they survive the next price change: 1h ==
  2x base input across every Anthropic entry, 5.1 cache hits == 0.025x base
  input, the TTL split reaches the arithmetic, the wire-format parse, and
  unknown-model → `None` + `status='unknown'`.
- **Upstream status (2026-09-02):** PR **#101068** —
  <https://github.com/NousResearch/hermes-agent/pull/101068>. Root cause also
  posted to issue **#100848**. The missing-model rows overlap four open PRs
  (#87942, #79426, #45238, #43317) sitting in competing scopes; the 1h
  cache-write half is not covered by any of them (#43317 identified the gap
  and explicitly left it). If any of those lands first, keep the 1h work and
  drop the table rows.
- **REQUIRES GATEWAY RELOAD AFTER CHERRY-PICK; THE RUNNING PROCESS WILL NOT
  PICK UP THE CHANGE.** Same mechanism and same command as the entry below —
  `launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`. A live gateway holds
  the old price table in memory, so every session keeps booking `$0.00` after
  the cherry-pick until it is restarted. Verify from `state.db`:
  `cost_status` must read `estimated` / `cost_source = official_docs_snapshot`
  for a session on the current model — `unknown` / `none` means stale code (or
  a genuinely unpriced model, which the startup warning names).

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
- **Guard test:** `tests/agent/test_cost_visibility.py` (40 tests) — footer
  math including the agent-rebuild case, once-per-crossing latching and `/new`
  re-arming, the handoff write/consume round-trip across a simulated restart,
  and the `$?`-not-`$0.00` rendering for an unpriced model.
- **REQUIRES GATEWAY RELOAD AFTER CHERRY-PICK; THE RUNNING PROCESS WILL NOT
  PICK UP THE CHANGE.** `launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`
  (leave `hermes serve` alone — the desktop session runs on it). Python does
  not reload modules for a live process, so a gateway started before the
  cherry-pick keeps serving the pre-patch `agent/turn_finalizer.py` and
  `agent/cost_visibility.py` from memory and the footer simply never appears.
  This cost a long detour on 2026-09-02: two cron smoke tests were debugged as
  a broken cron delivery path when both had in fact run against a stale
  process. Two traps make it look like a real bug — the persisted `state.db`
  row and the `response_len=` log line are both written at
  `turn_finalizer.py:375-464`, i.e. BEFORE the footer is appended at `:629`,
  so neither ever contains the footer even when everything works. Confirm the
  reload took effect from the `cost_visibility loaded — …` startup line, then
  verify a real send against `$HERMES_HOME/cost_visibility/ledger.json`
  (a fresh `seq` for the new session id) — not against `state.db`.

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
