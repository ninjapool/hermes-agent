# Local patches

Changes carried on our pinned branch that are **not** upstream. Each one has to
be re-applied (cherry-picked) when we move to a new pinned release, or
consciously dropped if upstream has landed an equivalent.

Check this file before and after every pin bump.

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
