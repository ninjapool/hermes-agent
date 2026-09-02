"""Cache stability of outbound image eviction (incident 2026-09-01).

The old ``_evict_old_screenshots`` kept a rolling "newest 3" window and ran in
the payload builder on every call. Because the window boundary advances with
every added image, each call rewrote a ``tool_result`` sitting BEFORE the cache
breakpoint, so the cached prefix was invalidated and re-written every call:
``cache_creation_input_tokens ~= cache_read_input_tokens`` (w_over_r ~= 1.0)
for a whole session. On a 129-call image session that was ~9.56M cache-write
tokens and ~90% of the bill.

These tests drive the REAL payload builder (``convert_messages_to_anthropic``)
and simulate Anthropic's prefix cache to measure w/r, rather than asserting on
the shape of the source.
"""

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agent.anthropic_adapter import (  # noqa: E402
    _evicted_prefix_count,
    _stripped_image_placeholder,
    convert_messages_to_anthropic,
)
from agent.prompt_caching import apply_anthropic_cache_control  # noqa: E402


def _image_tool_result(call_id: str, name: str, path: str) -> list:
    """One assistant tool_use + matching image-bearing tool result.

    The base64 blob is sized like a real downscaled screenshot (~100KB, the
    shape in the 2026-09-01 incident). Size matters to this test: cache cost is
    proportional to the bytes re-written, so a toy 1KB image would understate
    the damage a per-call rewrite does.
    """
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps({"image_url": path})},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": {
                "_multimodal": True,
                "text_summary": f"analysis of {path}",
                "content": [
                    {"type": "text", "text": f"Image loaded: {path}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + ("A" * 100_000)},
                    },
                ],
            },
        },
    ]


def _synthetic_session(n_calls: int, n_images: int) -> list:
    """25 calls, 20 images interleaved with ordinary text turns.

    The system prompt is deliberately bulky. A real Hermes session carries a
    system prompt plus tool schemas (~10-20K tokens) as the stable cached base,
    and w/r is only meaningful against a realistic base — with a near-empty
    prefix, appending any image dominates the ratio through sheer growth rather
    than through cache invalidation.
    """
    system = (
        "You are a helpful assistant.\n"
        + "\n".join(f"Tool schema line {i}: {'x' * 120}" for i in range(400))
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Review these site photos."},
    ]
    image_calls = {int(round(i * (n_calls - 1) / max(1, n_images - 1))) for i in range(n_images)}
    placed = 0
    for turn in range(n_calls):
        if turn in image_calls and placed < n_images:
            messages += _image_tool_result(
                f"call_{placed}", "vision_analyze", f"/photos/site_{placed:02d}.jpg"
            )
            placed += 1
        else:
            messages.append({"role": "assistant", "content": f"Noted observation {turn}."})
            messages.append({"role": "user", "content": f"Continue with step {turn}."})
    assert placed == n_images, f"placed {placed} images, wanted {n_images}"
    return messages


def _strip_markers(obj):
    """Remove cache_control markers before comparing prefix bytes.

    The marker designates the cache boundary; it is not itself cached content,
    and it legitimately moves forward as the conversation grows. Leaving it in
    the serialization would flag every call as a prefix rewrite and mask the
    real signal (image blocks being rewritten behind the breakpoint).
    """
    if isinstance(obj, dict):
        return {k: _strip_markers(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_markers(v) for v in obj]
    return obj


def _normalize_msg(msg: dict) -> dict:
    """Canonicalize a message for comparison.

    ``apply_anthropic_cache_control`` promotes string content to a single text
    block when it places a marker there, so the same logical message serializes
    two different ways depending on where the breakpoint currently sits. That
    is the cache planner's doing, not eviction's, so normalize it away.
    """
    out = _strip_markers(msg)
    content = out.get("content")
    if isinstance(content, str):
        out["content"] = [{"text": content, "type": "text"}]
    return out


def _prefix_blocks_upto_breakpoint(planned: list) -> list:
    """Per-message serialized bytes for everything before the LAST breakpoint.

    Returned as a LIST of per-message blobs rather than one blob: a growing
    conversation appends messages, and only an element-wise comparison can tell
    "appended a new turn" (fine) from "rewrote an earlier turn" (cache thrash).
    """
    last = -1
    for i, msg in enumerate(planned):
        content = msg.get("content")
        blocks = content if isinstance(content, list) else []
        if isinstance(msg.get("cache_control"), dict) or any(
            isinstance(b, dict) and isinstance(b.get("cache_control"), dict) for b in blocks
        ):
            last = i
    return [
        json.dumps(_normalize_msg(m), sort_keys=True, ensure_ascii=False).encode()
        for m in planned[: last + 1]
    ]


def _build_call(messages: list) -> tuple:
    """Run the real conversion + cache planner for one API call."""
    system, converted = convert_messages_to_anthropic(copy.deepcopy(messages))
    planned = apply_anthropic_cache_control(
        [{"role": "system", "content": system}] + converted if system else converted,
        native_anthropic=True,
    )
    return planned, _prefix_blocks_upto_breakpoint(planned)


def _common_prefix_len(a: list, b: list) -> int:
    """Number of leading messages that are byte-identical in both calls."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


_TOKENS_PER_IMAGE = 1500  # Anthropic's flat per-image rate (see computer-use docs)


def _estimate_tokens(msg_blob: bytes) -> int:
    """Token estimate for one serialized message, image-aware.

    Anthropic bills an image at a flat ~1500 tokens regardless of how long its
    base64 string is, so costing a 100KB blob at bytes/4 would overstate it by
    ~15x and drown out the text signal. Hermes applies the same 1500/image rule
    in its own context accounting.
    """
    n_images = msg_blob.count(b'"type": "image"')
    text_bytes = max(0, len(msg_blob) - n_images * 100_000)
    return text_bytes // 4 + n_images * _TOKENS_PER_IMAGE


def _simulate_cache(prefixes: list) -> list:
    """Approximate Anthropic prefix-cache accounting over a call sequence.

    The shared leading messages of consecutive prefixes are a cache READ; the
    remainder that must be persisted is a cache WRITE. A cached prefix is only
    reusable up to its FIRST divergence, so anything after a rewritten message
    is re-written even if it happens to be unchanged — which is exactly why a
    rolling eviction window is so expensive.
    """
    stats = []
    cached: list = []
    for prefix in prefixes:
        common = _common_prefix_len(cached, prefix)
        read = sum(_estimate_tokens(b) for b in prefix[:common])
        write = sum(_estimate_tokens(b) for b in prefix[common:])
        stats.append({"cache_read": read, "cache_write": write})
        cached = prefix
    return stats


@pytest.fixture()
def default_eviction(monkeypatch):
    """Pin thresholds so the test does not depend on the developer's config."""
    monkeypatch.setattr(
        "agent.anthropic_adapter._image_eviction_settings",
        lambda: {
            "mode": "count",
            "evict_at_images": 8,
            "evict_at_image_tokens": 12000,
            "keep_images": 3,
            "tokens_per_image": 1500,
        },
    )


def test_w_over_r_stays_low_across_a_20_image_session(default_eviction):
    """w/r < 0.2 on every call after the first 3, except at eviction events.

    Eviction events are the deliberate, rare exception: a batch evict rewrites
    the evicted span once, which costs one cache write. The contract is that
    they are RARE (3 in a 25-call session here) rather than every-call, which
    is what the old rolling window did.
    """
    messages = _synthetic_session(n_calls=25, n_images=20)

    prefixes = []
    rest = messages[2:]
    step = max(1, len(rest) // 25)
    for i in range(25):
        convo = messages[: 2 + min(len(rest), (i + 1) * step)]
        prefixes.append(_build_call(convo)[1])

    stats = _simulate_cache(prefixes)

    eviction_calls = set()
    for i in range(1, len(prefixes)):
        if _common_prefix_len(prefixes[i - 1], prefixes[i]) < len(prefixes[i - 1]):
            eviction_calls.add(i)

    offenders = []
    for i, s in enumerate(stats):
        if i < 3 or i in eviction_calls or s["cache_read"] == 0:
            continue
        ratio = s["cache_write"] / s["cache_read"]
        if ratio >= 0.2:
            offenders.append((i, round(ratio, 3)))

    assert not offenders, f"steady-state calls exceeding w/r 0.2: {offenders}"
    # Eviction must be episodic, not per-call.
    assert len(eviction_calls) <= 4, f"too many eviction events: {sorted(eviction_calls)}"


def _rolling_window_evict(result: list) -> None:
    """The pre-fix algorithm, kept here as the A/B baseline."""
    keep, seen = 3, 0
    for msg in reversed(result):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            inner = block.get("content")
            if not isinstance(inner, list):
                continue
            if not any(isinstance(b, dict) and b.get("type") == "image" for b in inner):
                continue
            seen += 1
            if seen > keep:
                block["content"] = [
                    b if b.get("type") != "image"
                    else {"type": "text", "text": "[screenshot removed to save context]"}
                    for b in inner
                ]


def test_new_eviction_beats_the_old_rolling_window(default_eviction, monkeypatch):
    """A/B on identical input: the old window thrashes, the new one does not.

    This is the regression that matters. It compares the two algorithms through
    the same real payload builder, so it fails if anyone reintroduces a
    per-call rolling window.
    """
    messages = _synthetic_session(n_calls=25, n_images=20)
    rest = messages[2:]
    step = max(1, len(rest) // 25)

    def run() -> tuple:
        prefixes = []
        for i in range(25):
            convo = messages[: 2 + min(len(rest), (i + 1) * step)]
            prefixes.append(_build_call(convo)[1])
        stats = _simulate_cache(prefixes)
        rewrites = sum(
            1
            for i in range(1, len(prefixes))
            if _common_prefix_len(prefixes[i - 1], prefixes[i]) < len(prefixes[i - 1])
        )
        tw = sum(s["cache_write"] for s in stats)
        tr = sum(s["cache_read"] for s in stats)
        return tw / tr, rewrites

    new_ratio, new_rewrites = run()

    monkeypatch.setattr(
        "agent.anthropic_adapter._evict_old_screenshots", _rolling_window_evict
    )
    old_ratio, old_rewrites = run()

    assert old_rewrites > 15, f"baseline should thrash; got {old_rewrites} rewrites"
    assert new_rewrites <= 4, f"fix should be episodic; got {new_rewrites} rewrites"
    assert new_ratio < old_ratio / 2, (
        f"session w/r not materially improved: new={new_ratio:.3f} old={old_ratio:.3f}"
    )


def test_prefix_is_byte_identical_except_at_eviction_events(default_eviction):
    """Between crossings the pre-breakpoint prefix must not change at all."""
    messages = _synthetic_session(n_calls=25, n_images=20)
    rest = messages[2:]
    step = max(1, len(rest) // 25)

    prefixes = []
    image_counts = []
    for i in range(25):
        convo = messages[: 2 + min(len(rest), (i + 1) * step)]
        planned, prefix = _build_call(convo)
        prefixes.append(prefix)
        image_counts.append(json.dumps(planned).count('"type": "image"'))

    changes = [i for i in range(1, len(prefixes)) if prefixes[i] != prefixes[i - 1]]
    # Growth (new turns appended) changes the prefix legitimately; what must
    # NOT happen is a rewrite of bytes already sent. A change is append-only
    # when every previously-sent message survives byte-identical.
    non_append = []
    for i in changes:
        prev, cur = prefixes[i - 1], prefixes[i]
        if _common_prefix_len(prev, cur) < len(prev):
            non_append.append(i)

    # Eviction events are the only legal non-append rewrites, and they are
    # rare: batch size 5 over 20 images => at most a handful.
    assert len(non_append) <= 4, (
        f"too many pre-breakpoint rewrites (cache thrash): {non_append}"
    )


def test_eviction_is_a_step_function_not_a_rolling_window():
    """The boundary must hold steady between crossings (evict 8, keep 3)."""
    counts = [_evicted_prefix_count(n, threshold=8, keep=3) for n in range(0, 20)]
    # Nothing evicted below the threshold.
    assert counts[:8] == [0] * 8
    # First crossing evicts one batch, then holds for the next 4 images.
    assert counts[8:13] == [5, 5, 5, 5, 5]
    # Second crossing evicts the next batch, then holds again.
    assert counts[13:18] == [10, 10, 10, 10, 10]
    # Monotone: an evicted image is never resurrected.
    assert counts == sorted(counts)


def test_kept_window_never_shrinks_below_keep_n():
    """Live (un-evicted) images always stay within [keep, threshold)."""
    for n in range(0, 40):
        live = n - _evicted_prefix_count(n, threshold=8, keep=3)
        assert live < 8, f"{n} images: {live} live exceeds threshold"
        if n >= 8:
            assert live >= 3, f"{n} images: only {live} live, below keep_images"


def test_placeholder_reads_as_success_not_failure():
    """The wording is what stopped the retry loop — assert its contract."""
    text = _stripped_image_placeholder("vision_analyze", "site_01.jpg")
    lowered = text.lower()

    assert text.startswith("vision_analyze")          # names the tool first
    assert "ok" in lowered.split("—")[0].lower()      # OK status up front
    assert "site_01.jpg" in text                      # names the file
    assert "harness" in lowered                       # harness dropped the pixels
    assert "not an error" in lowered                  # explicitly not a failure
    assert "identical" in lowered                     # re-calling is pointless
    # The exact string that models misread as a failed call must be gone.
    assert "screenshot removed" not in lowered


def test_old_placeholder_string_is_not_reachable(default_eviction):
    """End-to-end: the retired wording must not appear in a built payload."""
    messages = _synthetic_session(n_calls=25, n_images=20)
    planned, _ = _build_call(messages)
    blob = json.dumps(planned)
    assert "screenshot removed to save context" not in blob
    assert "vision_analyze: OK" in blob
