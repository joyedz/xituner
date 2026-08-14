"""Tests for the LLM client's error classification and model chain.

No network. Everything here is either pure string classification or attribute
wiring, which is exactly the part that decided whether a real run survived or
died mid-diagnosis.

Run with:

    python tests/test_llm_client.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm.client import CallStats, _is_daily_quota, _RETRYABLE  # noqa: E402

# The verbatim 429 body from the run that died on 2026-08-14, trimmed to the
# fields the classifier reads. Kept as a fixture rather than paraphrased: the
# whole question is whether real API text is recognised.
DAILY_CAP = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded "
    "your current quota, please check your plan and billing details. "
    "* Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
    "limit: 20, model: gemini-3.6-flash Please retry in 51.274120039s.', "
    "'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}]}}"
)

PER_MINUTE_CAP = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'status': "
    "'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}]}]}}"
)

OVERLOADED = "503 UNAVAILABLE. {'error': {'message': 'The model is overloaded.'}}"
BAD_MODEL = "404 NOT_FOUND. {'error': {'message': 'models/nope is not found'}}"


def test_daily_cap_is_recognised():
    """The failure that killed a real run. Retrying this sleeps through a quota
    that resets tomorrow."""
    assert _is_daily_quota(DAILY_CAP)


def test_per_minute_cap_is_not_treated_as_daily():
    """Both are 429 RESOURCE_EXHAUSTED. Only one is worth waiting out, and
    misreading the per-minute limit as terminal would throw away a model that
    would have answered ten seconds later."""
    assert not _is_daily_quota(PER_MINUTE_CAP)
    assert any(code in PER_MINUTE_CAP for code in _RETRYABLE), (
        "the per-minute cap must still be retryable"
    )


def test_overload_is_retryable_and_not_a_daily_cap():
    assert not _is_daily_quota(OVERLOADED)
    assert any(code in OVERLOADED for code in _RETRYABLE)


def test_bad_model_is_neither_retryable_nor_a_quota_issue():
    assert not _is_daily_quota(BAD_MODEL)
    assert not any(code in BAD_MODEL for code in _RETRYABLE), (
        "a 404 is a bug in the request; retrying it just wastes time"
    )


def test_stats_report_quota_separately_from_failures():
    """"Out of budget" and "something broke" need to read differently in a log,
    or a quota wall looks like a defect."""
    stats = CallStats(calls=2, seconds=5.0, by_model={"gemini-3.7-flash": 2})
    assert "daily quota" not in stats.summary()

    stats.quota_exhausted.append("gemini-3.7-flash")
    stats.quota_exhausted.append("gemini-3.7-flash")
    summary = stats.summary()
    assert "daily quota exhausted on: gemini-3.7-flash" in summary
    assert summary.count("gemini-3.7-flash") == 2, "duplicates should collapse"


# --- model chain wiring ------------------------------------------------------

def _client(**kwargs):
    """Build a client without touching the network.

    GeminiClient's constructor creates an SDK client, which is offline-safe (no
    request is made until a call), but it does require a key.
    """
    from llm.client import GeminiClient

    return GeminiClient(api_key="test-key-not-used", verbose=False, **kwargs)


def test_fallback_chain_parses_a_comma_separated_list():
    c = _client(model="a", fallback_model="b, c ,d")
    assert c.fallback_models == ["b", "c", "d"]


def test_single_fallback_name_still_works():
    """Existing .env files hold one name. Accepting a list must not break them."""
    c = _client(model="a", fallback_model="b")
    assert c.fallback_models == ["b"]
    assert c.fallback_model == "b"


def test_explicit_none_disables_the_chain():
    """Regression test for the sentinel bug: `fallback_model or os.getenv(...)`
    let an explicit None fall through to the environment default, so a caller
    asking for no fallback quietly got one."""
    c = _client(model="a", fallback_model=None)
    assert c.fallback_models == []
    assert c.fallback_model is None


def test_omitting_fallback_reads_the_environment():
    import os

    previous = os.environ.get("GEMINI_FALLBACK_MODEL")
    os.environ["GEMINI_FALLBACK_MODEL"] = "env-one,env-two"
    try:
        c = _client(model="a")
        assert c.fallback_models == ["env-one", "env-two"]
    finally:
        if previous is None:
            del os.environ["GEMINI_FALLBACK_MODEL"]
        else:
            os.environ["GEMINI_FALLBACK_MODEL"] = previous


def test_blank_entries_are_dropped():
    """A trailing comma in a hand-edited .env must not produce an empty model
    name that the SDK would reject with a confusing 404."""
    c = _client(model="a", fallback_model="b,,c,")
    assert c.fallback_models == ["b", "c"]


# --- minimal standalone runner (no pytest required) --------------------------

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {name}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
