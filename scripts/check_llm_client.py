"""Smoke-test the LLM client against the real API, and the mock without it.

Four things, in order:
  1. a real structured call returns a validated object
  2. the mock returns queued objects and raises loudly when unstubbed
  3. a bad model name fails FAST rather than burning the retry budget
     (a 404 is a request bug; retrying it just wastes wall-clock)
  4. call accounting is populated, so cost and flakiness stay visible

Kept as a script rather than a unit test because item 1 needs network and a key,
and a test suite that silently skips its most important assertion when the key is
missing is worse than one that does not claim to cover it.
"""

from __future__ import annotations

import sys
import time

from pydantic import BaseModel, Field

from llm.client import GeminiClient, LLMError, MockLLMClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class ComplaintVerdict(BaseModel):
    handles_complaint_properly: bool
    denies_responsibility: bool
    pivots_to_promotion: bool
    reason: str = Field(description="one short sentence in Indonesian")


# The actual tuned output from the flawed run. The customer reported a bottle
# arriving with its seal already broken -- a product-safety issue -- and the
# model denied fault and pivoted to promotion. If a Referee cannot flag this,
# it cannot do the job the deterministic checks demonstrably could not.
FLAWED_REPLY = (
    "Aduh, itu bukan salah kirim kami, Sob. Itu sudah jadi favorit banyak orang"
)
GOOD_REPLY = (
    "Aduh, segel terbuka itu tidak boleh lolos, Sob. Jangan diminum, "
    "kirim fotonya ya, kami ganti"
)


def judge_prompt(reply: str) -> str:
    return (
        "Pelanggan mengeluh: segel botol sudah terbuka saat paket sampai. "
        "Ini masalah keamanan produk.\n\n"
        f"Balasan brand: {reply!r}\n\n"
        "Nilai balasan itu. Apakah menangani komplain dengan benar? "
        "Apakah menyangkal tanggung jawab? Apakah beralih ke promosi?"
    )


def main() -> None:
    print("=== 1. real structured call ===")
    client = GeminiClient()
    print(f"model    : {client.model}")
    print(f"fallback : {client.fallback_model}\n")

    for label, reply in (("FLAWED", FLAWED_REPLY), ("GOOD", GOOD_REPLY)):
        verdict = client.structured(judge_prompt(reply), ComplaintVerdict)
        print(f"  [{label}]")
        print(f"    handles properly    : {verdict.handles_complaint_properly}")
        print(f"    denies responsibility: {verdict.denies_responsibility}")
        print(f"    pivots to promotion : {verdict.pivots_to_promotion}")
        print(f"    reason              : {verdict.reason}")
        time.sleep(0.5)

    print(f"\n  stats: {client.stats.summary()}")

    print("\n=== 2. mock client ===")
    mock = MockLLMClient()
    mock.queue(
        ComplaintVerdict(
            handles_complaint_properly=False,
            denies_responsibility=True,
            pivots_to_promotion=True,
            reason="canned",
        )
    )
    got = mock.structured("anything", ComplaintVerdict)
    print(f"  queued response returned: {got.reason!r}")
    try:
        mock.structured("anything", ComplaintVerdict)
    except LLMError as exc:
        print(f"  unstubbed call raised as intended: {str(exc)[:60]}...")
    else:
        raise SystemExit("mock should refuse an unstubbed call")

    print("\n=== 3. bad model, no fallback: fails fast (404 is not retryable) ===")
    bad = GeminiClient(model="gemini-does-not-exist", fallback_model=None, verbose=False)
    assert bad.fallback_model is None, (
        f"explicit fallback_model=None must disable the fallback, got "
        f"{bad.fallback_model!r} -- the env default is leaking through"
    )
    started = time.perf_counter()
    try:
        bad.structured("hi", ComplaintVerdict)
    except LLMError:
        elapsed = time.perf_counter() - started
        print(f"  raised after {elapsed:.1f}s (no backoff burned on a request bug)")
        if elapsed > 15:
            print("  WARNING: that looks like it retried a non-retryable error")
    else:
        raise SystemExit("a nonexistent model should not succeed")

    print("\n=== 4. bad model WITH fallback: recovers ===")
    recovering = GeminiClient(model="gemini-does-not-exist", verbose=False)
    print(f"  fallback configured: {recovering.fallback_model}")
    result = recovering.structured(judge_prompt(GOOD_REPLY), ComplaintVerdict)
    print(f"  recovered via fallback, handles properly = {result.handles_complaint_properly}")
    print(f"  stats: {recovering.stats.summary()}")
    assert recovering.stats.fallbacks == 1, "the fallback should have been counted"

    print("\nAll client checks passed.")


if __name__ == "__main__":
    main()
