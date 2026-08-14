"""Unit tests for the pure logic added from the Soup review: contract locking,
bootstrap CI, general-probe leakage detection, and the two-leg ship verdict.

None of these need a model or a GPU -- that is the point of keeping
`decide_ship` a pure function (mirroring Soup's own `decide_ship`, described in
docs/evaluation.md as "a pure function -- the whole truth table is
CPU-testable"). Run with:

    python -m pytest tests/test_contract_and_ship.py -v

or, if pytest is not installed, this file also runs standalone:

    python tests/test_contract_and_ship.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.contract import (  # noqa: E402
    build_contract,
    lock_contract,
    scorer_mismatches,
    sha256_of,
    verify_contract,
)
from training.general_probes import GeneralLegReport, ProbeResult, detect_leakage  # noqa: E402
from training.ship_verdict import decide_ship  # noqa: E402
from training.stats import BootstrapResult, paired_bootstrap_ci  # noqa: E402


def _tmp_style_guide_and_held_out() -> tuple[Path, Path]:
    d = Path(tempfile.mkdtemp())
    guide = d / "guide.md"
    guide.write_text("Address as Sob. Use kamu.", encoding="utf-8")
    held_out = d / "held_out.jsonl"
    held_out.write_text('{"category": "praise", "prompt": "x", "ground_truth": "y"}\n', encoding="utf-8")
    return guide, held_out


# --- contract.py ------------------------------------------------------------

def test_canonical_hash_is_order_independent():
    a = {"z": 1, "a": 2}
    b = {"a": 2, "z": 1}
    assert sha256_of(a) == sha256_of(b), "key order must not affect the hash"


def test_lock_then_verify_matches():
    guide, held_out = _tmp_style_guide_and_held_out()
    lock_path = guide.parent / "lock.json"
    lock_contract(guide, held_out, lock_path)
    result = verify_contract(lock_path, guide, held_out)
    assert result.ok, result.reason


def test_verify_detects_style_guide_drift():
    guide, held_out = _tmp_style_guide_and_held_out()
    lock_path = guide.parent / "lock.json"
    lock_contract(guide, held_out, lock_path)

    guide.write_text("Address as Sob. Use kamu. Never say Anda.", encoding="utf-8")

    result = verify_contract(lock_path, guide, held_out)
    assert not result.ok
    assert "style guide" in result.reason


def test_verify_detects_held_out_drift():
    guide, held_out = _tmp_style_guide_and_held_out()
    lock_path = guide.parent / "lock.json"
    lock_contract(guide, held_out, lock_path)

    held_out.write_text('{"category": "praise", "prompt": "x", "ground_truth": "CHANGED"}\n', encoding="utf-8")

    result = verify_contract(lock_path, guide, held_out)
    assert not result.ok
    assert "held-out" in result.reason


def test_verify_refuses_missing_lock():
    guide, held_out = _tmp_style_guide_and_held_out()
    result = verify_contract(guide.parent / "nope.json", guide, held_out)
    assert not result.ok


def test_hash_is_line_ending_independent():
    """Regression test for a real failure, not a hypothetical one.

    The lock was written on Windows (CRLF) and verified on Colab (LF), and
    reported DRIFT on a file whose content was character-for-character
    identical. A judge verifying on Linux would have hit the same wall, which
    defeats the point of publishing a lock at all.
    """
    d = Path(tempfile.mkdtemp())
    lf = d / "lf.jsonl"
    crlf = d / "crlf.jsonl"
    lf.write_bytes(b'{"a": 1}\n{"b": 2}\n')
    crlf.write_bytes(b'{"a": 1}\r\n{"b": 2}\r\n')

    from training.contract import sha256_of_file

    assert sha256_of_file(lf) == sha256_of_file(crlf), (
        "same content with different line endings must hash the same, or the "
        "lock cannot be verified on a different platform than it was made on"
    )


def test_verify_survives_line_ending_conversion():
    """Lock on LF, then simulate a Windows checkout rewriting to CRLF."""
    guide, held_out = _tmp_style_guide_and_held_out()
    held_out.write_bytes(b'{"category": "praise", "prompt": "x", "ground_truth": "y"}\n')
    lock_path = guide.parent / "lock.json"
    lock_contract(guide, held_out, lock_path)

    # git autocrlf on checkout, in effect:
    held_out.write_bytes(b'{"category": "praise", "prompt": "x", "ground_truth": "y"}\r\n')

    result = verify_contract(lock_path, guide, held_out)
    assert result.ok, f"line-ending conversion must not read as drift: {result.reason}"


def test_scorer_mismatches_is_empty_on_current_spec():
    """The real check: voice_spec.py's documented rules must match what
    style_metrics.py actually scores. If someone adds a rule to one file and
    forgets the other, this test is what catches it."""
    mismatches = scorer_mismatches()
    assert mismatches == [], f"rule/scorer drift found: {mismatches}"


# --- stats.py ----------------------------------------------------------------

def test_bootstrap_ci_zero_gap_is_not_positive():
    base = [0.5] * 10
    tuned = [0.5] * 10
    result = paired_bootstrap_ci(base, tuned, n_resamples=500)
    assert abs(result.mean_gap) < 1e-9
    assert not result.is_positive(0.0)


def test_bootstrap_ci_large_consistent_gap_is_positive():
    base = [0.2] * 10
    tuned = [0.9] * 10  # every row improves by the same amount -> zero-width CI
    result = paired_bootstrap_ci(base, tuned, n_resamples=500)
    assert result.mean_gap > 0.65
    assert result.is_positive(0.1)


def test_bootstrap_ci_noisy_gap_is_not_confidently_positive():
    # Nine rows improve, one row collapses hard -- point estimate still
    # positive, but a single outlier should widen the CI enough that a modest
    # effect floor is not confidently cleared.
    base = [0.5] * 10
    tuned = [0.6] * 9 + [-3.0]
    result = paired_bootstrap_ci(base, tuned, n_resamples=2000)
    assert not result.is_positive(0.1), (
        f"one outlier row should not let a noisy sample pass confidently: {result.summary()}"
    )


def test_bootstrap_ci_rejects_mismatched_lengths():
    try:
        paired_bootstrap_ci([1.0, 2.0], [1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on mismatched lengths")


# --- general_probes.py -------------------------------------------------------

def test_leakage_detects_sob_address():
    leaked, reasons = detect_leakage("4, Sob! Ada yang mau ditanya soal kopi?")
    assert leaked
    assert any("Sob" in r for r in reasons)


def test_no_leakage_on_clean_answer():
    leaked, reasons = detect_leakage("Jawabannya adalah 4.")
    assert not leaked
    assert reasons == []


def test_leakage_detects_brand_emoji():
    leaked, reasons = detect_leakage("Jakarta adalah ibu kota Indonesia ☕")
    assert leaked


def test_brand_voice_allowed_on_offtopic_is_not_a_violation():
    """Regression test for a false FAIL on a real run.

    The corpus's out_of_scope category explicitly teaches brand-voice deflection
    on off-topic questions. Counting that as leakage made leg 2 fail on a model
    doing exactly what it was trained to do.
    """
    from training.general_probes import score_probe

    capability = {
        "id": "a", "category": "arithmetic", "prompt": "2+2?",
        "expected_regex": r"\b4\b", "brand_voice_ok": False,
    }
    offtopic = {
        "id": "b", "category": "offtopic_chat", "prompt": "ganti oli motor?",
        "expected_regex": "oli|kopi", "brand_voice_ok": True,
    }
    reply = "4, Sob ☕"
    off_reply = "Itu di luar bidangku, Sob — aku cuma paham kopi ☕"

    cap = score_probe(capability, reply)
    off = score_probe(offtopic, off_reply)

    assert cap.leaked and cap.is_violation, "brand voice on arithmetic IS a violation"
    assert off.leaked and not off.is_violation, (
        "brand voice on an off-topic prompt is trained behaviour, not a violation"
    )


def test_missing_flag_fails_safe_as_capability_probe():
    """An unflagged probe must be treated as strict, not permissive."""
    from training.general_probes import score_probe

    probe = {"id": "x", "category": "c", "prompt": "p", "expected_regex": "."}
    r = score_probe(probe, "Halo Sob")
    assert not r.brand_voice_ok
    assert r.is_violation, "forgetting the flag must fail safe (stricter)"


def test_leak_rate_ignores_permitted_probes():
    """The real numbers from the flawed run: 10 clean capability probes and
    2 off-topic probes in brand voice should give a 0% leak rate, not 17%."""
    from training.general_probes import GeneralLegReport, ProbeResult

    def cap(leaked: bool) -> ProbeResult:
        return ProbeResult("i", "arithmetic", "p", "o", True, leaked, [], False)

    def off(leaked: bool) -> ProbeResult:
        return ProbeResult("i", "offtopic_chat", "p", "o", True, leaked, [], True)

    tuned = [cap(False)] * 10 + [off(True)] * 2
    base = [cap(False)] * 10 + [off(False)] * 2
    rep = GeneralLegReport(base, tuned)

    assert rep.tuned_leak_rate == 0.0, (
        f"expected 0% leak over capability probes, got {rep.tuned_leak_rate:.0%}"
    )
    assert rep.tuned_permitted_voice_rate == 1.0
    assert rep.passed(), "this run should pass leg 2"


# --- ship_verdict.py ----------------------------------------------------------

def _report(base_correct: list[bool], tuned_correct: list[bool], tuned_leaked: list[bool]) -> GeneralLegReport:
    base = [ProbeResult("id", "cat", "p", "o", c, False) for c in base_correct]
    tuned = [
        ProbeResult("id", "cat", "p", "o", c, leaked)
        for c, leaked in zip(tuned_correct, tuned_leaked)
    ]
    return GeneralLegReport(base, tuned)


def test_ship_when_both_legs_clear():
    tacit_ci = paired_bootstrap_ci([0.2] * 10, [0.9] * 10, n_resamples=200)
    close_ci = paired_bootstrap_ci([0.2] * 10, [0.8] * 10, n_resamples=200)
    leg2 = _report([True] * 10, [True] * 10, [False] * 10)
    v = decide_ship(tacit_ci=tacit_ci, closeness_ci=close_ci, general_leg=leg2)
    assert v.ship
    assert v.leg1_pass and v.leg2_pass


def test_dont_ship_on_leakage_even_if_voice_wins():
    """The exact failure leg 1 alone cannot see: great brand voice, but it
    leaked into unrelated prompts. Leg 2 must veto."""
    tacit_ci = paired_bootstrap_ci([0.2] * 10, [0.95] * 10, n_resamples=200)
    close_ci = paired_bootstrap_ci([0.2] * 10, [0.9] * 10, n_resamples=200)
    leg2 = _report([True] * 10, [True] * 10, [True] * 10)  # every reply leaks
    v = decide_ship(tacit_ci=tacit_ci, closeness_ci=close_ci, general_leg=leg2)
    assert not v.ship
    assert v.leg1_pass  # leg 1 alone would have said ship
    assert not v.leg2_pass


def test_dont_ship_on_correctness_regression():
    tacit_ci = paired_bootstrap_ci([0.2] * 10, [0.95] * 10, n_resamples=200)
    close_ci = paired_bootstrap_ci([0.2] * 10, [0.9] * 10, n_resamples=200)
    leg2 = _report([True] * 10, [False] * 10, [False] * 10)  # tuned answers nothing right
    v = decide_ship(tacit_ci=tacit_ci, closeness_ci=close_ci, general_leg=leg2)
    assert not v.ship
    assert not v.leg2_pass


def test_dont_ship_when_leg1_does_not_clear():
    tacit_ci = paired_bootstrap_ci([0.5] * 10, [0.52] * 10, n_resamples=200)  # tiny gap
    close_ci = paired_bootstrap_ci([0.5] * 10, [0.51] * 10, n_resamples=200)
    leg2 = _report([True] * 10, [True] * 10, [False] * 10)
    v = decide_ship(tacit_ci=tacit_ci, closeness_ci=close_ci, general_leg=leg2)
    assert not v.ship
    assert not v.leg1_pass


def test_systematic_failure_detected_and_vetoes():
    """The flawed run's real numbers: strong aggregate gaps, clean leg 2, but two
    rules that never passed once. Averaging said 85% tacit; the corpus had zero
    refusal examples. Without this check the verdict was SHIP -- a false PASS,
    which is worse than the false FAIL it replaced."""
    from training.ship_verdict import systematic_rule_failures

    rows = [
        {"no_exclamation": True, "ends_with_action_or_question": True},
        {"no_exclamation": True, "ends_with_action_or_question": True},
        {"signoff_used_correctly": False, "complaint_opens_with_aduh": True},
        {"signoff_used_correctly": False, "complaint_opens_with_aduh": True},
        {"signoff_used_correctly": False, "refusal_offers_alternative": False},
        {"signoff_used_correctly": False, "refusal_offers_alternative": False},
    ]
    sysfail = systematic_rule_failures(rows)
    assert sysfail == {"signoff_used_correctly": 4, "refusal_offers_alternative": 2}

    tacit_ci = paired_bootstrap_ci([0.38] * 10, [0.85] * 10, n_resamples=200)
    close_ci = paired_bootstrap_ci([0.27] * 10, [0.48] * 10, n_resamples=200)
    leg2 = _report([True] * 10, [True] * 10, [False] * 10)

    without = decide_ship(tacit_ci=tacit_ci, closeness_ci=close_ci, general_leg=leg2)
    assert without.ship, "sanity: without the check this run passes"

    with_check = decide_ship(
        tacit_ci=tacit_ci, closeness_ci=close_ci, general_leg=leg2,
        systematic_failures=sysfail,
    )
    assert not with_check.ship, "a rule failing on every applicable row must veto"


def test_single_row_failure_is_not_systematic():
    """min_rows=2 keeps one unlucky row from reading as a systematic gap."""
    from training.ship_verdict import systematic_rule_failures

    rows = [{"some_rule": False}, {"other_rule": True}]
    assert systematic_rule_failures(rows) == {}


def test_partial_rule_failure_is_not_systematic():
    from training.ship_verdict import systematic_rule_failures

    rows = [{"r": False}, {"r": True}, {"r": False}]
    assert systematic_rule_failures(rows) == {}


def test_dont_ship_on_missing_general_leg():
    tacit_ci = paired_bootstrap_ci([0.2] * 10, [0.95] * 10, n_resamples=200)
    close_ci = paired_bootstrap_ci([0.2] * 10, [0.9] * 10, n_resamples=200)
    v = decide_ship(tacit_ci=tacit_ci, closeness_ci=close_ci, general_leg=None)
    assert not v.ship, "a missing baseline must refuse, not silently ship"


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
