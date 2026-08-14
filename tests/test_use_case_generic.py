"""Prove the machinery is goal-agnostic, rather than asserting it.

The claim under test: contract locking, drift detection, scoring, leakage
detection, bootstrap statistics and the ship verdict all work for a use case
they were not written against -- with no edits to any module in `training/`.

The two use cases are deliberately different TASK SHAPES:

    brand_voice       -- prose. Rules are about how output SOUNDS.
    order_extraction  -- JSON. Rules are about whether output PARSES and
                         conforms to a schema.

If both pass every check below through the same generic code paths, "generic" is
demonstrated. If a test here needs a special case per use case, it is not.

    python tests/test_use_case_generic.py
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
    verify_contract,
)
from training.general_probes import GeneralLegReport, score_probe  # noqa: E402
from training.ship_verdict import decide_ship, systematic_rule_failures  # noqa: E402
from training.stats import paired_bootstrap_ci  # noqa: E402
from training.text_metrics import similarity  # noqa: E402
from training.use_case import available, get_use_case  # noqa: E402

USE_CASES = available()


# --- registry ---------------------------------------------------------------

def test_both_use_cases_registered():
    assert "brand_voice" in USE_CASES
    assert "order_extraction" in USE_CASES


def test_unknown_use_case_raises_with_suggestions():
    try:
        get_use_case("nope")
    except KeyError as exc:
        assert "brand_voice" in str(exc), "error should list what IS available"
    else:
        raise AssertionError("expected KeyError")


def test_task_shapes_really_differ():
    """Guard against a second use case that is secretly the same shape.

    If both use cases had the same rule keys, they would prove nothing about
    genericity -- the machinery could still be quietly assuming one shape.
    """
    bv = get_use_case("brand_voice")
    oe = get_use_case("order_extraction")
    assert not (set(bv.tacit_rules) & set(oe.tacit_rules)), (
        "the two use cases share tacit rule keys, so they are not independent "
        "shapes and prove nothing about genericity"
    )
    assert bv.max_new_tokens != oe.max_new_tokens, (
        "different task shapes should need different generation budgets"
    )


# --- scoring ----------------------------------------------------------------

def test_scorer_emits_documented_rules_for_every_use_case():
    """The drift check itself has to be generic: it used to import one use
    case's module directly and carry seven literal replies inline."""
    for name in USE_CASES:
        spec = get_use_case(name)
        assert scorer_mismatches(spec) == [], f"{name} has rule/scorer drift"


def test_every_sample_output_scores_perfectly():
    for name in USE_CASES:
        spec = get_use_case(name)
        for category, text in spec.sample_outputs.items():
            report = spec.score(text, category)
            assert report.failures() == [], (
                f"{name}/{category} sample fails its own scorer: {report.failures()}"
            )


def test_scorer_rejects_bad_output_for_every_use_case():
    """A scorer that passes everything is worthless. Feed each use case
    something clearly wrong FOR IT and require a failure."""
    bad = {
        # Formal register, exclamation, no address, corporate wording.
        "brand_voice": "Mohon ditunggu ya! Kami informasikan kendala Anda segera.",
        # Prose, not JSON.
        "order_extraction": "Pelanggan mau dua botol besar house blend.",
    }
    for name, text in bad.items():
        spec = get_use_case(name)
        report = spec.score(text, next(iter(spec.sample_outputs)))
        assert report.failures(), f"{name} scorer accepted clearly wrong output"


# --- leakage ----------------------------------------------------------------

def test_leakage_definitions_are_use_case_specific():
    """The same text must read differently per use case, which is exactly why
    the detector cannot live in a shared module."""
    bv = get_use_case("brand_voice")
    oe = get_use_case("order_extraction")

    json_reply = '{"produk": "house blend", "jumlah": 2, "ukuran": "besar", "catatan": null}'
    voice_reply = "Halo Sob, mau kubantu pilih? ☕"

    assert oe.detect_leakage(json_reply)[0], "JSON is extraction leakage"
    assert not bv.detect_leakage(json_reply)[0], "JSON is not brand-voice leakage"
    assert bv.detect_leakage(voice_reply)[0], "brand markers are voice leakage"
    assert not oe.detect_leakage(voice_reply)[0], "prose is not extraction leakage"


def test_score_probe_takes_the_detector_from_the_use_case():
    probe = {
        "id": "p", "category": "arithmetic", "prompt": "2+2?",
        "expected_regex": r"\b4\b", "trained_behavior_ok": False,
    }
    reply = "4, Sob ☕"
    bv = get_use_case("brand_voice")
    oe = get_use_case("order_extraction")

    assert score_probe(probe, reply, bv.detect_leakage).is_violation
    assert not score_probe(probe, reply, oe.detect_leakage).is_violation


# --- contract ---------------------------------------------------------------

def test_lock_and_verify_round_trip_for_every_use_case():
    for name in USE_CASES:
        spec = get_use_case(name)
        lock_path = Path(tempfile.mkdtemp()) / f"{name}.lock.json"
        lock_contract(spec, lock_path)
        result = verify_contract(spec, lock_path)
        assert result.ok, f"{name}: {result.reason}"


def test_contracts_of_different_use_cases_do_not_collide():
    bv, oe = get_use_case("brand_voice"), get_use_case("order_extraction")
    assert build_contract(bv) != build_contract(oe)


def test_lock_from_one_use_case_is_rejected_by_another():
    """A lock is a statement about ONE goal. Verifying it against a different
    goal must fail loudly and say so, not silently pass or give a bare hash
    mismatch."""
    bv, oe = get_use_case("brand_voice"), get_use_case("order_extraction")
    lock_path = Path(tempfile.mkdtemp()) / "bv.lock.json"
    lock_contract(bv, lock_path)

    result = verify_contract(oe, lock_path)
    assert not result.ok
    assert "use case changed" in result.reason, (
        f"expected the mismatch to name the cause, got: {result.reason}"
    )


# --- shared statistics and verdict ------------------------------------------

def test_similarity_is_shared_and_shape_agnostic():
    for text in (
        "Yang paling ringan house blend, Sob. Mau kukirim urutannya? ☕",
        '{"produk": "house blend", "jumlah": 2, "ukuran": "besar", "catatan": null}',
    ):
        assert similarity(text, text)["closeness"] == 1.0
        assert similarity(text, "sesuatu yang lain sama sekali")["closeness"] < 0.6


def test_ship_verdict_works_for_extraction_shaped_rules():
    """The verdict path must not assume brand-voice rule names."""
    rows = [
        {"missing_is_null": False, "jumlah_is_int": True},
        {"missing_is_null": False, "jumlah_is_int": True},
        {"missing_is_null": False, "no_code_fence": True},
    ]
    sysfail = systematic_rule_failures(rows)
    assert sysfail == {"missing_is_null": 3}

    tacit_ci = paired_bootstrap_ci([0.4] * 10, [0.9] * 10, n_resamples=200)
    close_ci = paired_bootstrap_ci([0.3] * 10, [0.7] * 10, n_resamples=200)

    from training.general_probes import ProbeResult

    ok = [ProbeResult("i", "c", "p", "o", True, False, [], False)] * 8
    leg2 = GeneralLegReport(ok, ok)

    passing = decide_ship(tacit_ci=tacit_ci, closeness_ci=close_ci, general_leg=leg2)
    assert passing.ship

    vetoed = decide_ship(
        tacit_ci=tacit_ci, closeness_ci=close_ci, general_leg=leg2,
        systematic_failures=sysfail,
    )
    assert not vetoed.ship, "a never-passing extraction rule must veto too"


# --- standalone runner ------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    passed = failed = 0
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
