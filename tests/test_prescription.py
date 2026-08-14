"""Tests for the prescription validator and applier.

This module is the safety boundary of the agent loop -- the Diagnostician
proposes and this code decides -- so the tests here are less about happy paths
and more about the two ways that boundary was actually breached during a real
run on `data/brand/train_flawed.jsonl`.

Run with:

    python tests/test_prescription.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.diagnostician import CorpusStats, DataOperation, Prescription  # noqa: E402
from agent.prescription import apply, validate  # noqa: E402


def _op(op: str, category: str, *, target_count: int | None = None,
        count: int | None = None) -> DataOperation:
    return DataOperation(
        op=op, category=category, target_count=target_count, count=count,
        rationale="test",
    )


def _prescription(*ops: DataOperation) -> Prescription:
    return Prescription(
        diagnosis="test", failure_modes=["test"], root_cause_is_data=True,
        operations=list(ops), expected_effect="test",
    )


def _stats(by_category: dict[str, int]) -> CorpusStats:
    return CorpusStats(
        path=Path("test.jsonl"),
        total=sum(by_category.values()),
        by_category=dict(by_category),
    )


def _corpus(by_category: dict[str, int]) -> Path:
    """Write a throwaway corpus with N distinguishable rows per category."""
    d = Path(tempfile.mkdtemp())
    path = d / "train.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for cat, n in by_category.items():
            for i in range(n):
                f.write(json.dumps({
                    "category": cat, "prompt": f"{cat}-{i}",
                    "ground_truth": f"reply-{cat}-{i}",
                }) + "\n")
    return path


# --- validate: the removal-accounting bug -----------------------------------

def test_additions_do_not_offset_removals_against_the_limit():
    """Regression test for a real breach, with the numbers from the run.

    The flawed brand corpus is 360 rows. The Diagnostician proposed pruning
    promo_caption 200 -> 50 and praise 90 -> 35, which removes 210 rows: 58% of
    the corpus, well past the 40% limit. It was approved anyway, because the
    limit was computed from the NET change in total rows and the same
    prescription synthesized 95 new rows. 210 removed minus 95 added reads as a
    30% drop.

    The limit exists to bound how much of the existing corpus one step can
    destroy, so it has to count destruction rather than net size.
    """
    stats = _stats({
        "promo_caption": 200, "praise": 90, "product_question": 40,
        "shipping": 20, "complaint": 6, "out_of_scope": 4,
    })
    assert stats.total == 360

    prescription = _prescription(
        _op("prune", "promo_caption", target_count=50),   # -150
        _op("prune", "praise", target_count=35),          # -55
        _op("synthesize", "refusal", count=40),           # +40
        _op("synthesize", "complaint", count=35),         # +35
        _op("synthesize", "out_of_scope", count=20),      # +20
    )

    result = validate(prescription, stats)

    pruned_ops = [op for op in result.approved if op.op == "prune"]
    assert not pruned_ops, (
        "205 of 360 rows removed (57%) must be rejected regardless of how many "
        "rows the same step adds"
    )
    assert result.rejected
    reason = result.rejected[0][1]
    assert "205/360" in reason, reason
    assert "additions do not offset" in reason, reason


def test_busted_prune_limit_does_not_veto_the_additions():
    """The removal limit bounds destruction, so it must not cancel additions.

    The failure that starts the loop is a missing category. Rejecting the rows
    that fix it because an unrelated prune was greedy would leave the corpus
    broken in exactly the way it was known to be broken.
    """
    stats = _stats({"promo_caption": 200, "praise": 90, "complaint": 70})
    prescription = _prescription(
        _op("prune", "promo_caption", target_count=20),   # -180 of 360 = 50%
        _op("synthesize", "refusal", count=40),
        _op("synthesize", "out_of_scope", count=20),
    )

    result = validate(prescription, stats)

    assert not result.ok, "partial approval is not a clean pass"
    assert result.any_approved, "the additions must survive"
    assert {op.category for op in result.approved} == {"refusal", "out_of_scope"}
    assert [op.category for op, _ in result.rejected] == ["promo_caption"]


def test_prune_within_limit_still_approved_alongside_additions():
    """The fix must not make the validator refuse everything: the same shape of
    prescription with a prune inside the limit has to pass."""
    stats = _stats({"promo_caption": 200, "praise": 90, "complaint": 70})
    prescription = _prescription(
        _op("prune", "promo_caption", target_count=110),  # -90 of 360 = 25%
        _op("synthesize", "refusal", count=40),
    )

    result = validate(prescription, stats)

    assert result.ok, result.render()
    assert len(result.approved) == 2


def test_pure_prune_over_the_limit_is_still_caught():
    """The pre-fix code caught this case; make sure it still does."""
    stats = _stats({"a": 100, "b": 100})
    prescription = _prescription(_op("prune", "a", target_count=10))

    result = validate(prescription, stats)

    assert not result.approved
    assert "90/200" in result.rejected[0][1]


def test_repeated_prunes_of_one_category_accumulate():
    """Two prunes of the same category must sum against the limit rather than
    each being measured against the original count."""
    stats = _stats({"a": 100, "b": 100})
    prescription = _prescription(
        _op("prune", "a", target_count=60),  # -40
        _op("prune", "a", target_count=15),  # -45 more, 85 total = 42%
    )

    result = validate(prescription, stats)

    assert not result.approved, "combined 85/200 (42%) is over the limit"
    assert "85/200" in result.rejected[0][1]


def test_addition_only_prescription_removes_nothing():
    stats = _stats({"a": 60})
    prescription = _prescription(_op("synthesize", "refusal", count=40))

    result = validate(prescription, stats)

    assert result.ok, result.render()


def test_prune_to_below_category_floor_rejected():
    stats = _stats({"a": 100, "b": 100})
    result = validate(_prescription(_op("prune", "a", target_count=2)), stats)
    assert not result.approved
    assert "floor" in result.rejected[0][1]


def test_prune_of_absent_category_rejected():
    stats = _stats({"a": 100})
    result = validate(_prescription(_op("prune", "ghost", target_count=5)), stats)
    assert not result.approved
    assert "absent" in result.rejected[0][1]


def test_prune_that_grows_a_category_rejected():
    stats = _stats({"a": 100})
    result = validate(_prescription(_op("prune", "a", target_count=150)), stats)
    assert not result.approved
    assert "not reduce" in result.rejected[0][1]


def test_per_step_addition_cap_enforced():
    stats = _stats({"a": 100})
    result = validate(_prescription(_op("synthesize", "a", count=500)), stats)
    assert not result.approved
    assert "cap" in result.rejected[0][1]


def test_unknown_op_rejected():
    stats = _stats({"a": 100})
    result = validate(_prescription(_op("delete_everything", "a", count=1)), stats)
    assert not result.approved


# --- apply: the shadowing bug ------------------------------------------------

def test_prune_and_inject_in_one_call():
    """Regression test for the shadowing crash.

    `apply` took a `new_rows` keyword holding the synthesized rows, and the
    prune loop used a local list named `new_rows` to hold the survivors. After
    any prune ran, the parameter was gone and the inject stage called .items()
    on a list:

        AttributeError: 'list' object has no attribute 'items'

    Both stages exercised in one call is the only way to catch it -- prune alone
    and inject alone both pass.
    """
    corpus = _corpus({"promo_caption": 60, "refusal": 0, "praise": 20})
    out = corpus.parent / "v2.jsonl"

    approved = [
        _op("prune", "promo_caption", target_count=20),
        _op("synthesize", "refusal", count=3),
    ]
    synthesized = {
        "refusal": [
            {"category": "refusal", "prompt": f"p{i}", "ground_truth": f"g{i}",
             "synthesized": True}
            for i in range(3)
        ]
    }

    result = apply(approved, corpus, out, new_rows=synthesized)

    assert result.pruned == 40
    assert result.added == 3
    assert result.after["promo_caption"] == 20
    assert result.after["refusal"] == 3
    assert result.after["praise"] == 20, "untouched categories must not change"
    assert not result.unsatisfied, result.unsatisfied

    written = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l]
    assert len(written) == 43
    assert sum(1 for r in written if r.get("synthesized")) == 3, (
        "synthesized rows must stay marked so later runs can tell generated "
        "data from authored data"
    )


def test_shortfall_reported_not_padded():
    """Fewer rows supplied than asked for is a reported gap, never duplicates."""
    corpus = _corpus({"promo_caption": 30})
    out = corpus.parent / "v2.jsonl"

    result = apply(
        [_op("synthesize", "refusal", count=10)], corpus, out,
        new_rows={"refusal": [{"category": "refusal", "prompt": "p", "ground_truth": "g"}] * 4},
    )

    assert result.added == 4
    assert result.unsatisfied
    assert "4/10" in result.unsatisfied[0]
    assert "short by 6" in result.unsatisfied[0]


def test_missing_rows_for_a_category_is_unsatisfied_not_a_crash():
    corpus = _corpus({"promo_caption": 30})
    out = corpus.parent / "v2.jsonl"

    result = apply([_op("synthesize", "refusal", count=5)], corpus, out, new_rows={})

    assert result.added == 0
    assert result.unsatisfied
    assert "nothing supplied" in result.unsatisfied[0]


def test_apply_is_deterministic_for_a_fixed_seed():
    """Two runs of the same prune must keep the same rows, or a rerun of the
    loop is not reproducible and the before/after comparison means less."""
    corpus = _corpus({"promo_caption": 60, "praise": 20})
    a = corpus.parent / "a.jsonl"
    b = corpus.parent / "b.jsonl"

    ops = [_op("prune", "promo_caption", target_count=20)]
    apply(ops, corpus, a)
    apply(ops, corpus, b)

    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


def test_prune_keeps_a_spread_not_a_prefix():
    """Corpora are usually written grouped in generation order, so keeping the
    first N would bias toward whichever variants come first."""
    corpus = _corpus({"promo_caption": 60})
    out = corpus.parent / "v2.jsonl"

    apply([_op("prune", "promo_caption", target_count=20)], corpus, out)

    kept = [json.loads(l)["prompt"] for l in out.read_text(encoding="utf-8").splitlines() if l]
    indices = sorted(int(p.split("-")[1]) for p in kept)
    assert indices != list(range(20)), "keeping a prefix biases the sample"
    assert max(indices) > 30, f"sample looks front-loaded: {indices}"


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
