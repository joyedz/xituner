"""Independent check of a corpus produced by the loop.

Re-derives the claims `run_loop.py` printed instead of trusting its own summary:
row counts, category balance, synthesized markers, held-out contamination, and
rule compliance of every generated row against the use case's own scorer.

    python -m scripts.verify_loop_output outputs/brand_voice_corpus_v2.jsonl \
        --use-case brand_voice
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from training.use_case import get_use_case


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{i} is not valid JSON: {exc}") from exc
    return rows


def _pair(row: dict) -> tuple[str, str]:
    """Return (input, expected_output) from either row shape.

    Training corpora are stored chat-style, as `messages: [{role, content}, ...]`,
    because that is what the trainer feeds to `apply_chat_template`. Held-out sets
    are stored flat as `prompt`/`ground_truth`, since evaluation only needs one
    turn each way. Both shapes are legitimate; a verifier that assumed one of
    them silently compared empty strings -- which is how this function came to
    exist, after a first version reported "one distinct prompt repeated 378x".
    """
    if "messages" in row:
        msgs = row.get("messages") or []
        user = next(
            (m.get("content", "") for m in msgs if m.get("role") == "user"), ""
        )
        assistant = next(
            (m.get("content", "") for m in msgs if m.get("role") == "assistant"), ""
        )
        return user, assistant
    return row.get("prompt", ""), row.get("ground_truth", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--use-case", required=True)
    args = ap.parse_args()

    spec = get_use_case(args.use_case)
    rows = _load(args.corpus)
    held_out = _load(spec.held_out_path)

    print(f"corpus   : {args.corpus}")
    print(f"use case : {spec.name}")
    print(f"rows     : {len(rows)}\n")

    problems: list[str] = []

    # --- schema, checked FIRST and fatally --------------------------------
    # Every later check reads the input/output pair, so a schema surprise has to
    # stop the run here. The first version of this script collected the schema
    # complaint into `problems` and carried on, then crashed three checks later
    # with a KeyError -- reporting the symptom instead of the cause it had
    # already found.
    malformed = [
        i for i, r in enumerate(rows, 1)
        if "category" not in r or not all(_pair(r))
    ]
    if malformed:
        print("FAIL")
        print(
            f"  - {len(malformed)} row(s) lack a category or a usable "
            f"input/output pair; first at line {malformed[0]}"
        )
        print(f"    keys present there: {sorted(rows[malformed[0] - 1].keys())}")
        print(
            "    expected either chat-style 'messages' with a user and an "
            "assistant turn, or flat 'prompt'/'ground_truth'"
        )
        return 1

    # --- composition -------------------------------------------------------
    by_cat = Counter(r.get("category", "uncategorised") for r in rows)
    synth = [r for r in rows if r.get("synthesized")]
    authored = [r for r in rows if not r.get("synthesized")]
    print("composition:")
    for cat, n in by_cat.most_common():
        s = sum(1 for r in synth if r.get("category") == cat)
        note = f"  ({s} synthesized)" if s else ""
        print(f"  {cat:<18} {n:>4}  {n / len(rows):>4.0%}{note}")
    print(f"\nsynthesized: {len(synth)}   authored: {len(authored)}")

    # --- contamination -----------------------------------------------------
    # The one failure that would invalidate every downstream number: a generated
    # row that duplicates an evaluation prompt turns the held-out set into
    # training data, and the comparison then measures memorisation.
    held_pairs = [_pair(r) for r in held_out]
    held_prompts = {p.strip().lower() for p, _ in held_pairs if p}
    held_truths = {g.strip().lower() for _, g in held_pairs if g}

    leaked_prompt, leaked_truth = [], []
    for r in rows:
        prompt, truth = _pair(r)
        if prompt.strip().lower() in held_prompts:
            leaked_prompt.append(prompt)
        if truth.strip().lower() in held_truths:
            leaked_truth.append(truth)

    if leaked_prompt:
        problems.append(
            f"{len(leaked_prompt)} row(s) reuse a held-out PROMPT -- the "
            f"evaluation set is contaminated: {leaked_prompt[0][:60]!r}"
        )
    if leaked_truth:
        problems.append(
            f"{len(leaked_truth)} row(s) reuse a held-out GROUND TRUTH: "
            f"{leaked_truth[0][:60]!r}"
        )
    print(
        f"held-out contamination: {len(leaked_prompt)} prompt(s), "
        f"{len(leaked_truth)} answer(s) over {len(held_out)} held-out rows"
    )

    # --- duplication -------------------------------------------------------
    # A repeated prompt is not automatically a defect. "caption untuk paket
    # bundling" legitimately has many valid answers, and the authored corpus
    # deliberately pairs one such request with several different captions -- that
    # is signal about the range of acceptable outputs, not padding.
    #
    # What IS a defect is a repeated (prompt, output) PAIR: an identical row
    # counted twice teaches nothing the first copy did not. So the check is on
    # pairs, and it is reported separately for synthesized rows, since the
    # synthesizer is the component that could pad a shortfall with repeats.
    def _pair_key(r: dict) -> tuple[str, str]:
        p, g = _pair(r)
        return p.strip().lower(), g.strip().lower()

    exact = Counter(_pair_key(r) for r in rows)
    exact_dupes = {k: n for k, n in exact.items() if n > 1}
    synth_exact = Counter(_pair_key(r) for r in synth)
    synth_dupes = {k: n for k, n in synth_exact.items() if n > 1}

    prompt_counts = Counter(_pair(r)[0].strip().lower() for r in rows)
    reused_prompts = {p: n for p, n in prompt_counts.items() if n > 1}
    if reused_prompts:
        worst_prompt = max(reused_prompts.values())
        print(
            f"reused prompts        : {len(reused_prompts)} distinct, worst {worst_prompt}x "
            f"(fine when the answers differ)"
        )
    else:
        print("reused prompts        : none")

    # Duplication is judged by who produced it. Repeats among AUTHORED rows were
    # inherited from the input corpus -- the loop did not create them and cannot
    # be failed for them -- but they are reported, because a row count read as
    # "N distinct examples" overstates the corpus when N includes repeats.
    # Repeats among SYNTHESIZED rows are the loop's own doing and do fail: the
    # synthesizer's contract is to report a shortfall, never to pad one.
    if exact_dupes:
        worst = max(exact_dupes.items(), key=lambda kv: kv[1])
        print(
            f"identical rows        : {len(exact_dupes)} pair(s) repeated, worst "
            f"{worst[1]}x  (inherited from the input corpus)"
        )
    else:
        print("identical rows        : none")

    distinct_total = len(exact)
    print(
        f"effective size        : {distinct_total} distinct of {len(rows)} rows "
        f"({distinct_total / len(rows):.0%})"
    )

    if synth_dupes:
        worst = max(synth_dupes.items(), key=lambda kv: kv[1])
        problems.append(
            f"{len(synth_dupes)} synthesized pair(s) are duplicates, worst "
            f"{worst[1]}x: {worst[0][0][:50]!r} -- the synthesizer must report a "
            "shortfall, not pad it with repeats"
        )
    else:
        synth_distinct = len({_pair_key(r) for r in synth})
        print(
            f"synthesized distinct  : {synth_distinct}/{len(synth)}"
            + ("  (no padding)" if synth and synth_distinct == len(synth) else "")
        )

    # --- rule compliance of generated rows ---------------------------------
    # Every synthesized row already passed the gates once. Re-scoring here is the
    # check that the rows WRITTEN TO DISK are the rows that passed, rather than
    # trusting that nothing was reordered or mangled on the way out.
    failures: list[tuple[int, str, list[str]]] = []
    for i, r in enumerate(synth, 1):
        _, output = _pair(r)
        report = spec.scorer(output, category=r.get("category"))
        broken = [k for k, ok in {**report.articulable, **report.tacit}.items() if ok is False]
        if broken:
            failures.append((i, r.get("category", "?"), broken))
    if failures:
        problems.append(
            f"{len(failures)}/{len(synth)} synthesized rows fail the use case's "
            f"own scorer, e.g. row {failures[0][0]} ({failures[0][1]}): "
            f"{', '.join(failures[0][2])}"
        )
        print(f"rule compliance       : {len(synth) - len(failures)}/{len(synth)} pass")
    else:
        print(f"rule compliance       : {len(synth)}/{len(synth)} pass")

    # --- verdict -----------------------------------------------------------
    print()
    if problems:
        print("FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PASS -- corpus is trainable and the evaluation set is untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
