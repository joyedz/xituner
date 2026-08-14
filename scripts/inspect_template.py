"""Print a tokenizer's chat template and special tokens, verbatim.

Written after a guess failed. The stop-token fix assumed Gemma ends assistant
turns with `<end_of_turn>`; the tokenizer reported no such token, so the fix
resolved to `<eos>` alone and changed nothing.

Rather than guess a second time, read the template and let it say which marker
actually terminates a turn.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a chat template.")
    parser.add_argument("--base-model", default="google/gemma-4-E2B-it")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base_model)

    print("=== special tokens map ===")
    for k, v in (tok.special_tokens_map or {}).items():
        print(f"  {k:<24} {v!r}")

    print("\n=== all added special tokens (id -> token) ===")
    added = getattr(tok, "added_tokens_decoder", {}) or {}
    for tid, tokobj in sorted(added.items()):
        content = getattr(tokobj, "content", str(tokobj))
        # Only the structurally interesting ones; the vocab has many reserved slots.
        if any(w in content.lower() for w in ("turn", "end", "eos", "bos", "im_", "eot")):
            print(f"  {tid:<8} {content!r}")

    print("\n=== rendered template, assistant turn included ===")
    msgs = [
        {"role": "user", "content": "PERTANYAAN"},
        {"role": "assistant", "content": "JAWABAN"},
    ]
    rendered = tok.apply_chat_template(msgs, tokenize=False)
    print(repr(rendered))

    print("\n=== rendered with add_generation_prompt (what we feed generate) ===")
    prompt_only = tok.apply_chat_template(
        [msgs[0]], tokenize=False, add_generation_prompt=True
    )
    print(repr(prompt_only))

    print("\n=== token ids of the full turn, tail end ===")
    ids = tok.apply_chat_template(msgs, tokenize=True)["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    tail = ids[-8:]
    print(f"  last 8 ids   : {tail}")
    print(f"  last 8 tokens: {tok.convert_ids_to_tokens(tail)}")
    print(
        "\nThe token that closes the assistant turn above is the one generation\n"
        "must stop on."
    )


if __name__ == "__main__":
    main()
