"""Shared generation helpers, so stop-token handling lives in exactly one place.

The stop-token logic was originally inline in one script and got it wrong twice:
first by passing only `eos_token_id`, then by hardcoding `<end_of_turn>` (which
does not exist on Gemma 4). Duplicating that logic into a second comparison
script would mean fixing the same bug twice, and eventually fixing it in only
one of them.
"""

from __future__ import annotations

# A full brand reply is ~40 tokens. Generating far past that only gives a model
# room to fall into a loop.
MAX_NEW_TOKENS = 120

# Pure greedy decoding on a small model degenerates into n-gram loops regardless
# of training quality, and a mild penalty is standard for any real deployment.
# It cannot manufacture voice rules the model never learned, so it does not
# flatter the result.
REPETITION_PENALTY = 1.15


def stop_token_ids(tokenizer) -> list[int]:
    """Every token that should end generation, not just `eos_token`.

    Passing only `eos_token_id` lets a chat model sail past the end of its answer
    and hallucinate a fresh conversation turn, because chat models terminate a
    turn with their own marker rather than with `<eos>`.

    The marker is DERIVED FROM THE RENDERED TEMPLATE rather than looked up by
    name, because names are not stable across model families -- or even across
    generations of one family:

        Gemma 3 : <end_of_turn>
        Gemma 4 : <turn|>        (id 106, rendered '<|turn>model\\n...<turn|>')

    An earlier version hardcoded `<end_of_turn>` and silently did nothing on
    Gemma 4: the token does not exist there, so the lookup fell back to `<eos>`
    alone and the overrun persisted. Reading the template avoids repeating that
    class of mistake on the next model.
    """
    ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)

    try:
        encoded = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "y"},
            ],
            tokenize=True,
        )
        turn_ids = encoded["input_ids"]
        if turn_ids and isinstance(turn_ids[0], (list, tuple)):
            turn_ids = turn_ids[0]

        special = set(tokenizer.all_special_ids or [])
        # Walk back from the end of the assistant turn, stepping over trailing
        # whitespace, and take the special tokens that close it.
        for tid in reversed(turn_ids):
            piece = tokenizer.convert_ids_to_tokens([tid])[0]
            if tid in special:
                ids.append(tid)
                continue
            if piece is not None and piece.strip(" \t\n\r▁Ġ") == "":
                continue
            break
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not derive turn marker from template: {exc})")

    return sorted(set(ids))


def render_prompt(tokenizer, user_text: str, system_text: str | None = None) -> str:
    """Render a chat prompt, falling back when the template rejects a system role.

    Gemma templates historically reject `role: system`. Rather than silently
    dropping the style guide -- which would quietly turn the "base + guide"
    column into a plain "base" column and invalidate the comparison -- fold it
    into the user turn.
    """
    if system_text:
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001
            merged = f"{system_text}\n\n---\n\nPesan masuk: {user_text}"
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": merged}],
                tokenize=False,
                add_generation_prompt=True,
            )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(
    model,
    tokenizer,
    user_text: str,
    device: str,
    *,
    system_text: str | None = None,
    stop_ids: list[int] | None = None,
    repetition_penalty: float = REPETITION_PENALTY,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Greedy decode, deterministic run to run."""
    import torch

    text = render_prompt(tokenizer, user_text, system_text)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            eos_token_id=stop_ids or stop_token_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
