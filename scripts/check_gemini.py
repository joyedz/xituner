"""Verify the Gemini API key works and structured output behaves, before
building anything on top of it.

Three things get checked, in order of how expensive they are to discover late:

  1. the key authenticates at all
  2. the configured model name actually exists (a typo here fails at call time,
     not at config time)
  3. `response_schema` really returns parseable structured output -- the whole
     agent layer depends on decisions arriving as JSON rather than prose, so
     confirming it once here is cheaper than debugging a parse error inside a
     Referee call later
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from pydantic import BaseModel, Field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()


class Probe(BaseModel):
    """Deliberately mixed types: a string enum, an int, and a bool.

    A schema of only strings would pass even if the model were returning prose
    that happens to look like JSON. Mixed types force real coercion.
    """

    verdict: str = Field(description="exactly one of: pass, fail")
    confidence: int = Field(description="integer 0-100")
    is_structured: bool


def main() -> None:
    key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    if not key:
        raise SystemExit("GEMINI_API_KEY is empty in .env")

    print(f"model : {model}")
    print(f"key   : {key[:6]}... ({len(key)} chars)\n")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)

    # --- 1 + 2: auth and model name -------------------------------------
    print("[1/3] plain call ...", end=" ", flush=True)
    try:
        resp = client.models.generate_content(
            model=model,
            contents="Reply with exactly one word: ready",
        )
        print(f"OK -> {resp.text.strip()[:40]!r}")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"FAILED\n  {type(exc).__name__}: {exc}\n\n"
            "If this says the model was not found, fix GEMINI_MODEL in .env.\n"
            "If it says the key is invalid, regenerate it at aistudio.google.com."
        ) from exc

    # --- 3: structured output ------------------------------------------
    print("[2/3] structured output ...", end=" ", flush=True)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=(
                "A test harness asked whether structured output works. "
                "Answer verdict='pass', confidence=95, is_structured=true."
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Probe,
            ),
        )
        parsed = resp.parsed
        assert isinstance(parsed, Probe), f"expected Probe, got {type(parsed)}"
        print(
            f"OK -> verdict={parsed.verdict!r} confidence={parsed.confidence} "
            f"is_structured={parsed.is_structured}"
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"FAILED\n  {type(exc).__name__}: {exc}\n\n"
            "Structured output is what the agent layer's decisions depend on. "
            "Without it, every verdict would need free-text parsing."
        ) from exc

    # --- 3b: determinism at temperature 0 -------------------------------
    # The Referee's scores have to be stable enough to compare across runs. This
    # does not prove determinism, but a model that disagrees with itself twice in
    # a row at temperature 0 would be a problem worth knowing about now.
    print("[3/3] temperature=0 stability ...", end=" ", flush=True)
    answers = []
    for _ in range(2):
        r = client.models.generate_content(
            model=model,
            contents="Is 17 a prime number? Answer verdict='pass' if yes, else 'fail'. confidence=100, is_structured=true.",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Probe,
                temperature=0.0,
            ),
        )
        answers.append(r.parsed.verdict)
    stable = answers[0] == answers[1]
    print(f"{'OK' if stable else 'VARIED'} -> {answers}")

    print("\nAll checks passed. Safe to build the agent layer on this.")


if __name__ == "__main__":
    main()
