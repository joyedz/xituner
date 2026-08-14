"""Find out what really happens when GEMINI_MODEL holds a nonexistent name.

This matters beyond tidiness. The contest requires Gemini 3.5 or newer. If a
typo in GEMINI_MODEL silently resolves to some other model, a run could be
served by something that does not satisfy that requirement while every log line
still says the configured name -- a compliance problem that no test would catch
and no output would reveal.

So: call a clearly invalid name and report exactly what comes back, including
which model the response says served it.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from pydantic import BaseModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()


class Tiny(BaseModel):
    word: str


def main() -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Tiny,
        temperature=0.0,
    )

    for name in ("gemini-does-not-exist", "totally-bogus-xyz", "gemini-99-flash"):
        print(f"--- {name} ---")
        try:
            resp = client.models.generate_content(
                model=name, contents="Reply with word='ok'", config=config
            )
            print(f"  SUCCEEDED (!)  parsed={resp.parsed}")
            # Which model actually served this?
            served = getattr(resp, "model_version", None)
            print(f"  model_version reported: {served!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"  raised {type(exc).__name__}: {str(exc)[:110]}")
        print()


if __name__ == "__main__":
    main()
