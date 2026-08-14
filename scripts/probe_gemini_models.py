"""Find which Gemini models are actually reachable right now.

Written after `gemini-3.7-flash` returned 503 "high demand" on two consecutive
attempts, including the plain call that had succeeded seconds earlier. That is a
capacity condition on the server, not a bug here -- but it decides two things
the agent layer needs to get right:

  1. retry with backoff is mandatory, not optional
  2. a fallback model is worth configuring, because a run that dies because one
     model is busy is a run wasted

Lists the models the key can see, then probes a shortlist for a real response.
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

# Ordered by preference: newest first, then the generation the hackathon
# mandates as a floor. Anything below 3.5 would not satisfy the requirement.
SHORTLIST = [
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-pro",
]


def main() -> None:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise SystemExit("GEMINI_API_KEY is empty in .env")

    from google import genai

    client = genai.Client(api_key=key)

    print("=== models this key can list (flash/pro only) ===")
    listed: list[str] = []
    try:
        for m in client.models.list():
            name = (m.name or "").replace("models/", "")
            if "flash" in name or "pro" in name:
                listed.append(name)
        for name in sorted(listed):
            print(f"  {name}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (list failed: {type(exc).__name__}: {exc})")

    print("\n=== reachability probe ===")
    candidates = SHORTLIST + [n for n in sorted(listed) if n not in SHORTLIST][:4]
    working: list[str] = []

    for name in candidates:
        print(f"  {name:<34}", end=" ", flush=True)
        try:
            resp = client.models.generate_content(
                model=name, contents="Reply with one word: ok"
            )
            text = (resp.text or "").strip()[:20]
            print(f"OK -> {text!r}")
            working.append(name)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "503" in msg or "UNAVAILABLE" in msg:
                print("503 overloaded")
            elif "404" in msg or "NOT_FOUND" in msg:
                print("404 not found for this key")
            elif "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print("429 rate limited / quota")
            else:
                print(f"{type(exc).__name__}: {msg[:70]}")
        time.sleep(1.0)  # be polite while probing

    print("\n=== result ===")
    if working:
        print(f"reachable now: {', '.join(working)}")
        print(f"\nSuggested .env:\n  GEMINI_MODEL={working[0]}")
        if len(working) > 1:
            print(f"  GEMINI_FALLBACK_MODEL={working[1]}")
    else:
        print(
            "Nothing responded. Either every candidate is overloaded right now, or\n"
            "the key lacks access. Retry in a few minutes before assuming the key\n"
            "is the problem -- the 503 message explicitly says spikes are temporary."
        )


if __name__ == "__main__":
    main()
