"""Gemini access for the agent layer, plus a mock so the loop is testable offline.

Every call returns a validated pydantic object, never free text. That is a
deliberate constraint: the Referee's verdict and the Diagnostician's prescription
are consumed by code, and a decision layer that emits prose forces a parser that
will eventually mis-parse something and fail silently.

Retry and fallback are not defensive padding
--------------------------------------------
Measured on 2026-08-14 against the free tier: `gemini-3.7-flash` answered a
plain call, then returned 503 "high demand" for the next two attempts within
seconds, while `gemini-3.5-flash` was simultaneously 503 and several 2.5 models
returned 404 for the same key. Capacity moves minute to minute.

An agent run makes many sequential calls -- one Referee judgement per held-out
row, then a Diagnostician call, per iteration. Without retry, any single 503
kills a run that has already spent GPU minutes on training. So: exponential
backoff on the retryable statuses, then one fallback model, then fail loudly.

Deliberately NOT here: a multi-provider abstraction. An earlier plan had
`glm.py` / `kimi.py` / `proxy_gemini.py` behind this interface; it was cut
because it cost days for zero judging points and introduced exactly the risk of
behaviour differing between the provider used in development and the one used in
the demo. One interface with one real implementation is hygiene; four is
premature.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)

# Statuses worth retrying: overloaded, rate limited, or a transient gateway
# error. A 400 or 404 is a bug in the request and retrying it just wastes time.
_RETRYABLE = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "502", "504")


class LLMError(RuntimeError):
    """Raised when a call could not be completed after retries and fallback."""


class _Unset:
    """Sentinel distinguishing "argument not given" from "explicitly None".

    Needed because `fallback_model or os.getenv(...)` silently ignored an
    explicit `fallback_model=None`: the falsy None fell through to the
    environment default, so a caller asking for NO fallback quietly got one. The
    bug surfaced as a test that expected a bad model name to fail and instead saw
    it succeed -- on the fallback.

    Explicit arguments have to win over the environment, or configuration
    becomes unpredictable in exactly the situations where it matters.
    """

    def __repr__(self) -> str:  # pragma: no cover
        return "<unset>"


UNSET = _Unset()


@dataclass
class CallStats:
    """Per-run call accounting, so cost and flakiness are visible not guessed."""

    calls: int = 0
    retries: int = 0
    fallbacks: int = 0
    failures: int = 0
    seconds: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        models = ", ".join(f"{m}x{n}" for m, n in sorted(self.by_model.items()))
        return (
            f"{self.calls} calls in {self.seconds:.1f}s "
            f"(retries {self.retries}, fallbacks {self.fallbacks}, "
            f"failures {self.failures}) [{models}]"
        )


class LLMClient(Protocol):
    """What the agent layer is allowed to assume about its decision layer."""

    def structured(
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T: ...

    @property
    def stats(self) -> CallStats: ...


class GeminiClient:
    """Structured-output Gemini client with backoff and a fallback model."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        fallback_model: str | None | _Unset = UNSET,
        *,
        temperature: float = 0.0,
        max_attempts: int = 4,
        base_delay: float = 2.0,
        verbose: bool = True,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Put it in .env (local) or use Colab "
                "Secrets + userdata.get('GEMINI_API_KEY') -- never paste a key "
                "into a notebook cell, since notebooks get committed."
            )
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
        # An explicit None means "no fallback"; omitting it means "use the env
        # default". The old `or` chain could not tell those apart.
        self.fallback_model = (
            (os.getenv("GEMINI_FALLBACK_MODEL") or None)
            if isinstance(fallback_model, _Unset)
            else fallback_model
        )
        # Referee scores get compared across runs, so a wandering temperature
        # would make a rerun disagree with itself for no reason.
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.verbose = verbose
        self._stats = CallStats()

        from google import genai

        self._client = genai.Client(api_key=self.api_key)

    @property
    def stats(self) -> CallStats:
        return self._stats

    def structured(
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        from google.genai import types

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=self.temperature,
            system_instruction=system,
        )

        models = [self.model] + ([self.fallback_model] if self.fallback_model else [])
        started = time.perf_counter()
        last_error: Exception | None = None

        for model_index, model in enumerate(models):
            if model_index > 0:
                self._stats.fallbacks += 1
                self._log(f"  falling back to {model}")

            for attempt in range(1, self.max_attempts + 1):
                try:
                    resp = self._client.models.generate_content(
                        model=model, contents=prompt, config=config
                    )
                    parsed = resp.parsed
                    if not isinstance(parsed, schema):
                        # Schema-conforming JSON that the SDK could not coerce is
                        # still a failure -- treat it as retryable rather than
                        # returning something the caller will mis-handle.
                        raise LLMError(
                            f"expected {schema.__name__}, got {type(parsed).__name__}"
                        )
                    self._stats.calls += 1
                    self._stats.seconds += time.perf_counter() - started
                    self._stats.by_model[model] = self._stats.by_model.get(model, 0) + 1
                    return parsed
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    message = str(exc)
                    retryable = any(code in message for code in _RETRYABLE)
                    if not retryable or attempt == self.max_attempts:
                        self._log(
                            f"  {model} attempt {attempt}/{self.max_attempts} "
                            f"failed{'' if retryable else ' (not retryable)'}: "
                            f"{message[:90]}"
                        )
                        break
                    # Full jitter: many sequential calls retrying in lockstep
                    # would hit the same busy endpoint at the same moment.
                    delay = random.uniform(0, self.base_delay * (2 ** (attempt - 1)))
                    self._stats.retries += 1
                    self._log(
                        f"  {model} attempt {attempt} got a retryable error, "
                        f"waiting {delay:.1f}s"
                    )
                    time.sleep(delay)

        self._stats.failures += 1
        self._stats.seconds += time.perf_counter() - started
        raise LLMError(
            f"all models exhausted ({', '.join(models)}). Last error: {last_error}"
        ) from last_error

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)


class MockLLMClient:
    """Returns canned objects, so the whole agent loop runs with no API key.

    This is not a convenience. The loop's guards -- prescription validation,
    iteration limits, the refusal to let an LLM mutate a corpus directly -- are
    deterministic, and testing them against a real model would make the tests
    slow, costly and non-reproducible. Canned responses let those paths be
    exercised exactly, including the malformed ones a real model occasionally
    produces.

    `responses` maps schema name -> list of instances, consumed in order. A
    schema with no queued response raises, so a test cannot pass by accidentally
    exercising an unstubbed call.
    """

    def __init__(self, responses: dict[str, list[BaseModel]] | None = None) -> None:
        self.responses = responses or {}
        self.prompts: list[tuple[str, str]] = []  # (schema name, prompt)
        self._stats = CallStats()

    @property
    def stats(self) -> CallStats:
        return self._stats

    def queue(self, obj: BaseModel) -> None:
        self.responses.setdefault(type(obj).__name__, []).append(obj)

    def structured(
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        self.prompts.append((schema.__name__, prompt))
        self._stats.calls += 1
        self._stats.by_model["mock"] = self._stats.by_model.get("mock", 0) + 1
        queued = self.responses.get(schema.__name__)
        if not queued:
            raise LLMError(
                f"MockLLMClient has no queued {schema.__name__}. Queue one with "
                "client.queue(...) -- an unstubbed call must fail loudly rather "
                "than let a test pass without exercising what it claims to."
            )
        return queued.pop(0)  # type: ignore[return-value]


def build_client(mock: bool = False, **kwargs) -> LLMClient:
    return MockLLMClient() if mock else GeminiClient(**kwargs)
