"""House Model client for Agenthon Track 1.

Connects to the organizer's hosted model endpoint via the audited proxy,
strictly enforces the per-unit 25-request budget and 4,000-output-token ceiling.
"""

from __future__ import annotations

import logging
import os
import pathlib
import threading
import time
from dataclasses import dataclass
from typing import Any

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

# Track 1 Competition Rules
MAX_REQUESTS_PER_UNIT = 25
MAX_OUTPUT_TOKENS = 4000


@dataclass(frozen=True)
class ChatResult:
    content: str
    finish_reason: str | None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


# Reasoning modes, sent as chat_template_kwargs (the House route forwards them; see the hub's
# docs/HOUSE-MODEL.md). Full thinking routinely spends the whole 4,000-token output cap on inline
# reasoning and never emits code.
#   low: low_effort=true      -- brief reasoning (DEFAULT). Two parallel A/Bs, 4 runs vs 6:
#        mean pass@1 0.205 vs ~0.182, no-output failures halved, ~20% fewer requests. The
#        organizers forward it but mark it untested, hence the fallback below.
#   off: enable_thinking=false -- documented by the organizers as the thinking control
#   on:  server default (full reasoning)
REASONING_MODES: dict[str, dict[str, Any] | None] = {
    "off": {"enable_thinking": False},
    "low": {"low_effort": True},
    "on": None,
}
DEFAULT_REASONING_MODE = "low"
# If "low" is not honoured (a reply reasons at length, or is cut off before any code), the
# client drops to "off" -- the documented control -- for the rest of the unit. Honoured low
# effort reasons for ~200 characters.
LOW_EFFORT_MAX_REASONING_CHARS = 4000
# Per-request HTTP timeout: a stalled call must not eat the unit's card time (the SDK default is
# 10 minutes). A full 4,000-token reply takes ~25-50 s locally.
REQUEST_TIMEOUT_SEC = float(os.environ.get("HOUSE_REQUEST_TIMEOUT", "300"))
# Transient failures (connection drops, timeouts, 429, 5xx) are retried a bounded number of times
# with backoff. Each retry is another admitted request, but giving up loses the whole unit.
# Rejections such as 400 (e.g. context overflow), 401 and 403 are not retried.
TRANSIENT_RETRIES = 2
_RETRY_BACKOFF_SEC = (2.0, 8.0)
_TRANSIENT_ERRORS = (
    openai.APIConnectionError,  # includes APITimeoutError
    openai.RateLimitError,
    openai.InternalServerError,
)


class HouseModelClient:
    """Client for interacting with the organizer-hosted model route ($MODEL_ENDPOINT)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        max_requests: int = MAX_REQUESTS_PER_UNIT,
    ) -> None:
        self.max_requests = max_requests
        self.request_count = 0
        # Candidates may query concurrently (parallel consensus): admission to the 25-request
        # budget must be atomic.
        self._lock = threading.Lock()

        # Resolve endpoint origin
        endpoint_origin = base_url or os.environ.get("MODEL_ENDPOINT")
        token = (
            api_key or os.environ.get("MODEL_TOKEN") or os.environ.get("OPENAI_API_KEY", "EMPTY")
        )
        self.model = (
            model_name or os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL", "default")
        )

        mode = os.environ.get("HOUSE_REASONING", DEFAULT_REASONING_MODE).strip().lower()
        if mode not in REASONING_MODES:
            logger.warning("Unknown HOUSE_REASONING=%r; using %r", mode, DEFAULT_REASONING_MODE)
            mode = DEFAULT_REASONING_MODE
        self.reasoning_mode = mode

        logger.info(
            "Initialized House Model client (model=%s, request_limit=%d, reasoning=%s)",
            self.model,
            self.max_requests,
            self.reasoning_mode,
        )

        # max_retries=0: an SDK retry is another admitted House request that would bypass the
        # 25-request counter below; the solve loop owns all retrying.
        if endpoint_origin:
            # According to SUBMISSION_CLI and baselines/README.md:
            # $MODEL_ENDPOINT is the route origin (e.g. http://model:8443).
            # The OpenAI-compatible API is served under /v1.
            resolved_base_url = endpoint_origin.rstrip("/")
            if not resolved_base_url.endswith("/v1"):
                resolved_base_url += "/v1"
            self.client = OpenAI(
                base_url=resolved_base_url,
                api_key=token,
                max_retries=0,
                timeout=REQUEST_TIMEOUT_SEC,
            )
        else:
            # Fallback for local development if standard OPENAI_BASE_URL / OPENAI_API_KEY are configured
            openai_base = os.environ.get("OPENAI_BASE_URL")
            if openai_base:
                self.client = OpenAI(
                    base_url=openai_base, api_key=token, max_retries=0, timeout=REQUEST_TIMEOUT_SEC
                )
            else:
                self.client = OpenAI(api_key=token, max_retries=0, timeout=REQUEST_TIMEOUT_SEC)

    def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        max_tokens: int = MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> ChatResult:
        """Send a completion request, retrying transient failures within the request budget."""
        for retry in range(TRANSIENT_RETRIES + 1):
            try:
                return self._chat_once(messages, max_tokens, temperature, **kwargs)
            except _TRANSIENT_ERRORS as exc:
                if retry == TRANSIENT_RETRIES or self.request_count >= self.max_requests:
                    raise
                delay = _RETRY_BACKOFF_SEC[min(retry, len(_RETRY_BACKOFF_SEC) - 1)]
                logger.warning(
                    "Transient House error (%s); retrying in %.0fs", exc.__class__.__name__, delay
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _chat_once(
        self,
        messages: list[ChatCompletionMessageParam],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> ChatResult:
        """One completion request, respecting the 25-request ceiling."""
        kwargs = dict(kwargs)
        template_kwargs = REASONING_MODES[self.reasoning_mode]
        if template_kwargs is not None and "extra_body" not in kwargs:
            kwargs["extra_body"] = {"chat_template_kwargs": template_kwargs}
        with self._lock:
            if self.request_count >= self.max_requests:
                logger.error(
                    "Model request limit reached (%d/%d)",
                    self.request_count,
                    self.max_requests,
                )
                raise RuntimeError(
                    f"Exceeded hard limit of {self.max_requests} model requests per unit. "
                    "Aborting further LLM calls to stay admissible under g2 gate."
                )
            self.request_count += 1
            number = self.request_count

        # Enforce maximum 4,000 output tokens per request
        effective_max_tokens = min(max_tokens, MAX_OUTPUT_TOKENS)

        started_at = time.perf_counter()
        prompt_characters = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                prompt_characters += len(content)
        logger.info(
            "Sending model request %d/%d (model=%s, messages=%d, prompt_chars=%d, "
            "max_tokens=%d, temperature=%.2f)",
            number,
            self.max_requests,
            self.model,
            len(messages),
            prompt_characters,
            effective_max_tokens,
            temperature,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=effective_max_tokens,
                temperature=temperature,
                **kwargs,
            )
        except Exception as exc:
            logger.warning(
                "Model request %d/%d failed after %.2fs: %s: %s",
                number,
                self.max_requests,
                time.perf_counter() - started_at,
                exc.__class__.__name__,
                str(exc)[:300],
            )
            raise

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        logger.info(
            "Model request %d/%d completed in %.2fs (response_chars=%d, "
            "prompt_tokens=%s, completion_tokens=%s, finish_reason=%s)",
            number,
            self.max_requests,
            time.perf_counter() - started_at,
            len(content),
            prompt_tokens if prompt_tokens is not None else "unknown",
            completion_tokens if completion_tokens is not None else "unknown",
            choice.finish_reason,
        )
        self._save_transcript(number, messages, content, choice.finish_reason)
        self._check_low_effort(content, choice.finish_reason)
        return ChatResult(content=content, finish_reason=choice.finish_reason)

    def _check_low_effort(self, content: str, finish_reason: str | None) -> None:
        """Fall back from "low" to "off" when a reply shows low effort was not applied."""
        if self.reasoning_mode != "low":
            return
        reasoning = content.split("</think>", 1)[0] if "</think>" in content else ""
        cut_off_without_code = finish_reason == "length" and "```" not in content
        if len(reasoning) > LOW_EFFORT_MAX_REASONING_CHARS or cut_off_without_code:
            logger.warning(
                "low_effort does not seem to be honoured (reasoning %d chars, finish_reason=%s); "
                "switching to reasoning=off for the rest of this unit",
                len(reasoning),
                finish_reason,
            )
            self.reasoning_mode = "off"

    def _save_transcript(
        self,
        number: int,
        messages: list[ChatCompletionMessageParam],
        content: str,
        finish_reason: str | None,
    ) -> None:
        """Debug aid: with AGENT_TRANSCRIPT_DIR set, keep each request/response pair on disk."""
        directory = os.environ.get("AGENT_TRANSCRIPT_DIR")
        if not directory:
            return
        try:
            path = pathlib.Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            stem = path / f"{number:02d}"
            prompt = "\n\n".join(f"[{m['role']}]\n{m.get('content')}" for m in messages)
            stem.with_suffix(".request.txt").write_text(prompt, encoding="utf-8")
            stem.with_suffix(".response.txt").write_text(
                f"[finish_reason={finish_reason}]\n{content}", encoding="utf-8"
            )
        except OSError:
            logger.debug("Could not write transcript to %s", directory, exc_info=True)
