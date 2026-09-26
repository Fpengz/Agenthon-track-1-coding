"""House Model client for Agenthon Track 1.

Connects to the organizer's hosted model endpoint via the audited proxy,
strictly enforces the per-unit 25-request budget and 4,000-output-token ceiling.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

# Track 1 Competition Rules
MAX_REQUESTS_PER_UNIT = 25
MAX_OUTPUT_TOKENS = 4000


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

        # Resolve endpoint origin
        endpoint_origin = base_url or os.environ.get("MODEL_ENDPOINT")
        token = api_key or os.environ.get("MODEL_TOKEN") or os.environ.get("OPENAI_API_KEY", "dummy-token")
        self.model = (
            model_name
            or os.environ.get("MODEL_NAME")
            or os.environ.get("OPENAI_MODEL", "default")
        )

        if endpoint_origin:
            # According to SUBMISSION_CLI and baselines/README.md:
            # $MODEL_ENDPOINT is the route origin (e.g. http://model:8443).
            # The OpenAI-compatible API is served under /v1.
            resolved_base_url = endpoint_origin.rstrip("/")
            if not resolved_base_url.endswith("/v1"):
                resolved_base_url += "/v1"
            self.client = OpenAI(base_url=resolved_base_url, api_key=token)
        else:
            # Fallback for local development if standard OPENAI_BASE_URL / OPENAI_API_KEY are configured
            openai_base = os.environ.get("OPENAI_BASE_URL")
            if openai_base:
                self.client = OpenAI(base_url=openai_base, api_key=token)
            else:
                self.client = OpenAI(api_key=token)

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Send a completion request to the House Model, respecting the 25-request ceiling."""
        if self.request_count >= self.max_requests:
            raise RuntimeError(
                f"Exceeded hard limit of {self.max_requests} model requests per unit. "
                "Aborting further LLM calls to stay admissible under g2 gate."
            )

        # Enforce maximum 4,000 output tokens per request
        effective_max_tokens = min(max_tokens, MAX_OUTPUT_TOKENS)

        self.request_count += 1
        logger.info(
            "Dispatching model request %d/%d (model=%s, max_tokens=%d)",
            self.request_count,
            self.max_requests,
            self.model,
            effective_max_tokens,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=effective_max_tokens,
            temperature=temperature,
            **kwargs,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        return content
