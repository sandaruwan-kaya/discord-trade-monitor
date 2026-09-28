"""
Injectable structured-output LLM client for agents.

Agents depend on the LLMClient protocol, never on OpenAI directly,
so tests can pass a fake and no network call is ever made.
"""
from typing import Protocol, TypeVar

from pydantic import BaseModel

import config


T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def structured(self, system: str, user: str, schema: type[T]) -> T:
        ...


class OpenAILLMClient:

    def __init__(self, model: str | None = None, client=None):
        self.model = model or config.agent_model()
        self._client = client

    def _get_client(self):
        if self._client is None:
            import os
            from openai import OpenAI

            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        return self._client

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        response = self._get_client().responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=schema
        )

        parsed = response.output_parsed

        if parsed is None:
            raise RuntimeError(
                f"LLM did not return a structured {schema.__name__}"
            )

        return parsed
