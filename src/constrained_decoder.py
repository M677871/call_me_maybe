"""Greedy token-by-token constrained decoder."""

from __future__ import annotations

import math
import heapq

from src.constraints import Constraint
from src.llm_adapter import LLMAdapter

TOP_K = 256


class ConstrainedDecoder:
    """Generate text while every new token keeps the output valid."""

    def __init__(self, llm: LLMAdapter) -> None:
        """Store the LLM adapter."""
        self.llm = llm

    def generate(
        self,
        prompt: str,
        constraint: Constraint,
        max_new_tokens: int = 50,
    ) -> str:
        """Generate the highest-scoring valid output."""
        prompt_ids = self.llm.encode(prompt)
        generated_ids: list[int] = []
        generated = ""
        for _ in range(max_new_tokens):
            if constraint.is_complete(generated):
                return generated

            inputs_ids = prompt_ids + generated_ids
            logits = self.llm.get_logits(inputs_ids)
            token_id = self._best_token(generated, logits, constraint)
            if token_id is None:
                raise RuntimeError("No valid token found during decoding.")

            token = self.llm.decode_token(token_id)
            if token == "":
                if constraint.is_complete(generated):
                    return generated
                continue

            generated_ids.append(token_id)
            generated += token

        if constraint.is_complete(generated):
            return generated
        raise RuntimeError("Constrained decoding reached token limit.")

    def _best_token(
        self,
        generated: str,
        logits: list[float],
        constraint: Constraint,
    ) -> int | None:
        """Return the best valid token, checking top logits first."""
        token_ids = self.llm.vocab_token_ids()
        top_ids = self._top_token_ids(token_ids, logits)

        token_id = self._best_token_from_ids(
            token_ids=top_ids,
            generated=generated,
            logits=logits,
            constraint=constraint,
        )
        if token_id is not None:
            return token_id

        return self._best_token_from_ids(
            token_ids=token_ids,
            generated=generated,
            logits=logits,
            constraint=constraint,
        )

    def _top_token_ids(
        self,
        token_ids: list[int],
        logits: list[float],
    ) -> list[int]:
        """Return the most likely token ids."""
        return heapq.nlargest(
            TOP_K,
            token_ids,
            key=lambda token_id: (
                logits[token_id]
                if token_id < len(logits)
                else -math.inf
            ),
        )

    def _best_token_from_ids(
        self,
        token_ids: list[int],
        generated: str,
        logits: list[float],
        constraint: Constraint,
    ) -> int | None:
        """Return the highest-score valid token from selected ids."""
        best_id: int | None = None
        best_score = -math.inf

        for token_id in token_ids:
            if token_id >= len(logits):
                continue

            token = self.llm.decode_token(token_id)
            if token == "":
                continue

            new_text = generated + token
            if not constraint.allows(new_text):
                continue

            score = logits[token_id]
            if score > best_score:
                best_score = score
                best_id = token_id

        return best_id
