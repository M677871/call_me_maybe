"""Token-by-token constrained decoding."""
from __future__ import annotations
import math
from src.constraints import Constraint
from src.llm_adapter import LLMAdapter


class ConstrainedDecoder:
    """Generate text while blocking invalid next tokens."""
    def __init__(self, llm: LLMAdapter) -> None:
        """Store LLM adapter."""
        self.llm = llm

    def generate(
        self,
        prompt: str,
        constraint: Constraint,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate text satisfying the given constraint."""
        generated = ""
        for _ in range(max_new_tokens):
            input_ids = self.llm.encode(prompt + generated)
            logits = self.llm.get_logits(input_ids)
            best_token_id = self._choose_best_valid_token(
                generated=generated,
                logits=logits,
                constraint=constraint,
            )
            if best_token_id is None:
                raise RuntimeError("No valid token available during decoding.")
            generated += self.llm.decode_token(best_token_id)
            if constraint.is_complete(generated):
                return generated
        raise RuntimeError("Constrained decoding reached max token limit.")

    def _choose_best_valid_token(
        self,
        generated: str,
        logits: list[float],
        constraint: Constraint,
    ) -> int | None:
        """Choose highest-logit token that keeps output valid."""
        best_token_id: int | None = None
        best_score = -math.inf
        for token_id in self.llm.vocab_token_ids():
            if token_id >= len(logits):
                continue
            token_text = self.llm.decode_token(token_id)
            candidate = generated + token_text
            if not constraint.allows(candidate):
                continue
            score = logits[token_id]
            if score > best_score:
                best_score = score
                best_token_id = token_id
        return best_token_id
