"""Safe adapter around the public llm_sdk API."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, cast

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class LLMAdapter:
    """Wrapper around the provided Small_LLM_Model."""

    def __init__(self) -> None:
        """Initialize the small LLM model."""
        self.model = Small_LLM_Model()
        self.vocabulary = self._load_vocabulary()

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids using the public SDK method."""
        encoded = self.model.encode(text)

        if hasattr(encoded, "tolist"):
            ids = encoded.tolist()
        else:
            ids = list(encoded)

        if ids and isinstance(ids[0], list):
            ids = ids[0]

        return [int(token_id) for token_id in ids]

    def decode_token(self, token_id: int) -> str:
        """Decode one token using public decode if available.

        Fall back to vocabulary when decode is unavailable.
        """
        if hasattr(self.model, "decode"):
            decoded = self.model.decode([token_id])
            return str(decoded)

        return self.vocabulary.get(token_id, "")

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Get next-token logits using the public SDK method."""
        logits = self.model.get_logits_from_input_ids(input_ids)

        if hasattr(logits, "tolist"):
            logits = logits.tolist()

        return [float(value) for value in logits]

    def vocab_token_ids(self) -> list[int]:
        """Return all known token ids."""
        return list(self.vocabulary.keys())

    def _load_vocabulary(self) -> dict[int, str]:
        """Load token vocabulary from the public SDK vocabulary path.

        Supported vocabulary formats:
        1. {"0": "hello"}                     -> id to token
        2. {"hello": 0}                       -> token to id
        3. {"model": {"vocab": {"hello": 0}}} -> Hugging Face tokenizer.json
        4. tiktoken lines: base64_token token_id
        """
        vocab_path = Path(self.model.get_path_to_vocab_file())

        vocabulary = self._load_json_vocabulary(vocab_path)
        if vocabulary:
            return vocabulary

        vocabulary = self._load_tiktoken_vocabulary(vocab_path)
        if vocabulary:
            return vocabulary

        raise ValueError(
            "Vocabulary file did not contain any valid token ids. "
            f"Path: {vocab_path}"
        )

    def _load_json_vocabulary(self, vocab_path: Path) -> dict[int, str]:
        """Try to load JSON vocabulary formats."""
        try:
            with vocab_path.open("r", encoding="utf-8") as file:
                raw_vocab: Any = json.load(file)
        except json.JSONDecodeError:
            return {}

        if not isinstance(raw_vocab, dict):
            return {}

        raw_vocab = self._extract_vocab_dict(raw_vocab)

        if not isinstance(raw_vocab, dict):
            return {}

        vocabulary: dict[int, str] = {}

        for key, value in raw_vocab.items():
            key_text = str(key).strip()

            # Format: {"123": "token"} - use isdecimal() instead of isdigit()
            # to avoid matching Unicode digits like ² which can't be converted
            if key_text.isdecimal():
                vocabulary[int(key_text)] = str(value)

            # Format: {"token": 123}
            elif isinstance(value, int):
                vocabulary[value] = str(key)

            # Format: {"token": "123"} - use isdecimal()
            elif isinstance(value, str) and value.strip().isdecimal():
                vocabulary[int(value.strip())] = str(key)

        return vocabulary

    def _extract_vocab_dict(self, raw_vocab: dict[str, Any]) -> dict[str, Any]:
        """Extract the actual vocab dict from wrapped tokenizer JSON."""
        model = raw_vocab.get("model")

        # Hugging Face tokenizer.json format:
        # {"model": {"vocab": {"hello": 0}}}
        if isinstance(model, dict) and isinstance(model.get("vocab"), dict):
            return cast(dict[str, Any], model["vocab"])

        # Some tokenizer files use:
        # {"vocab": {"hello": 0}}
        if isinstance(raw_vocab.get("vocab"), dict):
            return cast(dict[str, Any], raw_vocab["vocab"])

        return raw_vocab

    def _load_tiktoken_vocabulary(self, vocab_path: Path) -> dict[int, str]:
        """Try to load tiktoken format: base64_token token_id per line.

        Example line:
            IQ== 0

        Meaning:
            base64 token -> token id
        """
        vocabulary: dict[int, str] = {}

        with vocab_path.open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()

                if len(parts) != 2:
                    continue

                encoded_token, token_id_text = parts

                if not token_id_text.isdigit():
                    continue

                token_id = int(token_id_text)
                token = self._decode_base64_token(encoded_token)

                vocabulary[token_id] = token

        return vocabulary

    def _decode_base64_token(self, encoded_token: str) -> str:
        """Decode a base64 token safely."""
        try:
            token_bytes = base64.b64decode(encoded_token)
            return token_bytes.decode("utf-8", errors="replace")
        except Exception:
            return encoded_token
