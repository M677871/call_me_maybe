"""Small public wrapper around the provided llm_sdk package."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, cast

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class LLMAdapter:
    """Use only the public methods exposed by Small_LLM_Model."""

    def __init__(self) -> None:
        """Load the language model and its vocabulary."""
        try:
            self.model = Small_LLM_Model(trust_remote_code=False)
            self.vocabulary = self._load_vocabulary()
            self._token_ids = sorted(self.vocabulary.keys())
            self._logits_size: int | None = None
            self._decoded_tokens: dict[int, str] = {}
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"Failed to initialize the language model: {exc}"
            ) from exc

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        encoded = self.model.encode(text)
        if hasattr(encoded, "tolist"):
            raw_ids = encoded.tolist()
        else:
            raw_ids = list(encoded)
        if raw_ids and isinstance(raw_ids[0], list):
            raw_ids = raw_ids[0]
        return [int(token_id) for token_id in raw_ids]

    def decode_token(self, token_id: int) -> str:
        """Decode one token id into text."""
        if token_id in self._decoded_tokens:
            return self._decoded_tokens[token_id]
        text = self.vocabulary.get(token_id, "")
        if hasattr(self.model, "decode"):
            decoded = self.model.decode([token_id])
            if decoded is not None:
                text = str(decoded)
        self._decoded_tokens[token_id] = text
        return text

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits for the given input ids."""
        logits = self.model.get_logits_from_input_ids(input_ids)
        if hasattr(logits, "tolist"):
            logits = logits.tolist()
        values = [float(value) for value in logits]
        if self._logits_size != len(values):
            self._logits_size = len(values)
            self._token_ids = list(range(self._logits_size))
        return values

    def vocab_token_ids(self) -> list[int]:
        """Return all token ids known by the vocabulary."""
        return self._token_ids

    def _load_vocabulary(self) -> dict[int, str]:
        """Load vocabulary through the public SDK vocabulary path."""
        vocab_path = Path(self.model.get_path_to_vocab_file())
        vocabulary = self._load_json_vocabulary(vocab_path)
        if vocabulary:
            return vocabulary
        vocabulary = self._load_tiktoken_vocabulary(vocab_path)
        if vocabulary:
            return vocabulary
        raise ValueError(f"Unsupported vocabulary format: {vocab_path}")

    def _load_json_vocabulary(self, path: Path) -> dict[int, str]:
        """Load common JSON vocabulary formats."""
        try:
            with path.open("r", encoding="utf-8") as file:
                raw_data: Any = json.load(file)
        except json.JSONDecodeError:
            return {}
        if not isinstance(raw_data, dict):
            return {}

        raw_vocab = self._extract_vocab(raw_data)
        vocabulary: dict[int, str] = {}
        for key, value in raw_vocab.items():
            key_text = str(key).strip()
            if key_text.isdecimal():
                vocabulary[int(key_text)] = str(value)
            elif isinstance(value, int):
                vocabulary[value] = str(key)
            elif isinstance(value, str) and value.strip().isdecimal():
                vocabulary[int(value.strip())] = str(key)
        return vocabulary

    def _extract_vocab(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Extract the real vocabulary object from wrapped tokenizer JSON."""
        model_data = raw_data.get("model")
        if isinstance(model_data, dict):
            vocab = model_data.get("vocab")
            if isinstance(vocab, dict):
                return cast(dict[str, Any], vocab)
        vocab = raw_data.get("vocab")
        if isinstance(vocab, dict):
            return cast(dict[str, Any], vocab)
        return raw_data

    def _load_tiktoken_vocabulary(self, path: Path) -> dict[int, str]:
        """Load tiktoken format: base64 token then token id per line."""
        vocabulary: dict[int, str] = {}
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                token_text, token_id_text = parts
                if not token_id_text.isdigit():
                    continue
                token_id = int(token_id_text)
                vocabulary[token_id] = self._decode_base64(token_text)
        return vocabulary

    def _decode_base64(self, text: str) -> str:
        """Decode one base64 vocabulary token."""
        try:
            raw_bytes = base64.b64decode(text)
            return raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            return text
