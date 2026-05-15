from __future__ import annotations

import hashlib
import math
import re
from typing import Any, List


Vector = List[float]


def stable_hash(token: str) -> int:
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)


class HashingEmbeddingModel:
    """Dependency-free fallback embedding model for offline experiments."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.token_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?")

    def embed(self, text: str) -> Vector:
        vector = [0.0] * self.dimension
        tokens = self.token_pattern.findall(text.lower())
        for token in tokens:
            idx = stable_hash(token) % self.dimension
            sign = 1.0 if stable_hash("sign:" + token) % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0.0:
            vector = [v / norm for v in vector]
        return vector


class SentenceTransformerEmbeddingModel:
    """Wrapper for pretrained sentence-transformers models."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for pretrained embeddings. "
                "Install it or use --embedding-model hashing."
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> Vector:
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return [float(v) for v in vector]


def create_embedding_model(model_name: str) -> Any:
    if model_name == "hashing":
        return HashingEmbeddingModel()
    return SentenceTransformerEmbeddingModel(model_name)
