from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.data import CATALOGUE


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __post_init__(self) -> None:
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("vocabulary tokens must be unique")
        if self.tokens[:2] != (self.PAD, self.UNK):
            raise ValueError("PAD and UNK must be the first two tokens")

    @classmethod
    def from_catalogue(cls) -> Vocabulary:
        return cls((cls.PAD, cls.UNK, *(product.product_id for product in CATALOGUE)))

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    @property
    def id_to_token(self) -> dict[int, str]:
        return dict(enumerate(self.tokens))

    @property
    def product_ids(self) -> tuple[int, ...]:
        return tuple(range(2, len(self.tokens)))

    def encode(self, products: Iterable[str]) -> list[int]:
        mapping = self.token_to_id
        return [mapping.get(product, mapping[self.UNK]) for product in products]

    def decode(self, ids: Iterable[int], include_special: bool = False) -> list[str]:
        decoded = []
        for index in ids:
            token = self.tokens[index] if 0 <= index < len(self.tokens) else self.UNK
            if include_special or token not in (self.PAD, self.UNK):
                decoded.append(token)
        return decoded

    def pad(self, ids: Iterable[int], length: int, truncate: bool = True) -> list[int]:
        if length < 0:
            raise ValueError("length must be non-negative")
        values = list(ids)
        if len(values) > length:
            if not truncate:
                raise ValueError("sequence exceeds requested padding length")
            values = values[-length:] if length else []
        return values + [0] * (length - len(values))

    def product_index(self, product: str) -> int:
        return self.token_to_id.get(product, 1)

    def to_dict(self) -> dict[str, object]:
        return {"tokens": list(self.tokens)}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> Vocabulary:
        return cls(tuple(str(token) for token in value["tokens"]))
