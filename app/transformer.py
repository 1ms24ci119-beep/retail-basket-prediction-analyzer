from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from app.data import Session
from app.vocab import Vocabulary


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    block_size: int = 12
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    pad_id: int = 0

    def __post_init__(self) -> None:
        if self.vocab_size < 3 or self.block_size < 2 or self.d_model < 4:
            raise ValueError("vocabulary, context, and embedding sizes are too small")
        if self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers < 1 or not 0 <= self.dropout < 1:
            raise ValueError("layers must be positive and dropout must be in [0, 1)")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.head_size = config.d_model // config.n_heads
        self.n_heads = config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)))

    def forward(self, values: torch.Tensor, valid_keys: torch.Tensor) -> torch.Tensor:
        batch_size, length, width = values.shape
        q, k, v = self.qkv(values).chunk(3, dim=-1)
        q = q.view(batch_size, length, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, length, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, length, self.n_heads, self.head_size).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        allowed = self.causal_mask[:length, :length][None, None, :, :] & valid_keys[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch_size, length, width)
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, values: torch.Tensor, valid_keys: torch.Tensor) -> torch.Tensor:
        values = values + self.attention(self.attention_norm(values), valid_keys)
        return values + self.feed_forward(self.feed_forward_norm(values))


class RetailTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, time]")
        batch_size, length = token_ids.shape
        if length > self.config.block_size:
            raise ValueError("input exceeds configured context length")
        positions = torch.arange(length, device=token_ids.device)
        values = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        values = self.input_dropout(values)
        valid_keys = token_ids.ne(self.config.pad_id)
        for block in self.blocks:
            values = block(values, valid_keys)
        return self.output_head(self.final_norm(values))

    def predict_proba(self, prefix: Iterable[int], product_ids: Iterable[int], device: torch.device | str = "cpu") -> dict[int, float]:
        ids = list(prefix)[-self.config.block_size:]
        if not ids:
            raise ValueError("a non-empty basket prefix is required")
        self.eval()
        with torch.no_grad():
            tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            scores = self(tensor)[0, -1]
            probabilities = torch.softmax(scores, dim=-1)
        selected_ids = list(product_ids)
        selected = probabilities[selected_ids]
        selected = selected / selected.sum()
        return {token_id: float(probability) for token_id, probability in zip(selected_ids, selected.cpu())}

    def predict(self, prefix: Iterable[int], product_ids: Iterable[int], top_k: int = 5, device: torch.device | str = "cpu") -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        distribution = self.predict_proba(prefix, product_ids, device)
        return sorted(distribution.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collate(rows: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    return pad_sequence(rows, batch_first=True, padding_value=pad_id)


def _loss_for_loader(model: RetailTransformer, loader: DataLoader, device: torch.device, pad_id: int) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            token_count = int(targets.ne(pad_id).sum().item())
            if token_count == 0:
                continue
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id, reduction="sum")
            total_loss += float(loss.item())
            total_tokens += token_count
    return total_loss / max(1, total_tokens)


def train_transformer(
    train_sessions: list[Session],
    validation_sessions: list[Session],
    vocabulary: Vocabulary,
    config: TransformerConfig | None = None,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 0.002,
    seed: int = 1729,
    device: str = "auto",
) -> tuple[RetailTransformer, dict[str, list[float]], dict[str, object]]:
    if not train_sessions or not validation_sessions:
        raise ValueError("training and validation sessions must both be non-empty")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs and batch size must be positive and learning rate must be positive")
    set_seed(seed)
    device_obj = choose_device(device)
    config = config or TransformerConfig(vocab_size=len(vocabulary.tokens), block_size=12)
    model = RetailTransformer(config).to(device_obj)
    pad_id = vocabulary.token_to_id[vocabulary.PAD]
    train_tensors = [torch.tensor(vocabulary.encode(session.items), dtype=torch.long) for session in train_sessions]
    validation_tensors = [torch.tensor(vocabulary.encode(session.items), dtype=torch.long) for session in validation_sessions]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_tensors, batch_size=batch_size, shuffle=True, generator=generator,
                              collate_fn=lambda rows: _collate(rows, pad_id))
    validation_loader = DataLoader(validation_tensors, batch_size=batch_size, shuffle=False,
                                   collate_fn=lambda rows: _collate(rows, pad_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    history: dict[str, list[float]] = {"train_loss": [], "validation_loss": []}
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        history["train_loss"].append(_loss_for_loader(model, train_loader, device_obj, pad_id))
        history["validation_loss"].append(_loss_for_loader(model, validation_loader, device_obj, pad_id))
    metadata = {
        "seed": seed,
        "device": str(device_obj),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_sessions": len(train_sessions),
        "validation_sessions": len(validation_sessions),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, history, metadata


def save_checkpoint(model: RetailTransformer, path: Path, vocabulary: Vocabulary, metadata: dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict(), "metadata": metadata or {}}, path)
    vocabulary_path = path.with_suffix(".vocab.json")
    vocabulary_path.write_text(json.dumps(vocabulary.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[RetailTransformer, Vocabulary, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = TransformerConfig(**checkpoint["config"])
    model = RetailTransformer(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    vocabulary = Vocabulary.from_dict(json.loads(path.with_suffix(".vocab.json").read_text(encoding="utf-8")))
    return model, vocabulary, checkpoint["metadata"]
