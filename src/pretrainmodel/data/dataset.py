"""Map sample IDs to tensors.

The dataset is addressed *by sample ID*, not by integer position.  Positional
indexing would make the exactly-once invariant unverifiable across a world-size
change, because index 7 means a different window at every world size, whereas a
sample ID means the same window everywhere.

Shards are cached by path with a small LRU so a rank reading a strided slice of an
epoch does not reopen the same file for every sample.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from pretrainmodel.data.manifest import ShardManifest
from pretrainmodel.data.shard import ShardData, ShardedIndex, load_shard

__all__ = ["Batch", "WindowDataset", "collate"]


@dataclass(frozen=True, slots=True)
class Batch:
    """One training batch.  ``sample_ids`` travels with the tensors so the coverage
    collector observes what was *actually* consumed, not what was scheduled."""

    context_values: Tensor  # (B, Tc, S, F)
    context_observed: Tensor  # (B, Tc, S)
    target_values: Tensor  # (B, Th, S, F)
    target_observed: Tensor  # (B, Th, S)
    sample_ids: list[str]

    def to(self, device: torch.device) -> Batch:
        return Batch(
            context_values=self.context_values.to(device),
            context_observed=self.context_observed.to(device),
            target_values=self.target_values.to(device),
            target_observed=self.target_observed.to(device),
            sample_ids=self.sample_ids,
        )

    def __len__(self) -> int:
        return len(self.sample_ids)


class WindowDataset:
    """Resolve sample IDs to context/target tensors."""

    def __init__(
        self,
        root: Path,
        manifest: ShardManifest,
        index: ShardedIndex,
        *,
        cache_size: int = 4,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.index = index
        self._refs = index.by_id()
        self._cache: OrderedDict[str, ShardData] = OrderedDict()
        self._cache_size = max(1, cache_size)

    def _shard(self, rel_path: str) -> ShardData:
        cached = self._cache.get(rel_path)
        if cached is not None:
            self._cache.move_to_end(rel_path)
            return cached
        data = load_shard(self.root / rel_path)
        self._cache[rel_path] = data
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return data

    def get(self, sample_id: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ref = self._refs.get(sample_id)
        if ref is None:
            raise KeyError(f"unknown sample id {sample_id!r}")
        shard = self._shard(ref.shard_path)
        ctx = self.manifest.context_steps
        seq = self.manifest.seq_len
        values, observed = shard.window(ref.offset, seq)
        return (
            torch.from_numpy(np.ascontiguousarray(values[:ctx])),
            torch.from_numpy(np.ascontiguousarray(observed[:ctx])),
            torch.from_numpy(np.ascontiguousarray(values[ctx:seq])),
            torch.from_numpy(np.ascontiguousarray(observed[ctx:seq])),
        )

    def batch(self, sample_ids: list[str]) -> Batch:
        parts = [self.get(sid) for sid in sample_ids]
        return collate(parts, sample_ids)


def collate(parts: list[tuple[Tensor, Tensor, Tensor, Tensor]], sample_ids: list[str]) -> Batch:
    return Batch(
        context_values=torch.stack([p[0] for p in parts]),
        context_observed=torch.stack([p[1] for p in parts]),
        target_values=torch.stack([p[2] for p in parts]),
        target_observed=torch.stack([p[3] for p in parts]),
        sample_ids=list(sample_ids),
    )
