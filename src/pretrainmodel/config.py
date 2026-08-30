"""Typed, strictly validated run configuration.

Configuration is TOML and loading is strict in both directions: an unknown key is
a hard error (a typo must never silently disable a gate) and every declared field
is type-checked.  A resolved config serialises to a canonical form and hashes to a
stable digest recorded in every run manifest, so any artifact can be traced back
to the exact configuration that produced it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CheckpointConfig",
    "Config",
    "ConfigError",
    "CostConfig",
    "DataConfig",
    "DeterminismConfig",
    "DistributedConfig",
    "ModelConfig",
    "OptimConfig",
    "RunConfig",
    "TrainConfig",
    "config_hash",
    "load_config",
    "to_dict",
]


class ConfigError(ValueError):
    """A configuration file is structurally or semantically invalid."""


_VALID_DTYPES = ("fp32", "bf16")
_VALID_BACKENDS = ("gloo", "nccl")


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Identity and provenance settings for a single run."""

    name: str
    seed: int = 1234
    # A formal run is one whose artifacts may be cited as evidence.  It refuses to
    # start on a dirty tree or an unverified data manifest.  Exploratory runs set
    # this false and are marked non-citable in their manifest.
    formal: bool = False


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Dataset location and windowing.  Defines the logical sample space."""

    root: str
    context_steps: int
    horizon_steps: int
    num_sensors: int
    num_features: int = 1
    drop_last: bool = True
    shuffle: bool = True
    num_workers: int = 0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Spatiotemporal Transformer shape."""

    d_model: int
    num_layers: int
    num_heads: int
    ffn_mult: int = 4
    dropout: float = 0.0
    max_sensors: int = 4096
    time_features: bool = True


@dataclass(frozen=True, slots=True)
class OptimConfig:
    """Optimiser and learning-rate schedule."""

    lr: float
    warmup_steps: int
    total_steps: int
    weight_decay: float = 0.01
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    grad_clip: float = 1.0
    min_lr_ratio: float = 0.1


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Step budget and batching.  Every run carries a hard step and wall-clock cap."""

    micro_batch_size: int
    max_steps: int
    grad_accum_steps: int = 1
    max_wall_seconds: int = 1800
    eval_every: int = 0
    log_every: int = 10
    dtype: str = "fp32"


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Checkpoint destination and cadence."""

    dir: str
    every_steps: int = 0
    keep_last: int = 3


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    """Collective backend and topology expectations."""

    backend: str = "gloo"
    # Asserted, then verified against observed hostnames.  A run that claims
    # distinct hosts but observes one host fails rather than quietly logging a
    # multi-node claim that is not true.
    expect_distinct_hosts: bool = False
    expect_world_size: int = 0


@dataclass(frozen=True, slots=True)
class DeterminismConfig:
    """Controls for the deterministic same-world-size oracle."""

    deterministic_algorithms: bool = False
    async_checkpoint: bool = False


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Billing assumption recorded with every run for the cost ledger."""

    sku: str = "local-cpu"
    rate_usd_per_gpu_hour: float = 0.0


@dataclass(frozen=True, slots=True)
class Config:
    """A fully resolved run configuration."""

    run: RunConfig
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig
    checkpoint: CheckpointConfig
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    determinism: DeterminismConfig = field(default_factory=DeterminismConfig)
    cost: CostConfig = field(default_factory=CostConfig)

    @property
    def seq_len(self) -> int:
        """Total window length fed to the model."""
        return self.data.context_steps + self.data.horizon_steps

    def global_batch_size(self, world_size: int) -> int:
        """Effective global batch, which must be recorded in the run manifest."""
        return self.train.micro_batch_size * self.train.grad_accum_steps * world_size


# --------------------------------------------------------------------------- #
# Strict construction
# --------------------------------------------------------------------------- #


def _coerce(hint: Any, value: Any, where: str) -> Any:
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a table, got {type(value).__name__}")
        return _build(hint, value, where)

    origin = typing.get_origin(hint)
    if origin in (list, Sequence):
        args = typing.get_args(hint)
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        return [_coerce(args[0], item, f"{where}[{i}]") for i, item in enumerate(value)]

    # bool before int: bool is a subclass of int and must not satisfy an int field.
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected bool, got {type(value).__name__}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected int, got {type(value).__name__}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where}: expected float, got {type(value).__name__}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected str, got {type(value).__name__}")
        return value

    raise ConfigError(f"{where}: unsupported field type {hint!r}")


def _build[T](cls: type[T], raw: Mapping[str, Any], path: str) -> T:
    if not dataclasses.is_dataclass(cls):  # pragma: no cover - programming error
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = typing.get_type_hints(cls)
    fields = dataclasses.fields(cls)
    known = {f.name for f in fields}

    unknown = sorted(set(raw) - known)
    if unknown:
        location = f"[{path}]" if path else "top level"
        raise ConfigError(
            f"unknown key(s) at {location}: {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(known))}"
        )

    kwargs: dict[str, Any] = {}
    for f in fields:
        where = f"{path}.{f.name}" if path else f.name
        if f.name in raw:
            kwargs[f.name] = _coerce(hints[f.name], raw[f.name], where)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"missing required key: {where}")
    return typing.cast(T, cls(**kwargs))


# --------------------------------------------------------------------------- #
# Cross-field invariants
# --------------------------------------------------------------------------- #


def validate(cfg: Config) -> None:
    """Enforce invariants that no single field can express.

    Raises ``ConfigError`` on the first violation.  These are the checks that stop
    a run from producing evidence that looks valid but is not.
    """
    problems: list[str] = []

    if cfg.model.d_model % cfg.model.num_heads != 0:
        problems.append(
            f"model.d_model ({cfg.model.d_model}) must be divisible by "
            f"model.num_heads ({cfg.model.num_heads})"
        )
    if cfg.data.num_sensors > cfg.model.max_sensors:
        problems.append(
            f"data.num_sensors ({cfg.data.num_sensors}) exceeds "
            f"model.max_sensors ({cfg.model.max_sensors})"
        )
    if cfg.data.context_steps < 1 or cfg.data.horizon_steps < 1:
        problems.append("data.context_steps and data.horizon_steps must both be >= 1")
    if cfg.optim.warmup_steps > cfg.optim.total_steps:
        problems.append(
            f"optim.warmup_steps ({cfg.optim.warmup_steps}) exceeds "
            f"optim.total_steps ({cfg.optim.total_steps})"
        )
    if cfg.train.max_steps > cfg.optim.total_steps:
        problems.append(
            f"train.max_steps ({cfg.train.max_steps}) exceeds the schedule horizon "
            f"optim.total_steps ({cfg.optim.total_steps}); the LR schedule would be undefined"
        )
    if cfg.train.grad_accum_steps < 1 or cfg.train.micro_batch_size < 1:
        problems.append("train.grad_accum_steps and train.micro_batch_size must be >= 1")
    if cfg.train.max_steps < 1:
        problems.append("train.max_steps must be >= 1 (every run needs a hard step cap)")
    if cfg.train.max_wall_seconds < 1:
        problems.append("train.max_wall_seconds must be >= 1 (every run needs a wall-clock cap)")
    if cfg.train.dtype not in _VALID_DTYPES:
        problems.append(f"train.dtype must be one of {_VALID_DTYPES}, got {cfg.train.dtype!r}")
    if cfg.distributed.backend not in _VALID_BACKENDS:
        problems.append(
            f"distributed.backend must be one of {_VALID_BACKENDS}, got {cfg.distributed.backend!r}"
        )
    if len(cfg.optim.betas) != 2:
        problems.append(f"optim.betas must have exactly 2 entries, got {len(cfg.optim.betas)}")
    if not 0.0 <= cfg.optim.min_lr_ratio <= 1.0:
        problems.append("optim.min_lr_ratio must lie in [0, 1]")

    # The deterministic oracle (spec 8.1) is only meaningful under FP32, synchronous
    # checkpointing and a fixed data order.  Asking for it otherwise is a config bug,
    # not a tolerance to be relaxed at compare time.
    if cfg.determinism.deterministic_algorithms:
        if cfg.train.dtype != "fp32":
            problems.append(
                "determinism.deterministic_algorithms requires train.dtype = 'fp32'; "
                f"got {cfg.train.dtype!r}"
            )
        if cfg.determinism.async_checkpoint:
            problems.append(
                "determinism.deterministic_algorithms is incompatible with "
                "determinism.async_checkpoint"
            )
    if cfg.distributed.expect_distinct_hosts and cfg.distributed.expect_world_size < 2:
        problems.append(
            "distributed.expect_distinct_hosts requires distributed.expect_world_size >= 2"
        )

    if problems:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def to_dict(cfg: Config) -> dict[str, Any]:
    """Return the config as a plain nested dict, suitable for JSON serialisation."""
    return dataclasses.asdict(cfg)


def canonical_json(cfg: Config) -> str:
    """Canonical serialisation used for hashing: sorted keys, no insignificant space."""
    return json.dumps(to_dict(cfg), sort_keys=True, separators=(",", ":"))


def config_hash(cfg: Config) -> str:
    """Stable SHA-256 of the resolved config.

    Recorded in every run manifest.  Two runs sharing a config hash were configured
    identically; the digest is over the *resolved* config, so a default that changes
    in a later version changes the hash, which is the intended behaviour.
    """
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


def from_mapping(raw: Mapping[str, Any]) -> Config:
    """Build and validate a config from an already-parsed mapping."""
    cfg = _build(Config, raw, "")
    validate(cfg)
    return cfg


def load_config(path: str | Path) -> Config:
    """Load, strictly parse and validate a TOML configuration file."""
    p = Path(path)
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {p}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{p}: malformed TOML: {exc}") from exc
    try:
        return from_mapping(raw)
    except ConfigError as exc:
        raise ConfigError(f"{p}: {exc}") from exc
