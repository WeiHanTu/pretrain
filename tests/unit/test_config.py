"""Configuration loading, strictness and invariants.

These tests exist because a silently accepted bad key is the cheapest way to
produce evidence that looks valid and is not.  A typo in ``deterministic_algorithms``
would turn the bit-exactness gate into an unconstrained run that still reports pass.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from pretrainmodel.config import (
    Config,
    ConfigError,
    config_hash,
    from_mapping,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _minimal() -> dict[str, Any]:
    return {
        "run": {"name": "t"},
        "data": {
            "root": "data/x",
            "context_steps": 12,
            "horizon_steps": 12,
            "num_sensors": 8,
        },
        "model": {"d_model": 64, "num_layers": 2, "num_heads": 4, "max_sensors": 8},
        "optim": {"lr": 1e-3, "warmup_steps": 2, "total_steps": 10},
        "train": {"micro_batch_size": 2, "max_steps": 10},
        "checkpoint": {"dir": "checkpoints/t"},
    }


# --------------------------------------------------------------------------- #
# Shipped configs
# --------------------------------------------------------------------------- #


def test_config_dir_is_not_empty() -> None:
    assert list(CONFIG_DIR.glob("*.toml")), "no shipped configs found"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.toml")), ids=lambda p: p.name)
def test_shipped_configs_load(path: Path) -> None:
    """Every config in the repo must parse and satisfy all invariants."""
    cfg = load_config(path)
    assert cfg.run.name
    assert cfg.train.max_steps >= 1
    assert cfg.train.max_wall_seconds >= 1


def test_deterministic_config_is_actually_deterministic() -> None:
    """The oracle config must keep the settings its claim depends on.

    Guards against someone relaxing fp32 or enabling async checkpointing to make a
    failing bit-exactness gate pass.
    """
    cfg = load_config(CONFIG_DIR / "deterministic_resume.toml")
    assert cfg.determinism.deterministic_algorithms is True
    assert cfg.determinism.async_checkpoint is False
    assert cfg.train.dtype == "fp32"
    assert cfg.data.shuffle is False
    assert cfg.model.dropout == 0.0


def test_multinode_config_asserts_distinct_hosts() -> None:
    """The two-node config must make its topology claim falsifiable."""
    cfg = load_config(CONFIG_DIR / "two_node_l4.toml")
    assert cfg.distributed.expect_distinct_hosts is True
    assert cfg.distributed.expect_world_size >= 2


# --------------------------------------------------------------------------- #
# Strictness
# --------------------------------------------------------------------------- #


def test_unknown_key_is_rejected_and_named() -> None:
    raw = _minimal()
    raw["train"]["max_stpes"] = 10  # typo
    with pytest.raises(ConfigError) as exc:
        from_mapping(raw)
    assert "max_stpes" in str(exc.value)
    assert "train" in str(exc.value)


def test_unknown_top_level_section_is_rejected() -> None:
    raw = _minimal()
    raw["determinsm"] = {"deterministic_algorithms": True}
    with pytest.raises(ConfigError, match="determinsm"):
        from_mapping(raw)


def test_missing_required_key_is_rejected() -> None:
    raw = _minimal()
    del raw["train"]["max_steps"]
    with pytest.raises(ConfigError, match=r"train\.max_steps"):
        from_mapping(raw)


def test_bool_is_not_accepted_for_int_field() -> None:
    """bool subclasses int in Python; the loader must not let that through."""
    raw = _minimal()
    raw["train"]["max_steps"] = True
    with pytest.raises(ConfigError, match="expected int"):
        from_mapping(raw)


def test_int_is_not_accepted_for_bool_field() -> None:
    raw = _minimal()
    raw["run"]["formal"] = 1
    with pytest.raises(ConfigError, match="expected bool"):
        from_mapping(raw)


def test_string_is_not_accepted_for_float_field() -> None:
    raw = _minimal()
    raw["optim"]["lr"] = "1e-3"
    with pytest.raises(ConfigError, match="expected float"):
        from_mapping(raw)


def test_int_is_accepted_for_float_field() -> None:
    raw = _minimal()
    raw["optim"]["lr"] = 1
    assert from_mapping(raw).optim.lr == 1.0


def test_missing_file_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(CONFIG_DIR / "does_not_exist.toml")


def test_malformed_toml_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[run\nname = 'x'")
    with pytest.raises(ConfigError, match="malformed TOML"):
        load_config(bad)


# --------------------------------------------------------------------------- #
# Cross-field invariants
# --------------------------------------------------------------------------- #


def test_heads_must_divide_d_model() -> None:
    raw = _minimal()
    raw["model"]["num_heads"] = 5
    with pytest.raises(ConfigError, match="divisible"):
        from_mapping(raw)


def test_sensors_must_fit_embedding_table() -> None:
    raw = _minimal()
    raw["data"]["num_sensors"] = 99
    raw["model"]["max_sensors"] = 8
    with pytest.raises(ConfigError, match="max_sensors"):
        from_mapping(raw)


def test_warmup_may_not_exceed_schedule() -> None:
    raw = _minimal()
    raw["optim"]["warmup_steps"] = 99
    with pytest.raises(ConfigError, match="warmup_steps"):
        from_mapping(raw)


def test_max_steps_may_not_exceed_schedule_horizon() -> None:
    raw = _minimal()
    raw["train"]["max_steps"] = 999
    with pytest.raises(ConfigError, match="LR schedule would be undefined"):
        from_mapping(raw)


def test_deterministic_mode_rejects_bf16() -> None:
    """bf16 reductions are not associative, so exact equality cannot be claimed."""
    raw = _minimal()
    raw["train"]["dtype"] = "bf16"
    raw["determinism"] = {"deterministic_algorithms": True}
    with pytest.raises(ConfigError, match=r"requires train\.dtype"):
        from_mapping(raw)


def test_deterministic_mode_rejects_async_checkpoint() -> None:
    raw = _minimal()
    raw["determinism"] = {"deterministic_algorithms": True, "async_checkpoint": True}
    with pytest.raises(ConfigError, match="incompatible"):
        from_mapping(raw)


def test_distinct_hosts_requires_world_size_two() -> None:
    """A single-rank run can never be evidence of crossing a network boundary."""
    raw = _minimal()
    raw["distributed"] = {"expect_distinct_hosts": True, "expect_world_size": 1}
    with pytest.raises(ConfigError, match="expect_world_size"):
        from_mapping(raw)


def test_unknown_dtype_is_rejected() -> None:
    raw = _minimal()
    raw["train"]["dtype"] = "fp8"
    with pytest.raises(ConfigError, match=r"train\.dtype"):
        from_mapping(raw)


def test_unknown_backend_is_rejected() -> None:
    raw = _minimal()
    raw["distributed"] = {"backend": "mpi"}
    with pytest.raises(ConfigError, match=r"distributed\.backend"):
        from_mapping(raw)


def test_all_invariant_violations_are_reported_together() -> None:
    """A config with several faults reports all of them, not just the first."""
    raw = _minimal()
    raw["model"]["num_heads"] = 5
    raw["optim"]["warmup_steps"] = 999
    with pytest.raises(ConfigError) as exc:
        from_mapping(raw)
    assert "divisible" in str(exc.value)
    assert "warmup_steps" in str(exc.value)


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def test_config_hash_is_stable_across_loads() -> None:
    a = from_mapping(_minimal())
    b = from_mapping(_minimal())
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_with_any_value() -> None:
    base = from_mapping(_minimal())
    raw = _minimal()
    raw["optim"]["lr"] = 2e-3
    assert config_hash(from_mapping(raw)) != config_hash(base)


def test_config_hash_is_independent_of_key_order() -> None:
    raw = _minimal()
    reordered = {k: raw[k] for k in reversed(list(raw))}
    assert config_hash(from_mapping(raw)) == config_hash(from_mapping(reordered))


def test_config_hash_is_sha256_shaped() -> None:
    digest = config_hash(from_mapping(_minimal()))
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_shipped_configs_have_distinct_hashes() -> None:
    """Two configs that hash alike would make their artifacts indistinguishable."""
    hashes = {p.name: config_hash(load_config(p)) for p in CONFIG_DIR.glob("*.toml")}
    assert len(set(hashes.values())) == len(hashes), hashes


# --------------------------------------------------------------------------- #
# Derived quantities
# --------------------------------------------------------------------------- #


def test_global_batch_size_accounts_for_accumulation_and_world_size() -> None:
    raw = _minimal()
    raw["train"]["micro_batch_size"] = 4
    raw["train"]["grad_accum_steps"] = 3
    cfg = from_mapping(raw)
    assert cfg.global_batch_size(world_size=2) == 24


def test_seq_len_is_context_plus_horizon() -> None:
    cfg: Config = from_mapping(_minimal())
    assert cfg.seq_len == cfg.data.context_steps + cfg.data.horizon_steps


def test_every_shipped_config_is_valid_toml() -> None:
    for path in CONFIG_DIR.glob("*.toml"):
        tomllib.loads(path.read_text())
