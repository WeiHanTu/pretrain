"""FSDP2 sharding over an explicit DeviceMesh.

Each Transformer block is sharded before the root module (spec 6).  The order
matters: ``fully_shard`` applied to the root first would treat the whole model as
one flat parameter group, losing the per-block communication/compute overlap that
is the point of sharding blockwise.

The mesh is constructed explicitly rather than inferred.  An inferred mesh is the
easiest way to end up with a topology nobody can describe afterwards, which is the
opposite of what this project exists to demonstrate.

CPU note: ``fully_shard`` works over a Gloo CPU mesh and produces genuinely sharded
DTensor parameters (a (16, 8) weight becomes (8, 8) local on 2 ranks).  That makes
the whole *correctness* half of Phase B verifiable with no GPU -- only throughput,
memory and scaling numbers need real hardware.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

__all__ = [
    "build_device_mesh",
    "is_sharded",
    "shard_model",
    "sharding_report",
]


def build_device_mesh(world_size: int | None = None) -> DeviceMesh:
    """Create a 1-D mesh over the current process group.

    One dimension only: this project deliberately excludes tensor, pipeline and
    context parallelism (intend.md non-goals). A 1-D mesh is what FSDP2 needs and
    claiming more dimensions than are exercised would be dishonest.
    """
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("build_device_mesh requires an initialised process group")
    size = world_size if world_size is not None else dist.get_world_size()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return init_device_mesh(device_type, (size,), mesh_dim_names=("dp",))


def shard_model(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    """Apply FSDP2 to each Transformer block, then to the root module."""
    from torch.distributed.fsdp import fully_shard

    blocks = getattr(model, "blocks", None)
    if blocks is not None:
        for block in blocks:
            fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def is_sharded(model: nn.Module) -> bool:
    """Whether any parameter is a DTensor.

    Used to keep the manifest honest: a run that believes it sharded but holds
    plain replicated tensors did not exercise FSDP2 at all.
    """
    from torch.distributed.tensor import DTensor

    return any(isinstance(p, DTensor) for p in model.parameters())


def sharding_report(model: nn.Module) -> dict[str, object]:
    """Evidence that parameters are genuinely partitioned, not merely replicated.

    Records global versus local element counts. If ``local == global`` on more than
    one rank the parameters are replicated, and any memory-saving claim would be
    false.
    """
    from torch.distributed.tensor import DTensor

    total_global = 0
    total_local = 0
    sharded_params = 0
    examples: list[dict[str, object]] = []

    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            local = param.to_local()
            global_elements = math.prod(int(d) for d in param.shape)
            local_elements = math.prod(int(d) for d in local.shape)
            total_global += global_elements
            total_local += local_elements
            if local_elements < global_elements:
                sharded_params += 1
            if len(examples) < 3:
                examples.append(
                    {
                        "name": name,
                        "global_shape": list(param.shape),
                        "local_shape": list(local.shape),
                        "placements": [str(pl) for pl in param.placements],
                    }
                )
        else:
            n = param.numel()
            total_global += n
            total_local += n

    return {
        "is_sharded": sharded_params > 0,
        "sharded_parameter_tensors": sharded_params,
        "global_elements": total_global,
        "local_elements": total_local,
        "local_fraction": (total_local / total_global) if total_global else 0.0,
        "examples": examples,
    }
