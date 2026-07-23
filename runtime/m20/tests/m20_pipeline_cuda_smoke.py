#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "sglang"
    / "srt"
    / "layers"
    / "moe"
    / "kt_group_expert_buffer.py"
)
SPEC = importlib.util.spec_from_file_location(
    "kt_group_pipeline_cuda_under_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

KTGroupBufferConfig = MODULE.KTGroupBufferConfig
KTGroupExpertBufferManager = MODULE.KTGroupExpertBufferManager


class CudaLayer(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(
            torch.empty(2, 4, 3, dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.empty(2, 3, 2, dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )


class CudaWrapper:
    def __init__(self, device: torch.device, num_experts: int):
        self.num_gpu_experts = 2
        self.gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool)
        self.gpu_experts_mask[:2] = True
        self.logical_to_gpu_index = torch.full(
            (num_experts,), -1, dtype=torch.int32
        )
        self.logical_to_gpu_index[:2] = torch.arange(2, dtype=torch.int32)
        self.gpu_index_to_logical = torch.arange(2, dtype=torch.int32)
        self.gpu_experts_mask_cuda = self.gpu_experts_mask.to(device=device)
        self.logical_to_gpu_index_cuda = self.logical_to_gpu_index.to(
            device=device
        )
        self.wrapper = SimpleNamespace(
            gpu_experts_mask=self.gpu_experts_mask.clone()
        )


class PinnedLoader:
    group_async_host_memory_enabled = True

    def prefetch_experts(self, _layer_idx: int, _expert_ids) -> int:
        return 0

    def prepare_expert_for_group_slot(
        self,
        *,
        layer_idx: int,
        logical_expert_id: int,
        w13_dst: torch.Tensor,
        w2_dst: torch.Tensor,
    ):
        value = float(layer_idx * 20 + logical_expert_id)
        w13 = torch.full(
            tuple(w13_dst.shape),
            value,
            dtype=w13_dst.dtype,
            device="cpu",
            pin_memory=True,
        )
        w2 = torch.full(
            tuple(w2_dst.shape),
            value,
            dtype=w2_dst.dtype,
            device="cpu",
            pin_memory=True,
        )
        shard = int(w13.shape[0] // 2)
        return {
            "source": "pinned-smoke",
            "prefetch_wait_ms": 0.0,
            "h2d_bytes": (
                w13.numel() * w13.element_size()
                + w2.numel() * w2.element_size()
            ),
            "tensors": (w13[:shard], w13[shard:], w2),
            "keepalive": [w13, w2],
            "async_h2d_capable": True,
        }

    def enqueue_prepared_group_expert(
        self,
        *,
        prepared,
        w13_dst: torch.Tensor,
        w2_dst: torch.Tensor,
        stream: torch.cuda.Stream,
    ):
        gate, up, down = prepared["tensors"]
        with torch.cuda.stream(stream):
            w13_dst[: gate.shape[0]].copy_(gate, non_blocking=True)
            w13_dst[gate.shape[0] :].copy_(up, non_blocking=True)
            w2_dst.copy_(down, non_blocking=True)
        return prepared


def action_payload(manager, ticket_id: int, plans: dict, source: dict) -> dict:
    return {
        "ticket_id": ticket_id,
        "group_id": 0,
        "layer_plans": [[0, list(plans[0])]],
        "plan_hash": manager._plan_hash(0, plans),
        "logical_source_plans": [[0, list(source[0])]],
        "logical_source_buffer_id": 0,
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("M20 ticket pipeline CUDA smoke: SKIP (CUDA unavailable)")
        return 0

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    manager = KTGroupExpertBufferManager(
        KTGroupBufferConfig(
            num_layers=1,
            num_experts=11,
            group_size=1,
            slots_per_layer=2,
            buffer_count=2,
            load_mode="async",
            miss_policy="block",
            prefetch_policy="static",
            oracle_required=False,
            materialization="delta",
            max_replacements=1,
        ),
        strict_runtime=False,
    )
    layer = CudaLayer(device)
    manager.register_layer(0, layer, CudaWrapper(device, 11))
    loader = PinnedLoader()
    first = {0: (0, 1)}
    second = {0: (0, 2)}

    manager.begin_action(
        group_id=0,
        num_tokens=8,
        loader=loader,
        provider=None,
        plans=first,
        ticket_id=0,
    )
    assert manager.activate_group(0) == "gpu"
    assert manager.prefetch_action(action_payload(manager, 1, second, first))
    manager.finish_group(0)
    manager.end_action(0)

    manager.begin_action(
        group_id=0,
        num_tokens=8,
        loader=loader,
        provider=None,
        plans=second,
        ticket_id=1,
        logical_source_plans=first,
        logical_source_buffer_id=0,
    )
    adopted = manager._adopted_prefetch_record
    assert adopted is not None and adopted.ready_event is not None
    assert adopted.asynchronous
    assert manager.activate_group(0) == "gpu"
    materialization = manager.action_materialization_snapshot()
    assert materialization["h2d_experts"] == 1
    assert materialization["d2d_experts"] == 1
    torch.cuda.synchronize(device)
    assert float(layer.w13_weight[0, 0, 0]) == 0.0
    assert float(layer.w13_weight[1, 0, 0]) == 2.0
    manager.finish_group(0)
    metrics = manager.end_action(0)
    assert metrics["pipeline_prefetch_adopted"] == 1
    assert not manager.has_pending_action
    torch.cuda.synchronize(device)
    print("M20 ticket pipeline CUDA smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
