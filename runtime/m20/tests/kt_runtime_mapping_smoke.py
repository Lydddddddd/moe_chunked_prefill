#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTEPWrapperMethod,
    RuntimeOraclePrefetchProvider,
    mask_expert_ids_and_weights_for_cpu,
)


class DummyWrapper:
    def __init__(self, mask: torch.Tensor):
        self.gpu_experts_mask = mask.clone()


def test_stateless_oracle_demand() -> None:
    routed = np.array(
        [
            [[0, 1], [2, 3]],
            [[1, 1], [2, 4]],
            [[1, 2], [4, 4]],
            [[3, 3], [5, 5]],
        ],
        dtype=np.int32,
    )
    row = {
        "ok": True,
        "request_id": 0,
        "prompt_global_index": 42,
        "routed_shape": list(routed.shape),
        "routed_experts_base64": base64.b64encode(routed.tobytes()).decode(
            "ascii"
        ),
    }
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "routes.jsonl"
        trace_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        provider = RuntimeOraclePrefetchProvider(
            str(trace_path), require_prompt_identity=False
        )
        demand, confidence = provider.demand_for_batch(
            kt_metadata_list=[{"prompt_global_index": 42}],
            prefix_lens=[1],
            extend_lens=[2],
            layer_indices=[0, 1],
        )
        assert demand[0] == {1: 3.0, 2: 1.0}
        assert demand[1] == {2: 1.0, 4: 3.0}
        assert confidence == {0: 1.0, 1: 1.0}
        missing_demand, missing_confidence = provider.demand_for_batch(
            kt_metadata_list=[{"prompt_global_index": 99}],
            prefix_lens=[0],
            extend_lens=[2],
            layer_indices=[0],
        )
        assert missing_demand == {0: {}}
        assert missing_confidence == {0: 0.0}


def main() -> int:
    test_stateless_oracle_demand()
    route_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, -1]], dtype=torch.long)
    route_weights = torch.ones_like(route_ids, dtype=torch.float32)
    route_mask = torch.tensor([False, True, False, True], dtype=torch.bool)
    cpu_ids, cpu_weights = mask_expert_ids_and_weights_for_cpu(
        route_ids, route_weights, route_mask
    )
    assert torch.equal(
        cpu_ids,
        torch.tensor([[0, -1, 2, -1], [-1, 2, -1, -1]], dtype=torch.long),
    )
    assert torch.equal(
        cpu_weights,
        torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
    )

    method = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
    method._kt_runtime_mapping_full_copy = True
    method.global_num_experts = 8
    method.tp_rank = 0
    method.gpu_experts_mask = torch.tensor(
        [True, False, True, False, True, False, True, False], dtype=torch.bool
    )
    method.logical_to_gpu_index = torch.tensor(
        [0, -1, 1, -1, 2, -1, 3, -1], dtype=torch.int32
    )
    method.gpu_index_to_logical = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
    method.gpu_experts_mask_cuda = method.gpu_experts_mask.clone()
    method.logical_to_gpu_index_cuda = method.logical_to_gpu_index.clone()
    method.wrapper = DummyWrapper(method.gpu_experts_mask)

    new_by_slot = [7, 2, 5, 6]
    method._apply_runtime_gpu_mapping_delta(
        new_by_slot=new_by_slot,
        evict_slots=[0, 2],
        device=torch.device("cpu"),
    )

    expected_mask = torch.tensor(
        [False, False, True, False, False, True, True, True], dtype=torch.bool
    )
    expected_mapping = torch.tensor(
        [-1, -1, 1, -1, -1, 2, 3, 0], dtype=torch.int32
    )
    expected_reverse = torch.tensor(new_by_slot, dtype=torch.int32)
    assert torch.equal(method.gpu_experts_mask, expected_mask)
    assert torch.equal(method.gpu_experts_mask_cuda, expected_mask)
    assert torch.equal(method.logical_to_gpu_index, expected_mapping)
    assert torch.equal(method.logical_to_gpu_index_cuda, expected_mapping)
    assert torch.equal(method.gpu_index_to_logical, expected_reverse)
    assert torch.equal(method.wrapper.gpu_experts_mask, expected_mask)
    print("kt runtime full mapping copy smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
