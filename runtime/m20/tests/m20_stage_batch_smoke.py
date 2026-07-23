#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.kt_stage_batch import (
    StageBatchPayload,
    demux_stage_batch,
    pack_stage_batches,
)


RUNTIME_OWNER = object()
MODEL_CONFIG = SimpleNamespace()


def make_batch(request_id: int, token_count: int, split_index: int) -> ScheduleBatch:
    req = SimpleNamespace(
        rid=f"req-{request_id}",
        lora_id=None,
        kt_metadata={"prompt_global_index": request_id},
    )
    values = torch.arange(token_count, dtype=torch.int64) + request_id * 100
    batch = ScheduleBatch(
        reqs=[req],
        req_to_token_pool=RUNTIME_OWNER,
        token_to_kv_pool_allocator=RUNTIME_OWNER,
        tree_cache=RUNTIME_OWNER,
        model_config=MODEL_CONFIG,
        forward_mode=ForwardMode.SPLIT_PREFILL,
        input_ids=values,
        req_pool_indices=torch.tensor([request_id], dtype=torch.int64),
        seq_lens=torch.tensor([token_count + 7], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([token_count + 7], dtype=torch.int64),
        orig_seq_lens=torch.tensor([token_count + 7], dtype=torch.int32),
        out_cache_loc=values + 1000,
        seq_lens_sum=token_count + 7,
        prefix_lens=[7],
        extend_lens=[token_count],
        extend_num_tokens=token_count,
        extend_logprob_start_lens=[0],
        split_index=split_index,
        split_forward_count=2,
        device="cpu",
    )
    if split_index > 0:
        batch.split_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.SPLIT_PREFILL,
            batch_size=1,
            input_ids=batch.input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            orig_seq_lens=batch.orig_seq_lens,
            out_cache_loc=batch.out_cache_loc,
            seq_lens_sum=batch.seq_lens_sum,
            positions=torch.arange(token_count, dtype=torch.int64) + 7,
            extend_num_tokens=token_count,
            extend_seq_lens=torch.tensor([token_count], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([7], dtype=torch.int32),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_prefix_lens_cpu=[7],
            extend_seq_lens_cpu=[token_count],
            extend_logprob_start_lens_cpu=[0],
            hidden_states=torch.full((token_count, 3), float(request_id)),
            residual=torch.full((token_count, 3), float(request_id + 10)),
            split_index=split_index,
            lora_ids=[None],
            rids=[req.rid],
            kt_metadata_list=[dict(req.kt_metadata)],
            num_token_non_padded_cpu=token_count,
        )
    return batch


def test_group_zero_pack() -> None:
    payloads = [
        StageBatchPayload(make_batch(1, 3, 0)),
        StageBatchPayload(make_batch(2, 2, 0)),
    ]
    execution, context = pack_stage_batches(payloads, group_id=0, group_size=2)
    assert len(execution.reqs) == 2
    assert execution.extend_num_tokens == 5
    assert execution.input_ids.tolist() == [100, 101, 102, 200, 201]
    assert execution.req_pool_indices.tolist() == [1, 2]
    assert context.token_spans == ((0, 3), (3, 5))
    assert execution.split_forward_batch is None


def test_later_group_pack_and_demux() -> None:
    first = make_batch(3, 3, 2)
    second = make_batch(4, 2, 2)
    execution, context = pack_stage_batches(
        [StageBatchPayload(first), StageBatchPayload(second)],
        group_id=1,
        group_size=2,
    )
    packed = execution.split_forward_batch
    assert packed.batch_size == 2
    assert packed.hidden_states[:, 0].tolist() == [3.0, 3.0, 3.0, 4.0, 4.0]
    assert packed.extend_start_loc.tolist() == [0, 3]

    packed.hidden_states.add_(20)
    packed.residual.add_(30)
    packed.split_index = 4
    demux_stage_batch(execution, context)
    assert first.split_index == second.split_index == 4
    assert first.split_forward_batch.hidden_states[:, 0].tolist() == [23.0] * 3
    assert second.split_forward_batch.hidden_states[:, 0].tolist() == [24.0] * 2
    assert first.split_forward_batch.residual[:, 0].tolist() == [43.0] * 3
    assert second.split_forward_batch.residual[:, 0].tolist() == [44.0] * 2


def test_tail_group_uses_configured_group_size() -> None:
    tail = make_batch(5, 1, 4)
    tail.split_forward_count = 1
    execution, context = pack_stage_batches(
        [StageBatchPayload(tail)], group_id=2, group_size=2
    )
    assert execution is tail
    original_forward_batch = tail.split_forward_batch
    demux_stage_batch(execution, context)
    assert tail.split_forward_batch is original_forward_batch
    assert execution.split_index == 4
    assert execution.split_forward_count == 1


if __name__ == "__main__":
    test_group_zero_pack()
    test_later_group_pack_and_demux()
    test_tail_group_uses_configured_group_size()
    print("M20 stage batch smoke: PASS")
