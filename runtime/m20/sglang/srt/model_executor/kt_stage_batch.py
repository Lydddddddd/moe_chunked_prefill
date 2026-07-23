# SPDX-License-Identifier: Apache-2.0
"""Pack and demux text-only M20-B stage execution batches."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


@dataclass
class StageBatchPayload:
    batch: ScheduleBatch
    continuation_reqs: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class StageExecutionContext:
    group_id: int
    payloads: Tuple[StageBatchPayload, ...]
    token_spans: Tuple[Tuple[int, int], ...]

    @property
    def state_count(self) -> int:
        return len(self.payloads)


def _cat_tensors(
    values: Sequence[Optional[torch.Tensor]], name: str
) -> Optional[torch.Tensor]:
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError(f"stage batch field {name} is present for only some states")
    return torch.cat([value for value in values if value is not None])


def _cat_lists(values: Sequence[Optional[Sequence[Any]]], name: str) -> Optional[list]:
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError(f"stage batch field {name} is present for only some states")
    return [item for value in values if value is not None for item in value]


def _validate_payload(
    payload: StageBatchPayload, group_id: int, group_size: int
) -> None:
    batch = payload.batch
    if len(batch.reqs) != 1:
        raise RuntimeError("M20-B stage states must own exactly one request")
    if not batch.forward_mode.is_split_prefill():
        raise RuntimeError("stage payload is not SPLIT_PREFILL")
    if batch.return_logprob:
        raise RuntimeError("M20-B0b does not support return_logprob")
    if batch.has_grammar or batch.spec_info is not None or batch.dllm_config is not None:
        raise RuntimeError("M20-B0b supports plain text generation only")
    if batch.decoding_reqs:
        raise RuntimeError("M20-B0b cannot pack mixed prefill/decode batches")
    if batch.input_embeds is not None or batch.token_type_ids is not None:
        raise RuntimeError("M20-B0b does not support embeddings or token type IDs")
    if batch.multimodal_inputs and any(
        value is not None for value in batch.multimodal_inputs
    ):
        raise RuntimeError("M20-B0b does not support multimodal inputs")
    if int(batch.split_index) < 0:
        raise RuntimeError("stage payload has a negative split cursor")
    expected_start = int(group_id) * int(group_size)
    if int(batch.split_index) != expected_start:
        raise RuntimeError(
            "stage payload group mismatch: "
            f"expected_start={expected_start} actual={batch.split_index}"
        )
    if int(batch.extend_num_tokens or 0) <= 0:
        raise RuntimeError("stage payload has no extend tokens")


def _token_spans(payloads: Sequence[StageBatchPayload]) -> Tuple[Tuple[int, int], ...]:
    spans = []
    cursor = 0
    for payload in payloads:
        count = int(payload.batch.extend_num_tokens or 0)
        spans.append((cursor, cursor + count))
        cursor += count
    return tuple(spans)


def _pack_model_specific_states(
    forward_batches: Sequence[ForwardBatch], token_counts: Sequence[int]
) -> Optional[Dict[str, Any]]:
    values = [batch.model_specific_states for batch in forward_batches]
    if all(value is None for value in values):
        return None
    if not all(isinstance(value, dict) for value in values):
        raise RuntimeError("model_specific_states must be dictionaries for all states")
    keys = set(values[0])
    if any(set(value) != keys for value in values[1:]):
        raise RuntimeError("model_specific_states keys differ across stage states")
    packed: Dict[str, Any] = {}
    for key in sorted(keys):
        items = [value[key] for value in values]
        if all(
            isinstance(item, torch.Tensor)
            and item.dim() > 0
            and int(item.shape[0]) == token_count
            for item, token_count in zip(items, token_counts)
        ):
            packed[key] = torch.cat(items)
        elif all(item is items[0] for item in items):
            packed[key] = items[0]
        else:
            raise RuntimeError(
                f"unsupported model_specific_states field for packing: {key}"
            )
    return packed


def _slice_model_specific_states(
    values: Optional[Dict[str, Any]], start: int, end: int, total_tokens: int
) -> Optional[Dict[str, Any]]:
    if values is None:
        return None
    result: Dict[str, Any] = {}
    for key, value in values.items():
        if (
            isinstance(value, torch.Tensor)
            and value.dim() > 0
            and int(value.shape[0]) == total_tokens
        ):
            result[key] = value[start:end]
        else:
            result[key] = value
    return result


def pack_stage_batches(
    payloads: Sequence[StageBatchPayload], group_id: int, group_size: int
) -> Tuple[ScheduleBatch, StageExecutionContext]:
    if not payloads:
        raise RuntimeError("cannot pack an empty stage cohort")
    payloads = tuple(payloads)
    for payload in payloads:
        _validate_payload(payload, group_id, group_size)

    batches = [payload.batch for payload in payloads]
    first = batches[0]
    split_index = int(first.split_index)
    split_count = int(first.split_forward_count)
    for batch in batches[1:]:
        if (
            batch.model_config is not first.model_config
            or batch.req_to_token_pool is not first.req_to_token_pool
            or batch.token_to_kv_pool_allocator is not first.token_to_kv_pool_allocator
            or batch.tree_cache is not first.tree_cache
        ):
            raise RuntimeError("stage states do not share runtime memory ownership")
        if int(batch.split_index) != split_index or int(batch.split_forward_count) != split_count:
            raise RuntimeError("stage states have different split cursors")

    spans = _token_spans(payloads)
    context = StageExecutionContext(
        group_id=int(group_id), payloads=payloads, token_spans=spans
    )
    if len(payloads) == 1:
        return first, context

    execution = ScheduleBatch(
        reqs=[req for batch in batches for req in batch.reqs],
        req_to_token_pool=first.req_to_token_pool,
        token_to_kv_pool_allocator=first.token_to_kv_pool_allocator,
        tree_cache=first.tree_cache,
        model_config=first.model_config,
        forward_mode=ForwardMode.SPLIT_PREFILL,
        enable_overlap=False,
        batch_is_full=False,
        chunked_req=None,
        sampling_info=None,
        input_ids=_cat_tensors([batch.input_ids for batch in batches], "input_ids"),
        req_pool_indices=_cat_tensors(
            [batch.req_pool_indices for batch in batches], "req_pool_indices"
        ),
        seq_lens=_cat_tensors([batch.seq_lens for batch in batches], "seq_lens"),
        seq_lens_cpu=_cat_tensors(
            [batch.seq_lens_cpu for batch in batches], "seq_lens_cpu"
        ),
        orig_seq_lens=_cat_tensors(
            [batch.orig_seq_lens for batch in batches], "orig_seq_lens"
        ),
        out_cache_loc=_cat_tensors(
            [batch.out_cache_loc for batch in batches], "out_cache_loc"
        ),
        seq_lens_sum=sum(int(batch.seq_lens_sum) for batch in batches),
        return_logprob=False,
        prefix_lens=_cat_lists([batch.prefix_lens for batch in batches], "prefix_lens"),
        extend_lens=_cat_lists([batch.extend_lens for batch in batches], "extend_lens"),
        extend_num_tokens=sum(int(batch.extend_num_tokens) for batch in batches),
        extend_logprob_start_lens=_cat_lists(
            [batch.extend_logprob_start_lens for batch in batches],
            "extend_logprob_start_lens",
        ),
        multimodal_inputs=_cat_lists(
            [batch.multimodal_inputs for batch in batches], "multimodal_inputs"
        ),
        decoding_reqs=None,
        split_index=split_index,
        split_prefill_finished=False,
        split_forward_count=split_count,
        seq_lens_cpu_cache=_cat_tensors(
            [batch.seq_lens_cpu for batch in batches], "seq_lens_cpu_cache"
        ),
        has_stream=any(batch.has_stream for batch in batches),
        has_grammar=False,
        device=first.device,
        spec_algorithm=first.spec_algorithm,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        is_prefill_only=all(batch.is_prefill_only for batch in batches),
        hicache_consumer_index=-1,
        dp_cooperation_info=first.dp_cooperation_info,
    )
    if split_index > 0:
        forward_batches = [batch.split_forward_batch for batch in batches]
        if not all(isinstance(value, ForwardBatch) for value in forward_batches):
            raise RuntimeError("later-stage payload lost its ForwardBatch")
        token_counts = [int(batch.extend_num_tokens) for batch in batches]
        total_tokens = sum(token_counts)
        start_locs = []
        cursor = 0
        for count in token_counts:
            start_locs.append(cursor)
            cursor += count
        reference = forward_batches[0]
        execution.split_forward_batch = dataclasses.replace(
            reference,
            forward_mode=ForwardMode.SPLIT_PREFILL,
            batch_size=len(execution.reqs),
            input_ids=execution.input_ids,
            req_pool_indices=execution.req_pool_indices,
            seq_lens=execution.seq_lens,
            seq_lens_cpu=execution.seq_lens_cpu,
            orig_seq_lens=execution.orig_seq_lens,
            out_cache_loc=execution.out_cache_loc,
            seq_lens_sum=execution.seq_lens_sum,
            positions=_cat_tensors(
                [batch.positions for batch in forward_batches], "positions"
            ),
            extend_num_tokens=total_tokens,
            extend_seq_lens=_cat_tensors(
                [batch.extend_seq_lens for batch in forward_batches],
                "extend_seq_lens",
            ),
            extend_prefix_lens=_cat_tensors(
                [batch.extend_prefix_lens for batch in forward_batches],
                "extend_prefix_lens",
            ),
            extend_start_loc=torch.tensor(
                start_locs,
                dtype=reference.extend_start_loc.dtype,
                device=reference.extend_start_loc.device,
            ),
            extend_prefix_lens_cpu=list(execution.prefix_lens),
            extend_seq_lens_cpu=list(execution.extend_lens),
            extend_logprob_start_lens_cpu=execution.extend_logprob_start_lens,
            hidden_states=_cat_tensors(
                [batch.hidden_states for batch in forward_batches], "hidden_states"
            ),
            residual=_cat_tensors(
                [batch.residual for batch in forward_batches], "residual"
            ),
            model_specific_states=_pack_model_specific_states(
                forward_batches, token_counts
            ),
            split_index=split_index,
            sampling_info=execution.sampling_info,
            lora_ids=[req.lora_id for req in execution.reqs],
            rids=[req.rid for req in execution.reqs],
            kt_metadata_list=[dict(req.kt_metadata or {}) for req in execution.reqs],
            num_token_non_padded_cpu=total_tokens,
        )

    return execution, context


def demux_stage_batch(
    execution: ScheduleBatch, context: StageExecutionContext
) -> None:
    if context.state_count == 1:
        if execution is not context.payloads[0].batch:
            raise RuntimeError("singleton stage execution lost its original batch")
        return
    packed = execution.split_forward_batch
    if not isinstance(packed, ForwardBatch):
        raise RuntimeError("stage execution has no packed ForwardBatch to demux")
    total_tokens = int(execution.extend_num_tokens)
    for req_index, (payload, (start, end)) in enumerate(
        zip(context.payloads, context.token_spans)
    ):
        batch = payload.batch
        count = end - start
        member = dataclasses.replace(
            packed,
            forward_mode=ForwardMode.SPLIT_PREFILL,
            batch_size=1,
            input_ids=batch.input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            orig_seq_lens=batch.orig_seq_lens,
            out_cache_loc=batch.out_cache_loc,
            seq_lens_sum=int(batch.seq_lens_sum),
            positions=packed.positions[start:end],
            extend_num_tokens=count,
            extend_seq_lens=packed.extend_seq_lens[req_index : req_index + 1],
            extend_prefix_lens=packed.extend_prefix_lens[req_index : req_index + 1],
            extend_start_loc=torch.zeros_like(
                packed.extend_start_loc[req_index : req_index + 1]
            ),
            extend_prefix_lens_cpu=list(batch.prefix_lens),
            extend_seq_lens_cpu=list(batch.extend_lens),
            extend_logprob_start_lens_cpu=batch.extend_logprob_start_lens,
            hidden_states=packed.hidden_states[start:end],
            residual=(
                None if packed.residual is None else packed.residual[start:end]
            ),
            model_specific_states=_slice_model_specific_states(
                packed.model_specific_states, start, end, total_tokens
            ),
            sampling_info=batch.sampling_info,
            lora_ids=[batch.reqs[0].lora_id],
            rids=[batch.reqs[0].rid],
            kt_metadata_list=[dict(batch.reqs[0].kt_metadata or {})],
            num_token_non_padded_cpu=count,
        )
        batch.split_forward_batch = member
        batch.split_index = int(packed.split_index)
        batch.seq_lens_cpu_cache = batch.seq_lens_cpu
        batch.split_prefill_finished = False


__all__ = [
    "StageBatchPayload",
    "StageExecutionContext",
    "demux_stage_batch",
    "pack_stage_batches",
]
