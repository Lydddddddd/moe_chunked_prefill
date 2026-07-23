# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).

Diagnostic / escape-hatch environment variables (KT-DEBUG-ONLY; not for prod):

    SGLANG_KT_HYBRID_TIMING=1
        Per-call wall-time breakdown of submit / mask / gpu / sync / merge
        / cpu_wait stages. Logged at DEBUG for layers (0, 5, 20, 35) on TP0.

    SGLANG_KT_HYBRID_TIMING_DEEP=1
        Insert torch.cuda.synchronize() at each timing stage so DEEP numbers
        reflect real GPU work rather than async-launch return time. Slows
        decode meaningfully; only enable for one-shot triage.

    SGLANG_KT_HYBRID_TIMING_ALL_LAYERS=1
        With normal hybrid timing enabled, collect every MoE layer instead of
        the default representative layers (0, 5, 20, 35).

    SGLANG_KT_HYBRID_NO_CPU_STREAM=1
        Collapse the CPU-experts CUDA stream onto the main stream. Useful
        when isolating regressions caused by the multi-stream submit path.

    SGLANG_KT_BYPASS_GPU_MOE=1
        Force GPU-experts apply() to a zero return; routed expert output
        comes purely from the CPU side. "Plan-C" fallback for diagnosing
        whether a regression sits in the GPU MoE path or the merge math.

    SGLANG_KT_RUNTIME_FOREGROUND_ORACLE=1
        At each configured placement-update boundary, choose the current
        layer's GPU experts from the metadata-aligned oracle trace. This is a
        synchronous diagnostic path and does not itself enable prefetch.

    SGLANG_KT_RUNTIME_KEEP_OLD_ON_PREFETCH_MISS=1
        When a correctness-safe staging prefetch is missing or not ready at a
        placement boundary, keep the old live slots instead of synchronously
        loading a replacement on the forward path.

    SGLANG_KT_RUNTIME_MAPPING_FULL_COPY=1
        Rebuild the tiny (num_experts-sized) mapping tables on CPU and commit
        them with two contiguous CUDA copies. This avoids several advanced-
        indexing CUDA kernels for each expert-slot change.

    SGLANG_KT_RUNTIME_MIN_GAIN_ENTRIES_PER_SLOT=N
        Skip an oracle-predicted slot replacement unless the new expert has at
        least N more routed entries than the expert it would evict. Zero keeps
        the original unconditional top-k replacement behavior.

    SGLANG_KT_RUNTIME_SKIP_BATCH_SELECTION_WITH_ORACLE=1
        When exact oracle metadata is active, use the current placement as the
        fallback instead of synchronously counting and ranking current-batch
        topk_ids on every stage boundary. Oracle predictions replace this
        fallback for valid target layers.

    SGLANG_KT_RUNTIME_DEFER_PREFETCH_AFTER_CPU_SUBMIT=1
        Keep current-layer placement commits at the start of MoE apply, but
        defer submission of future-layer H2D work until the current CPU expert
        task has been submitted. This avoids competing with the critical D2H
        staging and CPU task launch.

    SGLANG_KT_RUNTIME_REGISTER_CPU_TENSORS=1
        Register safetensors-backed CPU expert pages with CUDA in place. Unlike
        Tensor.pin_memory(), this does not copy the weights. Registrations are
        tied to the bounded CPU tensor cache and removed on cache eviction.
"""

import copy
import ctypes
import base64
import json
import logging
import os
import time
import uuid
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
from sglang.srt.utils import get_compiler_backend, is_cuda

if is_cuda():
    from sglang.jit_kernel import gptq_marlin_repack

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper, generate_gpu_experts_masks

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)

# Global cache for GPU experts masks (initialized once per session)
_KT_GPU_EXPERTS_MASKS: Optional[torch.Tensor] = None
_KT_RUNTIME_HF_EXPERT_LOADER = None
_KT_RUNTIME_ORACLE_PREFETCH = None
_KT_RUNTIME_WRAPPER_REGISTRY: Dict[int, Any] = {}
_KT_RUNTIME_FORWARD_CONTEXT = threading.local()


def _to_int_list(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    try:
        return [int(x) for x in list(value)]
    except Exception:
        return None


def set_runtime_forward_context_from_forward_batch(forward_batch: Any) -> None:
    """Expose scheduler chunk metadata to the KT MoE runtime.

    The MoE wrapper is several layers below the scheduler and normally only sees
    routed token ids. This thread-local context lets runtime oracle prefetch
    align a MoE call to exact request/chunk token ranges.
    """

    metadata_list = getattr(forward_batch, "kt_metadata_list", None) or []
    context = {
        "rids": list(getattr(forward_batch, "rids", None) or []),
        "kt_metadata_list": [
            dict(item) if isinstance(item, dict) else {} for item in metadata_list
        ],
        "extend_prefix_lens_cpu": _to_int_list(
            getattr(forward_batch, "extend_prefix_lens_cpu", None)
        ),
        "extend_seq_lens_cpu": _to_int_list(
            getattr(forward_batch, "extend_seq_lens_cpu", None)
        ),
        "extend_num_tokens": getattr(forward_batch, "extend_num_tokens", None),
        "batch_size": int(getattr(forward_batch, "batch_size", 0) or 0),
        "forward_mode": str(getattr(forward_batch, "forward_mode", "")),
    }
    _KT_RUNTIME_FORWARD_CONTEXT.value = context


def get_runtime_forward_context() -> Optional[Dict[str, Any]]:
    context = getattr(_KT_RUNTIME_FORWARD_CONTEXT, "value", None)
    return context if isinstance(context, dict) else None


class RuntimeOraclePrefetchProvider:
    """Runtime oracle provider backed by KT routed expert traces.

    The preferred path aligns runtime prefill chunks by scheduler metadata:
    request_id/prompt id + [token_start, token_end). This is required once the
    scheduler reorders chunks. If metadata is unavailable, the provider keeps a
    legacy order-based fallback for old experiments.
    """

    def __init__(self, trace_path: str):
        self.trace_path = Path(trace_path)
        if not self.trace_path.exists():
            raise FileNotFoundError(f"oracle routed trace not found: {self.trace_path}")
        try:
            self.num_experts = int(
                os.environ.get("SGLANG_KT_RUNTIME_ORACLE_PREFETCH_NUM_EXPERTS", "128")
            )
        except ValueError:
            self.num_experts = 128
        self.num_experts = max(1, self.num_experts)
        self.verbose = (
            os.environ.get("SGLANG_KT_RUNTIME_ORACLE_PREFETCH_VERBOSE", "0") == "1"
        )
        self.require_prompt_identity = os.environ.get(
            "SGLANG_KT_RUNTIME_ORACLE_PREFETCH_REQUIRE_PROMPT_IDENTITY", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.prompt_identity_manifest_path = os.environ.get(
            "SGLANG_KT_RUNTIME_ORACLE_PREFETCH_PROMPT_IDENTITY_MANIFEST", ""
        )
        self.prompt_identities = self._load_prompt_identities(
            self.prompt_identity_manifest_path
        )
        self._identity_warning_keys: set[str] = set()
        if self.require_prompt_identity and not self.prompt_identities:
            raise ValueError(
                "KT oracle prefetch prompt identity is required but no valid manifest was loaded"
            )
        self.trace_metadata = self._load_trace_metadata(self.trace_path)
        self.trace_prompt_offset = self._safe_int(
            self.trace_metadata.get("prompt_offset")
        )
        if self.trace_prompt_offset is None:
            self.trace_prompt_offset = 0
        (
            self.route_rows,
            self.routes_by_request_id,
            self.routes_by_prompt_global_index,
            self.routes_by_prompt_index,
            self.routes_by_rid,
        ) = self._load_routes(
            self.trace_path, prompt_offset=int(self.trace_prompt_offset)
        )
        self.routes = [row["routed"] for row in self.route_rows]
        if not self.routes:
            raise ValueError(f"no valid oracle routed rows in {self.trace_path}")
        self._lock = threading.Lock()
        self._request_idx = 0
        self._token_offset = 0
        self._step = -1
        self._active: Optional[Dict[str, int]] = None
        self._select_count = 0
        self._metadata_step_count = 0
        self._fallback_step_count = 0
        logger.warning(
            "KT runtime oracle prefetch loaded: trace=%s requests=%d "
            "num_experts=%d trace_prompt_offset=%d",
            self.trace_path,
            len(self.routes),
            self.num_experts,
            int(self.trace_prompt_offset),
        )

    @staticmethod
    def _load_trace_metadata(trace_path: Path) -> Dict[str, Any]:
        """Read collector metadata when an older trace omits per-row IDs."""

        for name in ("activation_stats.metadata.json", "summary.json"):
            metadata_path = trace_path.parent / name
            if not metadata_path.exists():
                continue
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return {}

    @staticmethod
    def _load_prompt_identities(path: str) -> Dict[str, Tuple[str, str]]:
        if not path:
            return {}
        manifest_path = Path(path)
        if not manifest_path.exists():
            logger.warning(
                "KT oracle prompt identity manifest does not exist: %s", manifest_path
            )
            return {}
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "Failed to load KT oracle prompt identity manifest %s: %s",
                manifest_path,
                exc,
            )
            return {}
        if not isinstance(payload, dict):
            return {}
        workload_id = str(payload.get("workload_id") or "")
        entries = payload.get("entries")
        if not workload_id or not isinstance(entries, dict):
            return {}
        identities: Dict[str, Tuple[str, str]] = {}
        for raw_index, raw_entry in entries.items():
            if not isinstance(raw_entry, dict):
                continue
            try:
                prompt_index = int(raw_entry.get("prompt_global_index", raw_index))
            except (TypeError, ValueError):
                continue
            prompt_hash = str(raw_entry.get("prompt_sha256") or "")
            if prompt_hash:
                identities[str(prompt_index)] = (workload_id, prompt_hash)
        logger.info(
            "Loaded %d KT oracle prompt identities for workload=%s from %s",
            len(identities),
            workload_id,
            manifest_path,
        )
        return identities

    def _load_routes(
        self, trace_path: Path, *, prompt_offset: int
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Any], Dict[int, Any], Dict[int, Any], Dict[str, Any]]:
        import numpy as np

        rows: List[Dict[str, Any]] = []
        by_request_id: Dict[int, Any] = {}
        by_prompt_global_index: Dict[int, Any] = {}
        by_prompt_index: Dict[int, Any] = {}
        by_rid: Dict[str, Any] = {}
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not row.get("ok", True):
                    continue
                encoded = row.get("routed_experts_base64")
                shape = row.get("routed_shape")
                if not encoded or not isinstance(shape, list) or len(shape) != 3:
                    continue
                flat = np.frombuffer(
                    base64.b64decode(str(encoded).encode("utf-8")),
                    dtype=np.int32,
                )
                expected = int(shape[0]) * int(shape[1]) * int(shape[2])
                if flat.size != expected:
                    logger.warning(
                        "KT runtime oracle row skipped due to shape mismatch: "
                        "request_id=%s expected=%d actual=%d",
                        row.get("request_id"),
                        expected,
                        flat.size,
                    )
                    continue
                routed = flat.reshape(int(shape[0]), int(shape[1]), int(shape[2]))
                request_id = int(row.get("request_id", len(rows)))
                prompt_global_index = self._safe_int(
                    row.get("prompt_global_index")
                )
                prompt_index = self._safe_int(row.get("prompt_index"))
                # Old collector traces contain only a local request ID. Their
                # sidecar metadata records the selected prompt offset, which
                # lets the runtime reconstruct a stable global prompt ID.
                if prompt_global_index is None:
                    prompt_global_index = (
                        prompt_index
                        if prompt_index is not None
                        else int(prompt_offset) + request_id
                    )
                if prompt_index is None:
                    prompt_index = prompt_global_index
                item = {
                    "request_id": request_id,
                    "prompt_global_index": prompt_global_index,
                    "prompt_index": prompt_index,
                    "rid": row.get("rid"),
                    "routed": routed,
                }
                rows.append(item)
                by_request_id[request_id] = routed
                if item["prompt_global_index"] is not None:
                    by_prompt_global_index[int(item["prompt_global_index"])] = routed
                if item["prompt_index"] is not None:
                    by_prompt_index[int(item["prompt_index"])] = routed
                if item["rid"] is not None:
                    by_rid[str(item["rid"])] = routed
        rows.sort(key=lambda item: int(item["request_id"]))
        return rows, by_request_id, by_prompt_global_index, by_prompt_index, by_rid

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _lookup_route_record(
        self, segment: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        prompt_global_index = self._safe_int(segment.get("prompt_global_index"))
        if (
            prompt_global_index is not None
            and prompt_global_index in self.routes_by_prompt_global_index
        ):
            return (
                {
                    "routed": self.routes_by_prompt_global_index[prompt_global_index],
                    "request_id": None,
                    "prompt_global_index": prompt_global_index,
                },
                "prompt_global_index",
            )
        prompt_index = self._safe_int(segment.get("prompt_index"))
        if prompt_index is not None and prompt_index in self.routes_by_prompt_index:
            return (
                {
                    "routed": self.routes_by_prompt_index[prompt_index],
                    "request_id": None,
                    "prompt_global_index": prompt_index,
                },
                "prompt_index",
            )
        rid = segment.get("rid")
        if rid is not None and str(rid) in self.routes_by_rid:
            return (
                {
                    "routed": self.routes_by_rid[str(rid)],
                    "request_id": None,
                    "prompt_global_index": None,
                },
                "rid",
            )
        request_id = self._safe_int(segment.get("request_id"))
        if request_id is not None and request_id in self.routes_by_request_id:
            return (
                {
                    "routed": self.routes_by_request_id[request_id],
                    "request_id": request_id,
                    "prompt_global_index": None,
                },
                "request_id_fallback",
            )
        return None, "missing"

    def _prompt_identity_matches(self, segment: Dict[str, Any]) -> bool:
        if not self.require_prompt_identity:
            return True
        prompt_index = self._safe_int(
            segment.get("prompt_global_index", segment.get("prompt_index"))
        )
        expected = self.prompt_identities.get(str(prompt_index))
        actual = (
            str(segment.get("workload_id") or ""),
            str(segment.get("prompt_sha256") or ""),
        )
        if expected == actual:
            return True
        warning_key = f"{prompt_index}:{actual[0]}:{actual[1][:12]}"
        if warning_key not in self._identity_warning_keys:
            self._identity_warning_keys.add(warning_key)
            logger.warning(
                "KT oracle prefetch prompt identity mismatch; skip route: "
                "prompt_global_index=%s expected=%s actual_workload=%s actual_hash=%s",
                prompt_index,
                expected,
                actual[0],
                actual[1][:12],
            )
        return False

    def _lookup_routed(self, segment: Dict[str, Any]) -> Optional[Any]:
        item, _ = self._lookup_route_record(segment)
        return item["routed"] if item is not None else None

    def _segments_from_context(
        self, context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not isinstance(context, dict):
            return []
        metadata_list = context.get("kt_metadata_list")
        prefix_lens = context.get("extend_prefix_lens_cpu")
        extend_lens = context.get("extend_seq_lens_cpu")
        if (
            not isinstance(metadata_list, list)
            or not isinstance(prefix_lens, list)
            or not isinstance(extend_lens, list)
        ):
            return []
        count = min(len(metadata_list), len(prefix_lens), len(extend_lens))
        segments: List[Dict[str, Any]] = []
        for idx in range(count):
            metadata = metadata_list[idx] if isinstance(metadata_list[idx], dict) else {}
            token_start = self._safe_int(prefix_lens[idx])
            extend_len = self._safe_int(extend_lens[idx])
            if token_start is None or extend_len is None or extend_len <= 0:
                continue
            token_end = int(token_start) + int(extend_len)
            segment = {
                "request_id": self._safe_int(metadata.get("request_id")),
                "prompt_index": self._safe_int(metadata.get("prompt_index")),
                "prompt_global_index": self._safe_int(
                    metadata.get("prompt_global_index")
                ),
                "workload_id": metadata.get("workload_id"),
                "prompt_sha256": metadata.get("prompt_sha256"),
                "rid": metadata.get("rid"),
                "token_start": int(token_start),
                "token_end": int(token_end),
            }
            if not self._prompt_identity_matches(segment):
                continue
            trace_item, lookup_source = self._lookup_route_record(segment)
            if trace_item is None:
                continue
            segment["trace_lookup"] = lookup_source
            segment["trace_request_id"] = trace_item.get("request_id")
            segment["trace_prompt_global_index"] = trace_item.get(
                "prompt_global_index"
            )
            segments.append(segment)
        return segments

    def _expert_counts_from_segments(
        self,
        *,
        segments: List[Dict[str, Any]],
        target_layer: int,
    ) -> Optional[Any]:
        import numpy as np

        target_layer = int(target_layer)
        counts = np.zeros((self.num_experts,), dtype=np.int64)
        used_segments = 0
        for segment in segments:
            routed = self._lookup_routed(segment)
            if routed is None:
                continue
            if target_layer < 0 or target_layer >= int(routed.shape[1]):
                continue
            token_start = max(0, int(segment.get("token_start", 0)))
            token_end = min(int(segment.get("token_end", 0)), int(routed.shape[0]))
            if token_end <= token_start:
                continue
            values = routed[token_start:token_end, target_layer, :].reshape(-1)
            values = values[(values >= 0) & (values < self.num_experts)]
            if values.size == 0:
                continue
            counts += np.bincount(
                values.astype(np.int64, copy=False), minlength=self.num_experts
            )
            used_segments += 1
        if used_segments <= 0 or int(counts.sum()) <= 0:
            return None
        return counts

    def _ranked_experts_with_counts_from_segments(
        self,
        *,
        segments: List[Dict[str, Any]],
        target_layer: int,
        limit: int,
    ) -> Optional[Tuple[List[int], List[int]]]:
        import numpy as np

        counts = self._expert_counts_from_segments(
            segments=segments,
            target_layer=target_layer,
        )
        if counts is None:
            return None
        order = np.lexsort((np.arange(self.num_experts), -counts))
        ranked = [
            int(x)
            for x in order[: max(1, min(int(limit), self.num_experts))]
        ]
        return ranked, [int(x) for x in counts.tolist()]

    def _ranked_experts_from_segments(
        self,
        *,
        segments: List[Dict[str, Any]],
        target_layer: int,
        limit: int,
    ) -> Optional[List[int]]:
        prediction = self._ranked_experts_with_counts_from_segments(
            segments=segments,
            target_layer=target_layer,
            limit=limit,
        )
        return prediction[0] if prediction is not None else None

    def begin_step_if_layer0(self, layer_idx: int, num_tokens: int) -> None:
        if int(layer_idx) != 0:
            return
        context = get_runtime_forward_context()
        metadata_segments = self._segments_from_context(context)
        with self._lock:
            if metadata_segments:
                self._step += 1
                self._metadata_step_count += 1
                self._active = {
                    "step": int(self._step),
                    "source": "metadata",
                    "segments": metadata_segments,
                }
                if self.verbose and (
                    self._metadata_step_count <= 16
                    or self._metadata_step_count % 64 == 0
                ):
                    preview = metadata_segments[0]
                    logger.info(
                        "KT runtime oracle prefetch metadata step: step=%d "
                        "segments=%d first_request=%s first_tokens=[%d,%d) "
                        "runtime_tokens=%d trace_match=%s trace_prompt=%s",
                        self._step,
                        len(metadata_segments),
                        preview.get("request_id"),
                        int(preview["token_start"]),
                        int(preview["token_end"]),
                        int(num_tokens),
                        preview.get("trace_lookup"),
                        preview.get("trace_prompt_global_index"),
                    )
                return
            if context is not None and context.get("kt_metadata_list"):
                self._active = None
                if self.verbose and self._step < 16:
                    logger.info(
                        "KT runtime oracle prefetch metadata miss: layer=%d "
                        "runtime_tokens=%d context_batch=%s",
                        int(layer_idx),
                        int(num_tokens),
                        context.get("batch_size"),
                    )
                return

            while self._request_idx < len(self.routes):
                routed = self.routes[self._request_idx]
                if self._token_offset < int(routed.shape[0]):
                    break
                self._request_idx += 1
                self._token_offset = 0
            if self._request_idx >= len(self.routes):
                self._active = None
                return

            routed = self.routes[self._request_idx]
            token_start = int(self._token_offset)
            token_end = min(token_start + max(0, int(num_tokens)), int(routed.shape[0]))
            self._token_offset = token_end
            self._step += 1
            self._fallback_step_count += 1
            self._active = {
                "step": int(self._step),
                "source": "fallback_order",
                "request_idx": int(self._request_idx),
                "token_start": int(token_start),
                "token_end": int(token_end),
                "segments": [
                    {
                        "request_id": int(self._request_idx),
                        "token_start": int(token_start),
                        "token_end": int(token_end),
                    }
                ],
            }
            if self.verbose and self._step < 16:
                logger.info(
                    "KT runtime oracle prefetch fallback step: step=%d request=%d "
                    "tokens=[%d,%d) runtime_tokens=%d",
                    self._step,
                    self._request_idx,
                    token_start,
                    token_end,
                    int(num_tokens),
                )

    def ranked_experts_for_active_step(
        self,
        *,
        target_layer: int,
        limit: int,
    ) -> Optional[List[int]]:
        with self._lock:
            active = dict(self._active) if self._active is not None else None
        if active is None:
            return None
        segments = active.get("segments")
        if not isinstance(segments, list) or not segments:
            return None
        selected = self._ranked_experts_from_segments(
            segments=segments,
            target_layer=target_layer,
            limit=limit,
        )
        if not selected:
            return None
        self._select_count += 1
        if self.verbose and (
            self._select_count <= 16 or self._select_count % 64 == 0
        ):
            preview = segments[0]
            logger.info(
                "KT runtime oracle prefetch selected: step=%d source=%s "
                "segments=%d first_request=%s target_layer=%d "
                "first_tokens=[%d,%d) top=%s",
                int(active["step"]),
                active.get("source", "unknown"),
                len(segments),
                preview.get("request_id"),
                target_layer,
                int(preview.get("token_start", 0)),
                int(preview.get("token_end", 0)),
                selected[: min(8, len(selected))],
            )
        return selected

    def ranked_experts_with_counts_for_active_step(
        self,
        *,
        target_layer: int,
        limit: int,
    ) -> Optional[Tuple[List[int], List[int]]]:
        with self._lock:
            active = dict(self._active) if self._active is not None else None
        if active is None:
            return None
        segments = active.get("segments")
        if not isinstance(segments, list) or not segments:
            return None
        return self._ranked_experts_with_counts_from_segments(
            segments=segments,
            target_layer=target_layer,
            limit=limit,
        )

    def ranked_experts_for_next_step(
        self,
        *,
        target_layer: int,
        limit: int,
    ) -> Optional[List[int]]:
        """Return oracle hot experts for the next layer-0 runtime chunk."""
        with self._lock:
            active = dict(self._active) if self._active is not None else None
            request_idx = int(self._request_idx)
            token_start = int(self._token_offset)
        if active is None:
            return None
        segments = active.get("segments")
        if isinstance(segments, list) and segments:
            next_segments: List[Dict[str, Any]] = []
            for segment in segments:
                routed = self._lookup_routed(segment)
                if routed is None:
                    continue
                token_count = max(
                    1,
                    int(segment.get("token_end", 0))
                    - int(segment.get("token_start", 0)),
                )
                next_start = int(segment.get("token_end", 0))
                next_end = min(next_start + token_count, int(routed.shape[0]))
                if next_end <= next_start:
                    continue
                next_segment = dict(segment)
                next_segment["token_start"] = int(next_start)
                next_segment["token_end"] = int(next_end)
                next_segments.append(next_segment)
            selected = self._ranked_experts_from_segments(
                segments=next_segments,
                target_layer=target_layer,
                limit=limit,
            )
            if selected:
                if self.verbose and (
                    self._select_count <= 16 or self._select_count % 64 == 0
                ):
                    preview = next_segments[0]
                    logger.info(
                        "KT runtime oracle prefetch next-step selected: "
                        "current_step=%d source=%s segments=%d "
                        "first_request=%s target_layer=%d "
                        "first_tokens=[%d,%d) top=%s",
                        int(active["step"]),
                        active.get("source", "unknown"),
                        len(next_segments),
                        preview.get("request_id"),
                        int(target_layer),
                        int(preview.get("token_start", 0)),
                        int(preview.get("token_end", 0)),
                        selected[: min(8, len(selected))],
                    )
                return selected
            if active.get("source") == "metadata":
                return None

        # Legacy fallback for traces without runtime metadata.
        if "token_end" not in active or "token_start" not in active:
            return None
        token_count = max(
            1, int(active["token_end"]) - int(active["token_start"])
        )

        while request_idx < len(self.routes):
            routed = self.routes[request_idx]
            if token_start < int(routed.shape[0]):
                break
            request_idx += 1
            token_start = 0
        if request_idx >= len(self.routes):
            return None

        routed = self.routes[request_idx]
        target_layer = int(target_layer)
        if target_layer < 0 or target_layer >= int(routed.shape[1]):
            return None
        token_end = min(token_start + token_count, int(routed.shape[0]))
        if token_end <= token_start:
            return None

        values = routed[token_start:token_end, target_layer, :].reshape(-1)
        values = values[(values >= 0) & (values < self.num_experts)]
        if values.size == 0:
            return None
        import numpy as np

        counts = np.bincount(values.astype(np.int64), minlength=self.num_experts)
        order = np.lexsort((np.arange(self.num_experts), -counts))
        selected = [int(x) for x in order[: max(1, min(int(limit), self.num_experts))]]
        if self.verbose and (
            self._select_count <= 16 or self._select_count % 64 == 0
        ):
            logger.info(
                "KT runtime oracle prefetch next-step selected: current_step=%d "
                "request=%d target_layer=%d tokens=[%d,%d) top=%s",
                int(active["step"]),
                request_idx,
                target_layer,
                token_start,
                token_end,
                selected[: min(8, len(selected))],
            )
        return selected


def get_runtime_oracle_prefetch_provider() -> Optional[RuntimeOraclePrefetchProvider]:
    global _KT_RUNTIME_ORACLE_PREFETCH
    if os.environ.get("SGLANG_KT_RUNTIME_ORACLE_PREFETCH", "0") != "1":
        return None
    if _KT_RUNTIME_ORACLE_PREFETCH is not None:
        return _KT_RUNTIME_ORACLE_PREFETCH
    trace_path = os.environ.get("SGLANG_KT_RUNTIME_ORACLE_PREFETCH_TRACE", "")
    if not trace_path:
        logger.warning(
            "KT runtime oracle prefetch requested but "
            "SGLANG_KT_RUNTIME_ORACLE_PREFETCH_TRACE is empty"
        )
        return None
    try:
        _KT_RUNTIME_ORACLE_PREFETCH = RuntimeOraclePrefetchProvider(trace_path)
        return _KT_RUNTIME_ORACLE_PREFETCH
    except Exception as exc:
        logger.warning("KT runtime oracle prefetch is unavailable: %s", exc)
        return None


@dataclass
class KTConfig:
    """Configuration for KTransformers heterogeneous computing CPU part.

    Args:
        layer_idx: Layer index in the model
        gpu_experts_mask: Boolean tensor of shape [num_experts] indicating which experts are on GPU
        cpuinfer_threads: Number of CPU inference threads
        threadpool_count: Number of thread pools for CPU computation
        numa_nodes: Optional explicit NUMA node ids for each KT threadpool
        weight_path: Path to CPU quantized weights
        chunked_prefill_size: Chunk size for prefill computation
        method: CPU computation method (e.g., "int4")
        num_layers: Total number of layers in the model (optional)
        gpu_prefill_token_threshold: token threshold for enabling full GPU fallback
        kt_enable_dynamic_expert_update: Enable dynamic GPU expert updates based on runtime statistics
        expert_lora_path: Optional PEFT adapter directory for KT CPU expert LoRA
    """

    layer_idx: int
    gpu_experts_mask: torch.Tensor  # bool tensor of shape [num_experts]
    cpuinfer_threads: int
    threadpool_count: int
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: int
    method: str
    numa_nodes: Optional[List[int]] = None
    num_layers: Optional[int] = None
    gpu_prefill_token_threshold: Optional[int] = None
    kt_enable_dynamic_expert_update: bool = False
    expert_lora_path: Optional[str] = None


@dataclass
class KTExpertLoraWeights:
    gate_lora_a: torch.Tensor
    gate_lora_b: torch.Tensor
    up_lora_a: torch.Tensor
    up_lora_b: torch.Tensor
    down_lora_a: torch.Tensor
    down_lora_b: torch.Tensor
    rank: int
    alpha: float


_KT_SFT_METHOD_BY_INFERENCE_METHOD = {
    "AMXBF16": "AMXBF16_SFT",
    "BF16": "AMXBF16_SFT",
    "AMXINT8": "AMXINT8_SFT",
    "AMXINT4": "AMXINT4_SFT",
}


def _map_kt_method_to_sft_method(method: str) -> str:
    normalized = method.upper()
    if normalized in _KT_SFT_METHOD_BY_INFERENCE_METHOD:
        return _KT_SFT_METHOD_BY_INFERENCE_METHOD[normalized]
    raise ValueError(
        f"--kt-expert-lora-path currently supports only AMX/BF16 SFT-compatible "
        f"KT methods {sorted(_KT_SFT_METHOD_BY_INFERENCE_METHOD)}, got {method!r}."
    )


def _load_adapter_config(adapter_path: Path) -> Tuple[Optional[int], float]:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"KT expert LoRA adapter config not found: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    rank = config.get("r")
    if rank is not None:
        rank = int(rank)
    alpha = float(config.get("lora_alpha", rank if rank is not None else 1.0))
    return rank, alpha


def _find_adapter_weight_file(adapter_path: Path) -> Path:
    preferred = adapter_path / "adapter_model.safetensors"
    if preferred.is_file():
        return preferred
    candidates = sorted(adapter_path.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(
            f"No safetensors adapter weights found under {adapter_path}"
        )
    if len(candidates) > 1:
        logger.warning(
            "Multiple safetensors files found under %s; using %s for KT expert LoRA",
            adapter_path,
            candidates[0],
        )
    return candidates[0]


def _get_expert_lora_tensor(
    state_dict: Dict[str, torch.Tensor],
    layer_idx: int,
    expert_idx: int,
    proj_name: str,
    lora_name: str,
) -> torch.Tensor:
    suffixes = [
        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.{lora_name}.weight",
        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.{lora_name}.default.weight",
    ]
    prefixes = ["", "base_model.", "base_model.model.", "base_model.model.model."]
    for suffix in suffixes:
        for prefix in prefixes:
            key = prefix + suffix
            if key in state_dict:
                return state_dict[key]
    for key, tensor in state_dict.items():
        if any(key.endswith(suffix) for suffix in suffixes):
            return tensor
    raise KeyError(
        f"Missing KT expert LoRA tensor for layer={layer_idx}, expert={expert_idx}, "
        f"proj={proj_name}, lora={lora_name}. Expected suffix like {suffixes[0]!r}."
    )


_KT_LORA_STATE_DICT_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}


def _load_kt_expert_lora_weights(
    adapter_path: str,
    layer_idx: int,
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
    dtype: torch.dtype = torch.bfloat16,
) -> KTExpertLoraWeights:
    adapter_dir = Path(adapter_path)
    rank_from_config, alpha = _load_adapter_config(adapter_dir)
    weight_file = _find_adapter_weight_file(adapter_dir)
    weight_file_str = str(weight_file)
    if weight_file_str not in _KT_LORA_STATE_DICT_CACHE:
        from safetensors.torch import load_file

        _KT_LORA_STATE_DICT_CACHE[weight_file_str] = load_file(
            weight_file_str, device="cpu"
        )
    state_dict = _KT_LORA_STATE_DICT_CACHE[weight_file_str]
    sample = _get_expert_lora_tensor(
        state_dict, layer_idx, 0, "gate_proj", "lora_A"
    )
    rank = rank_from_config or int(sample.shape[0])
    if int(sample.shape[0]) != rank:
        raise ValueError(
            f"KT expert LoRA rank mismatch for layer {layer_idx}: "
            f"adapter_config r={rank}, sample tensor rank={int(sample.shape[0])}"
        )

    # SGLang may set the default torch device to CUDA during model loading.
    # KT SFT CPU kernels consume raw host pointers, so these adapter staging
    # buffers must be explicitly allocated on CPU.
    device = torch.device("cpu")
    gate_lora_a = torch.zeros((num_experts, rank, hidden_size), dtype=dtype, device=device)
    gate_lora_b = torch.zeros((num_experts, moe_intermediate_size, rank), dtype=dtype, device=device)
    up_lora_a = torch.zeros((num_experts, rank, hidden_size), dtype=dtype, device=device)
    up_lora_b = torch.zeros((num_experts, moe_intermediate_size, rank), dtype=dtype, device=device)
    down_lora_a = torch.zeros((num_experts, rank, moe_intermediate_size), dtype=dtype, device=device)
    down_lora_b = torch.zeros((num_experts, hidden_size, rank), dtype=dtype, device=device)

    targets = [
        ("gate_proj", "lora_A", gate_lora_a),
        ("gate_proj", "lora_B", gate_lora_b),
        ("up_proj", "lora_A", up_lora_a),
        ("up_proj", "lora_B", up_lora_b),
        ("down_proj", "lora_A", down_lora_a),
        ("down_proj", "lora_B", down_lora_b),
    ]
    for expert_idx in range(num_experts):
        for proj_name, lora_name, dst in targets:
            tensor = _get_expert_lora_tensor(
                state_dict, layer_idx, expert_idx, proj_name, lora_name
            )
            if tuple(tensor.shape) != tuple(dst[expert_idx].shape):
                raise ValueError(
                    f"KT expert LoRA shape mismatch for layer={layer_idx}, "
                    f"expert={expert_idx}, {proj_name}.{lora_name}: expected "
                    f"{tuple(dst[expert_idx].shape)}, got {tuple(tensor.shape)}"
                )
            dst[expert_idx].copy_(tensor.to(dtype=dtype, device=device))

    return KTExpertLoraWeights(
        gate_lora_a=gate_lora_a.contiguous(),
        gate_lora_b=gate_lora_b.contiguous(),
        up_lora_a=up_lora_a.contiguous(),
        up_lora_b=up_lora_b.contiguous(),
        down_lora_a=down_lora_a.contiguous(),
        down_lora_b=down_lora_b.contiguous(),
        rank=rank,
        alpha=alpha,
    )


_SHARED_FULL_CONTEXT = None
_SHARED_STAGING_BUFFER = None  # Global shared staging buffer for all MoE layers


class SharedStagingBuffer:
    """Global shared staging buffer for CPU expert input across all MoE layers.

    This avoids allocating a separate staging buffer per layer, which would
    consume significant GPU memory (chunked_prefill_size * hidden_size * N_layers).
    Instead, all layers share a single buffer since MoE layers are processed
    sequentially, not in parallel.
    """

    def __init__(
        self,
        max_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.buffer = torch.empty(
            (max_tokens, hidden_size),
            dtype=dtype,
            device=device,
        )
        buffer_size_mb = self.buffer.numel() * self.buffer.element_size() / 1024**2
        logger.info(
            f"[KT] Created shared staging buffer: {buffer_size_mb:.1f} MiB "
            f"(shape={self.buffer.shape}, dtype={dtype})"
        )

    def get_slice(self, num_tokens: int) -> torch.Tensor:
        """Get a slice of the buffer for the given number of tokens."""
        assert num_tokens <= self.max_tokens, (
            f"Batch size {num_tokens} exceeds staging buffer max size {self.max_tokens}"
        )
        return self.buffer[:num_tokens]


def get_or_create_shared_staging_buffer(
    max_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> SharedStagingBuffer:
    """Get or create the global shared staging buffer."""
    global _SHARED_STAGING_BUFFER
    if _SHARED_STAGING_BUFFER is None:
        _SHARED_STAGING_BUFFER = SharedStagingBuffer(
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
        )
    return _SHARED_STAGING_BUFFER


class SharedFullContext:
    def __init__(
        self,
        layer: torch.nn.Module,
        init_args: tuple,
        global_num_experts: int,
        moe_runner_config: "MoeRunnerConfig",
    ):
        self._build_layers(layer, init_args, global_num_experts, moe_runner_config)

        # Capture original tensors to support restoration before loading
        self.original_params = {
            name: param for name, param in self.gpu_layer.named_parameters()
        }
        self.original_buffers = {
            name: buf for name, buf in self.gpu_layer.named_buffers()
        }

        # Create CPU buffers once for weight loading (shared across layers)
        self._create_cpu_buffers()

        # For M3 MXFP8 layerwise prefill, cache the canonical uint8 ue8m0
        # scale Parameter objects so `_prepare_weight_mxfp8` can rebind to
        # them before each byte-copy. Native MXFP8 path keeps the scale in
        # this layout throughout (no convert to block-fp8).
        if getattr(self, "is_mxfp8_quant", False):
            self._init_mxfp8_aux()

    def _init_mxfp8_aux(self) -> None:
        """Cache the canonical uint8 ue8m0 [E, N, K//32] scale Parameters.

        Fp8MoEMethod.create_weights (use_mxfp8=True) allocated them; we just
        save references so the shadow gpu_layer can rebind back to the same
        objects across rounds of layerwise prefill.
        """
        layer = self.gpu_layer
        self._w13_scale_mxfp8_param = layer.w13_weight_scale_inv
        self._w2_scale_mxfp8_param = layer.w2_weight_scale_inv

    def _build_layers(self, layer, init_args, global_num_experts, moe_runner_config):
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            UnquantizedFusedMoEMethod,
        )

        hidden_size, intermediate_size_per_partition, params_dtype = init_args
        target_device = next(layer.parameters()).device

        # Create gpu_layer as a shallow copy, then override specific attributes
        self.gpu_layer = copy.copy(layer)
        # Clear module state that shouldn't be shared
        self.gpu_layer._parameters = {}
        self.gpu_layer._buffers = {}
        self.gpu_layer._modules = {}

        # Override expert counts for full GPU execution
        self.gpu_layer.num_experts = global_num_experts
        self.gpu_layer.num_local_experts = global_num_experts
        self.gpu_layer.num_gpu_experts = global_num_experts

        # Create quant_method for gpu_layer
        if self.gpu_layer.quant_config is not None:
            self.gpu_method = self.gpu_layer.quant_config.get_quant_method(
                self.gpu_layer, prefix=""
            )
        else:
            self.gpu_method = UnquantizedFusedMoEMethod(
                self.gpu_layer.use_triton_kernels
            )
        # V4-Flash routed experts are MXFP4-packed (FP4 e2m1 in int8 + ue8m0
        # scales) but quant_config.get_quant_method picks Fp8MoEMethod
        # (V4-Flash uses FP8 for attn / shared experts). The default
        # mxfp4_deepseek pipeline wraps Fp8MoEMethod with DeepSeekMxfp4MoEMethod
        # for V4 routed experts; replicate that wrap here so kt_ep_wrapper's
        # gpu_method correctly handles MXFP4 (shape K = hidden, not hidden/2
        # as the FP8 path assumes), and so the capability-driven dispatch in
        # mxfp4_deepseek.apply (trtllm vs triton_kernels) fires on the
        # wrapped path. Gate by `--kt-method MXFP4` (an explicit user choice
        # that means "this run uses MXFP4 routed experts on the kt path"),
        # NOT by SGLANG_V4_USE_TRITON_KERNELS (which is now a diagnostic
        # override only, see v4_triton_kernels_moe.use_v4_triton_kernels
        # docstring). Origin: kt-sglang 耦合 (V4-Flash routed experts MXFP4
        # detection in kt_ep_wrapper).
        try:
            import os as _os_v4
            from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
            from sglang.srt.layers.quantization.mxfp4_deepseek import (
                DeepSeekMxfp4MoEMethod,
            )
            from sglang.srt.server_args import get_global_server_args
            _v4_env = _os_v4.environ.get("SGLANG_V4_USE_TRITON_KERNELS")
            if _v4_env == "1":
                _do_v4_wrap = True
            elif _v4_env == "0":
                _do_v4_wrap = False
            else:
                _do_v4_wrap = (
                    (get_global_server_args().kt_method or "").upper() == "MXFP4"
                )
            if _do_v4_wrap and isinstance(self.gpu_method, Fp8MoEMethod):
                self.gpu_method = DeepSeekMxfp4MoEMethod(self.gpu_method, prefix="")
        except Exception as _v4_tk_wrap_exc:
            logger.warning(
                f"[kt-ep-wrapper] V4-Flash MXFP4 wrap skipped: {_v4_tk_wrap_exc}"
            )
        self.gpu_layer.quant_method = self.gpu_method

        self.gpu_method.create_weights(
            layer=self.gpu_layer,
            num_experts=global_num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
        )

        # Detect quantization type for weight loading based on actually created weights.
        # This is more robust than class-based detection when quant methods are wrapped
        # (e.g., KT wrapper -> compressed-tensors scheme), especially in layerwise prefill.
        self._detect_quant_type_from_created_weights()

        # Move all parameters to target device
        for param in self.gpu_layer.parameters():
            if param.device != target_device:
                param.data = param.data.to(target_device)

        # Create runner config - update both num_experts and num_local_experts for full GPU fallback
        # Set routed_scaling_factor=None to avoid double scaling:
        # - moe_sum_reduce would apply routed_scaling_factor internally
        # - deepseek_v2.py forward_normal also applies routed_scaling_factor for KTEPWrapperMethod
        # By setting it to None here, we ensure it's only applied once in forward_normal
        runner_config = replace(
            moe_runner_config,
            num_experts=global_num_experts,
            num_local_experts=global_num_experts,
            routed_scaling_factor=None,
        )
        self.gpu_layer.moe_runner_config = runner_config
        self.gpu_method.create_moe_runner(self.gpu_layer, runner_config)

    def _get_base_quant_method(self):
        """Unwrap nested quant methods to get the underlying base method.

        Some paths may wrap the real quant method with KT wrappers/schemes.
        """
        method = self.gpu_method
        visited = set()

        while method is not None and id(method) not in visited:
            visited.add(id(method))

            # KT wrapper pattern: method.gpu_method
            nested = getattr(method, "gpu_method", None)
            if nested is not None and nested is not method:
                method = nested
                continue

            # Compressed-tensors scheme pattern: method.scheme
            nested = getattr(method, "scheme", None)
            if nested is not None and nested is not method:
                method = nested
                continue

            break

        return method

    def _detect_quant_type_from_created_weights(self) -> None:
        """Detect quant type from weight attributes created on gpu_layer."""
        layer = self.gpu_layer

        # V4-Flash MXFP4 (must come before FP8 block — both register
        # `w13_weight_scale_inv`, but MXFP4 is FP4 nibble-packed weights with
        # ue8m0 scales rather than FP8 e4m3 weights with FP8 scales). Use the
        # quant method's class name as discriminator to avoid a circular import
        # of DeepSeekMxfp4MoEMethod. Origin: sglang 本身 (V4-Flash full-GPU
        # prefill fallback compat).
        if self.gpu_method.__class__.__name__ == "DeepSeekMxfp4MoEMethod":
            self.is_mxfp4_quant = True
            self.is_mxfp8_quant = False
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            return

        # INT4 Marlin
        if hasattr(layer, "w13_weight_packed") and hasattr(layer, "w2_weight_packed"):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = False
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            return

        # M3 MXFP8 block (must come before FP8 block — both register
        # `w13_weight_scale_inv`, but MXFP8 stores uint8 ue8m0 scales with
        # block_size=[1,32] while FP8 block uses fp32 scales with [128,128]).
        # The `format_ue8m0` attribute set by Fp8MoEMethod.create_weights
        # when use_mxfp8=True (fp8.py:914) is the canonical discriminator.
        # Origin: kt-sglang 耦合 (M3 MXFP8 layerwise prefill).
        if (
            hasattr(layer, "w13_weight_scale_inv")
            and hasattr(layer, "w2_weight_scale_inv")
            and getattr(layer.w13_weight_scale_inv, "format_ue8m0", False)
        ):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = True
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            return

        # FP8 block
        if hasattr(layer, "w13_weight_scale_inv") and hasattr(layer, "w2_weight_scale_inv"):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = False
            self.is_fp8_quant = True
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            return

        # FP8 per-channel
        if hasattr(layer, "w13_weight_scale") and hasattr(layer, "w2_weight_scale"):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = False
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = True
            self.is_bf16_quant = False
            return

        # BF16 / unquantized
        if hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight"):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = False
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = True
            return

        # Fallback to class-based detection for unknown layouts.
        self.is_mxfp4_quant = False
        self.is_mxfp8_quant = False
        self.is_fp8_quant = self._detect_fp8_quant()
        self.is_fp8_channel_quant = self._detect_fp8_channel_quant()
        self.is_bf16_quant = self._detect_bf16_quant()

    def _detect_fp8_quant(self) -> bool:
        """Detect if the quantization method is FP8 block quant.

        Returns:
            True if FP8 block quant, False otherwise (INT4 Marlin, BF16, etc.)
        """
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        method = self._get_base_quant_method()
        # Check for Fp8MoEMethod with block_quant
        if isinstance(method, Fp8MoEMethod) and getattr(method, "block_quant", False):
            return True

        # Check for CompressedTensorsW8A8Fp8MoEMethod with block_quant
        method_name = method.__class__.__name__
        if "W8A8Fp8" in method_name and getattr(method, "block_quant", False):
            return True

        return False

    def _detect_fp8_channel_quant(self) -> bool:
        """Detect if the quantization method is FP8 per-channel quant.

        Per-channel FP8 differs from block FP8:
        - Per-channel: scale shape is (num_experts, output_dim, 1), weight_scale name
        - Block FP8: scale shape is (num_experts, blocks_n, blocks_k), weight_scale_inv name

        Returns:
            True if FP8 per-channel quant, False otherwise
        """
        try:
            from compressed_tensors.quantization import QuantizationStrategy
        except ImportError:
            return False

        method = self._get_base_quant_method()
        method_name = method.__class__.__name__

        # Check for CompressedTensorsW8A8Fp8MoEMethod with channel strategy
        if "W8A8Fp8" in method_name:
            weight_quant = getattr(method, "weight_quant", None)
            if weight_quant is not None:
                if weight_quant.strategy == QuantizationStrategy.CHANNEL:
                    return True

        return False

    def _detect_bf16_quant(self) -> bool:
        """Detect if the quantization method is BF16/unquantized.

        Returns:
            True if BF16/unquantized, False otherwise (INT4 Marlin, FP8, etc.)
        """
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            UnquantizedFusedMoEMethod,
        )

        method = self._get_base_quant_method()
        # Check for UnquantizedFusedMoEMethod
        if isinstance(method, UnquantizedFusedMoEMethod):
            return True

        return False

    def _resolve_int4_quant_params(self):
        """Resolve INT4 quant params from potentially wrapped quant methods.

        Some quantization paths (e.g., compressed-tensors) expose INT4 metadata on
        the underlying scheme instead of the outer fused method wrapper.
        """
        candidates = []
        seen = set()

        def add_candidate(obj):
            if obj is None:
                return
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            candidates.append(obj)

        base_method = self._get_base_quant_method()
        add_candidate(self.gpu_method)
        add_candidate(getattr(self.gpu_method, "gpu_method", None))
        add_candidate(getattr(self.gpu_method, "scheme", None))
        add_candidate(base_method)
        add_candidate(getattr(base_method, "scheme", None))
        add_candidate(getattr(self.gpu_layer, "scheme", None))

        required = ("num_bits", "packed_factor", "group_size")
        for candidate in candidates:
            if all(hasattr(candidate, attr) for attr in required):
                return (
                    getattr(candidate, "num_bits"),
                    getattr(candidate, "packed_factor"),
                    getattr(candidate, "group_size"),
                    getattr(candidate, "actorder", None),
                )

        raise AttributeError(
            "Unable to resolve INT4 quantization params: expected attributes "
            "num_bits/packed_factor/group_size on quant method or scheme"
        )

    @property
    def weight_names(self) -> list:
        """Get weight names based on quantization type."""
        if getattr(self, "is_mxfp4_quant", False):
            # V4-Flash MXFP4 uses the same flat names as FP8 block (w13_weight,
            # w13_weight_scale_inv, w2_weight, w2_weight_scale_inv); the
            # underlying byte payload differs (FP4 nibble + ue8m0 scale) but
            # the staging buffers don't care about content.
            return self.WEIGHT_NAMES_FP8
        if getattr(self, "is_mxfp8_quant", False):
            # M3 MXFP8 reuses the FP8 block flat names. Byte payload is
            # MXFP8 (fp8 + uint8 ue8m0 [1,32]) — staging buffer dtype/shape
            # follow gpu_layer.w13_weight_scale_inv (uint8) so byte-copy
            # transports the canonical layout. fused_experts_mxfp8 consumes
            # it directly; no convert step.
            return self.WEIGHT_NAMES_FP8
        if self.is_fp8_quant:
            return self.WEIGHT_NAMES_FP8
        elif self.is_fp8_channel_quant:
            return self.WEIGHT_NAMES_FP8_CHANNEL
        elif self.is_bf16_quant:
            return self.WEIGHT_NAMES_BF16
        else:
            return self.WEIGHT_NAMES_INT4

    # Weight names for shared memory buffers (INT4 Marlin format)
    WEIGHT_NAMES_INT4 = [
        "w13_weight_packed",
        "w13_weight_scale",
        "w2_weight_packed",
        "w2_weight_scale",
    ]

    # Weight names for FP8 block quant format
    WEIGHT_NAMES_FP8 = [
        "w13_weight",
        "w13_weight_scale_inv",
        "w2_weight",
        "w2_weight_scale_inv",
    ]

    # Weight names for FP8 per-channel quant format
    # Per-channel differs from block quant:
    # - Scale shape: (num_experts, output_dim, 1) vs (num_experts, blocks_n, blocks_k)
    # - Weight name: w13_weight_scale vs w13_weight_scale_inv
    WEIGHT_NAMES_FP8_CHANNEL = [
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
    ]

    # Weight names for BF16/unquantized format (no scales)
    WEIGHT_NAMES_BF16 = [
        "w13_weight",
        "w2_weight",
    ]

    def _create_cpu_buffers(self):
        """Create CPU buffers in POSIX shared memory and register as pinned memory.

        Uses double buffering (2 experts) to reduce memory usage while maintaining
        pipeline efficiency: write(e+1) || copy(e) only needs 2 buffers.
        """
        # Set NUMA local allocation policy to allocate on local NUMA node
        libnuma = ctypes.CDLL("libnuma.so.1")
        if libnuma.numa_available() < 0:
            raise RuntimeError("NUMA is not available on this system")
        libnuma.numa_set_localalloc()

        self.cpu_buffers = {}
        self.shm_handles: Dict[str, shared_memory.SharedMemory] = {}
        tp_rank = get_tensor_model_parallel_rank()
        num_experts = self.gpu_layer.num_experts

        # Generate unique ID on rank 0 and broadcast to all ranks
        if tp_rank == 0:
            self.shm_unique_id = uuid.uuid4().hex[:8]
        else:
            self.shm_unique_id = None
        if dist.is_initialized():
            unique_id_list = [self.shm_unique_id]
            dist.broadcast_object_list(
                unique_id_list, src=0, group=get_tp_group().cpu_group
            )
            self.shm_unique_id = unique_id_list[0]

        for name in self.weight_names:
            gpu_tensor = getattr(self.gpu_layer, name)
            # Only allocate 2 experts worth of buffer (double buffering)
            expert_shape = gpu_tensor.shape[1:]  # Shape per expert
            if (
                getattr(self, "is_mxfp4_quant", False)
                and name in ("w13_weight_scale_inv", "w2_weight_scale_inv")
            ):
                buf_dtype = torch.bfloat16
            else:
                buf_dtype = gpu_tensor.dtype
            element_size = torch.empty((), dtype=buf_dtype).element_size()
            expert_nbytes = gpu_tensor.numel() // num_experts * element_size
            double_buf_nbytes = expert_nbytes * 2

            shm_name = f"kt_buf_{name}_r{tp_rank}_{self.shm_unique_id}"
            shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=double_buf_nbytes
            )
            self.shm_handles[name] = shm

            # Shape: [2, ...expert_shape...]
            cpu_buffer = torch.frombuffer(shm.buf, dtype=buf_dtype).reshape(
                (2,) + expert_shape
            )

            # Register as pinned memory for fast DMA
            if torch.cuda.is_available():
                torch.cuda.cudart().cudaHostRegister(
                    cpu_buffer.data_ptr(), double_buf_nbytes, 0
                )

            self.cpu_buffers[name] = cpu_buffer

        if dist.is_initialized():
            dist.barrier(group=get_tp_group().device_group)

        self.all_rank_buffer_ptrs = self._collect_all_rank_buffer_pointers()

        # Unlink shared memory after all ranks have collected pointers.
        # The memory remains accessible as long as we hold references via mmap.
        if dist.is_initialized():
            dist.barrier(group=get_tp_group().device_group)
        for shm in self.shm_handles.values():
            shm.unlink()

    def _collect_all_rank_buffer_pointers(self) -> Dict[str, List[int]]:
        """Collect CPU buffer pointers from all ranks."""
        tp_rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()
        buffer_names = list(self.cpu_buffers.keys())
        all_rank_ptrs: Dict[str, List[int]] = {name: [] for name in buffer_names}
        self._opened_shm_refs: Dict[str, shared_memory.SharedMemory] = {}

        for rank in range(tp_world_size):
            for name in buffer_names:
                if rank == tp_rank:
                    ptr = self.cpu_buffers[name].data_ptr()
                elif tp_rank == 0:
                    shm_name = f"kt_buf_{name}_r{rank}_{self.shm_unique_id}"
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name)
                        self._opened_shm_refs[f"{name}_r{rank}"] = shm
                        ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
                    except FileNotFoundError:
                        logger.error(
                            "Rank %d: Failed to open shared memory '%s'",
                            tp_rank,
                            shm_name,
                        )
                        ptr = 0
                else:
                    ptr = 0
                all_rank_ptrs[name].append(ptr)

        return all_rank_ptrs

    def _prepare_weight_int4(self, wrapper):
        """Prepare INT4 Marlin weights by writing from KT, copying to GPU, and postprocessing.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        Postprocessing extracted from CompressedTensorsWNA16MoEMethod.process_weights_after_loading
        in python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_bits, packed_factor, group_size, actorder = (
            self._resolve_int4_quant_params()
        )
        num_experts = layer.num_experts
        device = layer.w13_weight_packed.device

        # Create empty g_idx tensors for non-grouped actorder
        if actorder != "group":
            for name in [
                "w13_weight_g_idx",
                "w2_weight_g_idx",
                "w13_g_idx_sort_indices",
                "w2_g_idx_sort_indices",
            ]:
                setattr(
                    layer,
                    name,
                    torch.nn.Parameter(
                        torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                        requires_grad=False,
                    ),
                )

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_INT4:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            # Reshape gpu_t to match expert shape for per-expert copy
            expert_shape = cpu_buf.shape[1:]
            gpu_t.set_(gpu_t.view((num_experts,) + expert_shape))
            weight_infos.append((cpu_buf, gpu_t))

        w13_p, w13_s = layer.w13_weight_packed, layer.w13_weight_scale
        w2_p, w2_s = layer.w2_weight_packed, layer.w2_weight_scale
        w13_k, w13_n = w13_p.shape[1] * packed_factor, w13_p.shape[2]
        w2_k, w2_n = w2_p.shape[1] * packed_factor, w2_p.shape[2]
        w2_sk = w2_s.shape[1] * (group_size if group_size != -1 else packed_factor)
        perm = torch.empty(0, dtype=torch.int32, device=device)

        # Tmp buffers for transpose
        tmp_bufs = [
            torch.empty(t.size(1), t.size(2), dtype=t.dtype, device=device)
            for _, t in weight_infos
        ]

        def postprocess_expert(e):
            # Transpose
            for (_, gpu_t), tmp in zip(weight_infos, tmp_bufs):
                d1, d2 = gpu_t.size(1), gpu_t.size(2)
                tmp.copy_(gpu_t[e].reshape(d2, d1).T, non_blocking=True)
                gpu_t[e].copy_(tmp, non_blocking=True)
            # Repack weights
            w13_p[e].copy_(
                gptq_marlin_repack(w13_p[e], perm, w13_k, w13_n, num_bits).view(
                    w13_p[e].shape
                )
            )
            w2_p[e].copy_(
                gptq_marlin_repack(w2_p[e], perm, w2_k, w2_n, num_bits).view(
                    w2_p[e].shape
                )
            )
            # Permute scales
            w13_s[e].copy_(
                marlin_permute_scales(w13_s[e], w13_n, w13_s.shape[2], group_size).view(
                    w13_s[e].shape
                )
            )
            w2_s[e].copy_(
                marlin_permute_scales(w2_s[e], w2_sk, w2_s.shape[2], group_size).view(
                    w2_s[e].shape
                )
            )

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        events = [torch.cuda.Event() for _ in range(num_experts)]

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_packed_buf = self.cpu_buffers["w13_weight_packed"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale"]
            w2_packed_buf = self.cpu_buffers["w2_weight_packed"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_packed_expert_nbytes = (
                w13_packed_buf.numel() // 2 * w13_packed_buf.element_size()
            )
            w13_scale_expert_nbytes = (
                w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
            )
            w2_packed_expert_nbytes = (
                w2_packed_buf.numel() // 2 * w2_packed_buf.element_size()
            )
            w2_scale_expert_nbytes = (
                w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()
            )

            def submit_write_expert(expert_id):
                # Use expert_id % 2 for double buffering slot selection
                slot = expert_id % 2
                w13_packed_ptrs = [
                    ptr + slot * w13_packed_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_packed"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale"]
                ]
                w2_packed_ptrs = [
                    ptr + slot * w2_packed_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_packed"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_packed_ptrs,
                    w13_scale_ptrs,
                    w2_packed_ptrs,
                    w2_scale_ptrs,
                )

            # Submit expert 0 ahead of time
            submit_write_expert(0)

        for e in range(num_experts):
            # Sync write for expert e, submit write for expert e+1
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if e + 1 < num_experts:
                    # Before writing to slot (e+1)%2, make sure the previous
                    # copy from that slot has completed to avoid overwriting
                    # pinned host memory while DMA is in-flight.
                    if e > 0:
                        events[e - 1].synchronize()
                    submit_write_expert(e + 1)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                slot = e % 2  # Double buffering
                for cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[e].record(copy_stream)

            if e > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[e - 1])
                    postprocess_expert(e - 1)

        with torch.cuda.stream(post_stream):
            post_stream.wait_event(events[-1])
            postprocess_expert(num_experts - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

        # Reshape to final shape
        w13_p.set_(w13_p.view(num_experts, w13_k // 16, w13_n * (num_bits // 2)))
        w2_p.set_(w2_p.view(num_experts, w2_k // 16, w2_n * (num_bits // 2)))

    def _prepare_weight_fp8(self, wrapper, original_layer=None, gpu_experts_mask=None,
                            logical_to_gpu_index=None):
        """Prepare FP8 block quant weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        FP8 block quant is simpler than INT4 Marlin:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)

        The postprocess stage is a no-op for FP8 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.

        Optional DeepGemm ue8m0 conversion is handled after all experts are loaded.

        Optimization: If original_layer and gpu_experts_mask are provided, experts
        already on GPU are copied directly (fast GPU-to-GPU), while CPU experts
        use the KT wrapper pipeline.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_FP8:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((name, cpu_buf, gpu_t))

        # Separate GPU experts (direct copy) from CPU experts (KT transfer)
        gpu_expert_ids = []
        cpu_expert_ids = []
        if gpu_experts_mask is not None and original_layer is not None and logical_to_gpu_index is not None:
            for e in range(num_experts):
                if gpu_experts_mask[e].item():
                    gpu_expert_ids.append(e)
                else:
                    cpu_expert_ids.append(e)
        else:
            # Fallback: all experts from CPU
            cpu_expert_ids = list(range(num_experts))

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU) ---
        if gpu_expert_ids:
            for e in gpu_expert_ids:
                gpu_idx = logical_to_gpu_index[e].item()
                for name, _, dst in weight_infos:
                    src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                    dst[e].copy_(src[gpu_idx], non_blocking=True)

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not cpu_expert_ids:
            # All experts are on GPU, nothing more to do
            return

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        # Events indexed by position in cpu_expert_ids
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        def postprocess_expert(idx):
            # FP8 doesn't need actual postprocessing (no repack/permute).
            # This function provides a pipeline synchronization point and
            # can be extended for future FP8-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale_inv"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale_inv"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_weight_expert_nbytes = (
                w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
            )
            w13_scale_expert_nbytes = (
                w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
            )
            w2_weight_expert_nbytes = (
                w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
            )
            w2_scale_expert_nbytes = (
                w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()
            )

            def submit_write_expert(expert_id, slot):
                # Use provided slot for double buffering
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale_inv"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale_inv"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit first CPU expert ahead of time
            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2  # Double buffering based on iteration index

            # Sync write for expert e, submit write for next CPU expert
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    # Before writing to next_slot, ensure copy from that slot is complete.
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            # Postprocess expert idx-1: provides pipeline structure for future extensions
            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])
                    postprocess_expert(idx - 1)

        # Process last CPU expert
        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])
                postprocess_expert(len(cpu_expert_ids) - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    # NOTE: DeepGemm ue8m0 conversion is not used in KT fallback path.
    # The conversion is handled separately in the normal weight loading path.

    def _prepare_weight_mxfp8(self, wrapper, original_layer=None, gpu_experts_mask=None,
                              logical_to_gpu_index=None):
        """Byte-copy M3 MXFP8 weights from CPU staging buffer to GPU for the
        full-GPU layerwise prefill fallback.

        Shadow ``gpu_method`` stays in the MXFP8 view (``use_mxfp8=True``,
        ``weight_block_size=[1, 32]``). ``Fp8MoEMethod.apply`` then routes
        through ``get_triton_quant_info`` -> ``fused_experts_mxfp8``, which
        consumes the uint8 ue8m0 scale directly via ``tl.dot_scaled`` — no
        block-FP8 conversion, no precision loss.

        Origin: kt-sglang 耦合 (M3 MXFP8 layerwise prefill, native MXFP8).
        """
        # Reset shadow to MXFP8 view (idempotent; ensures the Parameter slot
        # points at the canonical uint8 ue8m0 tensor before byte-copy).
        self.gpu_method.use_mxfp8 = True
        self.gpu_method.weight_block_size = [1, 32]
        self.gpu_layer.w13_weight_scale_inv = self._w13_scale_mxfp8_param
        self.gpu_layer.w2_weight_scale_inv = self._w2_scale_mxfp8_param

        # Byte-copy via the FP8 pipeline (uint8 ue8m0 scale + fp8 weight
        # both copied bytewise from kt-kernel CPU staging buffer).
        # original_layer=None disables the GPU shortcut: the real layer's
        # scale slot may have been mutated by Fp8MoEMethod's post-load step
        # on other paths; force the CPU staging route for canonical bytes.
        self._prepare_weight_fp8(
            wrapper,
            original_layer=None,
            gpu_experts_mask=None,
            logical_to_gpu_index=None,
        )

    def _prepare_weight_mxfp4(self, wrapper, original_layer=None, gpu_experts_mask=None,
                              logical_to_gpu_index=None):
        """Prepare V4-Flash MXFP4 weights for the full-GPU prefill fallback.

        V4-Flash MXFP4 routed-experts share flat attribute names with FP8 block
        (`w13_weight` / `w13_weight_scale_inv` / `w2_weight` / `w2_weight_scale_inv`)
        but with different payload semantics: FP4 e2m1 nibble-packed weights +
        ue8m0 per-kgroup scales instead of FP8 e4m3 weights + FP8 scales. The
        staging-buffer byte-copy machinery in `_prepare_weight_fp8` does not
        care about content semantics, so we reuse it as-is for the 144 GPU +
        112 CPU expert load.

        After all 256 experts are filled into `gpu_layer.w13_weight` etc., we
        re-run `gpu_method.process_weights_after_loading(gpu_layer)`, which
        invokes `convert_v4_weights_to_triton_kernels` and stores the swizzled
        result in `gpu_layer._v4_tk_w13` / `_v4_tk_w13_pcg` / `_v4_tk_w2` /
        `_v4_tk_w2_pcg` — exactly what the downstream `gpu_method.apply` →
        `apply_v4_triton_kernels_moe` path expects to read. This requires the
        outer model loader to have skipped the post-swizzle deletes (gated on
        `kt_gpu_prefill_token_threshold > 0` in `mxfp4_deepseek.py`).

        **No caching**: SharedFullContext is a global singleton whose single
        `gpu_layer` holds one layer's swizzled weights at a time. After layer N
        loads, layer N-1's data is overwritten. A boolean or per-layer-set
        cache would be stale when a different layer has since loaded into the
        same gpu_layer. Every load() call must therefore run the full pipeline.

        Origin: sglang 本身 (V4-Flash full-GPU prefill fallback compat).
        """
        # Phase 1+2: byte-copy via the FP8 path (works for FP4-packed bytes).
        self._prepare_weight_fp8(
            wrapper,
            original_layer=original_layer,
            gpu_experts_mask=gpu_experts_mask,
            logical_to_gpu_index=logical_to_gpu_index,
        )

        # Phase 3: re-swizzle the now-256-expert flat tensors into the
        # triton_kernels form `gpu_method.apply` will consume. Ensure all
        # in-flight CPU→GPU copies from Phase 2 are visible first.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.gpu_method.process_weights_after_loading(self.gpu_layer)

    def _prepare_weight_fp8_channel(self, wrapper, original_layer=None, gpu_experts_mask=None,
                                     logical_to_gpu_index=None):
        """Prepare FP8 per-channel quant weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        FP8 per-channel quant differs from FP8 block quant:
        - Per-channel scale shape: (num_experts, output_dim, 1) vs (num_experts, blocks_n, blocks_k)
        - Weight name: w13_weight_scale vs w13_weight_scale_inv
        - Both use float8_e4m3fn weights

        Similar to block FP8:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)

        The postprocess stage is a no-op for FP8 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.

        Optimization: If original_layer and gpu_experts_mask are provided, experts
        already on GPU are copied directly (fast GPU-to-GPU), while CPU experts
        use the KT wrapper pipeline.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_FP8_CHANNEL:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((name, cpu_buf, gpu_t))

        # Separate GPU experts (direct copy) from CPU experts (KT transfer)
        gpu_expert_ids = []
        cpu_expert_ids = []
        if gpu_experts_mask is not None and original_layer is not None and logical_to_gpu_index is not None:
            for e in range(num_experts):
                if gpu_experts_mask[e].item():
                    gpu_expert_ids.append(e)
                else:
                    cpu_expert_ids.append(e)
        else:
            # Fallback: all experts from CPU
            cpu_expert_ids = list(range(num_experts))

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU) ---
        if gpu_expert_ids:
            for e in gpu_expert_ids:
                gpu_idx = logical_to_gpu_index[e].item()
                for name, _, dst in weight_infos:
                    src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                    dst[e].copy_(src[gpu_idx], non_blocking=True)

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not cpu_expert_ids:
            # All experts are on GPU, nothing more to do
            return

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        # Events indexed by position in cpu_expert_ids
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        def postprocess_expert(idx):
            # FP8 per-channel doesn't need actual postprocessing (no repack/permute).
            # This function provides a pipeline synchronization point and
            # can be extended for future FP8-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_weight_expert_nbytes = (
                w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
            )
            w13_scale_expert_nbytes = (
                w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
            )
            w2_weight_expert_nbytes = (
                w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
            )
            w2_scale_expert_nbytes = (
                w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()
            )

            def submit_write_expert(expert_id, slot):
                # Use provided slot for double buffering
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit first CPU expert ahead of time
            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2  # Double buffering based on iteration index

            # Sync write for expert e, submit write for next CPU expert
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    # Before writing to next_slot, ensure copy from that slot is complete.
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            # Postprocess expert idx-1: provides pipeline structure for future extensions
            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])
                    postprocess_expert(idx - 1)

        # Process last CPU expert
        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])
                postprocess_expert(len(cpu_expert_ids) - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    def _prepare_weight_bf16(self, wrapper, original_layer=None, gpu_experts_mask=None,
                             logical_to_gpu_index=None):
        """Prepare BF16/unquantized weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        BF16/unquantized is similar to FP8 block quant:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)
        - No scales at all (unlike FP8 which has scale_inv)

        The postprocess stage is a no-op for BF16 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.

        Optimization: If original_layer and gpu_experts_mask are provided, experts
        already on GPU are copied directly (fast GPU-to-GPU), while CPU experts
        use the KT wrapper pipeline.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_BF16:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((name, cpu_buf, gpu_t))

        # Separate GPU experts (direct copy) from CPU experts (KT transfer)
        gpu_expert_ids = []
        cpu_expert_ids = []
        if gpu_experts_mask is not None and original_layer is not None and logical_to_gpu_index is not None:
            for e in range(num_experts):
                if gpu_experts_mask[e].item():
                    gpu_expert_ids.append(e)
                else:
                    cpu_expert_ids.append(e)
        else:
            # Fallback: all experts from CPU
            cpu_expert_ids = list(range(num_experts))

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU) ---
        if gpu_expert_ids:
            for e in gpu_expert_ids:
                gpu_idx = logical_to_gpu_index[e].item()
                for name, _, dst in weight_infos:
                    src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                    dst[e].copy_(src[gpu_idx], non_blocking=True)

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not cpu_expert_ids:
            # All experts are on GPU, nothing more to do
            return

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        # Events indexed by position in cpu_expert_ids
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        def postprocess_expert(idx):
            # BF16 doesn't need actual postprocessing (no repack/permute/transpose).
            # This function provides a pipeline synchronization point and
            # can be extended for future BF16-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_weight_expert_nbytes = (
                w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
            )
            w2_weight_expert_nbytes = (
                w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
            )

            def submit_write_expert(expert_id, slot):
                # Use provided slot for double buffering
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                # For BF16, we pass empty scale pointer lists (no scales)
                w13_scale_ptrs = [0] * tp_world_size
                w2_scale_ptrs = [0] * tp_world_size
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit first CPU expert ahead of time
            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2  # Double buffering based on iteration index

            # Sync write for expert e, submit write for next CPU expert
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    # Before writing to next_slot, ensure copy from that slot is complete.
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            # Postprocess expert idx-1: provides pipeline structure for future extensions
            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])
                    postprocess_expert(idx - 1)

        # Process last CPU expert
        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])
                postprocess_expert(len(cpu_expert_ids) - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    def load(self, layer_idx, wrapper, original_layer=None, gpu_experts_mask=None,
             logical_to_gpu_index=None):
        """Load weights from disk to GPU via shared memory.

        Args:
            layer_idx: Layer index in the model
            wrapper: KT wrapper for CPU expert weight loading
            original_layer: Original MoE layer with GPU experts (optional)
            gpu_experts_mask: bool tensor [num_experts], True = on GPU (optional)
            logical_to_gpu_index: int tensor [num_experts], maps logical ID to GPU index (optional)
        """
        for name, param in self.original_params.items():
            setattr(self.gpu_layer, name, param)
        for name, buf in self.original_buffers.items():
            self.gpu_layer.register_buffer(name, buf)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tp_rank = get_tensor_model_parallel_rank()
        t0 = time.perf_counter()

        # Select appropriate prepare_weight method based on quantization type
        # FP8/BF16 methods support GPU expert optimization; INT4 uses full CPU pipeline
        if getattr(self, "is_mxfp4_quant", False):
            # V4-Flash MXFP4: byte-copy via FP8 path + re-swizzle into
            # triton_kernels form. Origin: sglang 本身.
            self._prepare_weight_mxfp4(wrapper, original_layer, gpu_experts_mask,
                                       logical_to_gpu_index)
        elif getattr(self, "is_mxfp8_quant", False):
            # M3 MXFP8: byte-copy via FP8 path with original_layer=None
            # (Phase 1 shortcut disabled) + Triton MXFP8->block-FP8 convert
            # on shadow gpu_layer so apply() runs the standard block-FP8
            # deep_gemm path. Origin: kt-sglang 耦合 (v2 bridge).
            self._prepare_weight_mxfp8(wrapper, original_layer, gpu_experts_mask,
                                       logical_to_gpu_index)
        elif self.is_fp8_quant:
            self._prepare_weight_fp8(wrapper, original_layer, gpu_experts_mask,
                                     logical_to_gpu_index)
        elif self.is_fp8_channel_quant:
            self._prepare_weight_fp8_channel(wrapper, original_layer, gpu_experts_mask,
                                             logical_to_gpu_index)
        elif self.is_bf16_quant:
            self._prepare_weight_bf16(wrapper, original_layer, gpu_experts_mask,
                                      logical_to_gpu_index)
        else:
            # INT4 Marlin format: write(e+1) || copy(e) || postprocess(e-1)
            self._prepare_weight_int4(wrapper)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = (time.perf_counter() - t0) * 1000.0

        if tp_rank == 0:
            logger.info(
                "KT layerwise prefill: layer %d prepare weight = %.2f ms",
                layer_idx,
                total_time,
            )


def generate_front_loading_masks(
    num_layers: int,
    num_experts: int,
    num_gpu_experts: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> torch.Tensor:
    """Generate masks by filling layers from first MoE layer onwards.

    Args:
        num_layers: Total number of layers in the model
        num_experts: Number of experts per layer
        num_gpu_experts: Total number of GPU experts to allocate
        first_k_dense_replace: Layer index where MoE layers start
        moe_layer_freq: Frequency of MoE layers (e.g., 1 = every layer, 2 = every other layer)

    Returns:
        Boolean mask tensor of shape [num_layers, num_experts]
    """
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")
    remaining = num_gpu_experts

    for layer_idx in range(num_layers):
        is_moe = layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0
        if not is_moe:
            # Dense layer - set all True (bypass KT wrapper)
            masks[layer_idx, :] = True
        elif remaining > 0:
            # MoE layer - allocate GPU experts
            num_for_this_layer = min(remaining, num_experts)
            masks[layer_idx, :num_for_this_layer] = True
            remaining -= num_for_this_layer

    return masks


def generate_uniform_masks(
    num_layers: int,
    num_experts: int,
    num_gpu_experts: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> torch.Tensor:
    """Generate masks with equal GPU experts per MoE layer.

    Args:
        num_layers: Total number of layers in the model
        num_experts: Number of experts per layer
        num_gpu_experts: Total number of GPU experts to allocate
        first_k_dense_replace: Layer index where MoE layers start
        moe_layer_freq: Frequency of MoE layers

    Returns:
        Boolean mask tensor of shape [num_layers, num_experts]
    """
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    # Identify MoE layers
    moe_layers = [
        i for i in range(num_layers)
        if i >= first_k_dense_replace and i % moe_layer_freq == 0
    ]
    num_moe_layers = len(moe_layers)

    if num_moe_layers == 0:
        return masks

    # Distribute GPU experts evenly
    experts_per_layer = num_gpu_experts // num_moe_layers
    remainder = num_gpu_experts % num_moe_layers

    for idx, layer_idx in enumerate(moe_layers):
        # First 'remainder' layers get one extra expert
        num_for_this_layer = experts_per_layer + (1 if idx < remainder else 0)
        num_for_this_layer = min(num_for_this_layer, num_experts)
        masks[layer_idx, :num_for_this_layer] = True

    # Set non-MoE layers to all True
    for layer_idx in range(num_layers):
        if layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:
            masks[layer_idx, :] = True

    return masks


def generate_random_masks(
    num_layers: int,
    num_experts: int,
    num_gpu_experts: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
    seed: int = 42,
) -> torch.Tensor:
    """Generate masks by randomly selecting GPU experts (fixed seed).

    Args:
        num_layers: Total number of layers in the model
        num_experts: Number of experts per layer
        num_gpu_experts: Total number of GPU experts to allocate
        first_k_dense_replace: Layer index where MoE layers start
        moe_layer_freq: Frequency of MoE layers
        seed: Random seed for reproducibility

    Returns:
        Boolean mask tensor of shape [num_layers, num_experts]
    """
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    # Collect all MoE (layer, expert) positions
    moe_positions = []
    for layer_idx in range(num_layers):
        is_moe = layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0
        if is_moe:
            for expert_idx in range(num_experts):
                moe_positions.append((layer_idx, expert_idx))

    # Randomly select positions
    if len(moe_positions) > 0:
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)
        num_to_select = min(num_gpu_experts, len(moe_positions))
        selected_indices = torch.randperm(len(moe_positions), generator=rng, device='cpu')[:num_to_select]

        for idx in selected_indices:
            layer_idx, expert_idx = moe_positions[idx]
            masks[layer_idx, expert_idx] = True

    # Set non-MoE layers to all True
    for layer_idx in range(num_layers):
        if layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:
            masks[layer_idx, :] = True

    return masks


def _init_kt_gpu_experts_masks(server_args: "ServerArgs") -> Optional[torch.Tensor]:
    """Initialize GPU experts masks from activation frequency data.

    Args:
        server_args: Global server arguments

    Returns:
        Masks tensor of shape [num_layers, num_experts], or None if KT not configured
    """
    global _KT_GPU_EXPERTS_MASKS

    if _KT_GPU_EXPERTS_MASKS is not None:
        return _KT_GPU_EXPERTS_MASKS

    # Get model config (unwrap VL configs that nest the text model config)
    hf_config = server_args.get_hf_config()

    # fix for kimi-k2.5 models where text_config holds the actual config
    if getattr(hf_config, "text_config", None) is not None:
        hf_config = hf_config.text_config

    num_layers = getattr(hf_config, "num_hidden_layers", None)
    # Try different attribute names for num_experts
    num_experts = getattr(hf_config, "num_local_experts", None)
    if num_experts is None:
        num_experts = getattr(hf_config, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(hf_config, "n_routed_experts", None)

    if num_layers is None or num_experts is None:
        logger.warning(
            "Could not determine num_layers or num_experts from model config."
        )
        return None

    # Get first_k_dense_replace to identify which layers are MoE layers
    first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 0) or 0
    moe_layer_freq = getattr(hf_config, "moe_layer_freq", 1)

    # NEW (2026-04-29): V4-Flash has hash-MoE layers at the front (num_hash_layers,
    # typically 3) which the HF DeepseekV3Config treats as first_k_dense_replace=3
    # by default. But hash layers DO have routed experts (n_routed_experts=256)
    # — they are NOT dense. Letting generate_uniform_masks set masks[0..2,:] = True
    # for hash layers makes the kt_ep_wrapper send all 256 experts to a GPU MoE
    # that only loaded num_gpu_experts_per_layer worth of weights, triggering
    # the fused_moe Hidden size mismatch assert. Subtract num_hash_layers so
    # hash layers are correctly classified as MoE for mask purposes.
    # Origin: kt-sglang 耦合 (V4-Flash hash-MoE handling in kt_ep_wrapper).
    num_hash_layers = getattr(hf_config, "num_hash_layers", 0) or 0
    if num_hash_layers > 0:
        first_k_dense_replace = max(0, first_k_dense_replace - num_hash_layers)

    # Normalize list-form moe_layer_freq (e.g., MiMo-V2-Flash: [0, 1, 1, ...])
    # to standard (first_k_dense_replace, moe_layer_freq=1) form
    if isinstance(moe_layer_freq, list):
        # Find first MoE layer index from the mask
        first_moe = next((i for i, v in enumerate(moe_layer_freq) if v), 0)
        first_k_dense_replace = max(first_k_dense_replace or 0, first_moe)
        moe_layer_freq = 1

    # Count actual MoE layers
    num_moe_layers = sum(
        1 for i in range(num_layers)
        if i >= first_k_dense_replace and i % moe_layer_freq == 0
    )
    total_experts = num_moe_layers * num_experts
    logger.debug(
        "[kt-mask] num_layers=%d num_experts=%d first_k_dense_replace=%s (type=%s) "
        "moe_layer_freq=%s (type=%s) computed_num_moe_layers=%d "
        "hf_config_class=%s.%s num_hash_layers=%s n_hash_layers=%s",
        num_layers, num_experts,
        first_k_dense_replace, type(first_k_dense_replace).__name__,
        moe_layer_freq, type(moe_layer_freq).__name__,
        num_moe_layers,
        type(hf_config).__module__, type(hf_config).__name__,
        getattr(hf_config, 'num_hash_layers', '<missing>'),
        getattr(hf_config, 'n_hash_layers', '<missing>'),
    )

    # Determine num_gpu_experts (total across all layers)
    if server_args.kt_gpu_experts_ratio is not None:
        # Use ratio to calculate total GPU experts
        num_gpu_experts = int(total_experts * server_args.kt_gpu_experts_ratio)
        if server_args.kt_num_gpu_experts is not None:
            logger.warning(
                f"--kt-gpu-experts-ratio={server_args.kt_gpu_experts_ratio} is set, "
                f"ignoring --kt-num-gpu-experts={server_args.kt_num_gpu_experts}. "
                f"Actual total GPU experts: {num_gpu_experts} "
                f"(= {total_experts} total experts × {server_args.kt_gpu_experts_ratio})"
            )
        else:
            logger.info(
                f"Using kt_gpu_experts_ratio={server_args.kt_gpu_experts_ratio}, "
                f"total GPU experts: {num_gpu_experts} "
                f"(= {total_experts} total experts × {server_args.kt_gpu_experts_ratio})"
            )
    elif server_args.kt_num_gpu_experts is not None:
        # kt_num_gpu_experts is per-layer, multiply by num_moe_layers
        num_gpu_experts = server_args.kt_num_gpu_experts * num_moe_layers
        logger.info(
            f"Using kt_num_gpu_experts={server_args.kt_num_gpu_experts} per layer, "
            f"total GPU experts: {num_gpu_experts} "
            f"(= {server_args.kt_num_gpu_experts} × {num_moe_layers} MoE layers)"
        )
    else:
        logger.warning("Either kt_num_gpu_experts or kt_gpu_experts_ratio is required but not set.")
        return None

    # Get GPU expert placement strategy
    strategy = server_args.kt_expert_placement_strategy

    # Generate masks based on strategy
    tp_rank = get_tensor_model_parallel_rank()

    if strategy == "frequency":
        # Load activation frequency from init_expert_location if it's a .pt file
        init_loc = server_args.init_expert_location
        has_activation_freq = init_loc and init_loc.endswith(".pt")

        if has_activation_freq:
            logger.info("Loading activation frequency from %s", init_loc)
            loaded_data = torch.load(init_loc, map_location="cpu", weights_only=True)
            # Handle both dict format (from ExpertDistributionRecorder) and raw tensor
            if isinstance(loaded_data, dict):
                if "logical_count" in loaded_data:
                    activation_counts = loaded_data["logical_count"]
                else:
                    raise ValueError(
                        f"Loaded dict does not contain 'logical_count' key. "
                        f"Available keys: {list(loaded_data.keys())}"
                    )
            else:
                activation_counts = loaded_data
            # Expected shape: [buffer_size, num_layers, num_experts]
            if activation_counts.dim() != 3:
                raise ValueError(
                    f"Expected activation counts tensor with 3 dims [buffer_size, num_layers, num_experts], "
                    f"got {activation_counts.dim()} dims with shape {activation_counts.shape}"
                )
            _, file_num_layers, file_num_experts = activation_counts.shape
            if file_num_layers != num_layers:
                raise ValueError(
                    f"Activation counts num_layers ({file_num_layers}) doesn't match "
                    f"model num_layers ({num_layers})"
                )
            if file_num_experts != num_experts:
                raise ValueError(
                    f"Activation counts num_experts ({file_num_experts}) doesn't match "
                    f"model num_experts ({num_experts})"
                )
            # Sum across buffer_size (dim0) to get total activation counts per expert
            activation_freq = activation_counts.sum(dim=0).float()  # [num_layers, num_experts]
            logger.info("Using frequency-based strategy with activation frequency data")
        else:
            # No activation frequency file, use zeros (uniform distribution)
            logger.warning(
                "Using frequency-based strategy WITHOUT activation frequency data "
                "(uniform distribution fallback)"
            )
            activation_freq = torch.zeros(num_layers, num_experts, dtype=torch.float32)
            # For layers that are actually MoE layers, set uniform distribution
            for layer_idx in range(num_layers):
                if layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0:
                    activation_freq[layer_idx, :] = 1.0

        # Generate masks on rank 0
        if tp_rank == 0:
            masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts)
            # For non-MoE layers, set all experts to GPU
            for layer_idx in range(num_layers):
                if layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:
                    masks[layer_idx, :] = True
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    elif strategy == "front-loading":
        if tp_rank == 0:
            logger.info("Using front-loading strategy for GPU expert placement")
            masks = generate_front_loading_masks(
                num_layers, num_experts, num_gpu_experts,
                first_k_dense_replace, moe_layer_freq
            )
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    elif strategy == "uniform":
        if tp_rank == 0:
            logger.info("Using uniform strategy for GPU expert placement")
            masks = generate_uniform_masks(
                num_layers, num_experts, num_gpu_experts,
                first_k_dense_replace, moe_layer_freq
            )
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    elif strategy == "random":
        if tp_rank == 0:
            logger.info("Using random strategy for GPU expert placement (seed=42)")
            masks = generate_random_masks(
                num_layers, num_experts, num_gpu_experts,
                first_k_dense_replace, moe_layer_freq, seed=42
            )
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    else:
        raise ValueError(f"Unknown kt_expert_placement_strategy: {strategy}")

    if dist.is_initialized():
        dist.broadcast(masks, src=0, group=get_tp_group().cpu_group)

    _KT_GPU_EXPERTS_MASKS = masks

    # Log per-layer GPU expert counts (rank 0 only, MoE layers only)
    if tp_rank == 0:
        per_layer_gpu_experts = masks.sum(dim=1).cpu().tolist()
        for layer_idx, num_gpu in enumerate(per_layer_gpu_experts):
            is_moe_layer = (
                layer_idx >= first_k_dense_replace
                and layer_idx % moe_layer_freq == 0
            )
            # Only log for actual MoE layers
            if is_moe_layer:
                logger.info(
                    "KT GPU experts: layer %d (MoE) has %d GPU experts",
                    layer_idx,
                    int(num_gpu),
                )

        # Count total GPU experts only for actual MoE layers
        total_moe_gpu_experts = sum(
            masks[i].sum().item()
            for i in range(num_layers)
            if i >= first_k_dense_replace and i % moe_layer_freq == 0
        )
        num_moe_layers = sum(
            1 for i in range(num_layers)
            if i >= first_k_dense_replace and i % moe_layer_freq == 0
        )
        logger.info(
            "Generated KT GPU experts masks using '%s' strategy: %d MoE layers (out of %d total layers) x %d experts, "
            "total GPU experts in MoE layers = %d",
            strategy, num_moe_layers, num_layers, num_experts, total_moe_gpu_experts
        )

    return _KT_GPU_EXPERTS_MASKS


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.

    Args:
        server_args: Global server arguments
        layer_idx: Layer index in the model

    Returns:
        KTConfig if KT is configured and not disabled, None otherwise
    """
    # Check if KT EP wrapper is disabled (e.g., for draft models in speculative decoding)
    from sglang.srt.layers.moe.utils import is_kt_ep_wrapper_disabled

    if is_kt_ep_wrapper_disabled():
        return None

    if server_args.kt_weight_path is None:
        return None

    # Get GPU experts masks (initializes if needed)
    masks = _init_kt_gpu_experts_masks(server_args)
    if masks is None:
        return None

    # Get num_layers from model config (unwrap VL configs)
    hf_config = server_args.get_hf_config()
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config
    num_layers = getattr(hf_config, "num_hidden_layers", None)

    # NOTE: hash-layer skip experiment was tried here (return None when
    # layer_idx < num_hash_layers); it didn't help because the underlying
    # fused_moe shape-mismatch in V4 hash MoE happens with or without KT wrap.
    # Reverted; root cause is in V4 MoE weight layout vs sglang fused_moe.

    # Get mask for this specific layer
    gpu_experts_mask = masks[layer_idx]

    # KT's CPU backend allocates its internal MoE workspaces from this value
    # (for example llamafile's group_max_len).  The experimental scheduler
    # packs several requests into one forward, so pass the aggregate token
    # capacity to KT instead of only the per-request chunk size.
    effective_chunked_prefill_size = server_args.chunked_prefill_size
    try:
        _multi_chunk_batch_size = int(
            os.environ.get("SGLANG_KT_MULTI_CHUNK_BATCH_SIZE", "0")
        )
    except ValueError:
        _multi_chunk_batch_size = 0
    if (
        effective_chunked_prefill_size is not None
        and effective_chunked_prefill_size > 0
        and _multi_chunk_batch_size > 1
    ):
        effective_chunked_prefill_size *= _multi_chunk_batch_size

    return KTConfig(
        layer_idx=layer_idx,
        gpu_experts_mask=gpu_experts_mask,
        cpuinfer_threads=server_args.kt_cpuinfer,
        threadpool_count=server_args.kt_threadpool_count,
        numa_nodes=server_args.kt_numa_nodes,
        weight_path=server_args.kt_weight_path,
        chunked_prefill_size=effective_chunked_prefill_size,
        method=server_args.kt_method,
        max_deferred_experts_per_token=server_args.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
        gpu_prefill_token_threshold=server_args.kt_gpu_prefill_token_threshold,
        kt_enable_dynamic_expert_update=server_args.kt_enable_dynamic_expert_update,
        expert_lora_path=getattr(server_args, "kt_expert_lora_path", None),
    )


def mask_and_remap_expert_ids(
    topk_ids: torch.Tensor,
    gpu_experts_mask: torch.Tensor,
    logical_to_gpu_index: torch.Tensor,
) -> torch.Tensor:
    """Mask CPU expert IDs and remap GPU expert IDs to weight indices.

    This function:
    1. Sets CPU expert IDs (gpu_experts_mask=False) to -1 so GPU kernel skips them
    2. Remaps GPU expert IDs to GPU weight indices (0 to num_gpu_experts-1)

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing logical expert IDs
        gpu_experts_mask: Boolean tensor of shape [num_experts] where True indicates GPU expert
        logical_to_gpu_index: Int tensor of shape [num_experts] mapping logical ID to GPU index

    Returns:
        Remapped topk_ids tensor with GPU indices for GPU experts, -1 for CPU experts
    """
    # Keep this helper in eager mode. In the dynamic expert update path the
    # mask changes every batch; torch.compile/Inductor has repeatedly crashed
    # here after several runtime swaps on Qwen3-30B-A3B.
    is_gpu_expert = gpu_experts_mask[topk_ids]
    # For GPU experts: remap to GPU weight index; for CPU experts: set to -1
    remapped_ids = torch.where(is_gpu_expert, logical_to_gpu_index[topk_ids], -1)
    return remapped_ids


def mask_expert_ids_for_cpu(
    topk_ids: torch.Tensor,
    gpu_experts_mask: torch.Tensor,
    dummy_expert_id: int = -1,
) -> torch.Tensor:
    """Mask GPU-resident experts before submitting work to the CPU backend.

    KT 0.5.0's LLAMAFILE backend only skips a static prefix of expert IDs.
    Keep the split authoritative in Python: GPU experts are replaced by -1
    for the CPU task, and CPU experts keep their logical IDs. LLAMAFILE checks
    negative IDs before indexing its expert buffers, so -1 avoids doing a
    zero-weight dummy expert GEMM.
    """
    valid = topk_ids.ge(0)
    lookup_ids = topk_ids.clamp_min(0)
    is_gpu_expert = gpu_experts_mask[lookup_ids] & valid
    should_skip = is_gpu_expert | ~valid
    dummy_ids = torch.full_like(topk_ids, int(dummy_expert_id))
    return torch.where(should_skip, dummy_ids, topk_ids)


def mask_expert_ids_and_weights_for_cpu(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gpu_experts_mask: torch.Tensor,
    dummy_expert_id: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return CPU-safe expert ids and weights for runtime dynamic placement.

    GPU-resident experts are already handled by the GPU MoE path. For the CPU
    backend we therefore zero their route weights and route them to -1. This
    preserves numerical split semantics and lets kt-kernel skip the route
    before launching expert GEMMs.
    """
    valid = topk_ids.ge(0)
    lookup_ids = topk_ids.clamp_min(0)
    is_gpu_expert = gpu_experts_mask[lookup_ids] & valid
    should_skip = is_gpu_expert | ~valid
    dummy_ids = torch.full_like(topk_ids, int(dummy_expert_id))
    cpu_topk_ids = torch.where(should_skip, dummy_ids, topk_ids)
    cpu_topk_weights = torch.where(
        should_skip, torch.zeros_like(topk_weights), topk_weights
    )
    return cpu_topk_ids, cpu_topk_weights


def select_top_experts_from_batch(
    topk_ids: torch.Tensor,
    num_experts: int,
    num_gpu_experts: int,
) -> torch.Tensor:
    """Select top N most frequently activated experts from batch routing results.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing logical expert IDs
        num_experts: Total number of experts in the layer
        num_gpu_experts: Number of experts to select for GPU

    Returns:
        Tensor of shape [num_gpu_experts] containing selected expert IDs (sorted)

    Edge cases:
        - If batch has fewer unique experts than num_gpu_experts, fills remaining
          slots with least-activated experts (maintaining determinism)
        - Handles ties by preferring lower expert IDs (deterministic)
    """
    # Count activation frequency for each expert in this batch
    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)

    # Flatten topk_ids and count occurrences
    flat_ids = topk_ids.flatten()
    # Filter out invalid IDs (< 0 or >= num_experts)
    valid_mask = (flat_ids >= 0) & (flat_ids < num_experts)
    valid_ids = flat_ids[valid_mask]

    if valid_ids.numel() > 0:
        expert_counts.index_add_(0, valid_ids, torch.ones_like(valid_ids, dtype=torch.int64))

    # Select top num_gpu_experts by frequency
    # For ties, torch.topk with sorted=True will prefer earlier indices (deterministic)
    _, selected_indices = torch.topk(
        expert_counts,
        k=min(num_gpu_experts, num_experts),
        largest=True,
        sorted=True  # Ensures deterministic tie-breaking
    )

    # Sort selected indices for easier debugging and consistent ordering
    selected_experts = selected_indices.sort()[0]

    return selected_experts


class RuntimeHFExpertWeightLoader:
    """Load individual Qwen3-MoE experts from HF safetensors into GPU slots.

    This is a runtime fallback for LLAMAFILE KT CPU experts. LLAMAFILE can
    compute CPU experts from GGUF, but it does not expose a C++ task to write
    one CPU expert into a GPU staging buffer. For the current Qwen3-30B-A3B
    BF16/TP=1 setup, the original HF safetensors contain per-expert BF16
    weights, and FusedMoE already has a loader that writes one logical expert
    into one local GPU slot. This class connects those two pieces.
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.index_path = self.model_path / "model.safetensors.index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"HF safetensors index not found: {self.index_path}"
            )
        self.weight_map = json.loads(
            self.index_path.read_text(encoding="utf-8")
        )["weight_map"]
        self._files: Dict[str, Any] = {}
        self._file_lock = threading.Lock()
        self._prefetch_enabled = (
            os.environ.get("SGLANG_KT_RUNTIME_ASYNC_PREFETCH", "0") == "1"
        )
        try:
            self._prefetch_workers = int(
                os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_WORKERS", "2")
            )
        except ValueError:
            self._prefetch_workers = 2
        try:
            self._prefetch_max_items = int(
                os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_MAX_ITEMS", "64")
            )
        except ValueError:
            self._prefetch_max_items = 64
        self._prefetch_workers = max(1, self._prefetch_workers)
        self._prefetch_max_items = max(1, self._prefetch_max_items)
        self._prefetch_executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(
                max_workers=self._prefetch_workers,
                thread_name_prefix="kt-hf-prefetch",
            )
            if self._prefetch_enabled
            else None
        )
        self._prefetch_lock = threading.Lock()
        self._prefetch_cache: "OrderedDict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._prefetch_futures: Dict[Tuple[int, int], Future] = {}
        self.prefetch_submitted = 0
        self.prefetch_cache_hits = 0
        self.prefetch_future_hits = 0
        self.prefetch_direct_loads = 0
        self.prefetch_wait_ms_total = 0.0
        self.last_load_source = "direct"
        self.last_prefetch_wait_ms = 0.0
        try:
            self._cpu_tensor_cache_max_items = int(
                os.environ.get("SGLANG_KT_RUNTIME_CPU_TENSOR_CACHE_ITEMS", "0")
            )
        except ValueError:
            self._cpu_tensor_cache_max_items = 0
        self._cpu_tensor_cache_max_items = max(0, self._cpu_tensor_cache_max_items)
        self._pin_cpu_tensors = (
            os.environ.get("SGLANG_KT_RUNTIME_PIN_CPU_TENSORS", "0") == "1"
        )
        self._register_cpu_tensors = (
            os.environ.get("SGLANG_KT_RUNTIME_REGISTER_CPU_TENSORS", "0")
            == "1"
        )
        if self._register_cpu_tensors and self._cpu_tensor_cache_max_items <= 0:
            logger.warning(
                "KT runtime CPU tensor host registration requires a bounded "
                "CPU tensor cache; registration disabled"
            )
            self._register_cpu_tensors = False
        self._cpu_tensor_cache_lock = threading.Lock()
        self._cpu_tensor_cache: "OrderedDict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._host_registration_lock = threading.Lock()
        self._host_registered_keys: set[Tuple[int, int]] = set()
        self.cpu_tensor_host_register_successes = 0
        self.cpu_tensor_host_register_failures = 0
        self.cpu_tensor_host_unregisters = 0
        self.cpu_tensor_cache_hits = 0
        self.cpu_tensor_cache_misses = 0
        self._gpu_prefetch_enabled = (
            os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH", "0") == "1"
        )
        try:
            self._gpu_prefetch_workers = int(
                os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH_WORKERS", "1")
            )
        except ValueError:
            self._gpu_prefetch_workers = 1
        self._gpu_prefetch_workers = max(1, self._gpu_prefetch_workers)
        self._gpu_prefetch_executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(
                max_workers=self._gpu_prefetch_workers,
                thread_name_prefix="kt-gpu-prefetch",
            )
            if self._gpu_prefetch_enabled
            else None
        )
        self.gpu_prefetch_submitted = 0
        if self._prefetch_enabled:
            logger.warning(
                "KT runtime async HF expert prefetch enabled: workers=%d max_items=%d",
                self._prefetch_workers,
                self._prefetch_max_items,
            )
        if self._gpu_prefetch_enabled:
            logger.warning(
                "KT runtime GPU expert slot prefetch enabled: workers=%d",
                self._gpu_prefetch_workers,
            )
        if self._cpu_tensor_cache_max_items > 0:
            logger.warning(
                "KT runtime CPU expert tensor cache enabled: max_items=%d "
                "pin=%s host_register=%s",
                self._cpu_tensor_cache_max_items,
                self._pin_cpu_tensors,
                self._register_cpu_tensors,
            )

    @staticmethod
    def _cuda_call_succeeded(result: Any) -> bool:
        value = getattr(result, "value", result)
        try:
            return int(value) == 0
        except (TypeError, ValueError):
            return False

    def _register_cpu_tensor_tuple(
        self,
        key: Tuple[int, int],
        tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> bool:
        if not self._register_cpu_tensors or not torch.cuda.is_available():
            return False
        with self._host_registration_lock:
            if key in self._host_registered_keys:
                return True
            registered: List[torch.Tensor] = []
            try:
                cudart = torch.cuda.cudart()
                for tensor in tensors:
                    result = cudart.cudaHostRegister(
                        tensor.data_ptr(),
                        tensor.numel() * tensor.element_size(),
                        0,
                    )
                    if not self._cuda_call_succeeded(result):
                        raise RuntimeError(f"cudaHostRegister returned {result}")
                    registered.append(tensor)
            except Exception as exc:
                cudart = torch.cuda.cudart()
                for tensor in registered:
                    try:
                        cudart.cudaHostUnregister(tensor.data_ptr())
                    except Exception:
                        pass
                self.cpu_tensor_host_register_failures += 1
                self._register_cpu_tensors = False
                logger.warning(
                    "KT runtime CPU tensor host registration disabled after "
                    "failure: key=%s error=%s",
                    key,
                    exc,
                )
                return False
            self._host_registered_keys.add(key)
            self.cpu_tensor_host_register_successes += 1
            return True

    def _unregister_cpu_tensor_tuple(
        self,
        key: Tuple[int, int],
        tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        with self._host_registration_lock:
            if key not in self._host_registered_keys:
                return
            cudart = torch.cuda.cudart()
            for tensor in tensors:
                result = cudart.cudaHostUnregister(tensor.data_ptr())
                if not self._cuda_call_succeeded(result):
                    logger.warning(
                        "KT runtime CPU tensor host unregister failed: "
                        "key=%s result=%s",
                        key,
                        result,
                    )
            self._host_registered_keys.remove(key)
            self.cpu_tensor_host_unregisters += 1

    def _get_tensor(self, name: str) -> torch.Tensor:
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"weight {name} not found in {self.index_path}")
        path = str((self.model_path / filename).resolve())
        with self._file_lock:
            handle = self._files.get(path)
            if handle is None:
                from safetensors import safe_open

                handle = safe_open(path, framework="pt", device="cpu")
                self._files[path] = handle
            return handle.get_tensor(name)

    def _load_expert_tensors_cpu(
        self, layer_idx: int, logical_expert_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (int(layer_idx), int(logical_expert_id))
        if self._cpu_tensor_cache_max_items > 0:
            with self._cpu_tensor_cache_lock:
                cached = self._cpu_tensor_cache.pop(key, None)
                if cached is not None:
                    self._cpu_tensor_cache[key] = cached
                    self.cpu_tensor_cache_hits += 1
                    return cached

        prefix = f"model.layers.{layer_idx}.mlp.experts.{logical_expert_id}"
        gate = self._get_tensor(f"{prefix}.gate_proj.weight")
        up = self._get_tensor(f"{prefix}.up_proj.weight")
        down = self._get_tensor(f"{prefix}.down_proj.weight")
        tensors = (gate, up, down)
        if self._pin_cpu_tensors:
            try:
                tensors = tuple(t.pin_memory() for t in tensors)  # type: ignore[assignment]
            except Exception as exc:
                self._pin_cpu_tensors = False
                logger.warning(
                    "KT runtime CPU tensor pinning disabled after failure: %s",
                    exc,
                )
        elif self._register_cpu_tensors:
            self._register_cpu_tensor_tuple(key, tensors)
        if self._cpu_tensor_cache_max_items > 0:
            evicted: List[
                Tuple[
                    Tuple[int, int],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                ]
            ] = []
            with self._cpu_tensor_cache_lock:
                self._cpu_tensor_cache[key] = tensors
                self.cpu_tensor_cache_misses += 1
                while len(self._cpu_tensor_cache) > self._cpu_tensor_cache_max_items:
                    evicted.append(self._cpu_tensor_cache.popitem(last=False))
            for evicted_key, evicted_tensors in evicted:
                self._unregister_cpu_tensor_tuple(
                    evicted_key, evicted_tensors
                )
        return tensors

    @property
    def prefetch_enabled(self) -> bool:
        return self._prefetch_enabled and self._prefetch_executor is not None

    @property
    def gpu_prefetch_enabled(self) -> bool:
        return (
            self._gpu_prefetch_enabled
            and self._gpu_prefetch_executor is not None
            and torch.cuda.is_available()
        )

    def _evict_prefetch_cache_locked(self) -> None:
        while (
            len(self._prefetch_cache) + len(self._prefetch_futures)
            > self._prefetch_max_items
            and self._prefetch_cache
        ):
            self._prefetch_cache.popitem(last=False)

    def _drain_completed_prefetch_locked(self) -> None:
        completed = [
            key for key, future in self._prefetch_futures.items() if future.done()
        ]
        for key in completed:
            future = self._prefetch_futures.pop(key)
            try:
                self._prefetch_cache[key] = future.result()
            except Exception as exc:
                logger.warning(
                    "KT runtime async HF expert prefetch failed: layer=%d "
                    "expert=%d error=%s",
                    key[0],
                    key[1],
                    exc,
                )
        self._evict_prefetch_cache_locked()

    def prefetch_expert(self, layer_idx: int, logical_expert_id: int) -> bool:
        if not self.prefetch_enabled:
            return False
        key = (int(layer_idx), int(logical_expert_id))
        with self._prefetch_lock:
            self._drain_completed_prefetch_locked()
            if key in self._prefetch_cache or key in self._prefetch_futures:
                return False
            self._evict_prefetch_cache_locked()
            if (
                len(self._prefetch_cache) + len(self._prefetch_futures)
                >= self._prefetch_max_items
            ):
                return False
            assert self._prefetch_executor is not None
            self._prefetch_futures[key] = self._prefetch_executor.submit(
                self._load_expert_tensors_cpu,
                key[0],
                key[1],
            )
            self.prefetch_submitted += 1
            return True

    def prefetch_experts(self, layer_idx: int, expert_ids: List[int]) -> int:
        submitted = 0
        for expert_id in expert_ids:
            if self.prefetch_expert(layer_idx, int(expert_id)):
                submitted += 1
        return submitted

    def _consume_prefetched_or_load(
        self, layer_idx: int, logical_expert_id: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], str, float]:
        key = (int(layer_idx), int(logical_expert_id))
        future: Optional[Future] = None
        with self._prefetch_lock:
            self._drain_completed_prefetch_locked()
            cached = self._prefetch_cache.pop(key, None)
            if cached is not None:
                self._prefetch_cache[key] = cached
                self.prefetch_cache_hits += 1
                return cached, "cache", 0.0
            future = self._prefetch_futures.pop(key, None)

        if future is not None:
            t_wait = time.perf_counter()
            tensors = future.result()
            wait_ms = (time.perf_counter() - t_wait) * 1000.0
            self.prefetch_future_hits += 1
            self.prefetch_wait_ms_total += wait_ms
            return tensors, "future", wait_ms

        self.prefetch_direct_loads += 1
        return self._load_expert_tensors_cpu(layer_idx, logical_expert_id), "direct", 0.0

    def load_expert_to_gpu_slot(
        self,
        *,
        layer: torch.nn.Module,
        layer_idx: int,
        logical_expert_id: int,
        gpu_slot: int,
    ) -> Dict[str, Any]:
        if not hasattr(layer, "_weight_loader_impl"):
            raise TypeError("runtime HF expert swap requires FusedMoE layer")
        if not (hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight")):
            raise NotImplementedError(
                "runtime HF expert swap currently supports BF16/unquantized "
                "FusedMoE weights with w13_weight and w2_weight only"
            )

        (gate, up, down), source, wait_ms = self._consume_prefetched_or_load(
            layer_idx=layer_idx,
            logical_expert_id=logical_expert_id,
        )
        self.last_load_source = source
        self.last_prefetch_wait_ms = wait_ms

        layer._weight_loader_impl(
            param=layer.w13_weight,
            loaded_weight=gate,
            weight_name="experts.gate_proj.weight",
            shard_id="w1",
            expert_id=gpu_slot,
        )
        layer._weight_loader_impl(
            param=layer.w13_weight,
            loaded_weight=up,
            weight_name="experts.up_proj.weight",
            shard_id="w3",
            expert_id=gpu_slot,
        )
        layer._weight_loader_impl(
            param=layer.w2_weight,
            loaded_weight=down,
            weight_name="experts.down_proj.weight",
            shard_id="w2",
            expert_id=gpu_slot,
        )
        return {
            "source": source,
            "prefetch_wait_ms": wait_ms,
            "keepalive": [gate, up, down],
        }

    def load_expert_to_gpu_slot_streamed(
        self,
        *,
        layer: torch.nn.Module,
        layer_idx: int,
        logical_expert_id: int,
        gpu_slot: int,
        stream: torch.cuda.Stream,
    ) -> Dict[str, Any]:
        """Load one BF16 expert into a GPU slot on a caller-owned CUDA stream.

        This path intentionally avoids calling FusedMoE's Python weight loader
        from a background thread. It is limited to the current TP=1 BF16
        runtime-swap setup and falls back to the normal loader if the expected
        tensor layout is not present.
        """
        if not (hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight")):
            return self.load_expert_to_gpu_slot(
                layer=layer,
                layer_idx=layer_idx,
                logical_expert_id=logical_expert_id,
                gpu_slot=gpu_slot,
            )

        (gate, up, down), source, wait_ms = self._consume_prefetched_or_load(
            layer_idx=layer_idx,
            logical_expert_id=logical_expert_id,
        )
        self.last_load_source = source
        self.last_prefetch_wait_ms = wait_ms

        def _fit_for_dst(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
            if tuple(src.shape) == tuple(dst.shape):
                return src
            if src.dim() == 2 and tuple(src.transpose(0, 1).shape) == tuple(dst.shape):
                return src.transpose(0, 1).contiguous()
            raise ValueError(
                f"cannot stream-copy expert tensor shape {tuple(src.shape)} "
                f"into destination shape {tuple(dst.shape)}"
            )

        try:
            w13_slot = layer.w13_weight.data[gpu_slot]
            w2_slot = layer.w2_weight.data[gpu_slot]
            shard = int(w13_slot.shape[0] // 2)
            gate_dst = w13_slot[:shard]
            up_dst = w13_slot[shard : shard * 2]
            down_dst = w2_slot
            gate = _fit_for_dst(gate, gate_dst)
            up = _fit_for_dst(up, up_dst)
            down = _fit_for_dst(down, down_dst)
            with torch.cuda.stream(stream):
                gate_dst.copy_(gate, non_blocking=True)
                up_dst.copy_(up, non_blocking=True)
                down_dst.copy_(down, non_blocking=True)
        except Exception:
            return self.load_expert_to_gpu_slot(
                layer=layer,
                layer_idx=layer_idx,
                logical_expert_id=logical_expert_id,
                gpu_slot=gpu_slot,
            )
        return {
            "source": source,
            "prefetch_wait_ms": wait_ms,
            "keepalive": [gate, up, down],
        }

    def stage_expert_to_gpu_slot_buffers(
        self,
        *,
        layer: torch.nn.Module,
        layer_idx: int,
        logical_expert_id: int,
        gpu_slot: int,
        stream: torch.cuda.Stream,
    ) -> Dict[str, Any]:
        """Stage one expert into temporary GPU buffers without touching live slots.

        This is the correctness-safe path for non-blocking runtime prefetch:
        the old live GPU slot remains valid until commit.  Commit later copies
        these staging buffers into the live slot on the serving stream and then
        updates logical-to-GPU mappings.
        """
        if not (hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight")):
            raise NotImplementedError(
                "runtime GPU staging prefetch currently supports BF16/unquantized "
                "FusedMoE weights with w13_weight and w2_weight only"
            )

        profile_enabled = os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_PROFILE") == "1"
        t_acquire = time.perf_counter()
        (gate, up, down), source, wait_ms = self._consume_prefetched_or_load(
            layer_idx=layer_idx,
            logical_expert_id=logical_expert_id,
        )
        acquire_ms = (time.perf_counter() - t_acquire) * 1000.0
        self.last_load_source = source
        self.last_prefetch_wait_ms = wait_ms

        def _fit_for_dst(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
            if tuple(src.shape) == tuple(dst.shape):
                return src
            if src.dim() == 2 and tuple(src.transpose(0, 1).shape) == tuple(dst.shape):
                return src.transpose(0, 1).contiguous()
            raise ValueError(
                f"cannot stage expert tensor shape {tuple(src.shape)} "
                f"into destination shape {tuple(dst.shape)}"
            )

        t_allocate = time.perf_counter()
        w13_slot = layer.w13_weight.data[gpu_slot]
        w2_slot = layer.w2_weight.data[gpu_slot]
        w13_stage = torch.empty_like(w13_slot)
        w2_stage = torch.empty_like(w2_slot)
        shard = int(w13_stage.shape[0] // 2)
        gate_dst = w13_stage[:shard]
        up_dst = w13_stage[shard : shard * 2]
        down_dst = w2_stage

        gate = _fit_for_dst(gate, gate_dst)
        up = _fit_for_dst(up, up_dst)
        down = _fit_for_dst(down, down_dst)
        allocate_ms = (time.perf_counter() - t_allocate) * 1000.0
        t_h2d = time.perf_counter()
        with torch.cuda.stream(stream):
            gate_dst.copy_(gate, non_blocking=True)
            up_dst.copy_(up, non_blocking=True)
            down_dst.copy_(down, non_blocking=True)
        h2d_enqueue_ms = (time.perf_counter() - t_h2d) * 1000.0

        return {
            "source": source,
            "prefetch_wait_ms": wait_ms,
            "staged_slot": {
                "slot": int(gpu_slot),
                "logical_expert_id": int(logical_expert_id),
                "w13": w13_stage,
                "w2": w2_stage,
            },
            "keepalive": [gate, up, down, w13_stage, w2_stage],
            "profile": {
                "enabled": profile_enabled,
                "acquire_ms": acquire_ms,
                "allocate_ms": allocate_ms,
                "h2d_enqueue_ms": h2d_enqueue_ms,
                "staging_bytes": int(
                    w13_stage.numel() * w13_stage.element_size()
                    + w2_stage.numel() * w2_stage.element_size()
                ),
            },
        }

    def _stage_experts_to_gpu_slot_buffers_for_prefetch(
        self,
        *,
        layer: torch.nn.Module,
        layer_idx: int,
        slot_to_expert: List[Tuple[int, int]],
        device_index: int,
        submitted_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        worker_started_at = time.perf_counter()
        queue_delay_ms = (
            max(0.0, worker_started_at - float(submitted_at)) * 1000.0
            if submitted_at is not None
            else 0.0
        )
        torch.cuda.set_device(device_index)
        stream = torch.cuda.Stream(device=device_index)
        sources: Dict[str, int] = {}
        prefetch_wait_ms = 0.0
        keepalive: List[torch.Tensor] = []
        staged_slots: List[Dict[str, Any]] = []
        acquire_ms = 0.0
        allocate_ms = 0.0
        h2d_enqueue_ms = 0.0
        staging_bytes = 0
        t0 = time.perf_counter()
        for slot, expert_id in slot_to_expert:
            info = self.stage_expert_to_gpu_slot_buffers(
                layer=layer,
                layer_idx=layer_idx,
                logical_expert_id=expert_id,
                gpu_slot=slot,
                stream=stream,
            )
            source = str(info.get("source", "unknown"))
            sources[source] = sources.get(source, 0) + 1
            prefetch_wait_ms += float(info.get("prefetch_wait_ms", 0.0) or 0.0)
            staged_slot = info.get("staged_slot")
            if isinstance(staged_slot, dict):
                staged_slots.append(staged_slot)
            profile = info.get("profile")
            if isinstance(profile, dict):
                acquire_ms += float(profile.get("acquire_ms", 0.0) or 0.0)
                allocate_ms += float(profile.get("allocate_ms", 0.0) or 0.0)
                h2d_enqueue_ms += float(
                    profile.get("h2d_enqueue_ms", 0.0) or 0.0
                )
                staging_bytes += int(profile.get("staging_bytes", 0) or 0)
            tensors = info.get("keepalive")
            if isinstance(tensors, list):
                keepalive.extend(t for t in tensors if isinstance(t, torch.Tensor))
        t_sync = time.perf_counter()
        stream.synchronize()
        stream_sync_ms = (time.perf_counter() - t_sync) * 1000.0
        return {
            "sources": sources,
            "prefetch_wait_ms": prefetch_wait_ms,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            "changed_slots": len(slot_to_expert),
            "staged_slots": staged_slots,
            "staging": True,
            "keepalive": keepalive,
            "background": True,
            "profile": {
                "queue_delay_ms": queue_delay_ms,
                "acquire_ms": acquire_ms,
                "allocate_ms": allocate_ms,
                "h2d_enqueue_ms": h2d_enqueue_ms,
                "stream_sync_ms": stream_sync_ms,
                "staging_bytes": staging_bytes,
            },
        }

    def submit_stage_gpu_prefetch(
        self,
        *,
        layer: torch.nn.Module,
        layer_idx: int,
        slot_to_expert: List[Tuple[int, int]],
        device: torch.device,
    ) -> Optional[Future]:
        if not self.gpu_prefetch_enabled or not slot_to_expert:
            return None
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        assert self._gpu_prefetch_executor is not None
        self.gpu_prefetch_submitted += 1
        submitted_at = time.perf_counter()
        return self._gpu_prefetch_executor.submit(
            self._stage_experts_to_gpu_slot_buffers_for_prefetch,
            layer=layer,
            layer_idx=layer_idx,
            slot_to_expert=list(slot_to_expert),
            device_index=int(device_index),
            submitted_at=submitted_at,
        )

    def _stream_experts_to_live_gpu_slots_for_prefetch(
        self,
        *,
        layer: torch.nn.Module,
        layer_idx: int,
        slot_to_expert: List[Tuple[int, int]],
        device_index: int,
        submitted_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Copy future-layer experts directly into their live slots.

        The caller must not run the target layer until this future completes.
        This removes the temporary GPU staging allocation and the later D2D
        copy. It is deliberately separate from the staging path because a
        ready-only miss is unsafe once a live slot has begun changing.
        """

        worker_started_at = time.perf_counter()
        queue_delay_ms = (
            max(0.0, worker_started_at - float(submitted_at)) * 1000.0
            if submitted_at is not None
            else 0.0
        )
        torch.cuda.set_device(device_index)
        stream = torch.cuda.Stream(device=device_index)
        sources: Dict[str, int] = {}
        prefetch_wait_ms = 0.0
        keepalive: List[torch.Tensor] = []
        acquire_ms = 0.0
        h2d_enqueue_ms = 0.0
        t0 = time.perf_counter()
        for slot, expert_id in slot_to_expert:
            t_copy = time.perf_counter()
            info = self.load_expert_to_gpu_slot_streamed(
                layer=layer,
                layer_idx=layer_idx,
                logical_expert_id=expert_id,
                gpu_slot=slot,
                stream=stream,
            )
            elapsed_ms = (time.perf_counter() - t_copy) * 1000.0
            source = str(info.get("source", "unknown"))
            sources[source] = sources.get(source, 0) + 1
            prefetch_wait_ms += float(info.get("prefetch_wait_ms", 0.0) or 0.0)
            # The direct path loads host tensors and enqueues H2D in one
            # helper; retain the full cost as acquire time for diagnostics.
            acquire_ms += elapsed_ms
            h2d_enqueue_ms += elapsed_ms
            tensors = info.get("keepalive")
            if isinstance(tensors, list):
                keepalive.extend(t for t in tensors if isinstance(t, torch.Tensor))
        t_sync = time.perf_counter()
        stream.synchronize()
        stream_sync_ms = (time.perf_counter() - t_sync) * 1000.0
        return {
            "sources": sources,
            "prefetch_wait_ms": prefetch_wait_ms,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            "changed_slots": len(slot_to_expert),
            "staged_slots": [],
            "staging": False,
            "live_slot_write": True,
            "keepalive": keepalive,
            "background": True,
            "profile": {
                "queue_delay_ms": queue_delay_ms,
                "acquire_ms": acquire_ms,
                "allocate_ms": 0.0,
                "h2d_enqueue_ms": h2d_enqueue_ms,
                "stream_sync_ms": stream_sync_ms,
                "staging_bytes": 0,
            },
        }

    def submit_streamed_gpu_prefetch(
        self,
        *,
        layer: torch.nn.Module,
        layer_idx: int,
        slot_to_expert: List[Tuple[int, int]],
        device: torch.device,
    ) -> Optional[Future]:
        if not self.gpu_prefetch_enabled or not slot_to_expert:
            return None
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        assert self._gpu_prefetch_executor is not None
        self.gpu_prefetch_submitted += 1
        submitted_at = time.perf_counter()
        return self._gpu_prefetch_executor.submit(
            self._stream_experts_to_live_gpu_slots_for_prefetch,
            layer=layer,
            layer_idx=layer_idx,
            slot_to_expert=list(slot_to_expert),
            device_index=int(device_index),
            submitted_at=submitted_at,
        )

    def _load_experts_to_gpu_slots_for_prefetch(
        self,
        *,
        layer: torch.nn.Module,
        layer_idx: int,
        slot_to_expert: List[Tuple[int, int]],
        device_index: int,
    ) -> Dict[str, Any]:
        torch.cuda.set_device(device_index)
        stream = torch.cuda.Stream(device=device_index)
        t0 = time.perf_counter()
        sources: Dict[str, int] = {}
        prefetch_wait_ms = 0.0
        with torch.cuda.stream(stream):
            for slot, expert_id in slot_to_expert:
                info = self.load_expert_to_gpu_slot(
                    layer=layer,
                    layer_idx=layer_idx,
                    logical_expert_id=expert_id,
                    gpu_slot=slot,
                )
                source = str(info.get("source", "unknown"))
                sources[source] = sources.get(source, 0) + 1
                prefetch_wait_ms += float(info.get("prefetch_wait_ms", 0.0) or 0.0)
        stream.synchronize()
        return {
            "sources": sources,
            "prefetch_wait_ms": prefetch_wait_ms,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            "changed_slots": len(slot_to_expert),
        }

    def submit_gpu_prefetch(
        self,
        *,
        layer: torch.nn.Module,
        layer_idx: int,
        slot_to_expert: List[Tuple[int, int]],
        device: torch.device,
    ) -> Optional[Future]:
        if not self.gpu_prefetch_enabled or not slot_to_expert:
            return None
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        assert self._gpu_prefetch_executor is not None
        self.gpu_prefetch_submitted += 1
        return self._gpu_prefetch_executor.submit(
            self._load_experts_to_gpu_slots_for_prefetch,
            layer=layer,
            layer_idx=layer_idx,
            slot_to_expert=list(slot_to_expert),
            device_index=int(device_index),
        )


def get_runtime_hf_expert_loader() -> Optional[RuntimeHFExpertWeightLoader]:
    global _KT_RUNTIME_HF_EXPERT_LOADER
    if _KT_RUNTIME_HF_EXPERT_LOADER is not None:
        return _KT_RUNTIME_HF_EXPERT_LOADER

    try:
        from sglang.srt.server_args import get_global_server_args

        model_path = get_global_server_args().model_path
        _KT_RUNTIME_HF_EXPERT_LOADER = RuntimeHFExpertWeightLoader(model_path)
        return _KT_RUNTIME_HF_EXPERT_LOADER
    except Exception as exc:
        logger.warning("KT runtime HF expert loader is unavailable: %s", exc)
        return None


def copy_experts_weights_int4(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy INT4 Marlin expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight_packed: Packed INT4 weights for gate+up projection
        - w13_weight_scale: FP16 scales for w13
        - w2_weight_packed: Packed INT4 weights for down projection
        - w2_weight_scale: FP16 scales for w2
    """
    weight_names = ["w13_weight_packed", "w13_weight_scale", "w2_weight_packed", "w2_weight_scale"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            # In src_layer, expert at logical_id is at index logical_id
            # In dst_layer, we write to gpu_index dst_idx
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def copy_experts_weights_fp8(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy FP8 block quant expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight: FP8 weights for gate+up projection
        - w13_weight_scale_inv: FP32 inverse scales for w13
        - w2_weight: FP8 weights for down projection
        - w2_weight_scale_inv: FP32 inverse scales for w2
    """
    weight_names = ["w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def copy_experts_weights_fp8_channel(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy FP8 per-channel quant expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight: FP8 weights for gate+up projection
        - w13_weight_scale: FP32 per-channel scales for w13
        - w2_weight: FP8 weights for down projection
        - w2_weight_scale: FP32 per-channel scales for w2
    """
    weight_names = ["w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def copy_experts_weights_bf16(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy BF16/unquantized expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight: BF16 weights for gate+up projection
        - w2_weight: BF16 weights for down projection
    """
    weight_names = ["w13_weight", "w2_weight"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def update_gpu_expert_mappings(
    selected_experts: torch.Tensor,
    num_experts: int,
    device: torch.device,
):
    """Update GPU expert mapping tables based on newly selected experts.

    Args:
        selected_experts: Tensor of logical expert IDs now on GPU (shape: [num_gpu_experts])
        num_experts: Total number of experts in layer
        device: Target CUDA device for mapping tensors

    Returns:
        Tuple of (gpu_experts_mask, logical_to_gpu_index, gpu_index_to_logical):
            - gpu_experts_mask: CPU bool tensor [num_experts], True = on GPU
            - logical_to_gpu_index: CUDA int32 tensor [num_experts], maps logical -> GPU index
            - gpu_index_to_logical: CPU int32 tensor [num_gpu_experts], reverse mapping
    """
    num_gpu_experts = len(selected_experts)

    # Create new mask (CPU tensor)
    gpu_experts_mask_cpu = torch.zeros(num_experts, dtype=torch.bool, device='cpu')
    gpu_experts_mask_cpu[selected_experts.cpu()] = True

    # Create logical_to_gpu_index (CUDA tensor)
    logical_to_gpu_index = torch.full(
        (num_experts,), -1, dtype=torch.int32, device=device
    )
    for gpu_idx, logical_id in enumerate(selected_experts):
        logical_to_gpu_index[logical_id] = gpu_idx

    # Create gpu_index_to_logical (CPU tensor for weight loading)
    gpu_index_to_logical_cpu = selected_experts.cpu().to(torch.int32)

    return gpu_experts_mask_cpu, logical_to_gpu_index, gpu_index_to_logical_cpu


def update_kt_wrapper_masks(
    wrapper: Optional["KTMoEWrapper"],
    gpu_experts_mask_cpu: torch.Tensor,
) -> None:
    """Update KT wrapper's internal GPU experts mask (rank 0 only).

    Args:
        wrapper: KTMoEWrapper instance (None if not rank 0)
        gpu_experts_mask_cpu: New GPU experts mask to apply

    The wrapper needs updated masks to correctly route tokens to CPU vs GPU experts.
    This is called on rank 0 only since only rank 0 has the wrapper instance.

    CRITICAL: wrapper.gpu_experts_mask is a pinned memory tensor whose pointer is shared
    with C++ code. We MUST use .copy_() to update in-place, not replace the reference.
    """
    if wrapper is None:
        return

    # Update wrapper's internal mask IN-PLACE when the backend exposes one.
    # LLAMAFILE AVX2 currently has no C++ runtime mask pointer, so we attach a
    # Python-side tensor for observability and rely on mask_expert_ids_for_cpu()
    # to enforce the CPU/GPU split before submit_forward().
    wrapper_mask = getattr(wrapper, "gpu_experts_mask", None)
    if isinstance(wrapper_mask, torch.Tensor):
        wrapper_mask.copy_(gpu_experts_mask_cpu)
    else:
        wrapper.gpu_experts_mask = gpu_experts_mask_cpu.clone()


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.

    This wrapper coordinates parallel execution of:
    - GPU experts (identified by gpu_experts_mask=True) using any quantization method
    - CPU experts (identified by gpu_experts_mask=False) using AMX/AVX instructions

    The wrapper implements the submit-compute-sync pattern:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize and merge CPU+GPU results

    Example:
        # Wrap any GPU method with AMX/AVX CPU expert support
        gpu_method = CompressedTensorsWNA16MoEMethod(quant_config, prefix)
        kt_config = KTConfig(layer_idx=0, gpu_experts_mask=mask, ...)
        method = KTEPWrapperMethod(gpu_method, kt_config)
    """

    # Tag for quant_method_registry.is_wrapped_method() — set as a class
    # attribute so isinstance-style checks in deepseek_v2 / glm4_moe work
    # without importing this module.
    _quant_wrapper_id = "kt_ep"

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTConfig,
    ):
        """Initialize the KT EP wrapper.

        Args:
            gpu_method: The quantization method to use for GPU experts
            kt_config: Configuration for KT CPU expert computation
        """
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config
        self.gpu_experts_mask = kt_config.gpu_experts_mask  # bool tensor [num_experts], on CPU
        self.num_gpu_experts = int(self.gpu_experts_mask.sum().item())
        self.kt_expert_lora_path = kt_config.expert_lora_path
        self.kt_expert_lora_enabled = bool(self.kt_expert_lora_path)
        self.kt_expert_lora_weights: Optional[KTExpertLoraWeights] = None
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_tensor_model_parallel_rank()
        if self.kt_expert_lora_enabled:
            if self.num_gpu_experts != 0:
                raise ValueError(
                    "--kt-expert-lora-path first supports CPU experts only. "
                    "Set --kt-num-gpu-experts 0 and do not enable "
                    "--kt-gpu-experts-ratio."
                )
            if kt_config.gpu_prefill_token_threshold:
                raise ValueError(
                    "--kt-expert-lora-path is not compatible with "
                    "--kt-gpu-prefill-token-threshold in the first single-adapter "
                    "implementation."
                )
            if kt_config.kt_enable_dynamic_expert_update:
                raise ValueError(
                    "--kt-expert-lora-path is not compatible with "
                    "--kt-enable-dynamic-expert-update in the first single-adapter "
                    "implementation."
                )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[kt-wrap-init] tp_rank=%d layer_idx=%s num_gpu_experts=%d "
                "mask_sum=%d mask_shape=%s gpu_method=%s",
                self.tp_rank,
                kt_config.layer_idx,
                self.num_gpu_experts,
                int(self.gpu_experts_mask.sum().item()),
                tuple(self.gpu_experts_mask.shape),
                type(gpu_method).__name__,
            )

        # Mapping tables for non-contiguous GPU expert allocation (CPU tensors)
        # Used by weight_loader to remap expert_id when loading weights
        gpu_expert_indices = torch.where(self.gpu_experts_mask)[0]
        self._kt_runtime_initial_gpu_experts = [
            int(x) for x in gpu_expert_indices.cpu().tolist()
        ]
        self.logical_to_gpu_index = torch.full(
            (len(self.gpu_experts_mask),), -1, dtype=torch.int32
        )
        self.logical_to_gpu_index[gpu_expert_indices] = torch.arange(
            len(gpu_expert_indices), dtype=torch.int32
        )
        self.gpu_index_to_logical = gpu_expert_indices.to(torch.int32)

        # CUDA tensors for inference (will be set in create_weights)
        self.gpu_experts_mask_cuda = None
        self.logical_to_gpu_index_cuda = None

        self.gpu_prefill_token_threshold = kt_config.gpu_prefill_token_threshold or 0
        self._full_init_args = None
        self.wrapper: Optional[KTMoEWrapper] = None
        self._kt_cpu_route_mask_required = (
            os.environ.get("SGLANG_KT_CPU_ROUTE_MASK", "0") == "1"
            and self.kt_config.method.upper() == "LLAMAFILE"
            and not self.kt_expert_lora_enabled
        )
        self._kt_cpu_native_route_mask = False
        self._kt_cpu_native_route_mask_disabled = (
            os.environ.get("SGLANG_KT_DISABLE_NATIVE_CPU_MASK", "0") == "1"
        )
        self._kt_runtime_hf_expert_swap_enabled = False
        self._kt_runtime_swap_count = 0
        self._runtime_layer_ref: Optional[torch.nn.Module] = None
        self._kt_runtime_gpu_prefetch_enabled = (
            os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH", "0") == "1"
        )
        self._kt_runtime_gpu_prefetch_lock = threading.Lock()
        self._kt_runtime_gpu_prefetch_pending: Optional[Dict[str, Any]] = None
        self._kt_runtime_gpu_prefetch_pending_counter = 0
        self._kt_runtime_gpu_prefetch_committed = 0
        self._kt_runtime_gpu_prefetch_skipped = 0
        self._kt_runtime_gpu_prefetch_no_pending = 0
        self._kt_runtime_gpu_prefetch_not_ready = 0
        self._kt_runtime_gpu_prefetch_missed_drop = 0
        self._kt_runtime_gpu_prefetch_pending_busy = 0
        self._kt_runtime_foreground_swap_no_pending = 0
        self._kt_runtime_foreground_swap_pending_not_ready = 0
        self._kt_runtime_gpu_prefetch_stream: Optional[torch.cuda.Stream] = None
        self._kt_runtime_gpu_prefetch_event: Optional[torch.cuda.Event] = None
        self._kt_runtime_gpu_prefetch_ready_only = (
            os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH_READY_ONLY", "1") == "1"
        )
        self._kt_runtime_gpu_prefetch_staging = (
            os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH_STAGING", "1") == "1"
        )
        self._kt_runtime_gpu_prefetch_direct_block_on_miss = (
            os.environ.get(
                "SGLANG_KT_RUNTIME_GPU_PREFETCH_DIRECT_BLOCK_ON_MISS", "1"
            )
            == "1"
        )
        self._kt_runtime_gpu_prefetch_discard_missed = (
            os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH_DISCARD_MISSED", "1")
            == "1"
        )
        self._kt_runtime_gpu_prefetch_background = (
            os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH_BACKGROUND", "0")
            == "1"
        )
        self._kt_runtime_foreground_oracle = (
            os.environ.get("SGLANG_KT_RUNTIME_FOREGROUND_ORACLE", "0") == "1"
        )
        self._kt_runtime_keep_old_on_prefetch_miss = (
            os.environ.get(
                "SGLANG_KT_RUNTIME_KEEP_OLD_ON_PREFETCH_MISS", "0"
            )
            == "1"
        )
        self._kt_runtime_mapping_full_copy = (
            os.environ.get("SGLANG_KT_RUNTIME_MAPPING_FULL_COPY", "0") == "1"
        )
        try:
            self._kt_runtime_min_gain_entries_per_slot = float(
                os.environ.get(
                    "SGLANG_KT_RUNTIME_MIN_GAIN_ENTRIES_PER_SLOT", "0"
                )
            )
        except ValueError:
            self._kt_runtime_min_gain_entries_per_slot = 0.0
        self._kt_runtime_min_gain_entries_per_slot = max(
            0.0, self._kt_runtime_min_gain_entries_per_slot
        )
        self._kt_runtime_low_gain_slots_skipped = 0
        self._kt_runtime_skip_batch_selection_with_oracle = (
            os.environ.get(
                "SGLANG_KT_RUNTIME_SKIP_BATCH_SELECTION_WITH_ORACLE", "0"
            )
            == "1"
        )
        self._kt_runtime_defer_prefetch_after_cpu_submit = (
            os.environ.get(
                "SGLANG_KT_RUNTIME_DEFER_PREFETCH_AFTER_CPU_SUBMIT", "0"
            )
            == "1"
        )
        self._kt_runtime_deferred_prefetch: Optional[
            Tuple[RuntimeHFExpertWeightLoader, torch.Tensor]
        ] = None
        try:
            self._kt_runtime_hot_ratio = float(
                os.environ.get("SGLANG_KT_RUNTIME_HOT_RATIO", "1.0")
            )
        except ValueError:
            self._kt_runtime_hot_ratio = 1.0
        self._kt_runtime_hot_ratio = min(1.0, max(0.0, self._kt_runtime_hot_ratio))
        self._kt_runtime_tail_policy = os.environ.get(
            "SGLANG_KT_RUNTIME_TAIL_POLICY", "none"
        ).strip().lower()
        try:
            self._kt_runtime_max_changed_slots = int(
                os.environ.get("SGLANG_KT_RUNTIME_MAX_CHANGED_SLOTS", "0")
            )
        except ValueError:
            self._kt_runtime_max_changed_slots = 0
        self._kt_runtime_max_changed_slots = max(0, self._kt_runtime_max_changed_slots)
        self._kt_runtime_global_counts_cpu = torch.zeros(
            len(self.gpu_experts_mask), dtype=torch.int64, device="cpu"
        )
        self._kt_runtime_last_batch_ranked_cpu: List[int] = []
        self._kt_runtime_prefetch_next_layer = (
            os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_NEXT_LAYER", "1") == "1"
        )
        self._kt_runtime_cpu_prefetch_enabled = (
            os.environ.get("SGLANG_KT_RUNTIME_CPU_PREFETCH_ENABLE", "1") == "1"
        )
        try:
            self._kt_runtime_prefetch_lookahead_layers = int(
                os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_LOOKAHEAD_LAYERS", "1")
            )
        except ValueError:
            self._kt_runtime_prefetch_lookahead_layers = 1
        self._kt_runtime_prefetch_lookahead_layers = max(
            1, self._kt_runtime_prefetch_lookahead_layers
        )
        try:
            self._kt_runtime_gpu_prefetch_lookahead_layers = int(
                os.environ.get(
                    "SGLANG_KT_RUNTIME_GPU_PREFETCH_LOOKAHEAD_LAYERS",
                    str(self._kt_runtime_prefetch_lookahead_layers),
                )
            )
        except ValueError:
            self._kt_runtime_gpu_prefetch_lookahead_layers = (
                self._kt_runtime_prefetch_lookahead_layers
            )
        self._kt_runtime_gpu_prefetch_lookahead_layers = max(
            1, self._kt_runtime_gpu_prefetch_lookahead_layers
        )
        try:
            self._kt_runtime_prefetch_stage_size = int(
                os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_STAGE_SIZE", "1")
            )
        except ValueError:
            self._kt_runtime_prefetch_stage_size = 1
        self._kt_runtime_prefetch_stage_size = max(
            1, self._kt_runtime_prefetch_stage_size
        )
        try:
            self._kt_runtime_cpu_prefetch_stage_span = int(
                os.environ.get("SGLANG_KT_RUNTIME_CPU_PREFETCH_STAGE_SPAN", "1")
            )
        except ValueError:
            self._kt_runtime_cpu_prefetch_stage_span = 1
        self._kt_runtime_cpu_prefetch_stage_span = max(
            1, self._kt_runtime_cpu_prefetch_stage_span
        )
        self._kt_runtime_prefetch_stage_boundary_only = (
            os.environ.get(
                "SGLANG_KT_RUNTIME_PREFETCH_STAGE_BOUNDARY_ONLY", "0"
            )
            == "1"
        )
        self._kt_runtime_update_stage_boundary_only = (
            os.environ.get("SGLANG_KT_RUNTIME_UPDATE_STAGE_BOUNDARY_ONLY", "0")
            == "1"
        )
        self._kt_runtime_prefetch_delta_only = (
            os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_DELTA_ONLY", "0") == "1"
        )
        self._kt_runtime_gpu_prefetch_target_boundary_only = (
            os.environ.get(
                "SGLANG_KT_RUNTIME_GPU_PREFETCH_TARGET_BOUNDARY_ONLY", "0"
            )
            == "1"
        )
        try:
            self._kt_runtime_gpu_prefetch_stage_span = int(
                os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH_STAGE_SPAN", "1")
            )
        except ValueError:
            self._kt_runtime_gpu_prefetch_stage_span = 1
        self._kt_runtime_gpu_prefetch_stage_span = max(
            1, self._kt_runtime_gpu_prefetch_stage_span
        )
        try:
            self._kt_runtime_gpu_prefetch_stage_stride = int(
                os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH_STAGE_STRIDE", "1")
            )
        except ValueError:
            self._kt_runtime_gpu_prefetch_stage_stride = 1
        self._kt_runtime_gpu_prefetch_stage_stride = max(
            1, self._kt_runtime_gpu_prefetch_stage_stride
        )
        self._kt_runtime_prefetch_next_step_layer0 = (
            os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_NEXT_STEP_LAYER0", "0")
            == "1"
        )
        try:
            self._kt_runtime_prefetch_top_k = int(
                os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_TOP_K", "16")
            )
        except ValueError:
            self._kt_runtime_prefetch_top_k = 16
        self._kt_runtime_prefetch_top_k = max(0, self._kt_runtime_prefetch_top_k)
        try:
            self._kt_runtime_prefetch_log_every = int(
                os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_LOG_EVERY", "64")
            )
        except ValueError:
            self._kt_runtime_prefetch_log_every = 64
        self._kt_runtime_prefetch_log_every = max(
            1, self._kt_runtime_prefetch_log_every
        )
        self._kt_runtime_prefetch_schedule_count = 0

        # Dual-stream parallelism: cpu_stream for CPU expert operations,
        # main stream for GPU computation (initialized in create_weights)
        self._cpu_stream: Optional[torch.cuda.Stream] = None
        self._sync_done_event: Optional[torch.cuda.Event] = None  # CPU computation done

        # Shared staging buffer reference (initialized in create_weights, shared across all layers)
        self._shared_staging_buffer: Optional[SharedStagingBuffer] = None
        # get_kt_config already expands this for a packed forward, and the
        # same effective size must be used by both Python and KT C++ buffers.
        self._staging_buffer_max_size: int = kt_config.chunked_prefill_size or 8192
        _KT_RUNTIME_WRAPPER_REGISTRY[int(self.kt_config.layer_idx)] = self

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for both GPU and CPU experts.

        Args:
            layer: The MoE layer module
            num_experts: Total number of experts (GPU + CPU)
            hidden_size: Hidden dimension size
            intermediate_size_per_partition: Intermediate size per TP partition
            params_dtype: Data type for parameters
            **extra_weight_attrs: Additional weight attributes
        """
        self.global_num_experts = num_experts
        self._runtime_layer_ref = layer
        self._full_init_args = (
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
        )

        # Get required parameters from layer object
        # top_k: number of experts selected per token
        num_experts_per_tok = layer.top_k

        # intermediate_size_full: full intermediate size before TP partitioning
        intermediate_size_full = (
            layer.intermediate_size_per_partition * layer.moe_tp_size
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
        if (
            self.kt_config.max_deferred_experts_per_token is not None
            and self.kt_config.num_layers is not None
            and self.kt_config.layer_idx == self.kt_config.num_layers - 1
        ):
            layer_max_deferred = 0

        # 1. Create weights for GPU experts using the wrapped method
        # GPU weights are indexed by gpu_index (0 to num_gpu_experts-1), not logical expert ID
        # The mapping logical_to_gpu_index is used to remap IDs during weight loading and inference
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        # Move mask and mapping tables to GPU for inference
        target_device = next(layer.parameters()).device
        self.gpu_experts_mask_cuda = self.gpu_experts_mask.to(device=target_device)
        self.logical_to_gpu_index_cuda = self.logical_to_gpu_index.to(device=target_device)

        # Initialize dual-stream for CPU-GPU parallelism (rank 0 only)
        if self.tp_rank == 0:
            self._cpu_stream = torch.cuda.Stream(device=target_device)
            self._sync_done_event = torch.cuda.Event()
            if self._kt_runtime_gpu_prefetch_enabled:
                self._kt_runtime_gpu_prefetch_stream = torch.cuda.Stream(
                    device=target_device
                )
                self._kt_runtime_gpu_prefetch_event = torch.cuda.Event()

            # Get or create shared staging buffer (shared across all MoE layers to save GPU memory)
            self._shared_staging_buffer = get_or_create_shared_staging_buffer(
                max_tokens=self._staging_buffer_max_size,
                hidden_size=hidden_size,
                dtype=params_dtype,
                device=target_device,
            )

        # 2. Initialize KT wrapper for CPU experts
        # CPU experts are identified by gpu_experts_mask=False
        if self.tp_rank == 0:
            # SwiGLU activation params for CPU experts. Source of truth is
            # MoeRunnerConfig, populated by the model file from HF config:
            #   - minimax_m3.py forwards config.swiglu_alpha / swiglu_limit
            #     as gemm1_alpha / gemm1_clamp_limit (swiglu_oai path)
            #   - deepseek_v2.py forwards config.swiglu_limit into the
            #     legacy swiglu_limit slot (DSV4 plain-silu clamp path)
            # kt-kernel C++ accepts a single (alpha, limit) pair and
            # disambiguates by alpha != 0 (swiglu_oai vs plain silu).
            _mrc = getattr(layer, "moe_runner_config", None)
            _cfg_alpha = getattr(_mrc, "gemm1_alpha", None) if _mrc is not None else None
            _cfg_clamp = getattr(_mrc, "gemm1_clamp_limit", None) if _mrc is not None else None
            _cfg_swglim = getattr(_mrc, "swiglu_limit", None) if _mrc is not None else None
            _kt_swiglu_alpha = float(_cfg_alpha) if _cfg_alpha is not None else 0.0
            _kt_swiglu_limit = float(
                _cfg_clamp if _cfg_clamp is not None else (_cfg_swglim or 0.0)
            )
            common_wrapper_kwargs = dict(
                layer_idx=self.kt_config.layer_idx,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                gpu_experts_mask=(
                    None
                    if self._kt_cpu_native_route_mask_disabled
                    else self.gpu_experts_mask
                ),
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                numa_nodes=self.kt_config.numa_nodes,
                weight_path=self.kt_config.weight_path,
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
            )
            if self._kt_cpu_native_route_mask_disabled:
                common_wrapper_kwargs["num_gpu_experts"] = self.num_gpu_experts
            if self.kt_expert_lora_enabled:
                if _kt_swiglu_limit != 0.0:
                    raise ValueError(
                        "--kt-expert-lora-path uses KT SFT wrappers, which do not "
                        "support the V4-2604B swiglu_limit path."
                    )
                self.kt_expert_lora_weights = _load_kt_expert_lora_weights(
                    adapter_path=self.kt_expert_lora_path,
                    layer_idx=self.kt_config.layer_idx,
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    moe_intermediate_size=intermediate_size_full,
                )
                self.wrapper = KTMoEWrapper(
                    **common_wrapper_kwargs,
                    method=_map_kt_method_to_sft_method(self.kt_config.method),
                    mode="sft",
                    num_gpu_experts=0,
                    lora_rank=self.kt_expert_lora_weights.rank,
                    lora_alpha=self.kt_expert_lora_weights.alpha,
                    max_cache_depth=1,
                )
            else:
                self.wrapper = KTMoEWrapper(
                    **common_wrapper_kwargs,
                    swiglu_limit=_kt_swiglu_limit,
                    swiglu_alpha=_kt_swiglu_alpha,
                    method=self.kt_config.method,
                    max_deferred_experts_per_token=layer_max_deferred,
                )
            self._kt_cpu_native_route_mask = bool(
                getattr(self.wrapper, "supports_runtime_gpu_expert_mask", False)
            )
            if (
                self._kt_cpu_route_mask_required
                and not self._kt_cpu_native_route_mask
            ):
                raise RuntimeError(
                    "SGLANG_KT_CPU_ROUTE_MASK=1 requires the patched kt-kernel "
                    "LLAMAFILE backend with supports_runtime_gpu_expert_mask"
                )
            if (
                self.kt_config.kt_enable_dynamic_expert_update
                and not getattr(
                    self.wrapper, "supports_online_expert_weight_update", True
                )
            ):
                hf_loader = get_runtime_hf_expert_loader()
                if hf_loader is not None:
                    self._kt_runtime_hf_expert_swap_enabled = True
                    if self.tp_rank == 0:
                        logger.warning(
                            "KT dynamic expert update for layer %d uses HF "
                            "safetensors runtime swap fallback because %s "
                            "does not expose an online expert weight write API.",
                            self.kt_config.layer_idx,
                            self.kt_config.method,
                        )
                else:
                    if self.tp_rank == 0:
                        logger.warning(
                            "KT dynamic expert update is disabled for layer %d: "
                            "%s backend does not support runtime CPU/GPU expert "
                            "swap and HF safetensors fallback is unavailable.",
                            self.kt_config.layer_idx,
                            self.kt_config.method,
                        )
                    self.kt_config.kt_enable_dynamic_expert_update = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Args:
            layer: The MoE layer module
        """
        # 1. Process GPU weights
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)

        # 2. Load CPU weights using KT wrapper
        if self.tp_rank == 0 and self.wrapper is not None:
            torch.cuda.synchronize()

            # Get expert location metadata for CPU expert mapping
            from sglang.srt.eplb.expert_location_dispatch import (
                get_global_expert_location_metadata,
            )

            metadata = get_global_expert_location_metadata()
            if (
                metadata is not None
                and getattr(metadata, "physical_to_logical_map_cpu", None) is not None
            ):
                physical_to_logical_map_cpu = (
                    metadata.physical_to_logical_map_cpu[self.kt_config.layer_idx]
                    .contiguous()
                )
            else:
                # Fallback for setups without EPLB metadata: identity mapping.
                physical_to_logical_map_cpu = torch.arange(
                    layer.num_experts, dtype=torch.int64, device="cpu"
                )
            self.wrapper.load_weights(physical_to_logical_map_cpu)
            if self.kt_expert_lora_enabled:
                if self.kt_expert_lora_weights is None:
                    raise RuntimeError(
                        "KT expert LoRA is enabled but adapter weights were not loaded."
                    )
                lora = self.kt_expert_lora_weights
                if os.environ.get("SGLANG_KT_EXPERT_LORA_DEBUG") == "1":
                    print(
                        "[KT expert LoRA debug] "
                        f"layer={self.kt_config.layer_idx} "
                        f"rank={lora.rank} alpha={lora.alpha} "
                        f"gate_a={tuple(lora.gate_lora_a.shape)}/{lora.gate_lora_a.dtype}/"
                        f"{lora.gate_lora_a.device}/ptr={lora.gate_lora_a.data_ptr()} "
                        f"gate_b={tuple(lora.gate_lora_b.shape)}/{lora.gate_lora_b.dtype}/"
                        f"{lora.gate_lora_b.device}/ptr={lora.gate_lora_b.data_ptr()} "
                        f"up_a={tuple(lora.up_lora_a.shape)}/{lora.up_lora_a.dtype}/"
                        f"{lora.up_lora_a.device}/ptr={lora.up_lora_a.data_ptr()} "
                        f"up_b={tuple(lora.up_lora_b.shape)}/{lora.up_lora_b.dtype}/"
                        f"{lora.up_lora_b.device}/ptr={lora.up_lora_b.data_ptr()} "
                        f"down_a={tuple(lora.down_lora_a.shape)}/{lora.down_lora_a.dtype}/"
                        f"{lora.down_lora_a.device}/ptr={lora.down_lora_a.data_ptr()} "
                        f"down_b={tuple(lora.down_lora_b.shape)}/{lora.down_lora_b.dtype}/"
                        f"{lora.down_lora_b.device}/ptr={lora.down_lora_b.data_ptr()}",
                        flush=True,
                    )
                self.wrapper.init_lora_weights(
                    lora.gate_lora_a,
                    lora.gate_lora_b,
                    lora.up_lora_a,
                    lora.up_lora_b,
                    lora.down_lora_a,
                    lora.down_lora_b,
                    torch.zeros_like(lora.gate_lora_a),
                    torch.zeros_like(lora.gate_lora_b),
                    torch.zeros_like(lora.up_lora_a),
                    torch.zeros_like(lora.up_lora_b),
                    torch.zeros_like(lora.down_lora_a),
                    torch.zeros_like(lora.down_lora_b),
                )
                logger.info(
                    "Loaded KT expert LoRA for layer %d from %s (rank=%d, alpha=%.3f)",
                    self.kt_config.layer_idx,
                    self.kt_expert_lora_path,
                    lora.rank,
                    lora.alpha,
                )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config

        # Create a separate config for GPU method without routed_scaling_factor.
        # This is because:
        # 1. GPU method's moe_sum_reduce would apply routed_scaling_factor internally
        # 2. KT CPU kernel does NOT apply routed_scaling_factor
        # 3. The combined output (GPU + CPU) would have inconsistent scaling
        # 4. routed_scaling_factor is applied uniformly in deepseek_v2.py forward_normal
        # So we disable it in GPU method to avoid double scaling on GPU part.
        gpu_runner_config = replace(moe_runner_config, routed_scaling_factor=None)
        if self.override_num_local_experts:
            gpu_runner_config = replace(
                gpu_runner_config, num_local_experts=self.num_gpu_experts
            )

        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, gpu_runner_config)

    def _submit_cpu_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        # These tensors are consumed by non-blocking D2H copies on the layer's
        # CPU stream. Tell the CUDA allocator about that stream so their storage
        # cannot be recycled while the copies are still pending.
        consumer_stream = torch.cuda.current_stream(hidden_states.device)
        hidden_states.record_stream(consumer_stream)
        topk_ids.record_stream(consumer_stream)
        topk_weights.record_stream(consumer_stream)
        if os.environ.get("SGLANG_KT_CPU_ROUTE_DIAG", "0") == "1":
            diag_count = int(getattr(self, "_kt_cpu_route_diag_count", 0))
            if diag_count < 1:
                valid = (topk_ids >= 0) & (topk_ids < self.global_num_experts)
                valid_ids = topk_ids[valid]
                gpu_hits = 0
                if valid_ids.numel() and self.gpu_experts_mask_cuda is not None:
                    gpu_hits = int(
                        self.gpu_experts_mask_cuda[valid_ids].sum().item()
                    )
                logger.warning(
                    "KT CPU route diag: layer=%d hidden_shape=%s ids_shape=%s "
                    "ids_min=%s ids_max=%s negative=%d out_of_range=%d "
                    "valid=%d gpu_hits=%d staging_capacity=%d",
                    int(self.kt_config.layer_idx),
                    tuple(hidden_states.shape),
                    tuple(topk_ids.shape),
                    int(topk_ids.min().item()) if topk_ids.numel() else None,
                    int(topk_ids.max().item()) if topk_ids.numel() else None,
                    int((topk_ids < 0).sum().item()),
                    int((topk_ids >= self.global_num_experts).sum().item()),
                    int(valid.sum().item()),
                    gpu_hits,
                    int(self._staging_buffer_max_size),
                )
                self._kt_cpu_route_diag_count = diag_count + 1
        if (
            self._kt_runtime_hf_expert_swap_enabled
            and not self._kt_cpu_native_route_mask
            and self.num_gpu_experts > 0
            and self.gpu_experts_mask_cuda is not None
        ):
            topk_ids, topk_weights = mask_expert_ids_and_weights_for_cpu(
                topk_ids, topk_weights, self.gpu_experts_mask_cuda
            )
        if self.kt_expert_lora_enabled:
            self.wrapper.submit_forward_inference(
                hidden_states,
                topk_ids,
                topk_weights,
                torch.cuda.current_stream(hidden_states.device).cuda_stream,
            )
        else:
            self.wrapper.submit_forward(
                hidden_states,
                topk_ids,
                topk_weights,
                torch.cuda.current_stream(hidden_states.device).cuda_stream,
            )

    def _sync_cpu_forward(self, ref_tensor: torch.Tensor) -> torch.Tensor:
        if self.kt_expert_lora_enabled:
            return self.wrapper.sync_forward_inference(
                torch.cuda.current_stream(ref_tensor.device).cuda_stream,
            )
        return self.wrapper.sync_forward(
            ref_tensor,
            torch.cuda.current_stream(ref_tensor.device).cuda_stream,
        )

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task to CPU (non-blocking)
        self._submit_cpu_forward(x, topk_ids, topk_weights)

    def sync(self, x: torch.Tensor) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(x)

        # Wait for CPU computation and retrieve results
        return self._sync_cpu_forward(x)

    def _submit_with_staged_input(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
        staged_hidden_states: torch.Tensor,
    ) -> None:
        """Submit CPU expert computation using staged hidden states.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
            staged_hidden_states: Pre-copied hidden states in staging buffer
        """
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        if self.tp_rank != 0 or self.wrapper is None:
            return

        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task using staged buffer
        self._submit_cpu_forward(staged_hidden_states, topk_ids, topk_weights)

    def _sync_with_staged_input(
        self, staged_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Synchronize CPU computation using staged hidden states reference.

        Args:
            staged_hidden_states: Staged buffer used in submit

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(staged_hidden_states)

        return self._sync_cpu_forward(staged_hidden_states)

    def _select_runtime_target_experts(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Select runtime GPU experts with optional hot/tail split.

        Default behavior is unchanged (`hot_ratio=1.0`, `tail_policy=none`):
        all GPU slots are selected from the current batch's hottest experts.
        `SGLANG_KT_RUNTIME_HOT_RATIO=0.875` plus
        `SGLANG_KT_RUNTIME_TAIL_POLICY=global` keeps a stable tail from
        cumulative route frequency, matching the M7 offline finding that a
        small global tail sharply reduces slot churn.
        """
        num_experts = self.global_num_experts
        num_gpu_experts = self.num_gpu_experts
        flat_ids = topk_ids.detach().reshape(-1)
        valid_ids = flat_ids[(flat_ids >= 0) & (flat_ids < num_experts)]

        batch_counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
        if valid_ids.numel() > 0:
            batch_counts.index_add_(
                0, valid_ids, torch.ones_like(valid_ids, dtype=torch.int64)
            )

        if valid_ids.numel() > 0:
            self._kt_runtime_global_counts_cpu.index_add_(
                0,
                valid_ids.to("cpu"),
                torch.ones(valid_ids.numel(), dtype=torch.int64, device="cpu"),
            )

        rank_count = min(num_gpu_experts, num_experts)
        if rank_count > 0:
            _, ranked_indices = torch.topk(
                batch_counts,
                k=rank_count,
                largest=True,
                sorted=True,
            )
            self._kt_runtime_last_batch_ranked_cpu = [
                int(x) for x in ranked_indices.to("cpu").tolist()
            ]

        hot_count = int(round(num_gpu_experts * self._kt_runtime_hot_ratio))
        hot_count = min(num_gpu_experts, max(0, hot_count))
        if self._kt_runtime_tail_policy in {"", "none"}:
            hot_count = num_gpu_experts
        tail_count = num_gpu_experts - hot_count

        selected: list[int] = []
        if hot_count > 0:
            _, hot_indices = torch.topk(
                batch_counts,
                k=min(hot_count, num_experts),
                largest=True,
                sorted=True,
            )
            selected.extend(int(x) for x in hot_indices.to("cpu").tolist())

        if tail_count > 0:
            if self._kt_runtime_tail_policy == "global":
                tail_scores = self._kt_runtime_global_counts_cpu.clone()
                for expert_id in selected:
                    tail_scores[expert_id] = -1
                _, tail_indices = torch.topk(
                    tail_scores,
                    k=min(tail_count, num_experts),
                    largest=True,
                    sorted=True,
                )
                selected.extend(int(x) for x in tail_indices.tolist())
            elif self._kt_runtime_tail_policy == "initial":
                for expert_id in self._kt_runtime_initial_gpu_experts:
                    if expert_id not in selected:
                        selected.append(expert_id)
                    if len(selected) >= num_gpu_experts:
                        break
            elif self._kt_runtime_tail_policy == "uniform":
                step = num_experts / float(max(1, tail_count))
                layer_offset = int(self.kt_config.layer_idx or 0)
                for idx in range(tail_count):
                    expert_id = int((idx + 0.5) * step + layer_offset) % num_experts
                    probe = expert_id
                    while probe in selected and len(selected) < num_experts:
                        probe = (probe + 1) % num_experts
                    if probe not in selected:
                        selected.append(probe)
                    if len(selected) >= num_gpu_experts:
                        break

        if len(selected) < num_gpu_experts:
            existing = set(selected)
            for expert_id in self._kt_runtime_initial_gpu_experts:
                if expert_id not in existing:
                    selected.append(expert_id)
                    existing.add(expert_id)
                if len(selected) >= num_gpu_experts:
                    break
            expert_id = 0
            while len(selected) < num_gpu_experts and expert_id < num_experts:
                if expert_id not in existing:
                    selected.append(expert_id)
                    existing.add(expert_id)
                expert_id += 1

        return torch.tensor(
            sorted(selected[:num_gpu_experts]), dtype=torch.int64, device="cpu"
        )

    def _ordered_runtime_experts(
        self,
        expert_ids: set[int],
        ranked: Optional[List[int]] = None,
    ) -> List[int]:
        if not expert_ids:
            return []
        ordered: List[int] = []
        seen = set()
        for expert_id in ranked or []:
            expert_id = int(expert_id)
            if expert_id in expert_ids and expert_id not in seen:
                ordered.append(expert_id)
                seen.add(expert_id)
        for expert_id in sorted(expert_ids):
            if expert_id not in seen:
                ordered.append(expert_id)
                seen.add(expert_id)
        return ordered

    def _build_runtime_swap_plan(
        self,
        target_set: set[int],
        ranked: Optional[List[int]] = None,
        scores: Optional[List[int]] = None,
    ) -> Optional[Tuple[List[int], List[int], List[int]]]:
        current_by_slot = [
            int(x) for x in self.gpu_index_to_logical.to("cpu").tolist()
        ]
        current_set = set(current_by_slot)
        if target_set == current_set:
            return None

        new_by_slot = list(current_by_slot)
        new_experts = self._ordered_runtime_experts(target_set - current_set, ranked)
        rank_map = {
            int(expert_id): int(rank)
            for rank, expert_id in enumerate(ranked or [])
        }
        fallback_rank = len(rank_map) + self.global_num_experts
        evict_slots = [
            idx
            for idx, expert_id in enumerate(current_by_slot)
            if expert_id not in target_set
        ]
        evict_slots.sort(
            key=lambda idx: (
                rank_map.get(int(current_by_slot[idx]), fallback_rank),
                idx,
            ),
            reverse=True,
        )
        if (
            self._kt_runtime_max_changed_slots > 0
            and len(new_experts) > self._kt_runtime_max_changed_slots
        ):
            new_experts = new_experts[: self._kt_runtime_max_changed_slots]
            evict_slots = evict_slots[: self._kt_runtime_max_changed_slots]
        skipped_low_gain = 0
        gains: List[int] = []
        if (
            scores is not None
            and len(scores) >= self.global_num_experts
            and self._kt_runtime_min_gain_entries_per_slot > 0
        ):
            kept_pairs: List[Tuple[int, int]] = []
            for slot, expert_id in zip(evict_slots, new_experts):
                old_expert_id = int(current_by_slot[slot])
                gain = int(scores[expert_id]) - int(scores[old_expert_id])
                if gain >= self._kt_runtime_min_gain_entries_per_slot:
                    kept_pairs.append((slot, expert_id))
                    gains.append(gain)
                else:
                    skipped_low_gain += 1
            evict_slots = [slot for slot, _expert_id in kept_pairs]
            new_experts = [expert_id for _slot, expert_id in kept_pairs]
            self._kt_runtime_low_gain_slots_skipped += skipped_low_gain
        if len(new_experts) != len(evict_slots):
            raise RuntimeError(
                "internal error while building KT runtime expert swap plan: "
                f"{len(new_experts)=}, {len(evict_slots)=}"
            )
        for slot, expert_id in zip(evict_slots, new_experts):
            new_by_slot[slot] = expert_id
        if (
            self.tp_rank == 0
            and os.environ.get("SGLANG_KT_RUNTIME_SWAP_PROFILE") == "1"
        ):
            logger.info(
                "KT runtime swap plan profile: layer=%d target=%d current=%d "
                "changed=%d max_changed=%d new_top=%s evict_slots=%s "
                "evict_old=%s min_gain=%.1f gains=%s low_gain_skipped=%d "
                "low_gain_skipped_total=%d",
                int(self.kt_config.layer_idx),
                len(target_set),
                len(current_set),
                len(evict_slots),
                int(self._kt_runtime_max_changed_slots),
                new_experts[: min(8, len(new_experts))],
                evict_slots[: min(8, len(evict_slots))],
                [current_by_slot[idx] for idx in evict_slots[: min(8, len(evict_slots))]],
                float(self._kt_runtime_min_gain_entries_per_slot),
                gains[: min(8, len(gains))],
                skipped_low_gain,
                self._kt_runtime_low_gain_slots_skipped,
            )
        if not evict_slots:
            return None
        return new_by_slot, evict_slots, new_experts

    def _apply_runtime_gpu_mapping(
        self, new_by_slot: List[int], device: torch.device
    ) -> None:
        selected_by_slot = torch.tensor(
            new_by_slot, dtype=torch.int64, device=device
        )
        gpu_experts_mask_cpu, logical_to_gpu_index_cuda, gpu_index_to_logical_cpu = (
            update_gpu_expert_mappings(
                selected_experts=selected_by_slot,
                num_experts=self.global_num_experts,
                device=device,
            )
        )
        self.gpu_experts_mask = gpu_experts_mask_cpu
        self.gpu_experts_mask_cuda.copy_(gpu_experts_mask_cpu)
        self.logical_to_gpu_index = logical_to_gpu_index_cuda.cpu()
        self.logical_to_gpu_index_cuda.copy_(logical_to_gpu_index_cuda)
        self.gpu_index_to_logical = gpu_index_to_logical_cpu
        if self.tp_rank == 0:
            update_kt_wrapper_masks(self.wrapper, gpu_experts_mask_cpu)

    def _apply_runtime_gpu_mapping_delta(
        self,
        new_by_slot: List[int],
        evict_slots: List[int],
        device: torch.device,
    ) -> None:
        if not evict_slots:
            return
        if self._kt_runtime_mapping_full_copy:
            selected_cpu = torch.tensor(
                new_by_slot, dtype=torch.long, device="cpu"
            )
            gpu_experts_mask_cpu = torch.zeros(
                self.global_num_experts, dtype=torch.bool, device="cpu"
            )
            gpu_experts_mask_cpu[selected_cpu] = True
            logical_to_gpu_index_cpu = torch.full(
                (self.global_num_experts,), -1, dtype=torch.int32, device="cpu"
            )
            logical_to_gpu_index_cpu[selected_cpu] = torch.arange(
                len(new_by_slot), dtype=torch.int32, device="cpu"
            )
            gpu_index_to_logical_cpu = selected_cpu.to(torch.int32)

            # Keep the existing CPU tensor objects alive: KT's native wrapper
            # may hold their pointers. CUDA mappings are tiny and contiguous,
            # so two copy_ calls are cheaper than four indexed update kernels.
            self.gpu_experts_mask.copy_(gpu_experts_mask_cpu)
            self.logical_to_gpu_index.copy_(logical_to_gpu_index_cpu)
            self.gpu_index_to_logical.copy_(gpu_index_to_logical_cpu)
            self.gpu_experts_mask_cuda.copy_(gpu_experts_mask_cpu)
            self.logical_to_gpu_index_cuda.copy_(logical_to_gpu_index_cpu)
            if self.tp_rank == 0:
                update_kt_wrapper_masks(self.wrapper, self.gpu_experts_mask)
            return

        slot_indices_cpu = torch.tensor(evict_slots, dtype=torch.long, device="cpu")
        old_experts_cpu = self.gpu_index_to_logical[slot_indices_cpu].to(torch.long)
        new_experts_cpu = torch.tensor(
            [int(new_by_slot[int(slot)]) for slot in evict_slots],
            dtype=torch.long,
            device="cpu",
        )
        slot_values_cpu = slot_indices_cpu.to(torch.int32)

        mask_device = self.gpu_experts_mask.device
        logical_device = self.logical_to_gpu_index.device
        reverse_device = self.gpu_index_to_logical.device

        self.gpu_experts_mask[old_experts_cpu.to(mask_device)] = False
        self.gpu_experts_mask[new_experts_cpu.to(mask_device)] = True
        self.logical_to_gpu_index[old_experts_cpu.to(logical_device)] = -1
        self.logical_to_gpu_index[new_experts_cpu.to(logical_device)] = (
            slot_values_cpu.to(logical_device)
        )
        self.gpu_index_to_logical[slot_indices_cpu.to(reverse_device)] = (
            new_experts_cpu.to(dtype=torch.int32, device=reverse_device)
        )

        old_experts_cuda = old_experts_cpu.to(device=device, non_blocking=True)
        new_experts_cuda = new_experts_cpu.to(device=device, non_blocking=True)
        slot_values_cuda = slot_values_cpu.to(device=device, non_blocking=True)
        self.gpu_experts_mask_cuda[old_experts_cuda] = False
        self.gpu_experts_mask_cuda[new_experts_cuda] = True
        self.logical_to_gpu_index_cuda[old_experts_cuda] = -1
        self.logical_to_gpu_index_cuda[new_experts_cuda] = slot_values_cuda
        if self.tp_rank == 0:
            wrapper_mask = (
                self.gpu_experts_mask
                if self.gpu_experts_mask.device.type == "cpu"
                else self.gpu_experts_mask.cpu()
            )
            update_kt_wrapper_masks(self.wrapper, wrapper_mask)

    def _commit_runtime_hf_gpu_prefetch_if_present(
        self, device: torch.device
    ) -> bool:
        if not self._kt_runtime_gpu_prefetch_enabled:
            return False
        with self._kt_runtime_gpu_prefetch_lock:
            pending = self._kt_runtime_gpu_prefetch_pending
        if pending is None:
            self._kt_runtime_gpu_prefetch_no_pending += 1
            return False

        t0 = time.perf_counter()
        result = pending.get("result", {})
        future = pending.get("future")
        event = pending.get("event")
        if future is not None:
            live_slot_write = bool(pending.get("live_slot_write"))
            must_wait_for_live_slot = (
                live_slot_write
                and self._kt_runtime_gpu_prefetch_direct_block_on_miss
            )
            if (
                self._kt_runtime_gpu_prefetch_ready_only
                and not future.done()
                and not must_wait_for_live_slot
            ):
                self._kt_runtime_gpu_prefetch_not_ready += 1
                if self._kt_runtime_gpu_prefetch_discard_missed:
                    with self._kt_runtime_gpu_prefetch_lock:
                        if self._kt_runtime_gpu_prefetch_pending is pending:
                            pending["missed"] = True
                self._kt_runtime_gpu_prefetch_skipped += 1
                if (
                    self.tp_rank == 0
                    and os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH_VERBOSE") == "1"
                ):
                    logger.info(
                        "KT runtime GPU expert prefetch not ready; keep old "
                        "placement: layer=%d source_layer=%s changed_slots=%s "
                        "missed=%s",
                        self.kt_config.layer_idx,
                        pending.get("source_layer"),
                        pending.get("changed_slots"),
                        pending.get("missed"),
                    )
                return False
            try:
                result = future.result()
            except Exception as exc:
                with self._kt_runtime_gpu_prefetch_lock:
                    if self._kt_runtime_gpu_prefetch_pending is pending:
                        self._kt_runtime_gpu_prefetch_pending = None
                self._kt_runtime_gpu_prefetch_skipped += 1
                if self.tp_rank == 0:
                    logger.warning(
                        "KT runtime GPU expert prefetch failed before commit: "
                        "layer=%d source_layer=%s changed_slots=%s error=%s",
                        self.kt_config.layer_idx,
                        pending.get("source_layer"),
                        pending.get("changed_slots"),
                        exc,
                    )
                return False
        elif event is not None:
            if self._kt_runtime_gpu_prefetch_ready_only and not event.query():
                self._kt_runtime_gpu_prefetch_not_ready += 1
                if self._kt_runtime_gpu_prefetch_discard_missed:
                    with self._kt_runtime_gpu_prefetch_lock:
                        if self._kt_runtime_gpu_prefetch_pending is pending:
                            pending["missed"] = True
                self._kt_runtime_gpu_prefetch_skipped += 1
                if (
                    self.tp_rank == 0
                    and os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH_VERBOSE") == "1"
                ):
                    logger.info(
                        "KT runtime GPU expert prefetch event not ready; keep old "
                        "placement: layer=%d source_layer=%s changed_slots=%s "
                        "missed=%s",
                        self.kt_config.layer_idx,
                        pending.get("source_layer"),
                        pending.get("changed_slots"),
                        pending.get("missed"),
                    )
                return False
            torch.cuda.current_stream(device).wait_event(event)
        else:
            with self._kt_runtime_gpu_prefetch_lock:
                if self._kt_runtime_gpu_prefetch_pending is pending:
                    self._kt_runtime_gpu_prefetch_pending = None
            return False

        if pending.get("missed"):
            with self._kt_runtime_gpu_prefetch_lock:
                if self._kt_runtime_gpu_prefetch_pending is pending:
                    self._kt_runtime_gpu_prefetch_pending = None
            self._kt_runtime_gpu_prefetch_skipped += 1
            self._kt_runtime_gpu_prefetch_missed_drop += 1
            if (
                self.tp_rank == 0
                and os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH_VERBOSE") == "1"
            ):
                logger.info(
                    "KT runtime GPU expert prefetch dropped after missed boundary: "
                    "layer=%d source_layer=%s changed_slots=%s",
                    self.kt_config.layer_idx,
                    pending.get("source_layer"),
                    pending.get("changed_slots"),
                )
            return False

        profile_enabled = os.environ.get("SGLANG_KT_RUNTIME_SWAP_PROFILE") == "1"
        prefetch_profile_enabled = (
            os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_PROFILE") == "1"
        )
        staged_slots = result.get("staged_slots")
        d2d_copy_ms = 0.0
        if staged_slots:
            layer_ref = self._runtime_layer_ref
            if not (
                layer_ref is not None
                and hasattr(layer_ref, "w13_weight")
                and hasattr(layer_ref, "w2_weight")
            ):
                with self._kt_runtime_gpu_prefetch_lock:
                    if self._kt_runtime_gpu_prefetch_pending is pending:
                        self._kt_runtime_gpu_prefetch_pending = None
                self._kt_runtime_gpu_prefetch_skipped += 1
                if self.tp_rank == 0:
                    logger.warning(
                        "KT runtime GPU expert prefetch cannot commit staged "
                        "weights: layer=%d missing live BF16 weight tensors",
                        self.kt_config.layer_idx,
                )
                return False
            try:
                t_copy = time.perf_counter()
                for staged in staged_slots:
                    slot = int(staged["slot"])
                    layer_ref.w13_weight.data[slot].copy_(
                        staged["w13"], non_blocking=True
                    )
                    layer_ref.w2_weight.data[slot].copy_(
                        staged["w2"], non_blocking=True
                    )
                if profile_enabled and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                d2d_copy_ms = (time.perf_counter() - t_copy) * 1000.0
            except Exception as exc:
                with self._kt_runtime_gpu_prefetch_lock:
                    if self._kt_runtime_gpu_prefetch_pending is pending:
                        self._kt_runtime_gpu_prefetch_pending = None
                self._kt_runtime_gpu_prefetch_skipped += 1
                if self.tp_rank == 0:
                    logger.warning(
                        "KT runtime GPU expert prefetch staged commit failed: "
                        "layer=%d source_layer=%s changed_slots=%s error=%s",
                        self.kt_config.layer_idx,
                        pending.get("source_layer"),
                        pending.get("changed_slots"),
                        exc,
                    )
                return False

        with self._kt_runtime_gpu_prefetch_lock:
            if self._kt_runtime_gpu_prefetch_pending is not pending:
                return False
            self._kt_runtime_gpu_prefetch_pending = None
        t_mapping = time.perf_counter()
        evict_slots = pending.get("evict_slots")
        if isinstance(evict_slots, list) and evict_slots:
            self._apply_runtime_gpu_mapping_delta(
                pending["new_by_slot"], [int(x) for x in evict_slots], device
            )
        else:
            self._apply_runtime_gpu_mapping(pending["new_by_slot"], device)
        if profile_enabled and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        mapping_ms = (time.perf_counter() - t_mapping) * 1000.0
        self._kt_runtime_gpu_prefetch_committed += 1
        commit_log_enabled = (
            os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH_COMMIT_LOG", "1")
            == "1"
        )
        if self.tp_rank == 0 and (
            commit_log_enabled or profile_enabled or prefetch_profile_enabled
        ):
            wait_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "KT runtime GPU expert prefetch committed: layer=%d "
                "source_layer=%s changed_slots=%d wait_ms=%.2f "
                "prefetch_elapsed_ms=%.2f prefetch_sources=%s commit_count=%d",
                self.kt_config.layer_idx,
                pending.get("source_layer"),
                int(pending.get("changed_slots") or 0),
                wait_ms,
                float(result.get("elapsed_ms", 0.0) or 0.0),
                result.get("sources", {}),
                self._kt_runtime_gpu_prefetch_committed,
            )
            if profile_enabled:
                logger.info(
                    "KT runtime GPU expert prefetch commit profile: layer=%d "
                    "source_layer=%s changed_slots=%d total_ms=%.2f "
                    "d2d_copy_ms=%.2f mapping_ms=%.2f staging=%s "
                    "sources=%s",
                    self.kt_config.layer_idx,
                    pending.get("source_layer"),
                    int(pending.get("changed_slots") or 0),
                    wait_ms,
                    d2d_copy_ms,
                    mapping_ms,
                    bool(result.get("staging", False)),
                    result.get("sources", {}),
                )
            if prefetch_profile_enabled:
                profile = result.get("profile", {})
                logger.info(
                    "KT runtime GPU prefetch state: layer=%d committed=%d "
                    "no_pending=%d not_ready=%d missed_drop=%d pending_busy=%d",
                    self.kt_config.layer_idx,
                    self._kt_runtime_gpu_prefetch_committed,
                    self._kt_runtime_gpu_prefetch_no_pending,
                    self._kt_runtime_gpu_prefetch_not_ready,
                    self._kt_runtime_gpu_prefetch_missed_drop,
                    self._kt_runtime_gpu_prefetch_pending_busy,
                )
                if isinstance(profile, dict):
                    logger.info(
                        "KT runtime GPU prefetch stage profile: layer=%d "
                        "source_layer=%s pending_id=%s elapsed_ms=%.2f "
                        "queue_delay_ms=%.2f "
                        "acquire_ms=%.2f allocate_ms=%.2f h2d_enqueue_ms=%.2f "
                        "stream_sync_ms=%.2f staging_bytes=%d",
                        self.kt_config.layer_idx,
                        pending.get("source_layer"),
                        pending.get("pending_id"),
                        float(result.get("elapsed_ms", 0.0) or 0.0),
                        float(profile.get("queue_delay_ms", 0.0) or 0.0),
                        float(profile.get("acquire_ms", 0.0) or 0.0),
                        float(profile.get("allocate_ms", 0.0) or 0.0),
                        float(profile.get("h2d_enqueue_ms", 0.0) or 0.0),
                        float(profile.get("stream_sync_ms", 0.0) or 0.0),
                        int(profile.get("staging_bytes", 0) or 0),
                    )
        return True

    def _set_runtime_hf_gpu_prefetch_pending(
        self,
        *,
        future: Optional[Future] = None,
        event: Optional[torch.cuda.Event] = None,
        result: Optional[Dict[str, Any]] = None,
        new_by_slot: List[int],
        evict_slots: Optional[List[int]] = None,
        changed_slots: int,
        source_layer: int,
        live_slot_write: bool = False,
    ) -> bool:
        with self._kt_runtime_gpu_prefetch_lock:
            existing = self._kt_runtime_gpu_prefetch_pending
            if existing is not None and existing.get("missed"):
                existing_future = existing.get("future")
                existing_event = existing.get("event")
                existing_done = False
                if existing_future is not None:
                    existing_done = bool(existing_future.done())
                elif existing_event is not None:
                    try:
                        existing_done = bool(existing_event.query())
                    except Exception:
                        existing_done = False
                else:
                    existing_done = True
                if existing_done:
                    self._kt_runtime_gpu_prefetch_pending = None
                    existing = None
            if existing is not None:
                self._kt_runtime_gpu_prefetch_skipped += 1
                self._kt_runtime_gpu_prefetch_pending_busy += 1
                return False
            self._kt_runtime_gpu_prefetch_pending_counter += 1
            self._kt_runtime_gpu_prefetch_pending = {
                "future": future,
                "event": event,
                "result": result or {},
                "new_by_slot": list(new_by_slot),
                "evict_slots": list(evict_slots or []),
                "changed_slots": int(changed_slots),
                "source_layer": int(source_layer),
                "live_slot_write": bool(live_slot_write),
                "pending_id": int(self._kt_runtime_gpu_prefetch_pending_counter),
                "scheduled_at": time.perf_counter(),
            }
            return True

    def _schedule_runtime_hf_gpu_prefetch(
        self,
        loader: RuntimeHFExpertWeightLoader,
        *,
        target_layer: int,
        predicted_selected: torch.Tensor,
        predicted_ranked: Optional[List[int]] = None,
        predicted_scores: Optional[List[int]] = None,
        device: torch.device,
    ) -> None:
        if not (
            self._kt_runtime_gpu_prefetch_enabled
            and loader.gpu_prefetch_enabled
            and get_tensor_model_parallel_world_size() == 1
        ):
            return
        target_method = _KT_RUNTIME_WRAPPER_REGISTRY.get(int(target_layer))
        if target_method is None:
            return
        if not getattr(target_method, "_kt_runtime_hf_expert_swap_enabled", False):
            return
        target_layer_ref = getattr(target_method, "_runtime_layer_ref", None)
        if target_layer_ref is None:
            return

        target_set = {int(x) for x in predicted_selected.to("cpu").tolist()}
        plan = target_method._build_runtime_swap_plan(
            target_set,
            ranked=(
                list(predicted_ranked)
                if predicted_ranked is not None
                else [int(x) for x in predicted_selected.to("cpu").tolist()]
            ),
            scores=predicted_scores,
        )
        if plan is None:
            return
        new_by_slot, evict_slots, _new_experts = plan
        slot_to_expert = [(slot, new_by_slot[slot]) for slot in evict_slots]
        stream = getattr(target_method, "_kt_runtime_gpu_prefetch_stream", None)
        event = getattr(target_method, "_kt_runtime_gpu_prefetch_event", None)
        if stream is None or event is None:
            return
        with target_method._kt_runtime_gpu_prefetch_lock:
            if target_method._kt_runtime_gpu_prefetch_pending is not None:
                target_method._kt_runtime_gpu_prefetch_skipped += 1
                return
        if getattr(target_method, "_kt_runtime_gpu_prefetch_background", False):
            direct_live_slots = not getattr(
                target_method, "_kt_runtime_gpu_prefetch_staging", True
            )
            if direct_live_slots:
                future = loader.submit_streamed_gpu_prefetch(
                    layer=target_layer_ref,
                    layer_idx=int(target_layer),
                    slot_to_expert=slot_to_expert,
                    device=device,
                )
            else:
                future = loader.submit_stage_gpu_prefetch(
                    layer=target_layer_ref,
                    layer_idx=int(target_layer),
                    slot_to_expert=slot_to_expert,
                    device=device,
                )
            if future is None:
                target_method._kt_runtime_gpu_prefetch_skipped += 1
                return
            accepted = target_method._set_runtime_hf_gpu_prefetch_pending(
                future=future,
                new_by_slot=new_by_slot,
                evict_slots=evict_slots,
                changed_slots=len(slot_to_expert),
                source_layer=int(self.kt_config.layer_idx),
                live_slot_write=direct_live_slots,
            )
            if not accepted:
                return
            if (
                self.tp_rank == 0
                and os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH_VERBOSE") == "1"
            ):
                logger.info(
                    "KT runtime GPU expert prefetch submitted background: "
                    "layer=%d target_layer=%d changed_slots=%d submitted_total=%d",
                    self.kt_config.layer_idx,
                    target_layer,
                    len(slot_to_expert),
                    loader.gpu_prefetch_submitted,
                )
            return

        sources: Dict[str, int] = {}
        prefetch_wait_ms = 0.0
        keepalive: List[torch.Tensor] = []
        staged_slots: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        try:
            for slot, expert_id in slot_to_expert:
                if getattr(target_method, "_kt_runtime_gpu_prefetch_staging", True):
                    info = loader.stage_expert_to_gpu_slot_buffers(
                        layer=target_layer_ref,
                        layer_idx=int(target_layer),
                        logical_expert_id=expert_id,
                        gpu_slot=slot,
                        stream=stream,
                    )
                    staged_slot = info.get("staged_slot")
                    if isinstance(staged_slot, dict):
                        staged_slots.append(staged_slot)
                else:
                    info = loader.load_expert_to_gpu_slot_streamed(
                        layer=target_layer_ref,
                        layer_idx=int(target_layer),
                        logical_expert_id=expert_id,
                        gpu_slot=slot,
                        stream=stream,
                    )
                source = str(info.get("source", "unknown"))
                sources[source] = sources.get(source, 0) + 1
                prefetch_wait_ms += float(info.get("prefetch_wait_ms", 0.0) or 0.0)
                tensors = info.get("keepalive")
                if isinstance(tensors, list):
                    keepalive.extend(t for t in tensors if isinstance(t, torch.Tensor))
        except Exception as exc:
            target_method._kt_runtime_gpu_prefetch_skipped += 1
            if self.tp_rank == 0:
                logger.warning(
                    "KT runtime GPU expert prefetch schedule failed: layer=%d "
                    "target_layer=%d changed_slots=%d error=%s",
                    self.kt_config.layer_idx,
                    target_layer,
                    len(slot_to_expert),
                    exc,
                )
            return
        event.record(stream)
        result = {
            "sources": sources,
            "prefetch_wait_ms": prefetch_wait_ms,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            "changed_slots": len(slot_to_expert),
            "staged_slots": staged_slots,
            "staging": bool(
                getattr(target_method, "_kt_runtime_gpu_prefetch_staging", True)
            ),
            "keepalive": keepalive,
        }
        loader.gpu_prefetch_submitted += 1
        accepted = target_method._set_runtime_hf_gpu_prefetch_pending(
            event=event,
            result=result,
            new_by_slot=new_by_slot,
            evict_slots=evict_slots,
            changed_slots=len(slot_to_expert),
            source_layer=int(self.kt_config.layer_idx),
        )
        if not accepted:
            return
        if (
            self.tp_rank == 0
            and os.environ.get("SGLANG_KT_RUNTIME_GPU_PREFETCH_VERBOSE") == "1"
        ):
            logger.info(
                "KT runtime GPU expert prefetch scheduled: layer=%d "
                "target_layer=%d changed_slots=%d submitted_total=%d",
                self.kt_config.layer_idx,
                target_layer,
                len(slot_to_expert),
                loader.gpu_prefetch_submitted,
            )

    def _runtime_predicted_selected_for_layer(
        self,
        *,
        target_layer: int,
        fallback_selected: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int], Optional[List[int]]]:
        predicted_selected = fallback_selected
        ranked = list(self._kt_runtime_last_batch_ranked_cpu)
        scores: Optional[List[int]] = None
        oracle = get_runtime_oracle_prefetch_provider()
        if oracle is not None:
            oracle_prediction = oracle.ranked_experts_with_counts_for_active_step(
                target_layer=target_layer,
                limit=max(self.num_gpu_experts, self._kt_runtime_prefetch_top_k),
            )
            oracle_ranked = oracle_prediction[0] if oracle_prediction else None
            if oracle_ranked:
                ranked = list(oracle_ranked)
                scores = list(oracle_prediction[1])
                selected_cpu = list(oracle_ranked[: self.num_gpu_experts])
                if len(selected_cpu) < self.num_gpu_experts:
                    existing = set(selected_cpu)
                    for expert_id in self._kt_runtime_initial_gpu_experts:
                        if expert_id not in existing:
                            selected_cpu.append(expert_id)
                            existing.add(expert_id)
                        if len(selected_cpu) >= self.num_gpu_experts:
                            break
                predicted_selected = torch.tensor(
                    sorted(selected_cpu[: self.num_gpu_experts]),
                    dtype=torch.int64,
                    device="cpu",
                )
        if not ranked:
            ranked = [int(x) for x in predicted_selected.to("cpu").tolist()]
        return predicted_selected, ranked, scores

    def _runtime_ordered_prefetch_candidates(
        self,
        *,
        target_layer: int,
        predicted_selected: torch.Tensor,
        ranked: List[int],
    ) -> List[int]:
        selected_set = {int(x) for x in predicted_selected.to("cpu").tolist()}
        if self._kt_runtime_prefetch_delta_only:
            target_method = _KT_RUNTIME_WRAPPER_REGISTRY.get(int(target_layer))
            if target_method is not None:
                current_set = {
                    int(x)
                    for x in target_method.gpu_index_to_logical.to("cpu").tolist()
                }
                selected_set = selected_set - current_set

        ordered: List[int] = []
        seen = set()
        for expert_id in ranked:
            expert_id = int(expert_id)
            if expert_id in seen or expert_id not in selected_set:
                continue
            ordered.append(expert_id)
            seen.add(expert_id)
            if len(ordered) >= self._kt_runtime_prefetch_top_k:
                break
        return ordered

    def _schedule_runtime_hf_prefetch(
        self, loader: RuntimeHFExpertWeightLoader, selected: torch.Tensor
    ) -> None:
        """Asynchronously stage predicted future-layer HF expert tensors on CPU.

        This is intentionally conservative: it does not write GPU expert slots
        from a background thread. It overlaps safetensors CPU reads with the
        current layer's CPU/GPU MoE work; the next layer's foreground swap then
        consumes staged tensors and performs the required GPU slot writes.
        """
        if not loader.prefetch_enabled or not self._kt_runtime_prefetch_next_layer:
            return
        if self._kt_runtime_prefetch_top_k <= 0:
            return
        if self.kt_config.num_layers is None:
            return
        current_layer = int(self.kt_config.layer_idx)
        stage_size = int(self._kt_runtime_prefetch_stage_size)
        cpu_target_layer = (
            int(self.kt_config.layer_idx) + self._kt_runtime_prefetch_lookahead_layers
        )
        gpu_target_layer = (
            current_layer + self._kt_runtime_gpu_prefetch_lookahead_layers
        )
        submitted = 0
        candidates = 0
        scheduled_target_layers: List[int] = []

        should_stage_cpu = (
            self._kt_runtime_cpu_prefetch_enabled
            and cpu_target_layer < int(self.kt_config.num_layers)
        )
        if (
            should_stage_cpu
            and self._kt_runtime_prefetch_stage_boundary_only
            and current_layer % stage_size != 0
        ):
            should_stage_cpu = False
        if should_stage_cpu:
            cpu_stop_layer = min(
                int(self.kt_config.num_layers),
                cpu_target_layer + self._kt_runtime_cpu_prefetch_stage_span,
            )
            for stage_cpu_target_layer in range(
                cpu_target_layer, cpu_stop_layer
            ):
                cpu_predicted_selected, cpu_ranked, cpu_scores = (
                    self._runtime_predicted_selected_for_layer(
                        target_layer=stage_cpu_target_layer,
                        fallback_selected=selected,
                    )
                )
                ordered: List[int] = []
                target_method = _KT_RUNTIME_WRAPPER_REGISTRY.get(
                    int(stage_cpu_target_layer)
                )
                if target_method is not None:
                    cpu_target_set = {
                        int(x)
                        for x in cpu_predicted_selected.to("cpu").tolist()
                    }
                    cpu_plan = target_method._build_runtime_swap_plan(
                        cpu_target_set,
                        ranked=cpu_ranked,
                        scores=cpu_scores,
                    )
                    if cpu_plan is not None:
                        ordered = [
                            int(x)
                            for x in cpu_plan[2][
                                : self._kt_runtime_prefetch_top_k
                            ]
                        ]
                else:
                    ordered = self._runtime_ordered_prefetch_candidates(
                        target_layer=stage_cpu_target_layer,
                        predicted_selected=cpu_predicted_selected,
                        ranked=cpu_ranked,
                    )
                submitted += loader.prefetch_experts(
                    stage_cpu_target_layer, ordered
                )
                candidates += len(ordered)
                scheduled_target_layers.append(stage_cpu_target_layer)

        should_stage_gpu = self._kt_runtime_gpu_prefetch_enabled
        gpu_target_layers: List[int] = []
        if should_stage_gpu:
            stage_span = int(self._kt_runtime_gpu_prefetch_stage_span)
            stage_stride = int(self._kt_runtime_gpu_prefetch_stage_stride)
            if stage_span > 1:
                if current_layer % stage_size == 0:
                    stop_layer = min(
                        int(self.kt_config.num_layers),
                        gpu_target_layer + stage_span,
                    )
                    gpu_target_layers = list(
                        range(gpu_target_layer, stop_layer, stage_stride)
                    )
            elif gpu_target_layer < int(self.kt_config.num_layers):
                gpu_target_layers = [gpu_target_layer]

        if (
            gpu_target_layers
            and self._kt_runtime_gpu_prefetch_target_boundary_only
            and self._kt_runtime_gpu_prefetch_stage_span <= 1
        ):
            gpu_target_layers = [
                layer for layer in gpu_target_layers if layer % stage_size == 0
            ]

        for stage_target_layer in gpu_target_layers:
            gpu_predicted_selected, gpu_ranked, gpu_scores = (
                self._runtime_predicted_selected_for_layer(
                    target_layer=stage_target_layer,
                    fallback_selected=selected,
                )
            )
            self._schedule_runtime_hf_gpu_prefetch(
                loader,
                target_layer=stage_target_layer,
                predicted_selected=gpu_predicted_selected,
                predicted_ranked=gpu_ranked,
                predicted_scores=gpu_scores,
                device=self.gpu_experts_mask_cuda.device,
            )
            if stage_target_layer not in scheduled_target_layers:
                scheduled_target_layers.append(stage_target_layer)

        oracle = get_runtime_oracle_prefetch_provider()
        should_stage_next_step_layer0 = (
            self._kt_runtime_prefetch_next_step_layer0
            and self._kt_runtime_gpu_prefetch_enabled
            and oracle is not None
            and self.kt_config.num_layers is not None
            and current_layer % stage_size == 0
            and current_layer + stage_size >= int(self.kt_config.num_layers)
        )
        if should_stage_next_step_layer0:
            next_ranked = oracle.ranked_experts_for_next_step(
                target_layer=0,
                limit=max(self.num_gpu_experts, self._kt_runtime_prefetch_top_k),
            )
            if next_ranked:
                selected_cpu = list(next_ranked[: self.num_gpu_experts])
                existing = set(selected_cpu)
                if len(selected_cpu) < self.num_gpu_experts:
                    for expert_id in self._kt_runtime_initial_gpu_experts:
                        if expert_id not in existing:
                            selected_cpu.append(expert_id)
                            existing.add(expert_id)
                        if len(selected_cpu) >= self.num_gpu_experts:
                            break
                next_predicted_selected = torch.tensor(
                    sorted(selected_cpu[: self.num_gpu_experts]),
                    dtype=torch.int64,
                    device="cpu",
                )
                self._schedule_runtime_hf_gpu_prefetch(
                    loader,
                    target_layer=0,
                    predicted_selected=next_predicted_selected,
                    predicted_ranked=next_ranked,
                    predicted_scores=None,
                    device=self.gpu_experts_mask_cuda.device,
                )
                if 0 not in scheduled_target_layers:
                    scheduled_target_layers.append(0)

        self._kt_runtime_prefetch_schedule_count += 1
        if (
            self.tp_rank == 0
            and os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_VERBOSE") == "1"
            and (
                self._kt_runtime_prefetch_schedule_count <= 16
                or self._kt_runtime_prefetch_schedule_count
                % self._kt_runtime_prefetch_log_every
                == 0
            )
        ):
            logger.info(
                "KT runtime HF async prefetch scheduled: layer=%d target_layers=%s "
                "submitted=%d candidates=%d loader_submitted=%d cache_hits=%d "
                "future_hits=%d direct_loads=%d wait_ms_total=%.2f "
                "cpu_cache_hits=%d cpu_cache_misses=%d",
                self.kt_config.layer_idx,
                scheduled_target_layers,
                submitted,
                candidates,
                loader.prefetch_submitted,
                loader.prefetch_cache_hits,
                loader.prefetch_future_hits,
                loader.prefetch_direct_loads,
                loader.prefetch_wait_ms_total,
                getattr(loader, "cpu_tensor_cache_hits", 0),
                getattr(loader, "cpu_tensor_cache_misses", 0),
            )

    def _update_gpu_experts_from_batch_hf_swap(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        if self._kt_runtime_defer_prefetch_after_cpu_submit:
            self._kt_runtime_deferred_prefetch = None
        if not self._kt_runtime_hf_expert_swap_enabled:
            return
        if self.num_gpu_experts <= 0:
            return
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError(
                "HF safetensors runtime expert swap currently supports TP=1 only"
            )

        topk_ids = dispatch_output.topk_output.topk_ids
        device = topk_ids.device
        current_layer = int(self.kt_config.layer_idx)
        stage_size = max(1, int(self._kt_runtime_prefetch_stage_size))
        num_layers = (
            int(self.kt_config.num_layers)
            if self.kt_config.num_layers is not None
            else None
        )
        oracle = get_runtime_oracle_prefetch_provider()
        if oracle is not None:
            oracle.begin_step_if_layer0(
                current_layer,
                int(topk_ids.shape[0]) if topk_ids.dim() > 0 else 0,
            )

        loader = get_runtime_hf_expert_loader()
        if loader is None:
            raise RuntimeError("KT runtime HF expert loader became unavailable")
        committed_gpu_prefetch = self._commit_runtime_hf_gpu_prefetch_if_present(device)
        pending_after_commit = self._kt_runtime_gpu_prefetch_pending

        update_boundary = (
            not self._kt_runtime_update_stage_boundary_only
            or current_layer % stage_size == 0
        )
        cpu_target_layer = current_layer + self._kt_runtime_prefetch_lookahead_layers
        gpu_target_layer = (
            current_layer + self._kt_runtime_gpu_prefetch_lookahead_layers
        )
        should_stage_cpu = (
            loader.prefetch_enabled
            and self._kt_runtime_prefetch_next_layer
            and self._kt_runtime_cpu_prefetch_enabled
            and self._kt_runtime_prefetch_top_k > 0
            and num_layers is not None
            and cpu_target_layer < num_layers
            and (
                not self._kt_runtime_prefetch_stage_boundary_only
                or current_layer % stage_size == 0
            )
        )
        if self._kt_runtime_gpu_prefetch_stage_span > 1:
            gpu_prefetch_layer_ok = (
                num_layers is not None
                and gpu_target_layer < num_layers
                and current_layer % stage_size == 0
            )
        else:
            gpu_prefetch_layer_ok = (
                num_layers is not None
                and gpu_target_layer < num_layers
                and (
                    not self._kt_runtime_gpu_prefetch_target_boundary_only
                    or gpu_target_layer % stage_size == 0
                )
            )
        should_stage_gpu = (
            loader.prefetch_enabled
            and self._kt_runtime_prefetch_next_layer
            and self._kt_runtime_gpu_prefetch_enabled
            and self._kt_runtime_prefetch_top_k > 0
            and gpu_prefetch_layer_ok
        )
        should_stage_next_step_layer0 = (
            self._kt_runtime_prefetch_next_step_layer0
            and self._kt_runtime_gpu_prefetch_enabled
            and oracle is not None
            and num_layers is not None
            and current_layer % stage_size == 0
            and current_layer + stage_size >= num_layers
        )
        needs_prefetch_schedule = (
            should_stage_cpu or should_stage_gpu or should_stage_next_step_layer0
        )
        if not needs_prefetch_schedule and (committed_gpu_prefetch or not update_boundary):
            return

        if (
            oracle is not None
            and self._kt_runtime_skip_batch_selection_with_oracle
        ):
            selected = self.gpu_index_to_logical.to(
                device="cpu", dtype=torch.int64
            ).clone()
            self._kt_runtime_last_batch_ranked_cpu = [
                int(x) for x in selected.tolist()
            ]
        else:
            selected = self._select_runtime_target_experts(topk_ids)
        selected_scores: Optional[List[int]] = None
        if self._kt_runtime_foreground_oracle and update_boundary:
            selected, oracle_ranked, selected_scores = self._runtime_predicted_selected_for_layer(
                target_layer=current_layer,
                fallback_selected=selected,
            )
            self._kt_runtime_last_batch_ranked_cpu = list(oracle_ranked)
            if (
                self.tp_rank == 0
                and os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_PROFILE") == "1"
            ):
                logger.info(
                    "KT runtime foreground oracle placement: layer=%d "
                    "stage_size=%d selected=%s",
                    current_layer,
                    stage_size,
                    [int(x) for x in selected.to("cpu").tolist()],
                )
        target_set = {int(x) for x in selected.tolist()}
        if self._kt_runtime_defer_prefetch_after_cpu_submit:
            self._kt_runtime_deferred_prefetch = (loader, selected.clone())
        else:
            self._schedule_runtime_hf_prefetch(loader, selected)
        if committed_gpu_prefetch:
            return
        if not update_boundary:
            return
        if (
            self._kt_runtime_gpu_prefetch_enabled
            and self._kt_runtime_keep_old_on_prefetch_miss
        ):
            if self.tp_rank == 0 and (
                os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_PROFILE") == "1"
            ):
                logger.info(
                    "KT runtime GPU prefetch miss kept old placement: "
                    "layer=%d pending=%s",
                    current_layer,
                    pending_after_commit is not None,
                )
            return

        plan = self._build_runtime_swap_plan(
            target_set,
            ranked=list(self._kt_runtime_last_batch_ranked_cpu),
            scores=selected_scores,
        )
        if plan is None:
            return
        new_by_slot, evict_slots, _new_experts = plan

        if self._kt_runtime_gpu_prefetch_enabled:
            if pending_after_commit is None:
                self._kt_runtime_foreground_swap_no_pending += 1
                foreground_reason = "no_pending"
            else:
                self._kt_runtime_foreground_swap_pending_not_ready += 1
                foreground_reason = "pending_not_committed"
        else:
            foreground_reason = "prefetch_disabled"

        t0 = time.perf_counter()
        prefetch_sources: Dict[str, int] = {}
        prefetch_wait_ms = 0.0
        profile_enabled = os.environ.get("SGLANG_KT_RUNTIME_SWAP_PROFILE") == "1"
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        t_load = time.perf_counter()
        for slot in evict_slots:
            load_info = loader.load_expert_to_gpu_slot(
                layer=layer,
                layer_idx=self.kt_config.layer_idx,
                logical_expert_id=new_by_slot[slot],
                gpu_slot=slot,
            )
            source = str(load_info.get("source", "unknown"))
            prefetch_sources[source] = prefetch_sources.get(source, 0) + 1
            prefetch_wait_ms += float(load_info.get("prefetch_wait_ms", 0.0) or 0.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        load_ms = (time.perf_counter() - t_load) * 1000.0

        t_mapping = time.perf_counter()
        self._apply_runtime_gpu_mapping_delta(new_by_slot, evict_slots, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        mapping_ms = (time.perf_counter() - t_mapping) * 1000.0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if self.tp_rank == 0:
            self._kt_runtime_swap_count += 1
            if os.environ.get("SGLANG_KT_RUNTIME_SWAP_VERBOSE") == "1":
                logger.info(
                    "KT runtime HF expert swap: layer=%d changed_slots=%d "
                    "swap_count=%d elapsed_ms=%.2f prefetch_sources=%s "
                    "prefetch_wait_ms=%.2f new_gpu_experts=%s",
                    self.kt_config.layer_idx,
                    len(evict_slots),
                    self._kt_runtime_swap_count,
                    elapsed_ms,
                    prefetch_sources,
                    prefetch_wait_ms,
                    new_by_slot,
                )
            else:
                logger.info(
                    "KT runtime HF expert swap: layer=%d changed_slots=%d "
                    "swap_count=%d elapsed_ms=%.2f prefetch_sources=%s "
                    "prefetch_wait_ms=%.2f",
                    self.kt_config.layer_idx,
                    len(evict_slots),
                    self._kt_runtime_swap_count,
                    elapsed_ms,
                    prefetch_sources,
                    prefetch_wait_ms,
                )
            if profile_enabled:
                logger.info(
                    "KT runtime HF expert swap profile: layer=%d changed_slots=%d "
                    "total_ms=%.2f load_ms=%.2f mapping_ms=%.2f "
                    "prefetch_sources=%s prefetch_wait_ms=%.2f",
                    self.kt_config.layer_idx,
                    len(evict_slots),
                    elapsed_ms,
                    load_ms,
                    mapping_ms,
                    prefetch_sources,
                    prefetch_wait_ms,
                )
            if os.environ.get("SGLANG_KT_RUNTIME_PREFETCH_PROFILE") == "1":
                logger.info(
                    "KT runtime foreground swap reason: layer=%d reason=%s "
                    "no_pending=%d pending_not_committed=%d "
                    "prefetch_no_pending=%d prefetch_not_ready=%d pending_busy=%d",
                    self.kt_config.layer_idx,
                    foreground_reason,
                    self._kt_runtime_foreground_swap_no_pending,
                    self._kt_runtime_foreground_swap_pending_not_ready,
                    self._kt_runtime_gpu_prefetch_no_pending,
                    self._kt_runtime_gpu_prefetch_not_ready,
                    self._kt_runtime_gpu_prefetch_pending_busy,
                )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.

        This is the main computation method that coordinates:
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results from CPU and GPU experts
        """
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        num_tokens = int(x.shape[0]) if x.dim() > 0 else 0
        _kt_timing = (
            os.environ.get("SGLANG_KT_HYBRID_TIMING") == "1"
            and self.tp_rank == 0
            and (
                os.environ.get("SGLANG_KT_HYBRID_TIMING_ALL_LAYERS") == "1"
                or getattr(self.kt_config, "layer_idx", None)
                in (0, 5, 20, 35)
            )
        )
        # Route statistics call .item()/torch.unique() on CUDA tensors and
        # therefore synchronize the serving stream. Keep them optional so
        # CPU-wait timing can remain compatible with background H2D overlap.
        _kt_route_stats = (
            _kt_timing
            and os.environ.get("SGLANG_KT_HYBRID_TIMING_ROUTE_STATS", "1")
            == "1"
        )
        _kt_t_apply_start = time.perf_counter() if _kt_timing else None
        _kt_t_after_submit = None
        _kt_t_after_mask = None
        _kt_t_after_gpu = None
        _kt_t_after_sync = None
        _kt_t_after_merge = None
        _kt_t_cpu_wait_ms = 0.0
        _kt_cpu_entries = 0
        _kt_gpu_entries = 0
        _kt_cpu_unique = 0
        _kt_gpu_unique = 0

        if (
            self.kt_config.kt_enable_dynamic_expert_update
            and self._kt_runtime_hf_expert_swap_enabled
            and not (
                torch.cuda.is_available()
                and torch.cuda.is_current_stream_capturing()
            )
        ):
            self._update_gpu_experts_from_batch_hf_swap(layer, dispatch_output)

        # Record GPU expert mask for distribution tracking (rank 0 only) after
        # any runtime swap, so traces reflect the placement used by this call.
        if self.tp_rank == 0:
            recorder = get_global_expert_distribution_recorder()
            recorder.on_gpu_expert_mask(
                self.kt_config.layer_idx, self.gpu_experts_mask_cuda
            )

        # Check for full GPU fallback. The full-GPU path's _build_full_context →
        # _prepare_weight_{mxfp4,fp8,fp8_channel,bf16,int4} helpers read flat
        # `w13_weight` / `w13_weight_packed` attributes off `layer`. V4-Flash
        # MXFP4 (triton_kernels path) optionally preserves these when
        # `kt_gpu_prefill_token_threshold > 0` is set (see
        # `mxfp4_deepseek.process_weights_after_loading`); accept either the
        # flat attr or the v4 triton-kernels marker as a hint that the loader
        # can populate the layer. Layouts without either are still skipped to
        # avoid crashing the scheduler. Origin: sglang 本身 (V4-Flash
        # full-GPU prefill fallback compat).
        _full_gpu_fallback_supported = (
            hasattr(layer, "w13_weight")
            or hasattr(layer, "w13_weight_packed")
            or getattr(layer, "_v4_tk_path", False)
        )
        if (
            self.gpu_prefill_token_threshold > 0
            and num_tokens >= self.gpu_prefill_token_threshold
            and _full_gpu_fallback_supported
            and not self._kt_runtime_hf_expert_swap_enabled
        ):
            ctx = self._build_full_context(layer)

            t_compute = time.perf_counter()
            result = ctx.gpu_method.apply(ctx.gpu_layer, dispatch_output)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            compute_time = (time.perf_counter() - t_compute) * 1000.0

            # Dynamic expert update: analyze batch and update GPU experts.
            # Skip for V4-Flash MXFP4 — `_update_gpu_experts_from_batch` →
            # `copy_experts_weights_int4` hardcodes int4 weight names
            # (`w13_weight_packed` etc.) and crashes on MXFP4 layouts. The
            # full-GPU fallback re-loads all 256 experts on every fire anyway,
            # so the dynamic-promote optimization is a no-op for MXFP4. Origin:
            # sglang 本身 (V4-Flash full-GPU prefill fallback compat).
            _mxfp4_skip_dyn_update = getattr(ctx, "is_mxfp4_quant", False)
            if self.kt_config.kt_enable_dynamic_expert_update and not _mxfp4_skip_dyn_update:
                t_update = time.perf_counter()
                self._update_gpu_experts_from_batch(
                    layer=layer,
                    ctx=ctx,
                    dispatch_output=dispatch_output,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                update_time = (time.perf_counter() - t_update) * 1000.0

                if self.tp_rank == 0:
                    logger.info(
                        "KT layerwise prefill: layer %d compute = %.2f ms, expert update = %.2f ms",
                        self.kt_config.layer_idx,
                        compute_time,
                        update_time,
                    )
            else:
                if self.tp_rank == 0:
                    logger.info(
                        "KT layerwise prefill: layer %d compute = %.2f ms",
                        self.kt_config.layer_idx,
                        compute_time,
                    )

            return result

        # Step 1: Copy hidden_states to staging buffer and submit CPU computation
        # Staging buffer allows GPU computation to proceed without waiting for D2H copy
        staging_buffer = None
        if self.tp_rank == 0 and self._cpu_stream is not None:
            # Use shared staging buffer (shared across all MoE layers to save GPU memory)
            assert self._shared_staging_buffer is not None, "Shared staging buffer not initialized"
            staging_buffer = self._shared_staging_buffer.get_slice(x.shape[0])

            # Copy to staging buffer on main stream
            staging_buffer.copy_(x, non_blocking=True)

            # SGLANG_KT_HYBRID_NO_CPU_STREAM=1 collapses cpu_stream onto main stream.
            _no_cpu_stream = os.environ.get("SGLANG_KT_HYBRID_NO_CPU_STREAM") == "1"
            if not _no_cpu_stream:
                # Fork to cpu_stream (waits for staging copy to complete)
                self._cpu_stream.wait_stream(torch.cuda.current_stream(x.device))
            from contextlib import nullcontext as _ctx_null
            _stream_ctx = _ctx_null() if _no_cpu_stream else torch.cuda.stream(self._cpu_stream)
            with _stream_ctx:
                # Submit uses staging_buffer, so GPU can modify original x freely
                self._submit_with_staged_input(
                    layer, dispatch_output, staging_buffer
                )
            if os.environ.get("SGLANG_KT_SYNC_AFTER_CPU_SUBMIT", "1") == "1":
                self._cpu_stream.synchronize()
            if os.environ.get("SGLANG_KT_SYNC_MAIN_BEFORE_GPU", "0") == "1":
                torch.cuda.current_stream(x.device).synchronize()
        deferred_prefetch = self._kt_runtime_deferred_prefetch
        if deferred_prefetch is not None:
            self._kt_runtime_deferred_prefetch = None
            deferred_loader, deferred_selected = deferred_prefetch
            self._schedule_runtime_hf_prefetch(
                deferred_loader, deferred_selected
            )
        if _kt_timing:
            if os.environ.get("SGLANG_KT_HYBRID_TIMING_DEEP") == "1":
                torch.cuda.synchronize(x.device)
            _kt_t_after_submit = time.perf_counter()

        # Step 2: Prepare GPU computation by masking and remapping expert IDs
        # CPU expert IDs are set to -1; GPU expert IDs are remapped to GPU weight indices
        topk_ids = topk_output.topk_ids
        masked_topk_ids = mask_and_remap_expert_ids(
            topk_ids, self.gpu_experts_mask_cuda, self.logical_to_gpu_index_cuda
        )
        if _kt_route_stats:
            try:
                _valid_topk_ids = topk_ids[
                    (topk_ids >= 0) & (topk_ids < self.global_num_experts)
                ]
                if _valid_topk_ids.numel() > 0:
                    _gpu_hit_mask = self.gpu_experts_mask_cuda[_valid_topk_ids]
                    _kt_gpu_entries = int(_gpu_hit_mask.sum().item())
                    _kt_cpu_entries = int(_valid_topk_ids.numel()) - _kt_gpu_entries
                    if _kt_gpu_entries > 0:
                        _kt_gpu_unique = int(
                            torch.unique(_valid_topk_ids[_gpu_hit_mask]).numel()
                        )
                    if _kt_cpu_entries > 0:
                        _kt_cpu_unique = int(
                            torch.unique(_valid_topk_ids[~_gpu_hit_mask]).numel()
                        )
            except Exception:
                _kt_cpu_entries = _kt_gpu_entries = 0
                _kt_cpu_unique = _kt_gpu_unique = 0

        # Create modified dispatch output for GPU computation
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)
        masked_dispatch_output = dispatch_output._replace(
            topk_output=masked_topk_output
        )
        if _kt_timing:
            if os.environ.get("SGLANG_KT_HYBRID_TIMING_DEEP") == "1":
                torch.cuda.synchronize(x.device)
            _kt_t_after_mask = time.perf_counter()
        if os.environ.get("SGLANG_KT_SYNC_AFTER_ROUTE_MASK", "0") == "1":
            torch.cuda.current_stream(x.device).synchronize()

        # Step 3: Execute GPU expert computation on main stream
        # No wait needed - staging buffer decouples CPU and GPU data access
        # When num_gpu_experts == 0 the gpu_method's weights have shapes that
        # are incompatible with its own apply() (e.g. on SM_120 with V4 Flash
        # where the only routed-expert quant method available, the FP8 fused
        # MoE Triton path, asserts hidden_states.shape[1] == w1.shape[2] -
        # padded_size, which fails because w1 is the empty 0-expert slice).
        # Skip the GPU GEMM entirely and start from zeros; the CPU path then
        # provides 100% of the routed-expert contribution.
        # Origin: kt-sglang 耦合 (sglang/kt_ep_wrapper.py).
        if not getattr(self, "_diag_logged", False) and logger.isEnabledFor(logging.DEBUG):
            self._diag_logged = True
            try:
                _mask_sum = int(self.gpu_experts_mask.sum().item())
            except Exception as e:  # pragma: no cover
                _mask_sum = f"err:{e}"
            logger.debug(
                "[kt-ep-diag] layer=%s num_gpu_experts=%d mask_sum=%s "
                "mask_shape=%s gpu_method=%s",
                getattr(self.kt_config, 'layer_idx', '?'),
                self.num_gpu_experts,
                _mask_sum,
                tuple(self.gpu_experts_mask.shape),
                type(self.gpu_method).__name__,
            )
        # SGLANG_KT_BYPASS_GPU_MOE=1 also short-circuits to zeros, because
        # the kt mask generator returns an all-True (num_gpu_experts ==
        # num_total_experts) per-layer mask in some configurations (e.g. V4
        # Flash + --kt-num-gpu-experts=0), which defeats the
        # num_gpu_experts==0 short-circuit. The env var lets the operator
        # force the bypass without untangling the mask generator.
        if self.num_gpu_experts == 0 or os.environ.get("SGLANG_KT_BYPASS_GPU_MOE") == "1":
            gpu_combine_input = None
            output = torch.zeros_like(x)
            # 2604B sub-mode adds a runtime path-checker assertion in the
            # model (deepseek_v4.py:1169 expects observed == 1 after every
            # MoE forward). The trtllm path bumps it inside its body; the
            # bypass path mirrors that here so the assertion still passes
            # when GPU MoE is short-circuited in favour of CPU experts.
            from sglang.srt.environ import envs as _envs
            if _envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
                from sglang.srt.debug_utils.deepseek_v4_debug_utils import (
                    deepseek_v4_moe_code_path_checker,
                )
                deepseek_v4_moe_code_path_checker.observed += 1
        else:
            gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)
            output = gpu_combine_input.hidden_states
        if _kt_timing:
            if os.environ.get("SGLANG_KT_HYBRID_TIMING_DEEP") == "1":
                torch.cuda.synchronize(x.device)
            _kt_t_after_gpu = time.perf_counter()

        # Step 4: Sync CPU results on cpu_stream, then synchronize streams
        if self.tp_rank == 0 and self._cpu_stream is not None:
            _no_cpu_stream = os.environ.get("SGLANG_KT_HYBRID_NO_CPU_STREAM") == "1"
            from contextlib import nullcontext as _ctx_null
            _stream_ctx = _ctx_null() if _no_cpu_stream else torch.cuda.stream(self._cpu_stream)
            with _stream_ctx:
                # Use staging_buffer for sync to get correct buffer reference
                _kt_t_sync_pre = time.perf_counter() if _kt_t_apply_start is not None else None
                cpu_output = self._sync_with_staged_input(staging_buffer)
                if _kt_t_sync_pre is not None:
                    _kt_t_cpu_wait_ms = (time.perf_counter() - _kt_t_sync_pre) * 1000.0
                if not _no_cpu_stream:
                    self._sync_done_event.record(self._cpu_stream)
            if _kt_timing:
                _kt_t_after_sync = time.perf_counter()

            # Main stream waits for cpu_stream to complete before merging results
            if not _no_cpu_stream:
                torch.cuda.current_stream(x.device).wait_event(self._sync_done_event)
            output = output + cpu_output
        if _kt_timing:
            _kt_t_after_merge = time.perf_counter()
            # Optional: synchronize GPU at end of apply() to capture true GPU
            # work latency (otherwise gpu_apply Python time only captures
            # kernel-launch CPU overhead, not actual GPU compute). DEEP mode
            # serialises streams so per-stage numbers reflect GPU work, not
            # async launch return.
            if os.environ.get("SGLANG_KT_HYBRID_TIMING_DEEP") == "1":
                torch.cuda.synchronize(x.device)
                _kt_t_after_merge = time.perf_counter()

        if _kt_t_apply_start is not None:
            _kt_total_ms = (_kt_t_after_merge - _kt_t_apply_start) * 1000.0
            _stage_submit_ms = (_kt_t_after_submit - _kt_t_apply_start) * 1000.0
            _stage_mask_ms = (_kt_t_after_mask - _kt_t_after_submit) * 1000.0
            _stage_gpu_ms = (_kt_t_after_gpu - _kt_t_after_mask) * 1000.0
            _stage_sync_ms = (
                (_kt_t_after_sync - _kt_t_after_gpu) * 1000.0
                if _kt_t_after_sync is not None else 0.0
            )
            _stage_merge_ms = (
                (_kt_t_after_merge - _kt_t_after_sync) * 1000.0
                if _kt_t_after_sync is not None
                else (_kt_t_after_merge - _kt_t_after_gpu) * 1000.0
            )
            _cls = type(self)
            if not hasattr(_cls, '_kt_layer_step'):
                _cls._kt_layer_step = {}
            _li = getattr(self.kt_config, 'layer_idx', -1)
            _cls._kt_layer_step[_li] = _cls._kt_layer_step.get(_li, 0) + 1
            _step = _cls._kt_layer_step[_li]
            if _step <= 16 or _step % 16 == 0:
                logger.info(
                    "[kt-time] layer=%s step=%d total=%.2fms submit=%.2f "
                    "mask=%.2f gpu=%.2f sync=%.2f merge=%.2f "
                    "cpu_wait=%.2fms num_tokens=%d cpu_entries=%d "
                    "gpu_entries=%d cpu_unique=%d gpu_unique=%d",
                    _li, _step, _kt_total_ms, _stage_submit_ms,
                    _stage_mask_ms, _stage_gpu_ms, _stage_sync_ms,
                    _stage_merge_ms, _kt_t_cpu_wait_ms, num_tokens,
                    _kt_cpu_entries, _kt_gpu_entries, _kt_cpu_unique,
                    _kt_gpu_unique,
                )
        return StandardCombineInput(hidden_states=output)

    def _update_gpu_experts_from_batch(
        self,
        layer: torch.nn.Module,
        ctx: "SharedFullContext",
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Update original layer's GPU experts based on current batch statistics.

        This method:
        1. Analyzes topk_ids to find most frequently activated experts
        2. Copies selected expert weights from ctx.gpu_layer to layer
        3. Updates all mapping tables (gpu_experts_mask, logical_to_gpu_index, etc.)
        4. Broadcasts changes across TP ranks for consistency

        Args:
            layer: Original MoE layer with subset of GPU experts
            ctx: SharedFullContext containing temporary full GPU layer
            dispatch_output: Current batch dispatch output with routing information
        """
        # Step 1: Select top experts (rank 0 computes, broadcasts to all ranks)
        topk_ids = dispatch_output.topk_output.topk_ids
        device = topk_ids.device

        if self.tp_rank == 0:
            selected_experts = select_top_experts_from_batch(
                topk_ids=topk_ids,
                num_experts=self.global_num_experts,
                num_gpu_experts=self.num_gpu_experts,
            )
        else:
            # Create placeholder on other ranks
            selected_experts = torch.zeros(
                self.num_gpu_experts, dtype=torch.int64, device=device
            )

        # Broadcast selected experts to all ranks for consistent weight updates
        if dist.is_initialized():
            dist.broadcast(selected_experts, src=0, group=get_tp_group().device_group)

        # Step 2: Copy weights from temporary layer to original layer
        if ctx.is_fp8_quant:
            copy_experts_weights_fp8(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )
        elif ctx.is_fp8_channel_quant:
            copy_experts_weights_fp8_channel(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )
        elif ctx.is_bf16_quant:
            copy_experts_weights_bf16(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )
        else:
            copy_experts_weights_int4(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )

        # Step 3: Update mapping tables
        gpu_experts_mask_cpu, logical_to_gpu_index_cuda, gpu_index_to_logical_cpu = (
            update_gpu_expert_mappings(
                selected_experts=selected_experts,
                num_experts=self.global_num_experts,
                device=device,
            )
        )

        # Update instance variables (both CPU and CUDA versions)
        # CRITICAL: Use .copy_() for CUDA tensors to maintain same buffer for CUDA graph compatibility
        # CUDA graph captures tensor memory addresses during decode phase, so we must update
        # in-place rather than replacing the tensor reference
        self.gpu_experts_mask = gpu_experts_mask_cpu  # CPU tensor, safe to replace
        self.gpu_experts_mask_cuda.copy_(gpu_experts_mask_cpu)  # In-place update for CUDA graph
        self.logical_to_gpu_index = logical_to_gpu_index_cuda.cpu()  # CPU version for weight loading
        self.logical_to_gpu_index_cuda.copy_(logical_to_gpu_index_cuda)  # In-place update for CUDA graph
        self.gpu_index_to_logical = gpu_index_to_logical_cpu  # CPU tensor, safe to replace

        # Step 4: Update KT wrapper (rank 0 only)
        if self.tp_rank == 0:
            update_kt_wrapper_masks(self.wrapper, gpu_experts_mask_cpu)

        # Log expert changes (rank 0 only)
        if self.tp_rank == 0:
            logger.debug(
                "KT dynamic update: layer %d updated GPU experts to: %s",
                self.kt_config.layer_idx,
                selected_experts.cpu().tolist(),
            )

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped GPU method.

        This allows the wrapper to transparently expose attributes and methods
        from the wrapped GPU quantization method.

        Args:
            name: Attribute name

        Returns:
            Attribute value from gpu_method
        """
        # Avoid infinite recursion for internal attributes
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self.gpu_method, name)

    def _build_full_context(self, layer: torch.nn.Module) -> "SharedFullContext":
        global _SHARED_FULL_CONTEXT

        if _SHARED_FULL_CONTEXT is None:
            _SHARED_FULL_CONTEXT = SharedFullContext(
                layer=layer,
                init_args=self._full_init_args,
                global_num_experts=self.global_num_experts,
                moe_runner_config=self.moe_runner_config,
            )

        _SHARED_FULL_CONTEXT.load(
            layer_idx=self.kt_config.layer_idx,
            wrapper=self.wrapper,
            original_layer=layer,
            gpu_experts_mask=self.gpu_experts_mask,
            logical_to_gpu_index=self.logical_to_gpu_index,
        )
        return _SHARED_FULL_CONTEXT


# ---------------------------------------------------------------------------
# Plugin registration: makes KTEPWrapperMethod available to FusedMoE without
# any base-file import. Activated by importing this module, which happens via
# sglang.srt.models.deepseek_v4 -> auto-discovered by ModelRegistry.
# ---------------------------------------------------------------------------

def _kt_ep_predicate(layer, server_args):
    return create_kt_config_from_server_args(server_args, layer.layer_id)


def _kt_ep_factory(layer, gpu_method, kt_config):
    return KTEPWrapperMethod(gpu_method, kt_config)


from sglang.srt.layers.moe.quant_method_registry import register_moe_quant_wrapper

# priority=20 → wraps after mxfp4 (matches PR #38 Phase 3 → outer wrapper)
register_moe_quant_wrapper(
    "kt_ep", _kt_ep_predicate, _kt_ep_factory, priority=20
)
