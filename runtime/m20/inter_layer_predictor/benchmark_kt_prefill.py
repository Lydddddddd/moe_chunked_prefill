#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import aiohttp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_FILE = ROOT / "workloads" / "text" / "sharegpt_long_qwen3_min2048_512.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "kt_migration" / "kt_prefill_smoke"


def _extract_prompt(obj: Any) -> str:
    if isinstance(obj, str):
        return obj.strip()
    if not isinstance(obj, dict):
        return ""
    for key in ["prompt", "rendered_prompt", "text", "instruction", "question", "input"]:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = obj.get("messages") or obj.get("conversation") or obj.get("conversations")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content") or message.get("value") or message.get("text")
                role = message.get("role") or message.get("from") or "user"
                if isinstance(content, str) and content.strip():
                    parts.append(f"{role}: {content.strip()}")
            elif isinstance(message, str) and message.strip():
                parts.append(message.strip())
        return "\n".join(parts).strip()
    return ""


def load_prompts(path: Path, limit: int, offset: int = 0) -> list[str]:
    prompts: list[str] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            if seen < offset:
                seen += 1
                continue
            try:
                prompt = _extract_prompt(json.loads(line))
            except json.JSONDecodeError:
                prompt = line
            if prompt:
                prompts.append(prompt)
                seen += 1
    return prompts


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_prompt_identity_manifest(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"prompt identity manifest must be a JSON object: {manifest_path}")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"prompt identity manifest has no entries map: {manifest_path}")
    payload["_path"] = str(manifest_path)
    return payload


def prepare_prompts(
    prompts: list[str],
    tokenizer_path: str,
    max_model_len: int,
) -> tuple[list[str], list[int]]:
    try:
        from transformers import AutoTokenizer
    except Exception:
        return prompts, [0 for _ in prompts]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    truncated_prompts: list[str] = []
    counts: list[int] = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=max_model_len,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"]
        counts.append(len(input_ids))
        truncated_prompts.append(tokenizer.decode(input_ids, skip_special_tokens=True))
    return truncated_prompts, counts


def load_verified_prompt_batch(
    *,
    prompt_file: Path,
    num_prompts: int,
    prompt_offset: int,
    tokenizer_path: str,
    max_model_len: int,
    identity_manifest: dict[str, Any],
    workload_id: str,
) -> tuple[list[str], list[int], list[dict[str, str]]]:
    prompts = load_prompts(prompt_file, num_prompts, prompt_offset)[:num_prompts]
    raw_prompt_hashes = [prompt_sha256(prompt) for prompt in prompts]
    entries = identity_manifest.get("entries", {}) if identity_manifest else {}
    prompt_identities: list[dict[str, str]] = []
    for request_id, prompt_hash in enumerate(raw_prompt_hashes):
        prompt_index = int(prompt_offset) + request_id
        expected = entries.get(str(prompt_index)) if isinstance(entries, dict) else None
        if identity_manifest:
            if not isinstance(expected, dict):
                raise ValueError(
                    "prompt identity manifest has no entry for "
                    f"prompt_global_index={prompt_index}"
                )
            expected_hash = str(expected.get("prompt_sha256") or "")
            if expected_hash != prompt_hash:
                raise ValueError(
                    "prompt identity mismatch for "
                    f"prompt_global_index={prompt_index}: workload/offset does not match manifest"
                )
        prompt_identities.append(
            {"workload_id": workload_id, "prompt_sha256": prompt_hash}
        )
    prepared, token_counts = prepare_prompts(
        prompts, tokenizer_path, max_model_len
    )
    return prepared, token_counts, prompt_identities


def _normalize_signature(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, dict):
        for key in [
            "expert_signature",
            "signature",
            "expert_ids",
            "experts",
            "hot_experts",
            "active_experts",
        ]:
            if key in value:
                return _normalize_signature(value[key])
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[int] = []
        for item in value:
            if isinstance(item, (list, tuple, set, dict)):
                out.extend(_normalize_signature(item))
                continue
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return sorted(set(out))
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def load_expert_signatures(path: str | None) -> dict[str, list[int]]:
    if not path:
        return {}
    signature_path = Path(path).resolve()
    if not signature_path.exists():
        raise FileNotFoundError(f"expert signature file does not exist: {signature_path}")

    def add_row(
        mapping: dict[str, list[int]],
        row: Any,
        fallback_key: str | None = None,
    ) -> None:
        signature = _normalize_signature(row)
        if not signature:
            return
        if isinstance(row, dict):
            keys = [
                fallback_key,
                row.get("rid"),
                row.get("request_rid"),
                row.get("request_id"),
                row.get("prompt_index"),
                row.get("kt_prompt_index"),
                row.get("prompt_global_index"),
            ]
        else:
            keys = [fallback_key]
        for key in keys:
            if key is not None:
                mapping[str(key)] = signature

    mapping: dict[str, list[int]] = {}
    if signature_path.suffix == ".jsonl":
        with signature_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    add_row(mapping, json.loads(line))
    else:
        payload = json.loads(signature_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = None
            for key in ["rows", "items", "records", "signatures"]:
                if isinstance(payload.get(key), list):
                    rows = payload[key]
                    break
            if rows is not None:
                for row in rows:
                    add_row(mapping, row)
            else:
                for key, value in payload.items():
                    add_row(mapping, value, fallback_key=str(key))
        elif isinstance(payload, list):
            for row in payload:
                add_row(mapping, row)
    return mapping


async def request_one(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
    prompt_offset: int,
    run_name: str,
    send_kt_metadata: bool,
    expert_signatures: dict[str, list[int]],
    prompt_identity: dict[str, str],
) -> dict[str, Any]:
    prompt_index = prompt_offset + request_id
    rid = f"{run_name}__p{prompt_index}__r{request_id}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    kt_metadata: dict[str, Any] = {
        "run_name": run_name,
        "request_id": request_id,
        "prompt_index": prompt_index,
        "prompt_global_index": prompt_index,
        "rid": rid,
        **prompt_identity,
    }
    signature = (
        expert_signatures.get(rid)
        or expert_signatures.get(str(prompt_index))
        or expert_signatures.get(str(request_id))
    )
    if signature:
        kt_metadata["expert_signature"] = signature
    if send_kt_metadata:
        payload["rid"] = rid
        payload["user"] = rid
        payload["metadata"] = kt_metadata
        payload["kt_metadata"] = kt_metadata
    start = time.perf_counter()
    status = 0
    text = ""
    error = ""
    generated_text = ""
    finish_reason = ""
    completion_tokens = 0
    try:
        async with session.post(url, json=payload) as response:
            status = response.status
            text = await response.text()
            if status >= 400:
                error = text[:1000]
            else:
                try:
                    response_json = json.loads(text)
                    choices = response_json.get("choices") or []
                    if choices:
                        choice = choices[0]
                        message = choice.get("message") or {}
                        generated_text = str(message.get("content") or "")
                        finish_reason = str(choice.get("finish_reason") or "")
                    usage = response_json.get("usage") or {}
                    completion_tokens = int(usage.get("completion_tokens") or 0)
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
    except Exception as exc:  # noqa: BLE001 - benchmark should record failures.
        error = repr(exc)
    end = time.perf_counter()
    return {
        "request_id": request_id,
        "prompt_index": prompt_index,
        "rid": rid,
        "status": status,
        "latency_s": end - start,
        "ok": status == 200 and not error,
        "error": error,
        "response_bytes": len(text.encode("utf-8")),
        "generated_text": generated_text,
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "sent_kt_metadata": send_kt_metadata,
        "sent_expert_signature_size": len(signature or []),
    }


async def run_requests(
    prompts: list[str],
    *,
    url: str,
    model: str,
    max_tokens: int,
    concurrency: int,
    timeout_s: float,
    prompt_offset: int,
    run_name: str,
    send_kt_metadata: bool,
    expert_signatures: dict[str, list[int]],
    prompt_identities: list[dict[str, str]],
) -> list[dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(limit=max(1, concurrency))
    results: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(max(1, concurrency))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def _wrapped(index: int, prompt: str) -> None:
            async with sem:
                results.append(
                    await request_one(
                        session,
                        url=url,
                        model=model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        request_id=index,
                        prompt_offset=prompt_offset,
                        run_name=run_name,
                        send_kt_metadata=send_kt_metadata,
                        expert_signatures=expert_signatures,
                        prompt_identity=(
                            prompt_identities[index]
                            if index < len(prompt_identities)
                            else {}
                        ),
                    )
                )

        await asyncio.gather(*[_wrapped(i, prompt) for i, prompt in enumerate(prompts)])
    return sorted(results, key=lambda row: int(row["request_id"]))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark KT/SGLang prefill-only server.")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen3")
    parser.add_argument("--model-path", default="/data/HF_MODELS/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup-num-prompts", type=int, default=0)
    parser.add_argument("--warmup-prompt-offset", type=int, default=0)
    parser.add_argument("--warmup-concurrency", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--run-name", default="kt_prefill")
    parser.add_argument("--expert-signature-file", default="")
    parser.add_argument(
        "--workload-id",
        default="",
        help="Stable workload identity carried with each runtime request.",
    )
    parser.add_argument(
        "--prompt-identity-manifest",
        default="",
        help="Optional JSON manifest that verifies prompt index/content before requests are sent.",
    )
    parser.add_argument(
        "--send-kt-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send stable rid plus metadata/kt_metadata for KT runtime tracing and reorder.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = Path(args.prompt_file).resolve()
    identity_manifest = load_prompt_identity_manifest(args.prompt_identity_manifest or None)
    workload_id = str(args.workload_id).strip() or str(
        identity_manifest.get("workload_id") or prompt_file.stem
    )
    prompts, token_counts, prompt_identities = load_verified_prompt_batch(
        prompt_file=prompt_file,
        num_prompts=args.num_prompts,
        prompt_offset=args.prompt_offset,
        tokenizer_path=args.model_path,
        max_model_len=args.max_model_len,
        identity_manifest=identity_manifest,
        workload_id=workload_id,
    )
    total_prompt_tokens = sum(token_counts)
    expert_signatures = load_expert_signatures(args.expert_signature_file or None)

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    warmup_summary: dict[str, Any] = {}
    if args.warmup_num_prompts > 0:
        warmup_prompts, warmup_token_counts, warmup_identities = load_verified_prompt_batch(
            prompt_file=prompt_file,
            num_prompts=args.warmup_num_prompts,
            prompt_offset=args.warmup_prompt_offset,
            tokenizer_path=args.model_path,
            max_model_len=args.max_model_len,
            identity_manifest=identity_manifest,
            workload_id=workload_id,
        )
        warmup_started = time.perf_counter()
        warmup_results = asyncio.run(
            run_requests(
                warmup_prompts,
                url=url,
                model=args.model,
                max_tokens=args.max_tokens,
                concurrency=(
                    args.warmup_concurrency
                    if args.warmup_concurrency > 0
                    else args.concurrency
                ),
                timeout_s=args.timeout_s,
                prompt_offset=args.warmup_prompt_offset,
                run_name=f"{args.run_name}__warmup",
                send_kt_metadata=args.send_kt_metadata,
                expert_signatures=expert_signatures,
                prompt_identities=warmup_identities,
            )
        )
        warmup_wall_time_s = time.perf_counter() - warmup_started
        warmup_ok_count = sum(1 for row in warmup_results if row.get("ok"))
        for row in warmup_results:
            request_id = int(row["request_id"])
            row["prompt_tokens"] = (
                warmup_token_counts[request_id]
                if request_id < len(warmup_token_counts)
                else 0
            )
        write_jsonl(output_dir / "warmup_request_metrics.jsonl", warmup_results)
        warmup_summary = {
            "num_prompts": len(warmup_prompts),
            "prompt_offset": args.warmup_prompt_offset,
            "concurrency": (
                args.warmup_concurrency
                if args.warmup_concurrency > 0
                else args.concurrency
            ),
            "prompt_tokens": sum(warmup_token_counts),
            "ok_count": warmup_ok_count,
            "error_count": len(warmup_results) - warmup_ok_count,
            "wall_time_s": warmup_wall_time_s,
        }
        if warmup_ok_count != len(warmup_results):
            raise RuntimeError(f"warmup request failed: {warmup_summary}")

    start = time.perf_counter()
    results = asyncio.run(
        run_requests(
            prompts,
            url=url,
            model=args.model,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            timeout_s=args.timeout_s,
            prompt_offset=args.prompt_offset,
            run_name=args.run_name,
            send_kt_metadata=args.send_kt_metadata,
            expert_signatures=expert_signatures,
            prompt_identities=prompt_identities,
        )
    )
    wall_time_s = time.perf_counter() - start

    latencies = [float(row["latency_s"]) for row in results if row.get("ok")]
    for row in results:
        request_id = int(row["request_id"])
        row["prompt_tokens"] = token_counts[request_id] if request_id < len(token_counts) else 0
    ok_count = sum(1 for row in results if row.get("ok"))
    summary = {
        "run_name": args.run_name,
        "base_url": args.base_url,
        "model": args.model,
        "model_path": args.model_path,
        "prompt_file": str(Path(args.prompt_file).resolve()),
        "workload_id": workload_id,
        "prompt_identity_manifest": identity_manifest.get("_path", ""),
        "output_dir": str(output_dir),
        "num_prompts_requested": args.num_prompts,
        "prompt_offset": args.prompt_offset,
        "num_prompts_loaded": len(prompts),
        "ok_count": ok_count,
        "error_count": len(results) - ok_count,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "send_kt_metadata": args.send_kt_metadata,
        "expert_signature_file": str(Path(args.expert_signature_file).resolve()) if args.expert_signature_file else "",
        "expert_signatures_loaded": len(expert_signatures),
        "prompt_tokens": total_prompt_tokens,
        "warmup": warmup_summary,
        "wall_time_s": wall_time_s,
        "requests_per_s": ok_count / wall_time_s if wall_time_s > 0 else None,
        "prefill_tokens_per_s": total_prompt_tokens / wall_time_s if wall_time_s > 0 else None,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p95_s": percentile(latencies, 0.95),
        "latency_max_s": max(latencies) if latencies else None,
        "notes": [
            "KT/SGLang OpenAI-compatible endpoint does not expose server-side TTFT here.",
            "For max_tokens=1 prefill-only smoke, request latency is used as TTFT approximation.",
        ],
    }

    write_jsonl(output_dir / "default_request_metrics.jsonl", results)
    (output_dir / "default_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "client_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if ok_count == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
