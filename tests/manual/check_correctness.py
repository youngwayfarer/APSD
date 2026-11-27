# SPDX-License-Identifier: Apache-2.0
"""End-to-end diagnostics for decoupled parallel speculative decoding.

This script intentionally uses the V1 in-process engine so it can trace the
scheduler/worker boundary without changing production code. It validates
algorithmic state transitions; exact equality with target-only decoding is
reported separately because GPU kernel shape changes can alter greedy argmax.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from collections.abc import Sequence
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

import torch

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.spec_decode.parallel_utils import ParallelDraftLatencyController
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner,
    OptimisticParallelDraftState,
)

TRACE: dict[str, list[dict[str, Any]]] = {
    "prepare": [],
    "finish": [],
    "prob_gather": [],
    "reconcile": [],
    "memory_gate": [],
    "sm_profile": [],
}


class CyclingSMProfileScheduler(Scheduler):
    """Diagnostic-only scheduler that cycles through every Draft profile."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._diagnostic_sm_profile_index = 0

    def _select_parallel_spec_decode_sm_profile_id(
        self,
        planned_draft_work: bool,
        num_scheduled_tokens=None,
        scheduled_spec_decode_tokens=None,
    ) -> int | None:
        profiles = self.scheduler_config.parallel_spec_decode_sm_profiles
        if profiles is None:
            return None
        if not planned_draft_work:
            profile_id = 0
        else:
            if len(profiles) < 2:
                raise RuntimeError(
                    "Cycling diagnostics require a nonzero-Draft profile"
                )
            profile_id = 1 + (
                self._diagnostic_sm_profile_index % (len(profiles) - 1)
            )
            self._diagnostic_sm_profile_index += 1
        TRACE["sm_profile"].append(
            {
                "step": self.current_step,
                "profile_id": profile_id,
                "planned_draft_work": planned_draft_work,
            }
        )
        return profile_id


def _expected_pending(
    old_pending: list[int],
    generated: list[int],
    drafted: list[int],
) -> list[int]:
    """Independent oracle for the synchronization rule in the design."""
    if not generated:
        return old_pending + drafted

    common = min(len(old_pending), len(generated))
    if old_pending[:common] != generated[:common]:
        return []
    if len(generated) <= len(old_pending):
        return old_pending[len(generated) :] + drafted

    extra = generated[len(old_pending) :]
    if drafted[: len(extra)] != extra:
        return []
    return drafted[len(extra) :]


def _install_trace_hooks(
    force_memory_cycle: bool = False,
    trace_memory: bool = False,
    force_draft_budget: int | None = None,
) -> None:
    original_prepare = GPUModelRunner._prepare_optimistic_parallel_draft
    original_finish = GPUModelRunner._finish_optimistic_parallel_draft
    original_prob_gather = GPUModelRunner._get_parallel_spec_decode_draft_probs
    original_reconcile = Scheduler._reconcile_decoupled_spec_token_ids
    original_drafter_fits = Scheduler._drafter_fits_in_memory
    original_verify_tokens = Scheduler._get_num_target_verify_tokens
    original_draft_eligible = Scheduler._is_parallel_draft_eligible
    forced_window_start: int | None = None

    if force_draft_budget is not None:

        def select_forced_draft_budget(
            self: ParallelDraftLatencyController,
        ) -> int:
            return min(force_draft_budget, self.max_draft_tokens)

        ParallelDraftLatencyController.select_num_draft_tokens = (
            select_forced_draft_budget
        )

    def memory_is_forced_high(scheduler: Scheduler) -> bool:
        # Diagnostic-only cycle: normal -> memory pressure -> recovery.
        return bool(
            force_memory_cycle
            and forced_window_start is not None
            and forced_window_start
            <= scheduler.current_step
            < forced_window_start + 3
        )

    def traced_drafter_fits(self: Scheduler) -> bool:
        nonlocal forced_window_start
        usage = self.kv_cache_manager.block_pool.get_usage()
        threshold = self.scheduler_config.spec_decode_max_mem_utilization
        base_result = original_drafter_fits(self)
        if (
            force_memory_cycle
            and forced_window_start is None
            and base_result
        ):
            forced_window_start = self.current_step + 1
        forced_high = memory_is_forced_high(self)
        result = base_result and not forced_high
        TRACE["memory_gate"].append(
            {
                "kind": "draft",
                "step": self.current_step,
                "usage": usage,
                "threshold": threshold,
                "actual_pressure": usage >= threshold,
                "forced_high": forced_high,
                "result": result,
            }
        )
        return result

    def traced_verify_tokens(
        self: Scheduler,
        request,
        remaining_token_budget: int,
        base_num_new_tokens: int,
    ) -> int:
        usage = self.kv_cache_manager.block_pool.get_usage()
        threshold = self.scheduler_config.spec_decode_max_mem_utilization
        result = original_verify_tokens(
            self,
            request,
            remaining_token_budget,
            base_num_new_tokens,
        )
        forced_high = memory_is_forced_high(self)
        if forced_high:
            result = 0
        TRACE["memory_gate"].append(
            {
                "kind": "verify",
                "step": self.current_step,
                "usage": usage,
                "threshold": threshold,
                "actual_pressure": usage > threshold,
                "forced_high": forced_high,
                "result": result,
            }
        )
        return result

    def traced_draft_eligible(self: Scheduler, request) -> bool:
        return original_draft_eligible(
            self,
            request,
        ) and not memory_is_forced_high(self)

    def traced_prepare(
        self: GPUModelRunner,
        scheduler_output,
        common_attn_metadata,
        num_scheduled_tokens: int,
    ) -> OptimisticParallelDraftState | None:
        state = original_prepare(
            self,
            scheduler_output,
            common_attn_metadata,
            num_scheduled_tokens,
        )
        req_ids = self.input_batch.req_ids.copy()
        maintenance_ids = [] if state is None or state.maintenance is None else (
            state.maintenance.req_ids.copy()
        )
        extension_ids = [] if state is None or state.extension is None else (
            state.extension.req_ids.copy()
        )
        prefill_maintenance_ids = sorted(
            set(maintenance_ids)
            & set(scheduler_output.draft_kv_maintenance_req_ids)
        )
        catchup_ids = sorted(set(maintenance_ids) - set(prefill_maintenance_ids))
        TRACE["prepare"].append(
            {
                "req_ids": req_ids,
                "full_batch_size": len(req_ids),
                "enabled": sorted(scheduler_output.draft_enabled_req_ids),
                "maintenance": maintenance_ids,
                "prefill_maintenance": prefill_maintenance_ids,
                "catchup": catchup_ids,
                "extension": extension_ids,
                "pending_lens": {
                    req_id: len(
                        scheduler_output.pending_spec_decode_tokens.get(req_id, [])
                    )
                    for req_id in req_ids
                },
                "verify_lens": {
                    req_id: len(
                        scheduler_output.scheduled_spec_decode_tokens.get(req_id, [])
                    )
                    for req_id in req_ids
                },
                "scheduled": {
                    req_id: scheduler_output.num_scheduled_tokens.get(req_id, 0)
                    for req_id in req_ids
                },
                "state_created": state is not None,
                "profile_id": (
                    scheduler_output.parallel_spec_decode_sm_profile_id
                ),
                "state_profile_id": (
                    None
                    if state is None
                    else state.parallel_spec_decode_sm_profile_id
                ),
            }
        )
        return state

    def traced_finish(
        self: GPUModelRunner,
        state: OptimisticParallelDraftState,
        target_token_ids: list[list[int]],
    ) -> None:
        original_finish(self, state, target_token_ids)
        assert isinstance(self._draft_token_ids, list)
        target_ms = state.target_start_event.elapsed_time(
            state.target_done_event
        )
        draft_start_delay_ms = state.target_start_event.elapsed_time(
            state.draft_start_event
        )
        draft_total_ms = state.draft_start_event.elapsed_time(
            state.draft_done_event
        )
        draft_finish_ms = state.target_start_event.elapsed_time(
            state.draft_done_event
        )
        TRACE["finish"].append(
            {
                "req_ids": state.req_ids,
                "maintenance": (
                    [] if state.maintenance is None else state.maintenance.req_ids
                ),
                "extension": state.extension_req_ids,
                "target_lens": [len(ids) for ids in target_token_ids],
                "draft_lens": [len(ids) for ids in self._draft_token_ids],
                "selected_draft_tokens": state.num_draft_tokens,
                "target_ms": target_ms,
                "draft_start_delay_ms": draft_start_delay_ms,
                "draft_total_ms": draft_total_ms,
                "deadline_missed": draft_finish_ms > target_ms,
                "profile_id": state.parallel_spec_decode_sm_profile_id,
                "prob_state_lens": {
                    req_id: len(prob_state.token_ids)
                    for req_id, prob_state in (
                        self._parallel_pending_draft_states.items()
                    )
                },
            }
        )

    def traced_prob_gather(self: GPUModelRunner, metadata):
        result = original_prob_gather(self, metadata)
        TRACE["prob_gather"].append(
            None if result is None else list(result.shape)
        )
        return result

    def traced_reconcile(
        old_pending: list[int],
        generated: list[int],
        drafted: list[int],
    ) -> list[int]:
        result = original_reconcile(
            old_pending,
            generated,
            drafted,
        )
        expected = _expected_pending(old_pending, generated, drafted)
        TRACE["reconcile"].append(
            {
                "old": old_pending.copy(),
                "generated": generated.copy(),
                "drafted": drafted.copy(),
                "result": result.copy(),
                "oracle_match": result == expected,
            }
        )
        return result

    GPUModelRunner._prepare_optimistic_parallel_draft = traced_prepare
    GPUModelRunner._finish_optimistic_parallel_draft = traced_finish
    GPUModelRunner._get_parallel_spec_decode_draft_probs = traced_prob_gather
    Scheduler._reconcile_decoupled_spec_token_ids = staticmethod(traced_reconcile)
    if force_memory_cycle or trace_memory:
        Scheduler._drafter_fits_in_memory = traced_drafter_fits
        Scheduler._get_num_target_verify_tokens = traced_verify_tokens
    if force_memory_cycle:
        Scheduler._is_parallel_draft_eligible = traced_draft_eligible


def _prompts(scenario: str) -> list[str]:
    if scenario == "single":
        return [
            "请简单介绍一下推测解码，"
            "并说明它为什么能加速生成。"
        ]
    if scenario == "mixed":
        long_prompt = "请逐条分析并总结以下材料：" + (
            "并行推测解码需要维护已确认前缀和乐观后缀。" * 45
        )
        return [
            "用一句话解释推测解码。",
            long_prompt,
            "列出两个大语言模型推理优化方法。",
        ]
    if scenario == "pressure":
        long_request = "并行解码需要维护正确前缀。" * 35
        fillers = [
            f"显存压力填充请求{i}。" + "推测后缀占用KV缓存。" * 45
            for i in range(7)
        ]
        return [long_request, *fillers]
    raise ValueError(f"Unknown scenario: {scenario}")


def _sm_partition_kwargs(
    target_sm_count: int | None,
    draft_sm_count: int | None,
    sm_profiles: Sequence[tuple[int, int]] | None,
) -> dict[str, Any]:
    if (target_sm_count is None) != (draft_sm_count is None):
        raise ValueError("Target and Draft SM counts must be set together")
    if sm_profiles is not None and target_sm_count is not None:
        raise ValueError("Fixed SM counts and dynamic profiles are mutually exclusive")
    if sm_profiles is not None:
        return {
            "parallel_spec_decode_sm_profiles": tuple(sm_profiles),
            "scheduler_cls": CyclingSMProfileScheduler,
        }
    if target_sm_count is None:
        return {}
    assert draft_sm_count is not None
    return {
        "parallel_spec_decode_target_sm_count": target_sm_count,
        "parallel_spec_decode_draft_sm_count": draft_sm_count,
    }


def _parse_sm_profile(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "SM profile must use TARGET,DRAFT format"
        )
    try:
        target_sm_count, draft_sm_count = map(int, parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "SM profile counts must be integers"
        ) from exc
    return target_sm_count, draft_sm_count


def _run_llm(
    target: str,
    draft: str | None,
    prompts: Sequence[str],
    max_tokens: int,
    max_num_batched_tokens: int,
    sampling: str,
    seed_start: int | None = None,
    max_speculative_tokens: int = 4,
    memory_threshold: float = 0.95,
    draft_time_budget_ratio: float = 0.8,
    per_request_max_tokens: list[int] | None = None,
    target_sm_count: int | None = None,
    draft_sm_count: int | None = None,
    sm_profiles: Sequence[tuple[int, int]] | None = None,
    gpu_memory_utilization: float = 0.5,
) -> tuple[list[list[int]], list[list[int]]]:
    kwargs: dict[str, Any] = {
        "model": target,
        "async_scheduling": False,
        "enforce_eager": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": 1024,
        "max_num_seqs": 8,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_chunked_prefill": True,
    }
    if draft is not None:
        kwargs.update(
            speculative_config={
                "method": "draft_model",
                "model": draft,
                "max_speculative_tokens": max_speculative_tokens,
                "enforce_eager": True,
                "max_model_len": 1024,
                "rejection_sample_method": "standard",
                "draft_sample_method": "probabilistic",
            },
            enable_parallel_spec_decode=True,
            spec_decode_max_mem_utilization=memory_threshold,
            parallel_spec_decode_draft_time_budget_ratio=(
                draft_time_budget_ratio
            ),
        )
        kwargs.update(
            _sm_partition_kwargs(
                target_sm_count,
                draft_sm_count,
                sm_profiles,
            )
        )

    llm = LLM(**kwargs)
    try:
        tokenizer = llm.get_tokenizer()
        prompt_token_ids = [tokenizer.encode(prompt) for prompt in prompts]
        if per_request_max_tokens is not None:
            assert len(per_request_max_tokens) == len(prompts)
            sampling_params = [
                SamplingParams(
                    temperature=0 if sampling == "greedy" else 0.8,
                    seed=None if sampling == "greedy" else 20_000 + index,
                    max_tokens=request_max_tokens,
                    ignore_eos=True,
                )
                for index, request_max_tokens in enumerate(
                    per_request_max_tokens
                )
            ]
        elif sampling == "greedy":
            sampling_params: SamplingParams | list[SamplingParams] = SamplingParams(
                temperature=0,
                max_tokens=max_tokens,
                ignore_eos=True,
            )
        elif sampling == "random":
            if seed_start is None:
                sampling_params = SamplingParams(
                    temperature=0.8,
                    seed=123,
                    max_tokens=max_tokens,
                    ignore_eos=True,
                )
            else:
                sampling_params = [
                    SamplingParams(
                        temperature=0.8,
                        seed=seed_start + index,
                        max_tokens=max_tokens,
                        ignore_eos=True,
                    )
                    for index in range(len(prompts))
                ]
        else:
            assert len(prompts) == 3
            sampling_params = [
                SamplingParams(
                    temperature=0,
                    max_tokens=max_tokens,
                    ignore_eos=True,
                ),
                SamplingParams(
                    temperature=0.8,
                    seed=123,
                    max_tokens=max_tokens,
                    ignore_eos=True,
                ),
                SamplingParams(
                    temperature=0,
                    max_tokens=max_tokens,
                    ignore_eos=True,
                ),
            ]
        outputs = llm.generate(
            list(prompts),
            sampling_params,
            use_tqdm=False,
        )
        return (
            [list(output.outputs[0].token_ids) for output in outputs],
            prompt_token_ids,
        )
    finally:
        del llm
        gc.collect()
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


def _first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _validate_trace(
    scenario: str,
    sampling: str,
    max_speculative_tokens: int = 4,
    force_memory_cycle: bool = False,
    require_real_memory_cycle: bool = False,
    require_mixed_target_verify: bool = False,
    require_draft_subbatch: bool = False,
    require_prefill_maintenance: bool = False,
    require_catchup_then_extension: bool = False,
    sm_profiles: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    created = [event for event in TRACE["prepare"] if event["state_created"]]
    enabled = [event for event in created if event["enabled"]]
    assert created, "Optimistic parallel state was never created."
    assert enabled, "No request entered the draft-enabled path."
    assert len(TRACE["finish"]) == len(created), (
        "Every created parallel state must finish exactly once."
    )
    assert TRACE["reconcile"], "No target/draft reconciliation was observed."
    assert all(event["oracle_match"] for event in TRACE["reconcile"]), (
        "Scheduler reconciliation disagrees with the independent oracle."
    )

    for event in TRACE["finish"]:
        extension_ids = set(event["extension"])
        for req_id, draft_len in zip(event["req_ids"], event["draft_lens"]):
            expected_len = (
                event["selected_draft_tokens"]
                if req_id in extension_ids
                else 0
            )
            assert draft_len == expected_len, (
                f"Unexpected emitted draft length for {req_id}: "
                f"expected {expected_len}, got {draft_len}."
            )
        assert 1 <= event["selected_draft_tokens"] <= max_speculative_tokens

    mixed_steps = [
        event
        for event in enabled
        if 0 < len(event["enabled"]) < len(event["req_ids"])
    ]
    mixed_target_steps = [
        event
        for event in created
        if 0
        < sum(length > 0 for length in event["verify_lens"].values())
        < len(event["req_ids"])
    ]
    strict_draft_subbatches = [
        event
        for event in created
        if 0 < len(event["extension"]) < event["full_batch_size"]
    ]
    prefill_maintenance_steps = [
        event for event in created if event["prefill_maintenance"]
    ]
    last_catchup_event: dict[str, int] = {}
    catchup_then_extension: list[dict[str, Any]] = []
    for event_index, event in enumerate(created):
        for req_id in event["catchup"]:
            last_catchup_event[req_id] = event_index
        for req_id in event["extension"]:
            catchup_event = last_catchup_event.pop(req_id, None)
            if catchup_event is not None:
                catchup_then_extension.append(
                    {
                        "req_id": req_id,
                        "catchup_event": catchup_event,
                        "extension_event": event_index,
                    }
                )
    if require_draft_subbatch:
        assert strict_draft_subbatches, (
            "No step used a strict Draft extension subbatch."
        )
    if require_prefill_maintenance:
        assert prefill_maintenance_steps, (
            "No prefill request entered Draft KV maintenance."
        )
    if require_catchup_then_extension:
        assert catchup_then_extension, (
            "No request transitioned from catch-up-only to Draft extension."
        )
    if scenario == "mixed":
        assert mixed_steps, (
            "The run did not produce a batch containing both draft-enabled "
            "and draft-disabled requests."
        )
        if require_mixed_target_verify:
            assert mixed_target_steps, (
                "The run did not produce a batch containing both target "
                "verification and standard autoregressive decoding."
            )
        assert any(
            any(
                event["scheduled"][req_id] > 1
                for req_id in event["req_ids"]
                if req_id not in event["enabled"]
            )
            for event in mixed_steps
        ), "The mixed batch did not include a prefill row."

    retained = sum(bool(event["result"]) for event in TRACE["reconcile"])
    cleared = sum(
        bool(event["old"] or event["drafted"]) and not event["result"]
        for event in TRACE["reconcile"]
    )
    probability_states = sum(
        bool(event["prob_state_lens"]) for event in TRACE["finish"]
    )
    max_probability_state_len = max(
        (
            length
            for event in TRACE["finish"]
            for length in event["prob_state_lens"].values()
        ),
        default=0,
    )
    max_pending_len = max(
        (len(event["result"]) for event in TRACE["reconcile"]),
        default=0,
    )
    probability_gathers = sum(shape is not None for shape in TRACE["prob_gather"])
    if sampling != "greedy":
        assert probability_gathers, "Rejection sampler never received Draft probs."
    forced_high_steps: list[int] = []
    if force_memory_cycle:
        forced_high_steps = sorted(
            {
                event["step"]
                for event in TRACE["memory_gate"]
                if event["forced_high"]
            }
        )
        assert len(forced_high_steps) == 3
        assert forced_high_steps == list(
            range(forced_high_steps[0], forced_high_steps[0] + 3)
        )
        assert all(
            event["result"] in (False, 0)
            for event in TRACE["memory_gate"]
            if event["forced_high"]
        ), "Draft or Target verify remained enabled under forced pressure."
        enabled_steps = {
            event["step"]
            for event in TRACE["memory_gate"]
            if event["kind"] == "draft" and event["result"]
        }
        assert any(step < forced_high_steps[0] for step in enabled_steps)
        assert any(step > forced_high_steps[-1] for step in enabled_steps)
        assert any(
            not event["extension"]
            for event in TRACE["prepare"]
        ), "The forced skip did not produce a no-extension parallel step."
    memory_usages = [event["usage"] for event in TRACE["memory_gate"]]
    actual_pressure_events = [
        event for event in TRACE["memory_gate"] if event["actual_pressure"]
    ]
    recovered_after_pressure = False
    if require_real_memory_cycle:
        assert actual_pressure_events, "Real KV usage never crossed the threshold."
        draft_events = [
            event
            for event in TRACE["memory_gate"]
            if event["kind"] == "draft"
        ]
        pressure_indices = [
            index
            for index, event in enumerate(draft_events)
            if event["actual_pressure"]
        ]
        assert pressure_indices, "Draft never observed real memory pressure."
        first_pressure = pressure_indices[0]
        assert all(
            not draft_events[index]["result"] for index in pressure_indices
        ), "Draft remained enabled above the real memory threshold."
        assert any(
            event["result"] and not event["actual_pressure"]
            for event in draft_events[:first_pressure]
        ), "Draft was not enabled before real memory pressure."
        recovered_after_pressure = any(
            event["result"] and not event["actual_pressure"]
            for event in draft_events[first_pressure + 1 :]
        )
        assert recovered_after_pressure, (
            "Draft did not recover after KV usage fell below the threshold."
        )
        assert all(
            event["result"] == 0
            for event in actual_pressure_events
            if event["kind"] == "verify"
        ), "Target verify remained enabled above the memory threshold."
    draft_budget_histogram: dict[int, int] = {}
    for event in TRACE["finish"]:
        budget = event["selected_draft_tokens"]
        draft_budget_histogram[budget] = draft_budget_histogram.get(budget, 0) + 1
    selected_profile_ids: list[int] = []
    if sm_profiles is not None:
        assert TRACE["sm_profile"], "Dynamic SM profile selection was not observed."
        for event in TRACE["sm_profile"]:
            profile_id = event["profile_id"]
            planned_draft_work = event["planned_draft_work"]
            assert (profile_id == 0) == (not planned_draft_work)
            selected_profile_ids.append(profile_id)
        expected_draft_profiles = set(range(1, len(sm_profiles)))
        assert expected_draft_profiles <= set(selected_profile_ids), (
            "Not every configured Draft profile was selected: "
            f"expected={sorted(expected_draft_profiles)}, "
            f"observed={sorted(set(selected_profile_ids))}."
        )
        for event in TRACE["prepare"]:
            if event["state_created"]:
                assert event["profile_id"] == event["state_profile_id"]
        for event in TRACE["finish"]:
            assert event["profile_id"] in expected_draft_profiles
    return {
        "parallel_states": len(created),
        "draft_enabled_states": len(enabled),
        "mixed_states": len(mixed_steps),
        "mixed_target_states": len(mixed_target_steps),
        "strict_draft_subbatch_states": len(strict_draft_subbatches),
        "prefill_maintenance_states": len(prefill_maintenance_steps),
        "catchup_then_extension": catchup_then_extension,
        "full_batch_sizes": [event["full_batch_size"] for event in created],
        "maintenance_subbatch_sizes": [
            len(event["maintenance"]) for event in created
        ],
        "catchup_subbatch_sizes": [
            len(event["catchup"]) for event in created
        ],
        "extension_subbatch_sizes": [
            len(event["extension"]) for event in created
        ],
        "maintenance_req_ids": [event["maintenance"] for event in created],
        "catchup_req_ids": [event["catchup"] for event in created],
        "extension_req_ids": [event["extension"] for event in created],
        "reconciliations": len(TRACE["reconcile"]),
        "retained_suffixes": retained,
        "cleared_suffixes": cleared,
        "probability_state_observations": probability_states,
        "max_probability_state_len": max_probability_state_len,
        "probability_gathers": probability_gathers,
        "max_pending_len": max_pending_len,
        "forced_memory_high_steps": forced_high_steps,
        "memory_usage_min": min(memory_usages, default=0.0),
        "memory_usage_max": max(memory_usages, default=0.0),
        "actual_memory_pressure_observations": len(actual_pressure_events),
        "recovered_after_memory_pressure": recovered_after_pressure,
        "draft_budget_histogram": draft_budget_histogram,
        "draft_deadline_misses": sum(
            event["deadline_missed"] for event in TRACE["finish"]
        ),
        "target_ms_total": sum(
            event["target_ms"] for event in TRACE["finish"]
        ),
        "draft_ms_total": sum(
            event["draft_total_ms"] for event in TRACE["finish"]
        ),
        "selected_sm_profile_ids": selected_profile_ids,
    }


def _find_kv_catchup_case(
    prompt_token_ids: list[int],
) -> tuple[list[int], int]:
    """Return confirmed prefix and next optimistic token after a rejection."""
    confirmed = prompt_token_ids.copy()
    waiting_for_extension = False
    for event in TRACE["reconcile"]:
        if waiting_for_extension and event["drafted"]:
            return confirmed.copy(), event["drafted"][0]
        old = event["old"]
        generated = event["generated"]
        drafted = event["drafted"]
        common = min(len(old), len(generated))
        rejected = old[:common] != generated[:common]
        if not rejected and len(generated) > len(old):
            bonus = generated[len(old) :]
            rejected = drafted[: len(bonus)] != bonus
        confirmed.extend(event["generated"])
        waiting_for_extension = waiting_for_extension or rejected
    raise AssertionError("No rejection followed by a Draft catch-up was observed.")


def _fresh_draft_token(
    draft: str,
    prefix: list[int],
    gpu_memory_utilization: float = 0.5,
) -> int:
    llm = LLM(
        model=draft,
        async_scheduling=False,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=1024,
        enable_chunked_prefill=True,
    )
    try:
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=prefix)],
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
        return int(outputs[0].outputs[0].token_ids[0])
    finally:
        del llm
        gc.collect()
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


def _hashed_distribution_tvs(
    target_outputs: list[list[int]],
    parallel_outputs: list[list[int]],
    num_buckets: int = 32,
) -> list[float]:
    """Compare empirical token marginals with several deterministic hashes."""
    target_tokens = [token for output in target_outputs for token in output]
    parallel_tokens = [token for output in parallel_outputs for token in output]
    assert len(target_tokens) == len(parallel_tokens)

    tvs = []
    for salt in (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D):
        target_counts = [0] * num_buckets
        parallel_counts = [0] * num_buckets
        for token in target_tokens:
            target_counts[((token * salt) ^ (token >> 7)) % num_buckets] += 1
        for token in parallel_tokens:
            parallel_counts[((token * salt) ^ (token >> 7)) % num_buckets] += 1
        total = len(target_tokens)
        tvs.append(
            0.5
            * sum(
                abs(target_count - parallel_count) / total
                for target_count, parallel_count in zip(
                    target_counts, parallel_counts
                )
            )
        )
    return tvs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument(
        "--scenario",
        choices=("single", "mixed", "pressure"),
        default="single",
    )
    parser.add_argument(
        "--sampling",
        choices=("greedy", "random", "mixed"),
        default="greedy",
    )
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--compare-target", action="store_true")
    parser.add_argument("--force-memory-cycle", action="store_true")
    parser.add_argument("--check-kv-catchup", action="store_true")
    parser.add_argument("--distribution-seeds", type=int, default=0)
    parser.add_argument("--trace-memory", action="store_true")
    parser.add_argument("--require-real-memory-cycle", action="store_true")
    parser.add_argument("--memory-threshold", type=float, default=0.95)
    parser.add_argument("--max-speculative-tokens", type=int, default=4)
    parser.add_argument("--draft-time-budget-ratio", type=float, default=0.8)
    parser.add_argument(
        "--force-draft-budget",
        type=int,
        default=None,
        help="Diagnostic-only fixed Draft length; bypasses dynamic selection.",
    )
    parser.add_argument("--require-mixed-target-verify", action="store_true")
    parser.add_argument("--require-draft-subbatch", action="store_true")
    parser.add_argument("--require-prefill-maintenance", action="store_true")
    parser.add_argument("--require-catchup-then-extension", action="store_true")
    parser.add_argument("--repeat-prompts", type=int, default=1)
    parser.add_argument("--target-sm-count", type=int, default=None)
    parser.add_argument("--draft-sm-count", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument(
        "--sm-profile",
        action="append",
        type=_parse_sm_profile,
        default=None,
        metavar="TARGET,DRAFT",
        help=(
            "Repeatable dynamic Green Context profile. Profile 0 must reserve "
            "all SMs for Target and every later profile must reserve nonzero "
            "SMs for Draft."
        ),
    )
    args = parser.parse_args()

    prompts = _prompts(args.scenario)
    prompts *= args.repeat_prompts
    seed_start = None
    if args.distribution_seeds:
        assert args.scenario == "single" and args.sampling == "random"
        prompts *= args.distribution_seeds
        seed_start = 10_000
    token_budget = {
        "mixed": 128,
        "pressure": 64,
    }.get(args.scenario, 512)
    per_request_max_tokens = (
        [args.max_tokens, *([8] * 7)]
        if args.scenario == "pressure"
        else None
    )
    target_outputs = None
    if args.compare_target or args.distribution_seeds:
        target_outputs, _ = _run_llm(
            args.target,
            None,
            prompts,
            args.max_tokens,
            token_budget,
            args.sampling,
            seed_start=seed_start,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    if args.force_draft_budget is not None:
        assert 1 <= args.force_draft_budget <= args.max_speculative_tokens
    _install_trace_hooks(
        args.force_memory_cycle,
        args.trace_memory or args.require_real_memory_cycle,
        args.force_draft_budget,
    )
    parallel_outputs, prompt_token_ids = _run_llm(
        args.target,
        args.draft,
        prompts,
        args.max_tokens,
        token_budget,
        args.sampling,
        seed_start,
        args.max_speculative_tokens,
        args.memory_threshold,
        args.draft_time_budget_ratio,
        per_request_max_tokens,
        args.target_sm_count,
        args.draft_sm_count,
        args.sm_profile,
        args.gpu_memory_utilization,
    )
    summary = _validate_trace(
        args.scenario,
        args.sampling,
        max_speculative_tokens=args.max_speculative_tokens,
        force_memory_cycle=args.force_memory_cycle,
        require_real_memory_cycle=args.require_real_memory_cycle,
        require_mixed_target_verify=args.require_mixed_target_verify,
        require_draft_subbatch=args.require_draft_subbatch,
        require_prefill_maintenance=args.require_prefill_maintenance,
        require_catchup_then_extension=args.require_catchup_then_extension,
        sm_profiles=args.sm_profile,
    )
    summary["output_lengths"] = [len(ids) for ids in parallel_outputs]
    summary["output_hashes"] = [
        hashlib.sha256(
            ",".join(map(str, token_ids)).encode("ascii")
        ).hexdigest()
        for token_ids in parallel_outputs
    ]

    if target_outputs is not None:
        summary["target_exact_matches"] = sum(
            target == parallel
            for target, parallel in zip(target_outputs, parallel_outputs)
        )
        summary["first_mismatches"] = [
            _first_mismatch(target, parallel)
            for target, parallel in zip(target_outputs, parallel_outputs)
        ]
        if args.distribution_seeds:
            distribution_tvs = _hashed_distribution_tvs(
                target_outputs,
                parallel_outputs,
            )
            assert max(distribution_tvs) < 0.15, (
                "Target-only and parallel empirical distributions diverged: "
                f"hashed TVs={distribution_tvs}."
            )
            summary["distribution_seeds"] = args.distribution_seeds
            summary["distribution_hashed_tvs"] = distribution_tvs

    if args.check_kv_catchup:
        assert args.scenario == "single" and args.sampling == "greedy"
        prefix, optimistic_token = _find_kv_catchup_case(prompt_token_ids[0])
        fresh_token = _fresh_draft_token(
            args.draft,
            prefix,
            args.gpu_memory_utilization,
        )
        assert optimistic_token == fresh_token, (
            "Draft KV catch-up disagrees with fresh prefill: "
            f"optimistic={optimistic_token}, fresh={fresh_token}."
        )
        summary["kv_catchup_prefix_len"] = len(prefix)
        summary["kv_catchup_token"] = optimistic_token

    print("PARALLEL_SD_CORRECTNESS=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
