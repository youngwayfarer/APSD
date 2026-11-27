# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager, nullcontext
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn

import vllm.v1.worker.gpu_model_runner as gpu_model_runner_module
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    CompilationMode,
    CUDAGraphMode,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.weight_transfer.base import SparseWeightPatch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import CommonAttentionMetadata, MultipleOf
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.kv_cache_utils import estimate_max_model_len, get_kv_cache_configs
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode import llm_base_proposer
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.parallel_utils import DraftKVPathState, DraftRoundDeadline
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner, ParallelDraftRow
from vllm.v1.worker.utils import select_common_block_size

BLOCK_SIZE = 16
NUM_BLOCKS = 10
DEVICE_TYPE = current_platform.device_type


def initialize_kv_cache(runner: GPUModelRunner):
    """
    Only perform necessary steps in GPUModelRunner.initialize_kv_cache()
    """
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=runner.model_config.get_num_kv_heads(runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size, shared_by=["layer.0"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec)
        ],
    )
    runner.kv_cache_config = kv_cache_config
    runner.input_batch = InputBatch(
        max_num_reqs=runner.max_num_reqs,
        max_model_len=runner.max_model_len,
        max_num_batched_tokens=runner.max_num_tokens,
        device=runner.device,
        pin_memory=runner.pin_memory,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=[kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size],
        kernel_block_sizes=[
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        ],
    )
    runner.initialize_attn_backend(kv_cache_config)


def get_vllm_config():
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config


@pytest.fixture
def model_runner():
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        model_config = vllm_config.model_config
        num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        head_size = model_config.get_head_size()
        vllm_config.compilation_config.static_forward_context["layer.0"] = Attention(
            num_heads, head_size, 0.1
        )
        runner = GPUModelRunner(vllm_config, DEVICE_TYPE)
        initialize_kv_cache(runner)
        yield runner


model_runner_2 = model_runner


def _schedule_new_request(*req_ids: str) -> SchedulerOutput:
    new_reqs = []
    num_scheduled_tokens = {}
    total_num_scheduled_tokens = 0
    for req_id in req_ids:
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=[1, 2, 3],
                mm_features=[],
                sampling_params=SamplingParams(),
                pooling_params=None,
                block_ids=([0],),
                num_computed_tokens=0,
                lora_request=None,
            )
        )
        num_scheduled_tokens[req_id] = 3
        total_num_scheduled_tokens += num_scheduled_tokens[req_id]

    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _schedule_cached_requests(
    req_ids: list[str],
    num_scheduled_tokens: dict[str, int],
    new_token_ids: list[list[int]],
    num_computed_tokens: list[int],
    num_output_tokens: list[int],
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=set(),
            new_token_ids=new_token_ids,
            all_token_ids={},
            new_block_ids=[None] * len(req_ids),
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        ),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _is_req_scheduled(model_runner, req_id: str) -> bool:
    return req_id in model_runner.input_batch.req_id_to_index


def _is_req_added(model_runner, req_id: str) -> bool:
    return req_id in model_runner.requests


def _is_sampling_metadata_changed(
    model_runner, sampling_metadata_before: SamplingMetadata
):
    return model_runner.input_batch.sampling_metadata is not (sampling_metadata_before)


def _is_req_state_block_table_match(model_runner, req_id: str) -> bool:
    req_index = model_runner.input_batch.req_id_to_index[req_id]
    block_table = model_runner.input_batch.block_table[0]
    req_state = model_runner.requests[req_id]
    if block_table.num_blocks_per_row[req_index] != len(req_state.block_ids[0]):
        return False
    num_blocks = block_table.num_blocks_per_row[req_index]
    return (
        block_table.block_table.np[req_index, :num_blocks] == req_state.block_ids[0]
    ).all()


def _make_mock_backend_for_kernel_block_size(
    supported_sizes: list[int | MultipleOf],
):
    class _MockBackend:
        @staticmethod
        def get_supported_kernel_block_sizes():
            return supported_sizes

    return _MockBackend()


def _make_kv_cache_spec() -> FullAttentionSpec:
    return FullAttentionSpec(block_size=1, num_kv_heads=1, head_size=1, dtype="float16")


def test_select_common_block_size_prefers_manager_block_size():
    backend_a = _make_mock_backend_for_kernel_block_size([MultipleOf(32)])
    backend_b = _make_mock_backend_for_kernel_block_size([64, MultipleOf(16)])

    selected_size = select_common_block_size(128, [backend_a, backend_b])
    assert selected_size == 128


def test_select_common_block_size_uses_largest_shared_int():
    backend_a = _make_mock_backend_for_kernel_block_size([128, 64])
    backend_b = _make_mock_backend_for_kernel_block_size([64, 32])

    selected_size = select_common_block_size(256, [backend_a, backend_b])
    assert selected_size == 64


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("world_size", "is_last_rank", "expected_calls"),
    [(1, True, 0), (2, True, 0), (2, False, 1)],
)
def test_sample_tokens_receives_pp_sampled_ids_only_on_non_last_rank(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    is_last_rank: bool,
    expected_calls: int,
):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.execute_model_state = None
    runner.kv_connector_output = None
    runner.use_async_scheduling = True
    receive_calls = 0

    def receive_prev_sampled_token_ids():
        nonlocal receive_calls
        receive_calls += 1

    runner._pp_receive_prev_sampled_token_ids_to_input_batch = (
        receive_prev_sampled_token_ids
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=world_size, is_last_rank=is_last_rank),
    )

    output = GPUModelRunner.sample_tokens(runner, None)
    assert output in (EMPTY_MODEL_RUNNER_OUTPUT, None)
    assert receive_calls == expected_calls


@pytest.mark.skip_global_cleanup
def test_sample_tokens_skips_pp_group_lookup_without_async_scheduling(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.execute_model_state = None
    runner.kv_connector_output = None
    runner.use_async_scheduling = False

    monkeypatch.setattr(
        gpu_model_runner_module,
        "get_pp_group",
        pytest.fail,
    )

    output = GPUModelRunner.sample_tokens(runner, None)
    assert output in (EMPTY_MODEL_RUNNER_OUTPUT, None)


@pytest.mark.parametrize(
    ("entrypoint", "implementation_name", "args", "implementation_args"),
    [
        (
            "execute_model",
            "_execute_model",
            ("scheduler_output",),
            ("scheduler_output", None),
        ),
        (
            "sample_tokens",
            "_sample_tokens",
            ("grammar_output",),
            ("grammar_output",),
        ),
    ],
)
def test_parallel_target_entrypoints_run_on_target_stream(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    implementation_name: str,
    args: tuple[object, ...],
    implementation_args: tuple[object, ...],
):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    target_stream = Mock(name="target_stream")
    runner._parallel_target_stream = target_stream
    events: list[str] = []
    expected_output = object()

    @contextmanager
    def stream_context(stream):
        assert stream is target_stream
        events.append("enter")
        yield
        events.append("exit")

    def implementation(*implementation_args):
        assert implementation_args == expected_implementation_args
        events.append("body")
        return expected_output

    expected_implementation_args = implementation_args
    setattr(runner, implementation_name, implementation)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)

    output = getattr(GPUModelRunner, entrypoint)(runner, *args)

    assert output is expected_output
    assert events == ["enter", "body", "exit"]


@pytest.mark.parametrize(
    ("entrypoint", "implementation_name", "args", "implementation_args"),
    [
        (
            "execute_model",
            "_execute_model",
            ("scheduler_output",),
            ("scheduler_output", None),
        ),
        (
            "sample_tokens",
            "_sample_tokens",
            ("grammar_output",),
            ("grammar_output",),
        ),
    ],
)
def test_parallel_target_entrypoints_keep_current_stream_without_partition(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    implementation_name: str,
    args: tuple[object, ...],
    implementation_args: tuple[object, ...],
):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner._parallel_target_stream = None
    expected_output = object()
    implementation = Mock(return_value=expected_output)
    setattr(runner, implementation_name, implementation)
    stream_context = Mock(side_effect=AssertionError("unexpected stream switch"))
    monkeypatch.setattr(torch.cuda, "stream", stream_context)

    output = getattr(GPUModelRunner, entrypoint)(runner, *args)

    assert output is expected_output
    implementation.assert_called_once_with(*implementation_args)
    stream_context.assert_not_called()


def _make_dynamic_profile_runner():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    profiles = (
        SimpleNamespace(
            profile_id=0,
            target_stream=Mock(name="target_0"),
            draft_stream=None,
            target_sm_count=84,
            draft_sm_count=0,
        ),
        SimpleNamespace(
            profile_id=1,
            target_stream=Mock(name="target_1"),
            draft_stream=Mock(name="draft_1"),
            target_sm_count=76,
            draft_sm_count=8,
        ),
        SimpleNamespace(
            profile_id=2,
            target_stream=Mock(name="target_2"),
            draft_stream=Mock(name="draft_2"),
            target_sm_count=68,
            draft_sm_count=16,
        ),
    )
    runner._parallel_sm_profile_pool = SimpleNamespace(
        profiles=profiles,
        get=lambda profile_id: profiles[profile_id],
    )
    runner._parallel_sm_profile_lease = None
    runner._parallel_target_stream = None
    runner._parallel_draft_stream = Mock(name="wrong_global_draft")
    return runner, profiles


def _dynamic_profile_scheduler_output(
    profile_id: int | None,
    *,
    planned_draft_work: bool,
    total_num_scheduled_tokens: int = 1,
):
    return SimpleNamespace(
        parallel_spec_decode_sm_profile_id=profile_id,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        draft_enabled_req_ids={"request"} if planned_draft_work else set(),
        draft_kv_maintenance_req_ids=set(),
    )


@pytest.mark.parametrize(
    ("profile_id", "planned_draft_work", "message"),
    [
        (None, False, "missing"),
        (3, True, "outside"),
        (0, True, "profile 0"),
        (1, False, "without planned Draft"),
    ],
)
def test_dynamic_sm_profile_lease_rejects_scheduler_contract_violations(
    profile_id: int | None,
    planned_draft_work: bool,
    message: str,
):
    runner, _ = _make_dynamic_profile_runner()
    scheduler_output = _dynamic_profile_scheduler_output(
        profile_id,
        planned_draft_work=planned_draft_work,
    )

    with pytest.raises(RuntimeError, match=message):
        runner._acquire_parallel_sm_profile_lease(scheduler_output)


def test_dynamic_sm_profile_lease_rejects_id_without_pool():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner._parallel_sm_profile_pool = None
    runner._parallel_sm_profile_lease = None
    scheduler_output = _dynamic_profile_scheduler_output(
        1,
        planned_draft_work=True,
    )

    with pytest.raises(RuntimeError, match="without a dynamic SM profile pool"):
        runner._acquire_parallel_sm_profile_lease(scheduler_output)


def test_dynamic_sm_profile_switches_target_streams_without_overlap(
    monkeypatch: pytest.MonkeyPatch,
):
    runner, profiles = _make_dynamic_profile_runner()
    entered_streams = []
    runner._execute_model = Mock(return_value=object())

    @contextmanager
    def stream_context(stream):
        entered_streams.append(stream)
        yield

    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    for profile_id in (0, 1, 2, 0):
        output = _dynamic_profile_scheduler_output(
            profile_id,
            planned_draft_work=profile_id != 0,
        )
        GPUModelRunner.execute_model(runner, output)
        assert runner._parallel_sm_profile_lease is None

    assert entered_streams == [
        profiles[0].target_stream,
        profiles[1].target_stream,
        profiles[2].target_stream,
        profiles[0].target_stream,
    ]


def test_dynamic_sm_profile_sampling_reuses_execute_profile(
    monkeypatch: pytest.MonkeyPatch,
):
    runner, profiles = _make_dynamic_profile_runner()
    scheduler_output = _dynamic_profile_scheduler_output(
        1,
        planned_draft_work=True,
    )
    runner._acquire_parallel_sm_profile_lease(scheduler_output)
    runner.execute_model_state = SimpleNamespace(
        parallel_spec_decode_sm_profile_id=1,
    )
    expected = object()
    runner._sample_tokens = Mock(return_value=expected)
    entered_streams = []

    @contextmanager
    def stream_context(stream):
        entered_streams.append(stream)
        yield

    monkeypatch.setattr(torch.cuda, "stream", stream_context)

    result = GPUModelRunner.sample_tokens(runner, None)

    assert result is expected
    assert entered_streams == [profiles[1].target_stream]
    assert runner._parallel_sm_profile_lease is None


def test_dynamic_sm_profile_draft_uses_captured_profile_stream(
    monkeypatch: pytest.MonkeyPatch,
):
    runner, profiles = _make_dynamic_profile_runner()
    scheduler_output = _dynamic_profile_scheduler_output(
        2,
        planned_draft_work=True,
    )
    runner._acquire_parallel_sm_profile_lease(scheduler_output)
    runner._parallel_draft_workspace_slot = 1
    runner.drafter = Mock()
    entered_streams = []

    @contextmanager
    def stream_context(stream):
        entered_streams.append(stream)
        yield

    monkeypatch.setattr(gpu_model_runner_module, "DraftModelProposer", object)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "use_workspace_slot",
        lambda _slot: nullcontext(),
    )
    done_event = Mock()
    monkeypatch.setattr(torch.cuda, "Event", lambda **_kwargs: done_event)
    state = SimpleNamespace(
        parallel_spec_decode_sm_profile_id=2,
        inputs_ready_event=Mock(),
        draft_start_event=Mock(),
        maintenance=None,
        extension=None,
        draft_done_event=None,
    )

    runner._launch_optimistic_parallel_draft(state)

    assert entered_streams == [profiles[2].draft_stream]
    assert entered_streams != [runner._parallel_draft_stream]


def test_parallel_draft_state_is_published_before_executor_submission():
    runner = object.__new__(GPUModelRunner)
    runner._optimistic_parallel_draft_state = None
    runner._launch_optimistic_parallel_draft = Mock()
    future = Mock()
    state = SimpleNamespace(draft_launch_future=None)

    def submit(_fn, submitted_state):
        assert runner._optimistic_parallel_draft_state is state
        assert submitted_state is state
        return future

    runner._parallel_draft_executor = SimpleNamespace(submit=submit)

    runner._submit_optimistic_parallel_draft(state)

    assert state.draft_launch_future is future


def test_target_failure_drains_draft_before_releasing_profile_lease(
    monkeypatch: pytest.MonkeyPatch,
):
    runner, profiles = _make_dynamic_profile_runner()
    scheduler_output = _dynamic_profile_scheduler_output(
        1,
        planned_draft_work=True,
    )
    future = Mock()

    def fail_draft():
        assert runner._parallel_sm_profile_lease is not None
        raise RuntimeError("draft failed")

    def synchronize_draft_stream():
        assert runner._parallel_sm_profile_lease is not None

    future.result.side_effect = fail_draft
    profiles[1].draft_stream.synchronize.side_effect = synchronize_draft_stream
    state = SimpleNamespace(
        draft_launch_future=future,
        draft_done_event=None,
        parallel_spec_decode_sm_profile_id=1,
    )
    runner._optimistic_parallel_draft_state = state
    runner._execute_model = Mock(side_effect=RuntimeError("target failed"))
    runner._parallel_sm_profile_scope = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())

    with pytest.raises(RuntimeError, match="target failed"):
        GPUModelRunner.execute_model(runner, scheduler_output)

    future.result.assert_called_once_with()
    profiles[1].draft_stream.synchronize.assert_called_once_with()
    assert runner._optimistic_parallel_draft_state is None
    assert runner._parallel_sm_profile_lease is None


def test_sampling_failure_drains_draft_before_releasing_profile_lease(
    monkeypatch: pytest.MonkeyPatch,
):
    runner, _ = _make_dynamic_profile_runner()
    scheduler_output = _dynamic_profile_scheduler_output(
        1,
        planned_draft_work=True,
    )
    runner._acquire_parallel_sm_profile_lease(scheduler_output)
    runner.execute_model_state = SimpleNamespace(
        parallel_spec_decode_sm_profile_id=1,
    )
    done_event = Mock()
    state = SimpleNamespace(
        draft_launch_future=Mock(),
        draft_done_event=done_event,
        parallel_spec_decode_sm_profile_id=1,
    )
    runner._optimistic_parallel_draft_state = state
    runner._sample_tokens = Mock(side_effect=RuntimeError("sampling failed"))
    runner._parallel_sm_profile_scope = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())

    state.draft_launch_future.result.side_effect = lambda: (
        assert_active_parallel_sm_profile_lease(runner)
    )
    done_event.synchronize.side_effect = lambda: (
        assert_active_parallel_sm_profile_lease(runner)
    )

    with pytest.raises(RuntimeError, match="sampling failed"):
        GPUModelRunner.sample_tokens(runner, None)

    state.draft_launch_future.result.assert_called_once_with()
    done_event.synchronize.assert_called_once_with()
    assert runner._optimistic_parallel_draft_state is None
    assert runner._parallel_sm_profile_lease is None


def test_sampling_profile_mismatch_drains_draft_before_releasing_lease():
    runner, _ = _make_dynamic_profile_runner()
    scheduler_output = _dynamic_profile_scheduler_output(
        1,
        planned_draft_work=True,
    )
    runner._acquire_parallel_sm_profile_lease(scheduler_output)
    runner.execute_model_state = SimpleNamespace(
        parallel_spec_decode_sm_profile_id=2,
    )
    done_event = Mock()
    state = SimpleNamespace(
        draft_launch_future=Mock(),
        draft_done_event=done_event,
        parallel_spec_decode_sm_profile_id=1,
    )
    state.draft_launch_future.result.side_effect = lambda: (
        assert_active_parallel_sm_profile_lease(runner)
    )
    done_event.synchronize.side_effect = lambda: (
        assert_active_parallel_sm_profile_lease(runner)
    )
    runner._optimistic_parallel_draft_state = state
    runner._sample_tokens = Mock()

    with pytest.raises(
        RuntimeError,
        match="Sampling SM profile does not match the active lease",
    ):
        GPUModelRunner.sample_tokens(runner, None)

    state.draft_launch_future.result.assert_called_once_with()
    done_event.synchronize.assert_called_once_with()
    runner._sample_tokens.assert_not_called()
    assert runner._optimistic_parallel_draft_state is None
    assert runner._parallel_sm_profile_lease is None


def assert_active_parallel_sm_profile_lease(runner: GPUModelRunner) -> None:
    assert runner._parallel_sm_profile_lease is not None


def test_draft_model_proposer_advance_kv_cache_runs_prefix_only() -> None:
    proposer = object.__new__(DraftModelProposer)
    proposer.method = "draft_model"
    proposer.pass_hidden_states_to_model = False
    proposer.parallel_drafting = False
    proposer._last_draft_probs = torch.ones(1)
    proposer._forward_prefix = Mock()
    token_ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    metadata = Mock()

    proposer.advance_kv_cache(token_ids, positions, metadata)

    proposer._forward_prefix.assert_called_once_with(
        token_ids,
        positions,
        metadata,
    )
    assert proposer._last_draft_probs is None


@pytest.mark.parametrize(
    ("method", "pass_hidden_states_to_model", "parallel_drafting", "match"),
    [
        ("eagle", False, False, "independent draft model"),
        ("draft_model", True, False, "independent draft model"),
        ("draft_model", False, True, "parallel drafting models"),
    ],
)
def test_draft_model_proposer_advance_kv_cache_rejects_unsupported_modes(
    method: str,
    pass_hidden_states_to_model: bool,
    parallel_drafting: bool,
    match: str,
) -> None:
    proposer = object.__new__(DraftModelProposer)
    proposer.method = method
    proposer.pass_hidden_states_to_model = pass_hidden_states_to_model
    proposer.parallel_drafting = parallel_drafting
    proposer._forward_prefix = Mock()

    with pytest.raises(ValueError, match=match):
        proposer.advance_kv_cache(torch.tensor([1]), torch.tensor([0]), Mock())

    proposer._forward_prefix.assert_not_called()


class _SupportedPrefixMetadata:
    pass


def _make_prefix_metadata_proposer(
    monkeypatch: pytest.MonkeyPatch,
    per_group_metadata: list[object],
) -> tuple[DraftModelProposer, SimpleNamespace]:
    proposer = object.__new__(DraftModelProposer)
    proposer.method = "draft_model"
    proposer.pass_hidden_states_to_model = False
    proposer.parallel_drafting = False
    proposer.num_speculative_tokens = 2
    proposer.max_speculative_tokens = 2
    proposer._last_draft_probs = None
    proposer.allowed_attn_types = (_SupportedPrefixMetadata,)
    proposer.input_ids = torch.zeros(2, dtype=torch.int32)
    proposer.positions = torch.tensor([0, 1])
    proposer.uses_mrope = False
    proposer.block_size = 1
    proposer.arange = torch.arange(2, dtype=torch.int32)
    proposer.token_arange_np = np.arange(2, dtype=np.int32)
    proposer.vllm_config = Mock()
    proposer.model = Mock(return_value=torch.zeros(2, 1))
    proposer._set_positions = Mock()
    proposer._determine_batch_execution_and_padding = Mock(
        side_effect=[(None, 2, 2), (None, 1, 1)]
    )
    proposer.build_model_inputs_first_pass = Mock(
        return_value=({"input_ids": proposer.input_ids}, 2)
    )
    proposer._get_slot_mapping = Mock(return_value=torch.zeros(2, dtype=torch.int64))
    proposer._sample_draft_tokens = Mock(return_value=(torch.tensor([1]), None))
    proposer._update_positions_dependent_metadata = Mock(return_value=torch.tensor([1]))
    proposer._get_positions = Mock(return_value=torch.tensor([1]))
    proposer.build_per_group_and_layer_attn_metadata = Mock(
        side_effect=[([group_metadata], {}) for group_metadata in per_group_metadata]
    )
    monkeypatch.setattr(
        llm_base_proposer,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )
    metadata = SimpleNamespace(
        batch_size=Mock(return_value=1),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([2], dtype=torch.int32),
        _num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([2], dtype=torch.int32),
        _num_computed_tokens_cache=torch.tensor([0], dtype=torch.int32),
        is_prefilling=torch.tensor([True]),
        num_actual_tokens=2,
        max_query_len=2,
        slot_mapping=torch.zeros(2, dtype=torch.int64),
    )

    def replace_metadata(**kwargs):
        values = vars(metadata).copy()
        values.update(kwargs)
        return SimpleNamespace(**values)

    metadata.replace = replace_metadata
    return proposer, metadata


def test_propose_from_prefix_validates_packed_prefix_metadata_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsupported_prefix_metadata = object()
    proposer, metadata = _make_prefix_metadata_proposer(
        monkeypatch,
        [unsupported_prefix_metadata, _SupportedPrefixMetadata()],
    )
    expected_error = (
        "Unsupported attention metadata type for optimistic "
        f"parallel drafting: {type(unsupported_prefix_metadata)}"
    )

    with pytest.raises(ValueError) as exc_info:
        proposer.propose_from_prefix(
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
            metadata,
            Mock(),
        )

    assert str(exc_info.value) == expected_error
    assert metadata.num_actual_tokens == 2
    assert metadata.max_query_len == 2
    torch.testing.assert_close(
        metadata.query_start_loc, torch.tensor([0, 2], dtype=torch.int32)
    )
    assert proposer.build_per_group_and_layer_attn_metadata.call_count == 1


def test_propose_from_prefix_allows_unsupported_later_draft_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer, metadata = _make_prefix_metadata_proposer(
        monkeypatch,
        [_SupportedPrefixMetadata(), object()],
    )

    result = proposer.propose_from_prefix(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        metadata,
        Mock(),
    )

    torch.testing.assert_close(result, torch.tensor([[1, 1]]))
    build_calls = proposer.build_per_group_and_layer_attn_metadata.call_args_list
    assert len(build_calls) == 2
    assert build_calls[0].args == (metadata,)
    assert build_calls[0].kwargs == {}
    decode_metadata = build_calls[1].args[0]
    assert decode_metadata is not metadata
    assert build_calls[1].kwargs == {"draft_index": 1}
    torch.testing.assert_close(
        metadata.query_start_loc, torch.tensor([0, 2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        decode_metadata.query_start_loc, torch.tensor([0, 1], dtype=torch.int32)
    )
    assert metadata.num_actual_tokens == 2
    assert metadata.max_query_len == 2
    torch.testing.assert_close(metadata.seq_lens, torch.tensor([2], dtype=torch.int32))
    torch.testing.assert_close(metadata.is_prefilling, torch.tensor([True]))
    torch.testing.assert_close(decode_metadata.is_prefilling, torch.tensor([False]))


def test_propose_from_prefix_stops_when_current_round_no_longer_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer, metadata = _make_prefix_metadata_proposer(
        monkeypatch,
        [_SupportedPrefixMetadata()],
    )

    class FakeEvent:
        def __init__(self, timestamp_ms: float):
            self.timestamp_ms = timestamp_ms
            self.record = Mock()
            self.synchronize = Mock()

        def elapsed_time(self, other: "FakeEvent") -> float:
            return other.timestamp_ms - self.timestamp_ms

    first_round_start = FakeEvent(0.0)
    first_round_end = FakeEvent(6.0)
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: first_round_end)
    deadline = DraftRoundDeadline(remaining_ms=10.0)

    result = proposer.propose_from_prefix(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        metadata,
        Mock(),
        round_deadline=deadline,
        first_round_start_event=first_round_start,
    )

    torch.testing.assert_close(result, torch.tensor([[1]]))
    assert deadline.remaining_ms == pytest.approx(4.0)
    first_round_end.record.assert_called_once_with()
    first_round_end.synchronize.assert_called_once_with()
    assert proposer._sample_draft_tokens.call_count == 1


def test_propose_from_prefix_rejects_more_than_per_step_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer, metadata = _make_prefix_metadata_proposer(
        monkeypatch,
        [_SupportedPrefixMetadata()],
    )

    with pytest.raises(ValueError, match="max_speculative_tokens"):
        proposer.propose_from_prefix(
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
            metadata,
            Mock(),
            max_tokens_to_generate=3,
        )


def test_propose_from_prefix_deadline_still_respects_per_step_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer, metadata = _make_prefix_metadata_proposer(
        monkeypatch,
        [_SupportedPrefixMetadata()] * 3,
    )

    class FakeEvent:
        def __init__(self, timestamp_ms: float):
            self.timestamp_ms = timestamp_ms
            self.record = Mock()
            self.synchronize = Mock()

        def elapsed_time(self, other: "FakeEvent") -> float:
            return other.timestamp_ms - self.timestamp_ms

    # The deadline has room for more work, but the per-request step cap is two.
    first_round_start = FakeEvent(0.0)
    events = iter((FakeEvent(1.0), FakeEvent(1.0), FakeEvent(2.0)))
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: next(events))
    deadline = DraftRoundDeadline(remaining_ms=10.0)

    result = proposer.propose_from_prefix(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        metadata,
        Mock(),
        max_tokens_to_generate=2,
        round_deadline=deadline,
        first_round_start_event=first_round_start,
    )

    torch.testing.assert_close(result, torch.tensor([[1, 1]]))
    assert result.shape[1] == proposer.max_speculative_tokens
    assert deadline.remaining_ms == pytest.approx(8.0)
    assert proposer._sample_draft_tokens.call_count == 2


def test_select_common_block_size_no_valid_option():
    backend_a = _make_mock_backend_for_kernel_block_size([64])
    backend_b = _make_mock_backend_for_kernel_block_size([MultipleOf(16)])

    with pytest.raises(ValueError):
        select_common_block_size(48, [backend_a, backend_b])


def test_update_states_new_request(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_finished(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # finish req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids={req_id},
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert not _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)


def test_update_states_request_resumed(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # unschedule req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)

    # resume req
    cached_req_data = CachedRequestData(
        req_ids=[req_id],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[([0],)],
        num_computed_tokens=[0],
        num_output_tokens=[0],
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_get_nans_in_logits(model_runner, dist_init):
    req_ids = ("req_0", "req_1")

    scheduler_output = _schedule_new_request(*req_ids)
    model_runner._update_states(scheduler_output)

    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
        ],
        device=DEVICE_TYPE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 0, "req_1": 0}

    logits = torch.tensor(
        [
            [1.0, float("nan"), 3.0],
            [4.0, float("nan"), float("nan")],
        ],
        device=DEVICE_TYPE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 1, "req_1": 2}

    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, float("nan"), float("nan")],
        ],
        device=DEVICE_TYPE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 0, "req_1": 2}

    result = model_runner._get_nans_in_logits(logits=None)
    assert result == {"req_0": 0, "req_1": 0}

    logits = torch.tensor(
        [
            [1.0, float("nan"), 3.0],
        ],
        device=DEVICE_TYPE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 1, "req_1": 0}

    logits = torch.tensor(
        [
            [float("nan"), float("nan"), 2.0],
            [1.0, 2.0, 3.0],
            [float("nan"), 2.0, 3.0],
        ],
        device=DEVICE_TYPE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 2, "req_1": 0}


def test_update_states_no_changes(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # schedule req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert not _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_unscheduled(model_runner, dist_init):
    req_ids = ("req_0", "req_1")

    # new reqs
    scheduler_output = _schedule_new_request(*req_ids)

    model_runner._update_states(scheduler_output)

    assert _is_req_added(model_runner, req_ids[0])
    assert _is_req_scheduled(model_runner, req_ids[0])

    assert _is_req_added(model_runner, req_ids[1])
    assert _is_req_scheduled(model_runner, req_ids[1])

    # unschedule req_1
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_ids[0]: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)

    assert _is_req_added(model_runner, req_ids[0])
    assert _is_req_scheduled(model_runner, req_ids[0])

    assert _is_req_added(model_runner, req_ids[1])
    assert not _is_req_scheduled(model_runner, req_ids[1])


def test_update_states_pp_non_async_multi_request_keeps_token_buffers_consistent(
    model_runner, model_runner_2, dist_init, monkeypatch
):
    req_ids = ["req_0", "req_1"]
    non_last_runner = model_runner
    last_runner = model_runner_2
    non_last_runner.use_async_scheduling = False
    last_runner.use_async_scheduling = False

    # Both ranks start from the same request set.
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False, world_size=2),
    )
    non_last_runner._update_states(_schedule_new_request(*req_ids))
    last_runner._update_states(_schedule_new_request(*req_ids))

    sampled_by_last_rank = {req_ids[0]: 101, req_ids[1]: 201}
    # Emulate last-rank bookkeeping result from previous step:
    # sampled tokens already cached in CPU token buffers.
    for req_id, token_id in sampled_by_last_rank.items():
        req_index = last_runner.input_batch.req_id_to_index[req_id]
        start_idx = int(last_runner.input_batch.num_tokens_no_spec[req_index])
        end_idx = start_idx + 1
        last_runner.input_batch.token_ids_cpu[req_index, start_idx:end_idx] = [token_id]
        last_runner.input_batch.is_token_ids[req_index, start_idx:end_idx] = True
        last_runner.input_batch.num_tokens_no_spec[req_index] = end_idx
        last_runner.requests[req_id].output_token_ids.append(token_id)

    scheduler_output = _schedule_cached_requests(
        req_ids=req_ids,
        num_scheduled_tokens={req_ids[0]: 1, req_ids[1]: 1},
        new_token_ids=[[101], [201]],
        num_computed_tokens=[3, 3],  # prompt tokens only
        num_output_tokens=[1, 1],
    )
    # non-last rank appends new_token_ids in _update_states.
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False, world_size=2),
    )
    non_last_runner._update_states(scheduler_output)
    # last rank should keep its already-bookkept CPU buffers unchanged.
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True, world_size=2),
    )
    last_runner._update_states(scheduler_output)

    # Verify consistency between PP ranks after _update_states.
    for req_id in req_ids:
        non_last_idx = non_last_runner.input_batch.req_id_to_index[req_id]
        last_idx = last_runner.input_batch.req_id_to_index[req_id]
        non_last_len = int(non_last_runner.input_batch.num_tokens_no_spec[non_last_idx])
        last_len = int(last_runner.input_batch.num_tokens_no_spec[last_idx])
        assert non_last_len == last_len
        assert (
            non_last_runner.input_batch.token_ids_cpu[
                non_last_idx, :non_last_len
            ].tolist()
            == last_runner.input_batch.token_ids_cpu[last_idx, :last_len].tolist()
        )


def test_update_states_pp_async_multi_request_keeps_rank_state_consistent(
    model_runner, model_runner_2, dist_init, monkeypatch
):
    req_ids = ["req_0", "req_1"]
    non_last_runner = model_runner
    last_runner = model_runner_2
    non_last_runner.use_async_scheduling = True
    last_runner.use_async_scheduling = True

    # Both ranks start from the same request set.
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False, world_size=2),
    )
    non_last_runner._update_states(_schedule_new_request(*req_ids))
    last_runner._update_states(_schedule_new_request(*req_ids))

    # Simulate async previous-step sampled tokens known on both ranks.
    # non-last rank may receive them via PP communication; last rank has
    # them from local sampling/bookkeeping.
    sampled_by_last_rank = {req_ids[0]: 111, req_ids[1]: 222}
    for runner in (non_last_runner, last_runner):
        for req_id, token_id in sampled_by_last_rank.items():
            req_index = runner.input_batch.req_id_to_index[req_id]
            start_idx = int(runner.input_batch.num_tokens_no_spec[req_index])
            end_idx = start_idx + 1
            runner.input_batch.token_ids_cpu[req_index, start_idx:end_idx] = [token_id]
            runner.input_batch.is_token_ids[req_index, start_idx:end_idx] = True
            runner.input_batch.num_tokens_no_spec[req_index] = end_idx
            runner.requests[req_id].output_token_ids.append(token_id)

    scheduler_output = _schedule_cached_requests(
        req_ids=req_ids,
        num_scheduled_tokens={req_ids[0]: 1, req_ids[1]: 1},
        new_token_ids=[],
        num_computed_tokens=[4, 4],
        num_output_tokens=[1, 1],
    )
    # non-last rank: async PP branch (new_token_ids empty).
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False, world_size=2),
    )
    non_last_runner._update_states(scheduler_output)
    # last rank: keep already-bookkept state aligned with scheduler view.
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True, world_size=2),
    )
    last_runner._update_states(scheduler_output)

    for req_id in req_ids:
        non_last_idx = non_last_runner.input_batch.req_id_to_index[req_id]
        last_idx = last_runner.input_batch.req_id_to_index[req_id]
        non_last_len = int(non_last_runner.input_batch.num_tokens_no_spec[non_last_idx])
        last_len = int(last_runner.input_batch.num_tokens_no_spec[last_idx])
        assert non_last_len == last_len
        assert (
            non_last_runner.input_batch.token_ids_cpu[
                non_last_idx, :non_last_len
            ].tolist()
            == last_runner.input_batch.token_ids_cpu[last_idx, :last_len].tolist()
        )


def test_kv_cache_stride_order(monkeypatch, model_runner):
    # This test checks if GPUModelRunner initializes correctly when an attention
    # backend enforces a non-default KV cache stride order.
    n_heads = model_runner.model_config.get_num_kv_heads(model_runner.parallel_config)
    head_size = model_runner.model_config.get_head_size()

    # Get the expected shape from the backend's get_kv_cache_shape method
    # to ensure compatibility with different backends (triton vs flexattention)
    attn_backend = None
    for attn_group in model_runner._attn_group_iterator():
        attn_backend = attn_group.backend
        break

    assert attn_backend is not None, "No attention backend found"
    expected_kv_cache_shape = list(
        attn_backend.get_kv_cache_shape(NUM_BLOCKS, BLOCK_SIZE, n_heads, head_size)
    )

    # TODO mla test
    default_stride = tuple(range(5))
    # Permutation that gets you back to expected kv shape
    for test_stride in ((1, 4, 0, 2, 3), (0, 1, 2, 3, 4)):

        def rnd_stride_order(
            include_num_layers_dimension: bool = False, test_stride=test_stride
        ):
            assert not include_num_layers_dimension
            return test_stride

        # Patch the attention backend class and re-trigger the KV cache creation
        for attn_group in model_runner._attn_group_iterator():
            attn_backend = attn_group.backend
            monkeypatch.setattr(
                attn_backend, "get_kv_cache_stride_order", rnd_stride_order
            )

        model_runner.attn_groups = []
        model_runner.kv_caches = []
        model_runner.initialize_kv_cache(model_runner.kv_cache_config)

        # Shape is unchanged, but layout may differ
        kv_cache_shape = model_runner.kv_caches[0].shape
        assert list(kv_cache_shape) == expected_kv_cache_shape
        if default_stride == test_stride:
            assert all(kv.is_contiguous() for kv in model_runner.kv_caches)
        else:
            assert all(not kv.is_contiguous() for kv in model_runner.kv_caches)


def test_update_config(model_runner):
    # Simple update
    model_runner.update_config({"load_config": {"load_format": "dummy"}})
    assert model_runner.load_config.load_format == "dummy"
    # Raise error on non-existing config
    with pytest.raises(AssertionError):
        model_runner.update_config({"do_not_exist_config": "dummy"})


def test_load_model_weights_inplace(dist_init, model_runner, model_runner_2):
    # In this test, model_runner loads model + weights in one go, while
    # model_runner_2 loads dummy weights first then load real weights inplace
    model_runner.load_model()
    original_load_format = model_runner_2.load_config.load_format
    model_runner_2.update_config({"load_config": {"load_format": "dummy"}})
    model_runner_2.load_model()  # Initial model loading with dummy weights
    assert str(model_runner.get_model().state_dict()) != str(
        model_runner_2.get_model().state_dict()
    )
    model_runner_2.update_config({"load_config": {"load_format": original_load_format}})
    model_runner_2.reload_weights()  # Load real weights inplace
    assert str(model_runner.get_model().state_dict()) == str(
        model_runner_2.get_model().state_dict()
    )


def test_reload_weights_before_load_model(model_runner):
    with pytest.raises(ValueError):
        model_runner.reload_weights()


def test_sample_passes_reordered_draft_probs_to_rejection_sampler():
    runner = object.__new__(GPUModelRunner)
    runner.use_async_scheduling = False
    runner.scheduler_config = SimpleNamespace(enable_parallel_spec_decode=False)
    runner.input_batch = SimpleNamespace(
        sampling_metadata=Mock(spec=SamplingMetadata),
        update_async_output_token_ids=Mock(),
        req_ids=["req_a", "req_b", "req_c"],
    )
    runner.rejection_sampler = Mock(return_value="sampler_output")
    runner.sampler = Mock()
    runner._draft_prob_req_ids = ["req_c", "req_a", "req_b"]
    runner._draft_probs = torch.arange(3 * 3 * 4, dtype=torch.float32).reshape(3, 3, 4)

    spec_decode_metadata = SpecDecodeMetadata.make_dummy(
        [[1, 2], [], [3]],
        device=torch.device("cpu"),
    )
    logits = torch.randn(6, 4)

    output = GPUModelRunner._sample(runner, logits, spec_decode_metadata)

    assert output == "sampler_output"
    passed_draft_probs = runner.rejection_sampler.call_args.args[1]
    expected_draft_probs = torch.cat(
        [
            runner._draft_probs[1, :2],
            runner._draft_probs[0, :1],
        ],
        dim=0,
    )
    assert torch.equal(passed_draft_probs, expected_draft_probs)


def _make_parallel_prob_runner():
    runner = object.__new__(GPUModelRunner)
    runner.scheduler_config = SimpleNamespace(
        enable_parallel_spec_decode=True,
    )
    runner._parallel_pending_draft_states = {}
    runner.requests = {}
    return runner


def test_parallel_draft_capacity_is_bounded_per_target_step():
    runner = object.__new__(GPUModelRunner)
    runner.num_spec_tokens = 3
    runner.max_speculative_tokens = 3

    assert runner._get_parallel_draft_token_capacity(set()) == 1
    assert runner._get_parallel_draft_token_capacity({"draft"}) == 3


def _make_parallel_execution_resource_runner(
    *,
    target_sm_count: int | None,
    draft_sm_count: int | None,
    profiles: tuple[tuple[int, int], ...] | None = None,
):
    runner = object.__new__(GPUModelRunner)
    runner.scheduler_config = SimpleNamespace(
        enable_parallel_spec_decode=True,
        parallel_spec_decode_target_sm_count=target_sm_count,
        parallel_spec_decode_draft_sm_count=draft_sm_count,
        parallel_spec_decode_sm_profiles=profiles,
    )
    runner.compilation_config = SimpleNamespace(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
    )
    runner.device = torch.device("cuda:0")
    runner.speculative_config = SimpleNamespace(uses_draft_model=lambda: True)
    runner.supports_mm_inputs = False
    runner.use_async_scheduling = False
    runner.parallel_config = SimpleNamespace(
        use_ubatching=False,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
    )
    runner._initialize_parallel_draft_worker = Mock()
    return runner


def _mock_parallel_executor(monkeypatch: pytest.MonkeyPatch):
    executor = Mock()
    executor.submit.return_value.result = Mock()
    executor_factory = Mock(return_value=executor)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "ThreadPoolExecutor",
        executor_factory,
    )
    return executor, executor_factory


def test_parallel_execution_resources_keep_ordinary_stream_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _make_parallel_execution_resource_runner(
        target_sm_count=None,
        draft_sm_count=None,
    )
    ordinary_stream = Mock(name="ordinary_stream")
    stream_factory = Mock(return_value=ordinary_stream)
    partition_factory = Mock()
    monkeypatch.setattr(torch.cuda, "Stream", stream_factory)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "create_parallel_spec_decode_sm_partition",
        partition_factory,
        raising=False,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.current_platform,
        "is_cuda",
        lambda: True,
    )
    executor, executor_factory = _mock_parallel_executor(monkeypatch)

    runner._initialize_parallel_execution_resources()

    assert runner._parallel_sm_partition is None
    assert runner._parallel_target_stream is None
    assert runner._parallel_draft_stream is ordinary_stream
    partition_factory.assert_not_called()
    executor_factory.assert_called_once_with(
        max_workers=1,
        thread_name_prefix="vllm-parallel-draft",
    )
    executor.submit.assert_called_once_with(runner._initialize_parallel_draft_worker)
    executor.submit.return_value.result.assert_called_once_with()


def test_parallel_execution_resources_map_green_context_streams(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _make_parallel_execution_resource_runner(
        target_sm_count=64,
        draft_sm_count=20,
    )
    partition = SimpleNamespace(
        target_stream=Mock(name="target_stream"),
        draft_stream=Mock(name="draft_stream"),
        target_sm_count=64,
        draft_sm_count=20,
    )
    partition_factory = Mock(return_value=partition)
    ordinary_stream_factory = Mock()
    monkeypatch.setattr(torch.cuda, "Stream", ordinary_stream_factory)
    monkeypatch.setattr(
        torch.cuda,
        "stream",
        lambda stream: nullcontext(),
    )
    monkeypatch.setattr(torch.cuda, "current_blas_handle", Mock())
    monkeypatch.setattr(
        gpu_model_runner_module,
        "create_parallel_spec_decode_sm_partition",
        partition_factory,
        raising=False,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.current_platform,
        "is_cuda",
        lambda: True,
    )
    _mock_parallel_executor(monkeypatch)

    runner._initialize_parallel_execution_resources()

    partition_factory.assert_called_once_with(
        runner.device,
        target_sm_count=64,
        draft_sm_count=20,
    )
    assert runner._parallel_sm_partition is partition
    assert runner._parallel_target_stream is partition.target_stream
    assert runner._parallel_draft_stream is partition.draft_stream
    ordinary_stream_factory.assert_not_called()


def test_parallel_execution_resources_create_dynamic_profile_pool(
    monkeypatch: pytest.MonkeyPatch,
):
    profiles = ((84, 0), (76, 8), (60, 24))
    runner = _make_parallel_execution_resource_runner(
        target_sm_count=None,
        draft_sm_count=None,
        profiles=profiles,
    )
    profile_states = [
        SimpleNamespace(
            profile_id=0,
            target_stream=Mock(name="target_0"),
            draft_stream=None,
            target_sm_count=84,
            draft_sm_count=0,
        ),
        SimpleNamespace(
            profile_id=1,
            target_stream=Mock(name="target_1"),
            draft_stream=Mock(name="draft_1"),
            target_sm_count=76,
            draft_sm_count=8,
        ),
        SimpleNamespace(
            profile_id=2,
            target_stream=Mock(name="target_2"),
            draft_stream=Mock(name="draft_2"),
            target_sm_count=60,
            draft_sm_count=24,
        ),
    ]
    pool = SimpleNamespace(profiles=tuple(profile_states))
    pool_factory = Mock(return_value=pool)
    stream_contexts = []

    @contextmanager
    def stream_context(stream):
        stream_contexts.append(stream)
        yield

    monkeypatch.setattr(
        gpu_model_runner_module,
        "create_parallel_spec_decode_sm_profile_pool",
        pool_factory,
        raising=False,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.current_platform,
        "is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    blas_handle = Mock()
    monkeypatch.setattr(torch.cuda, "current_blas_handle", blas_handle)
    ordinary_stream_factory = Mock()
    monkeypatch.setattr(torch.cuda, "Stream", ordinary_stream_factory)
    _mock_parallel_executor(monkeypatch)

    runner._initialize_parallel_execution_resources()

    pool_factory.assert_called_once_with(runner.device, profiles)
    assert runner._parallel_sm_profile_pool is pool
    assert runner._parallel_sm_partition is None
    assert runner._parallel_target_stream is None
    assert runner._parallel_draft_stream is None
    assert stream_contexts == [
        profile_states[0].target_stream,
        profile_states[1].target_stream,
        profile_states[2].target_stream,
    ]
    assert blas_handle.call_count == 3
    ordinary_stream_factory.assert_not_called()


def test_parallel_execution_resources_roll_back_pool_on_target_init_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _make_parallel_execution_resource_runner(
        target_sm_count=None,
        draft_sm_count=None,
        profiles=((84, 0), (76, 8)),
    )
    pool = Mock()
    pool.profiles = (
        SimpleNamespace(target_stream=Mock(), draft_stream=None),
        SimpleNamespace(target_stream=Mock(), draft_stream=Mock()),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "create_parallel_spec_decode_sm_profile_pool",
        Mock(return_value=pool),
    )
    monkeypatch.setattr(
        gpu_model_runner_module.current_platform,
        "is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "current_blas_handle",
        Mock(side_effect=RuntimeError("target BLAS init failed")),
    )
    executor_factory = Mock()
    monkeypatch.setattr(
        gpu_model_runner_module,
        "ThreadPoolExecutor",
        executor_factory,
    )

    with pytest.raises(RuntimeError, match="target BLAS init failed"):
        runner._initialize_parallel_execution_resources()

    pool.close.assert_called_once_with()
    executor_factory.assert_not_called()
    assert runner._parallel_sm_profile_pool is None
    assert runner._parallel_sm_profile_lease is None


def test_parallel_execution_resources_roll_back_partition_and_executor(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _make_parallel_execution_resource_runner(
        target_sm_count=64,
        draft_sm_count=20,
    )
    partition = Mock(
        target_stream=Mock(),
        draft_stream=Mock(),
        target_sm_count=64,
        draft_sm_count=20,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "create_parallel_spec_decode_sm_partition",
        Mock(return_value=partition),
    )
    monkeypatch.setattr(
        gpu_model_runner_module.current_platform,
        "is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_blas_handle", Mock())
    executor, _ = _mock_parallel_executor(monkeypatch)
    executor.submit.return_value.result.side_effect = RuntimeError(
        "draft worker init failed"
    )

    with pytest.raises(RuntimeError, match="draft worker init failed"):
        runner._initialize_parallel_execution_resources()

    executor.shutdown.assert_called_once_with(wait=True, cancel_futures=False)
    partition.close.assert_called_once_with()
    assert runner._parallel_draft_executor is None
    assert runner._parallel_sm_partition is None
    assert runner._parallel_target_stream is None
    assert runner._parallel_draft_stream is None


def test_parallel_draft_worker_initializes_every_profile_draft_stream(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cuda:0")
    runner._parallel_draft_stream = None
    draft_1 = Mock(name="draft_1")
    draft_2 = Mock(name="draft_2")
    runner._parallel_sm_profile_pool = SimpleNamespace(
        profiles=(
            SimpleNamespace(draft_stream=None),
            SimpleNamespace(draft_stream=draft_1),
            SimpleNamespace(draft_stream=draft_2),
        )
    )
    initialized_streams = []

    @contextmanager
    def stream_context(stream):
        initialized_streams.append(stream)
        yield

    set_device = Mock()
    blas_handle = Mock()
    monkeypatch.setattr(torch.accelerator, "set_device_index", set_device)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "current_blas_handle", blas_handle)

    runner._initialize_parallel_draft_worker()

    set_device.assert_called_once_with(runner.device)
    assert initialized_streams == [draft_1, draft_2]
    assert blas_handle.call_count == 2


def test_warm_up_parallel_sm_profiles_runs_each_profile_sequentially(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = object.__new__(GPUModelRunner)
    target_0 = Mock(name="target_0")
    target_1 = Mock(name="target_1")
    draft_1 = Mock(name="draft_1")
    profiles = (
        SimpleNamespace(
            profile_id=0,
            target_stream=target_0,
            draft_stream=None,
        ),
        SimpleNamespace(
            profile_id=1,
            target_stream=target_1,
            draft_stream=draft_1,
        ),
    )
    runner._parallel_sm_profile_pool = SimpleNamespace(profiles=profiles)
    runner._parallel_draft_workspace_slot = 1
    runner._dummy_run = Mock()
    runner.drafter = SimpleNamespace(dummy_run=Mock())
    events = []

    @contextmanager
    def stream_context(stream):
        events.append(("enter", stream))
        yield
        events.append(("exit", stream))

    class ImmediateFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    def submit(fn, *args):
        return ImmediateFuture(fn(*args))

    runner._parallel_draft_executor = SimpleNamespace(submit=Mock(side_effect=submit))
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "Event", lambda: Mock())
    monkeypatch.setattr(
        gpu_model_runner_module,
        "use_workspace_slot",
        lambda _slot: nullcontext(),
    )

    runner.warm_up_parallel_sm_profiles()

    assert runner._dummy_run.call_count == 2
    for call in runner._dummy_run.call_args_list:
        assert call.args == (1,)
        assert call.kwargs["skip_eplb"] is True
        assert call.kwargs["skip_parallel_drafter_warmup"] is True
    runner.drafter.dummy_run.assert_called_once_with(
        1,
        use_cudagraphs=False,
    )
    assert events == [
        ("enter", target_0),
        ("exit", target_0),
        ("enter", target_1),
        ("exit", target_1),
        ("enter", draft_1),
        ("exit", draft_1),
        ("enter", target_1),
        ("exit", target_1),
    ]
    target_0.synchronize.assert_called_once_with()
    target_1.wait_event.assert_called_once()
    target_1.synchronize.assert_called_once_with()


def test_shutdown_parallel_execution_resources_is_ordered_and_idempotent():
    runner = object.__new__(GPUModelRunner)
    calls = []
    future = Mock()
    future.result.side_effect = lambda: calls.append("future")
    runner._optimistic_parallel_draft_state = SimpleNamespace(
        draft_launch_future=future
    )
    executor = Mock()
    executor.shutdown.side_effect = lambda **_kwargs: calls.append("executor")
    runner._parallel_draft_executor = executor
    pool = Mock()
    pool.close.side_effect = lambda: calls.append("pool")
    runner._parallel_sm_profile_pool = pool
    runner._parallel_sm_partition = None

    runner._shutdown_parallel_execution_resources()
    runner._shutdown_parallel_execution_resources()

    assert calls == ["future", "executor", "pool"]
    assert runner._parallel_draft_executor is None
    assert runner._parallel_sm_profile_pool is None


def test_shutdown_parallel_execution_resources_cleans_up_after_draft_failure():
    runner = object.__new__(GPUModelRunner)
    calls = []
    future = Mock()

    def fail_future():
        calls.append("future")
        raise RuntimeError("draft failed")

    future.result.side_effect = fail_future
    runner._optimistic_parallel_draft_state = SimpleNamespace(
        draft_launch_future=future
    )
    executor = Mock()
    executor.shutdown.side_effect = lambda **_kwargs: calls.append("executor")
    runner._parallel_draft_executor = executor
    pool = Mock()
    pool.close.side_effect = lambda: calls.append("pool")
    runner._parallel_sm_profile_pool = pool
    runner._parallel_sm_partition = None
    runner._parallel_target_stream = None
    runner._parallel_draft_stream = None
    runner._parallel_sm_profile_lease = Mock()

    with pytest.raises(RuntimeError, match="draft failed"):
        runner._shutdown_parallel_execution_resources()

    assert calls == ["future", "executor", "pool"]
    assert runner._optimistic_parallel_draft_state is None
    assert runner._parallel_draft_executor is None
    assert runner._parallel_sm_profile_pool is None
    assert runner._parallel_sm_profile_lease is None


def test_shutdown_continues_model_cleanup_after_parallel_draft_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import rotary_embedding
    from vllm.v1.worker import workspace

    runner = object.__new__(GPUModelRunner)
    calls = []

    def fail_parallel_cleanup():
        calls.append("parallel")
        raise RuntimeError("draft failed")

    runner._shutdown_parallel_execution_resources = Mock(
        side_effect=fail_parallel_cleanup
    )
    runner._cleanup_profiling_kv_cache = Mock(side_effect=lambda: calls.append("kv"))
    static_forward_context = Mock()
    static_forward_context.clear.side_effect = lambda: calls.append("static")
    runner.compilation_config = SimpleNamespace(
        static_forward_context=static_forward_context
    )
    runner.model = object()
    rope_cache = Mock()
    rope_cache.clear.side_effect = lambda: calls.append("rope")
    monkeypatch.setattr(rotary_embedding, "_ROPE_DICT", rope_cache)
    monkeypatch.setattr(
        workspace,
        "reset_workspace_manager",
        lambda: calls.append("workspace"),
    )
    monkeypatch.setattr(
        gpu_model_runner_module.current_platform,
        "is_rocm",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="draft failed"):
        runner.shutdown()

    assert calls == ["parallel", "kv", "static", "rope", "workspace"]
    assert runner.model is None


def test_parallel_execution_resources_require_eager_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _make_parallel_execution_resource_runner(
        target_sm_count=64,
        draft_sm_count=20,
    )
    runner.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    monkeypatch.setattr(
        gpu_model_runner_module.current_platform,
        "is_cuda",
        lambda: True,
    )
    with pytest.raises(
        RuntimeError,
        match="requires enforce_eager=True",
    ):
        runner._initialize_parallel_execution_resources()


def test_parallel_execution_resources_reject_torch_compile_without_cudagraph(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _make_parallel_execution_resource_runner(
        target_sm_count=64,
        draft_sm_count=20,
    )
    runner.compilation_config.mode = CompilationMode.VLLM_COMPILE
    monkeypatch.setattr(
        gpu_model_runner_module.current_platform,
        "is_cuda",
        lambda: True,
    )
    partition = SimpleNamespace(
        target_stream=Mock(),
        draft_stream=Mock(),
        target_sm_count=64,
        draft_sm_count=20,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "create_parallel_spec_decode_sm_partition",
        Mock(return_value=partition),
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_blas_handle", Mock())
    _mock_parallel_executor(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="requires enforce_eager=True",
    ):
        runner._initialize_parallel_execution_resources()


def test_parallel_execution_resources_require_cuda(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _make_parallel_execution_resource_runner(
        target_sm_count=64,
        draft_sm_count=20,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.current_platform,
        "is_cuda",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="requires CUDA"):
        runner._initialize_parallel_execution_resources()


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("use_async_scheduling", True),
        ("use_ubatching", True),
        ("tensor_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("data_parallel_size", 2),
        ("decode_context_parallel_size", 2),
    ],
)
def test_parallel_execution_resources_reject_unsupported_topology(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value,
):
    runner = _make_parallel_execution_resource_runner(
        target_sm_count=64,
        draft_sm_count=20,
    )
    if attribute == "use_async_scheduling":
        runner.use_async_scheduling = value
    else:
        setattr(runner.parallel_config, attribute, value)
    monkeypatch.setattr(
        gpu_model_runner_module.current_platform,
        "is_cuda",
        lambda: True,
    )

    with pytest.raises(RuntimeError, match="unsupported parallel topology"):
        runner._initialize_parallel_execution_resources()


def test_parallel_draft_compact_scatter_empty_subbatches_create_no_fence(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = object.__new__(GPUModelRunner)
    runner._supports_optimistic_parallel_draft = Mock(return_value=True)
    runner._build_parallel_draft_subbatches = Mock(return_value=(None, None))
    runner._parallel_pending_draft_states = {}
    runner._parallel_draft_generators = {}
    runner.input_batch = SimpleNamespace(req_ids=["skip"])
    runner.requests = {}
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        pending_spec_decode_tokens={},
    )
    event_factory = Mock()
    monkeypatch.setattr(torch.cuda, "Event", event_factory)

    state = runner._prepare_optimistic_parallel_draft(
        scheduler_output,
        Mock(),
        num_scheduled_tokens=1,
    )

    assert state is None
    event_factory.assert_not_called()


def test_parallel_draft_target_start_precedes_inputs_ready(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = object.__new__(GPUModelRunner)
    runner._supports_optimistic_parallel_draft = Mock(return_value=True)
    runner._supports_parallel_draft_sampling_subbatch = Mock(return_value=True)
    runner._snapshot_parallel_draft_paths = Mock(
        return_value=({"draft": []}, {"draft": [10]})
    )
    extension = SimpleNamespace(
        req_ids=["draft"],
        source_row_indices=[0],
        common_attn_metadata=SimpleNamespace(max_seq_len=14),
    )
    runner._build_parallel_draft_subbatches = Mock(return_value=(None, extension))
    runner._get_parallel_draft_token_capacity = Mock(return_value=5)
    runner._build_parallel_draft_sampling_metadata = Mock(return_value=Mock())
    runner._parallel_pending_draft_states = {}
    runner._parallel_sm_profile_lease = None
    runner.effective_drafter_max_model_len = 16
    runner.input_batch = SimpleNamespace(req_ids=["draft"])

    record_order = []
    events = [Mock() for _ in range(4)]
    events[0].record.side_effect = lambda: record_order.append("inputs_ready")
    events[1].record.side_effect = lambda: record_order.append("target_start")
    monkeypatch.setattr(torch.cuda, "Event", Mock(side_effect=events))

    state = runner._prepare_optimistic_parallel_draft(
        SimpleNamespace(),
        Mock(),
        num_scheduled_tokens=1,
    )

    assert state is not None
    assert record_order == ["target_start", "inputs_ready"]
    # The model-length bound reduces the configured per-step cap before launch.
    assert state.draft_token_capacity == 2
    assert state.num_draft_tokens == 0


@pytest.mark.parametrize(
    ("logitsprocs", "holder", "expected"),
    [
        (
            SimpleNamespace(
                argmax_invariant=[],
                non_argmax_invariant=[SimpleNamespace(min_toks={})],
            ),
            None,
            True,
        ),
        (
            SimpleNamespace(
                argmax_invariant=[],
                non_argmax_invariant=[SimpleNamespace(min_toks={0: object()})],
            ),
            None,
            False,
        ),
        (
            SimpleNamespace(argmax_invariant=[], non_argmax_invariant=[]),
            SimpleNamespace(has_tracked_requests=lambda: True),
            False,
        ),
    ],
)
def test_parallel_draft_subbatch_rejects_stateful_sampling_metadata(
    logitsprocs,
    holder,
    expected: bool,
):
    runner = object.__new__(GPUModelRunner)
    runner.input_batch = SimpleNamespace(
        sampling_metadata=SimpleNamespace(
            logitsprocs=logitsprocs,
            thinking_budget_state_holder=holder,
        )
    )

    assert runner._supports_parallel_draft_sampling_subbatch() is expected


def _make_parallel_draft_subbatch_runner():
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.drafter = SimpleNamespace(block_size=4)
    runner.num_spec_tokens = 2
    runner.scheduler_config = SimpleNamespace(
        parallel_spec_decode_draft_token_budget=100,
        max_num_batched_tokens=100,
    )
    runner.input_batch = SimpleNamespace(
        req_ids=["prefill", "ready-draft", "lagging-draft", "skip"],
        block_table=[SimpleNamespace(num_blocks_per_row=np.array([2, 2, 2, 2]))],
    )
    runner.requests = {
        "prefill": SimpleNamespace(
            prompt_token_ids=[10, 11, 12, 13], output_token_ids=[]
        ),
        "ready-draft": SimpleNamespace(
            prompt_token_ids=[20, 21], output_token_ids=[22, 23]
        ),
        "lagging-draft": SimpleNamespace(
            prompt_token_ids=[30, 31], output_token_ids=[32, 33]
        ),
        "skip": SimpleNamespace(prompt_token_ids=[40, 41], output_token_ids=[42]),
    }
    runner._parallel_draft_kv_states = {
        "prefill": DraftKVPathState([10, 11, 12]),
        # The cached path need only cover draft_input_path[:-1].
        "ready-draft": DraftKVPathState([20, 21, 22, 23]),
        "lagging-draft": DraftKVPathState([30, 31, 99, 100]),
    }

    block_table = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32)
    query_start_loc_cpu = torch.tensor([0, 2, 3, 4, 5], dtype=torch.int32)
    positions = torch.tensor([2, 3, 3, 3, 2], dtype=torch.int64)
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc_cpu.clone(),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=torch.tensor([4, 4, 4, 3], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([4, 4, 4, 3], dtype=torch.int32),
        _num_computed_tokens_cpu=torch.tensor([2, 3, 3, 2], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([4, 4, 4, 3], dtype=torch.int32),
        num_reqs=4,
        num_actual_tokens=5,
        max_query_len=2,
        max_seq_len=4,
        block_table_tensor=block_table,
        slot_mapping=torch.tensor([6, 7, 15, 23, 30], dtype=torch.int64),
        positions=positions,
        is_prefilling=torch.tensor([True, False, False, False]),
    )
    scheduler_output = SimpleNamespace(
        pending_spec_decode_tokens={
            "ready-draft": [24],
            "lagging-draft": [34],
        },
        draft_enabled_req_ids={"prefill", "ready-draft", "lagging-draft"},
        draft_kv_maintenance_req_ids={"prefill"},
    )
    return runner, scheduler_output, common_attn_metadata


def test_parallel_draft_subbatch_requires_input_path_snapshot():
    runner, scheduler_output, target_metadata = _make_parallel_draft_subbatch_runner()

    with pytest.raises(TypeError, match="draft_input_paths"):
        runner._build_parallel_draft_subbatches(
            scheduler_output,
            target_metadata,
            target_metadata.num_actual_tokens,
        )


def _build_parallel_draft_subbatches_from_snapshot(
    runner: GPUModelRunner,
    scheduler_output,
    target_metadata: CommonAttentionMetadata,
):
    _, draft_input_paths = runner._snapshot_parallel_draft_paths(scheduler_output)
    return runner._build_parallel_draft_subbatches(
        scheduler_output,
        target_metadata,
        target_metadata.num_actual_tokens,
        draft_input_paths,
    )


def test_parallel_draft_common_attention_metadata_fields_are_audited():
    rebuilt_fields = {
        "query_start_loc",
        "query_start_loc_cpu",
        "seq_lens",
        "num_reqs",
        "num_actual_tokens",
        "max_query_len",
        "max_seq_len",
        "block_table_tensor",
        "slot_mapping",
        "positions",
        "seq_lens_cpu_upper_bound",
        "_seq_lens_cpu",
        "_num_computed_tokens_cpu",
        "_num_computed_tokens_cache",
    }
    gathered_fields = {
        "encoder_seq_lens",
        "encoder_seq_lens_cpu",
        "dcp_local_seq_lens",
        "dcp_local_seq_lens_cpu",
        "is_prefilling",
    }
    # Fast-prefill indices address the Target packed tensor and are rejected
    # before compact Draft metadata is built.
    rejected_fields = {"logits_indices_padded", "num_logits_indices"}
    safely_shared_fields = {"causal"}
    classifications = (
        rebuilt_fields,
        gathered_fields,
        rejected_fields,
        safely_shared_fields,
    )
    audited_fields = set().union(*classifications)

    assert sum(map(len, classifications)) == len(audited_fields)
    assert {field.name for field in fields(CommonAttentionMetadata)} == audited_fields


def test_parallel_draft_role_classification_and_compact_metadata():
    runner, scheduler_output, target_metadata = _make_parallel_draft_subbatch_runner()

    maintenance, extension = _build_parallel_draft_subbatches_from_snapshot(
        runner,
        scheduler_output,
        target_metadata,
    )

    assert maintenance is not None
    assert maintenance.req_ids == ["prefill"]
    assert maintenance.source_row_indices == [0]
    assert torch.equal(maintenance.token_ids, torch.tensor([13]))
    assert torch.equal(maintenance.positions, torch.tensor([3]))
    maintenance_metadata = maintenance.common_attn_metadata
    assert torch.equal(
        maintenance_metadata.query_start_loc_cpu,
        torch.tensor([0, 1], dtype=torch.int32),
    )
    assert torch.equal(
        maintenance_metadata.query_start_loc,
        torch.tensor([0, 1], dtype=torch.int32),
    )
    assert torch.equal(
        maintenance_metadata.seq_lens,
        torch.tensor([4], dtype=torch.int32),
    )
    assert torch.equal(
        maintenance_metadata.block_table_tensor,
        torch.tensor([[1, 2]], dtype=torch.int32),
    )
    assert torch.equal(
        maintenance_metadata.slot_mapping,
        torch.tensor([7]),
    )

    assert extension is not None
    assert extension.req_ids == ["ready-draft", "lagging-draft"]
    assert extension.source_row_indices == [1, 2]
    assert torch.equal(extension.token_ids, torch.tensor([24, 32, 33, 34]))
    assert torch.equal(extension.positions, torch.tensor([4, 2, 3, 4]))
    extension_metadata = extension.common_attn_metadata
    assert extension_metadata.block_table_tensor.shape[0] == 2
    assert torch.equal(
        extension_metadata.block_table_tensor,
        torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
    )
    assert torch.equal(
        extension_metadata.query_start_loc_cpu,
        torch.tensor([0, 1, 4], dtype=torch.int32),
    )
    assert torch.equal(
        extension_metadata.seq_lens, torch.tensor([5, 5], dtype=torch.int32)
    )
    assert torch.equal(extension_metadata.slot_mapping, torch.tensor([16, 22, 23, 24]))
    assert "skip" not in maintenance.req_ids + extension.req_ids


def test_parallel_draft_budget_prioritizes_shortest_catchup():
    runner, scheduler_output, target_metadata = _make_parallel_draft_subbatch_runner()
    runner.scheduler_config.parallel_spec_decode_draft_token_budget = 1

    maintenance, extension = _build_parallel_draft_subbatches_from_snapshot(
        runner,
        scheduler_output,
        target_metadata,
    )

    # ready-draft has zero catch-up and consumes the complete first-round budget,
    # even though prefill appears earlier in the Target batch.
    assert maintenance is None
    assert extension is not None
    assert extension.req_ids == ["ready-draft"]


def test_parallel_draft_budget_allows_partial_catchup_progress():
    runner, scheduler_output, target_metadata = _make_parallel_draft_subbatch_runner()
    runner.scheduler_config.parallel_spec_decode_draft_token_budget = 4

    maintenance, extension = _build_parallel_draft_subbatches_from_snapshot(
        runner,
        scheduler_output,
        target_metadata,
    )

    # First-round cost: ready extension=1, prefill catch-up=1. The remaining
    # two tokens partially advance lagging-draft, whose full fused cost is 3.
    assert maintenance is not None
    assert maintenance.req_ids == ["prefill", "lagging-draft"]
    assert torch.equal(maintenance.token_ids, torch.tensor([13, 32, 33]))
    assert extension is not None
    assert extension.req_ids == ["ready-draft"]


def test_parallel_draft_subbatch_uses_single_input_path_snapshot():
    runner, scheduler_output, target_metadata = _make_parallel_draft_subbatch_runner()
    pending_token_ids, draft_input_paths = runner._snapshot_parallel_draft_paths(
        scheduler_output
    )
    scheduler_output.pending_spec_decode_tokens["ready-draft"] = [99]
    runner.requests["ready-draft"].output_token_ids.append(98)

    _, extension = runner._build_parallel_draft_subbatches(
        scheduler_output,
        target_metadata,
        target_metadata.num_actual_tokens,
        draft_input_paths,
    )

    assert pending_token_ids["ready-draft"] == [24]
    assert draft_input_paths["ready-draft"] == [20, 21, 22, 23, 24]
    assert extension is not None
    assert extension.req_ids[0] == "ready-draft"
    assert extension.token_ids[0].item() == 24


def test_parallel_draft_subbatch_respects_drafter_token_capacity():
    runner, scheduler_output, target_metadata = _make_parallel_draft_subbatch_runner()
    runner.drafter.max_num_tokens = 3

    maintenance, extension = _build_parallel_draft_subbatches_from_snapshot(
        runner,
        scheduler_output,
        target_metadata,
    )

    assert maintenance is not None
    assert maintenance.req_ids == ["prefill", "lagging-draft"]
    assert torch.equal(maintenance.token_ids, torch.tensor([13, 32, 33]))
    assert torch.equal(maintenance.positions, torch.tensor([3, 2, 3]))
    assert torch.equal(
        maintenance.common_attn_metadata.query_start_loc_cpu,
        torch.tensor([0, 1, 3], dtype=torch.int32),
    )
    assert extension is not None
    assert extension.req_ids == ["ready-draft"]
    assert torch.equal(extension.token_ids, torch.tensor([24]))


def test_parallel_draft_subbatch_does_not_share_target_metadata_tensors():
    runner, scheduler_output, target_metadata = _make_parallel_draft_subbatch_runner()
    target_seq_lens = target_metadata.seq_lens.clone()
    target_block_table = target_metadata.block_table_tensor.clone()

    maintenance, extension = _build_parallel_draft_subbatches_from_snapshot(
        runner,
        scheduler_output,
        target_metadata,
    )

    assert maintenance is not None
    assert extension is not None
    for subbatch in (maintenance, extension):
        metadata = subbatch.common_attn_metadata
        assert metadata.seq_lens.data_ptr() != target_metadata.seq_lens.data_ptr()
        assert (
            metadata.block_table_tensor.data_ptr()
            != target_metadata.block_table_tensor.data_ptr()
        )
        assert (
            metadata.slot_mapping.data_ptr() != target_metadata.slot_mapping.data_ptr()
        )

    maintenance.common_attn_metadata.seq_lens.add_(10)
    maintenance.common_attn_metadata.block_table_tensor.fill_(0)
    assert torch.equal(target_metadata.seq_lens, target_seq_lens)
    assert torch.equal(target_metadata.block_table_tensor, target_block_table)


def test_parallel_draft_subbatch_fast_prefill_indices_fail_closed():
    runner, scheduler_output, target_metadata = _make_parallel_draft_subbatch_runner()
    target_metadata.logits_indices_padded = torch.tensor([1, 2, 4, 4])
    target_metadata.num_logits_indices = 3

    maintenance, extension = _build_parallel_draft_subbatches_from_snapshot(
        runner,
        scheduler_output,
        target_metadata,
    )

    # Compaction would remove full-batch rows 1 and 3 while packing multi-token
    # maintenance rows, so the original packed-token indices are not reusable.
    assert maintenance is None
    assert extension is None
    assert torch.equal(
        target_metadata.logits_indices_padded, torch.tensor([1, 2, 4, 4])
    )
    assert target_metadata.num_logits_indices == 3


def test_parallel_draft_subbatch_leaf_rejects_fast_prefill_metadata():
    runner, _, target_metadata = _make_parallel_draft_subbatch_runner()
    target_metadata.logits_indices_padded = torch.tensor([0])
    target_metadata.num_logits_indices = 1

    with pytest.raises(AssertionError, match="Fast-prefill"):
        runner._build_parallel_draft_subbatch(
            [ParallelDraftRow("ready-draft", 1, [24], [4])],
            target_metadata,
        )


def test_parallel_draft_role_missing_state_fuses_catchup_into_first_extension():
    runner, scheduler_output, target_metadata = _make_parallel_draft_subbatch_runner()
    runner._parallel_draft_kv_states.pop("ready-draft")

    maintenance, extension = _build_parallel_draft_subbatches_from_snapshot(
        runner,
        scheduler_output,
        target_metadata,
    )

    assert maintenance is not None
    assert maintenance.req_ids == ["prefill"]
    assert extension is not None
    assert extension.req_ids == ["lagging-draft", "ready-draft"]
    assert torch.equal(
        maintenance.token_ids,
        torch.tensor([13]),
    )
    assert torch.equal(
        extension.token_ids,
        torch.tensor([32, 33, 34, 20, 21, 22, 23, 24]),
    )
    assert runner._parallel_draft_kv_states["ready-draft"].token_ids == []


def test_parallel_draft_role_stale_future_path_remains_extension_ready():
    runner, scheduler_output, target_metadata = _make_parallel_draft_subbatch_runner()
    runner._parallel_draft_kv_states["ready-draft"] = DraftKVPathState(
        [20, 21, 22, 23, 90, 91]
    )

    maintenance, extension = _build_parallel_draft_subbatches_from_snapshot(
        runner,
        scheduler_output,
        target_metadata,
    )

    assert maintenance is not None
    assert extension is not None
    assert extension.req_ids == ["ready-draft", "lagging-draft"]
    assert torch.equal(extension.token_ids, torch.tensor([24, 32, 33, 34]))
    assert torch.equal(extension.positions, torch.tensor([4, 2, 3, 4]))


def test_parallel_draft_subbatch_rejects_unallocated_block_position():
    runner, scheduler_output, target_metadata = _make_parallel_draft_subbatch_runner()
    runner.input_batch.block_table[0].num_blocks_per_row[1] = 1

    with pytest.raises(AssertionError, match="allocated block-table row"):
        _build_parallel_draft_subbatches_from_snapshot(
            runner,
            scheduler_output,
            target_metadata,
        )


def test_parallel_draft_subbatch_maps_valid_hybrid_kernel_block_slot():
    runner, scheduler_output, target_metadata = _make_parallel_draft_subbatch_runner()
    runner.input_batch.block_table[0] = SimpleNamespace(
        block_size=4,
        blocks_per_kv_block=2,
        num_blocks_per_row=np.array([2, 2, 2, 2]),
    )
    target_metadata.block_table_tensor[1] = torch.tensor([30, 31])

    _, extension = _build_parallel_draft_subbatches_from_snapshot(
        runner,
        scheduler_output,
        target_metadata,
    )

    assert extension is not None
    # Position 4 uses kernel block 1 at offset 0, independent of the larger
    # allocation block represented by blocks_per_kv_block.
    assert extension.common_attn_metadata.slot_mapping[0].item() == 124


def test_parallel_draft_role_state_cleanup_for_kv_lifecycle_changes():
    runner = object.__new__(GPUModelRunner)
    runner.drafter = SimpleNamespace(block_size=4)
    runner._parallel_draft_kv_states = {
        req_id: DraftKVPathState([1, 2, 3, 4])
        for req_id in ("finished", "preempted", "resumed", "future-block")
    }
    runner._parallel_draft_kv_states["reset-block"] = DraftKVPathState([1, 2, 3, 4, 5])
    runner._parallel_pending_draft_states = {}
    runner._parallel_draft_generators = {}
    runner.requests = {
        "reset-block": SimpleNamespace(block_ids=([10, 11],)),
        "future-block": SimpleNamespace(block_ids=([20, 21],)),
    }
    scheduler_output = SimpleNamespace(
        preempted_req_ids={"preempted"},
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids={"resumed"}),
        new_block_ids_to_zero=[11, 21],
    )

    runner._clear_finished_parallel_draft_states({"finished"})
    runner._clear_parallel_draft_kv_states(scheduler_output)

    assert set(runner._parallel_draft_kv_states) == {"future-block"}


def test_update_states_clears_finished_parallel_draft_state_first():
    runner = object.__new__(GPUModelRunner)
    runner._clear_finished_parallel_draft_states = Mock()
    runner._clear_parallel_draft_kv_states = Mock(
        side_effect=RuntimeError("stop after lifecycle cleanup")
    )
    scheduler_output = SimpleNamespace(finished_req_ids={"finished"})

    with pytest.raises(RuntimeError, match="stop after lifecycle cleanup"):
        runner._update_states(scheduler_output)

    runner._clear_finished_parallel_draft_states.assert_called_once_with({"finished"})


def test_finished_parallel_draft_state_cleanup_clears_all_owned_state():
    runner = object.__new__(GPUModelRunner)
    runner._parallel_draft_kv_states = {
        "finished": DraftKVPathState([1]),
        "live": DraftKVPathState([2]),
    }
    runner._parallel_pending_draft_states = {
        "finished": SimpleNamespace(),
        "live": SimpleNamespace(),
    }
    runner._parallel_draft_generators = {
        "finished": object(),
        "live": object(),
    }

    runner._clear_finished_parallel_draft_states({"finished"})

    assert set(runner._parallel_draft_kv_states) == {"live"}
    assert set(runner._parallel_pending_draft_states) == {"live"}
    assert set(runner._parallel_draft_generators) == {"live"}


def test_parallel_draft_role_state_cleanup_uses_allocation_block_size():
    runner = object.__new__(GPUModelRunner)
    runner.drafter = SimpleNamespace(block_size=4)
    runner.input_batch = SimpleNamespace(
        block_table=[SimpleNamespace(block_size=4, blocks_per_kv_block=2)]
    )
    runner._parallel_draft_kv_states = {"hybrid": DraftKVPathState([1, 2, 3, 4, 5])}
    runner.requests = {
        "hybrid": SimpleNamespace(block_ids=([10, 11],)),
    }
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        new_block_ids_to_zero=[11],
    )

    runner._clear_parallel_draft_kv_states(scheduler_output)

    assert "hybrid" in runner._parallel_draft_kv_states


def test_parallel_draft_compact_scatter_launches_maintenance_then_extension(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = object.__new__(GPUModelRunner)
    runner._parallel_draft_stream = Mock()
    runner._parallel_draft_workspace_slot = 1
    runner.drafter = Mock()
    call_order = []
    runner.drafter.advance_kv_cache.side_effect = lambda **kwargs: call_order.append(
        "maintenance"
    )
    runner.drafter.propose_from_prefix.return_value = torch.tensor([[1, 2]])
    runner.drafter.propose_from_prefix.side_effect = lambda **kwargs: (
        call_order.append("extension") or torch.tensor([[1, 2]])
    )
    runner.drafter.take_last_draft_probs.side_effect = lambda: (
        call_order.append("probs") or None
    )
    runner._build_parallel_draft_sampling_metadata = Mock()
    draft_sampling_metadata = Mock()
    first_round_start_event = Mock()
    done_event = Mock()
    done_event.record.side_effect = lambda: call_order.append("done")
    monkeypatch.setattr(gpu_model_runner_module, "DraftModelProposer", object)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    @contextmanager
    def use_workspace_slot(slot: int):
        call_order.append(f"workspace-{slot}-enter")
        try:
            yield
        finally:
            call_order.append(f"workspace-{slot}-exit")

    monkeypatch.setattr(
        gpu_model_runner_module,
        "use_workspace_slot",
        use_workspace_slot,
        raising=False,
    )
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda **kwargs: next(iter_events),
    )
    iter_events = iter((first_round_start_event, done_event))
    maintenance = SimpleNamespace(
        token_ids=torch.tensor([10, 11, 12]),
        positions=torch.tensor([0, 1, 2]),
        common_attn_metadata=Mock(),
    )
    extension = SimpleNamespace(
        req_ids=["draft-b", "draft-a"],
        source_row_indices=[1, 2],
        token_ids=torch.tensor([20, 30]),
        positions=torch.tensor([3, 4]),
        common_attn_metadata=Mock(),
    )
    state = SimpleNamespace(
        inputs_ready_event=Mock(),
        draft_start_event=Mock(),
        maintenance=maintenance,
        extension=extension,
        extension_req_ids=extension.req_ids,
        draft_sampling_metadata=draft_sampling_metadata,
        draft_token_capacity=17,
        num_draft_tokens=0,
        draft_token_ids=None,
        draft_probs=None,
        draft_done_event=None,
        target_predicted_ms=10.0,
    )

    runner._launch_optimistic_parallel_draft(state)

    assert call_order == [
        "workspace-1-enter",
        "maintenance",
        "extension",
        "probs",
        "done",
        "workspace-1-exit",
    ]
    runner.drafter.advance_kv_cache.assert_called_once_with(
        target_token_ids=maintenance.token_ids,
        target_positions=maintenance.positions,
        common_attn_metadata=maintenance.common_attn_metadata,
    )
    runner._build_parallel_draft_sampling_metadata.assert_not_called()
    propose_kwargs = runner.drafter.propose_from_prefix.call_args.kwargs
    assert propose_kwargs["target_token_ids"] is extension.token_ids
    assert propose_kwargs["target_positions"] is extension.positions
    assert propose_kwargs["common_attn_metadata"] is extension.common_attn_metadata
    assert propose_kwargs["sampling_metadata"] is draft_sampling_metadata
    assert propose_kwargs["max_tokens_to_generate"] == 17
    assert isinstance(propose_kwargs["round_deadline"], DraftRoundDeadline)
    assert propose_kwargs["round_deadline"].remaining_ms == 10.0
    assert propose_kwargs["first_round_start_event"] is first_round_start_event
    # The proposer stops according to its per-round timing decision.
    assert state.num_draft_tokens == 2
    state.draft_start_event.record.assert_called_once_with()
    first_round_start_event.record.assert_called_once_with()
    done_event.record.assert_called_once_with()


def test_parallel_draft_submission_uses_persistent_executor():
    runner = object.__new__(GPUModelRunner)
    runner._parallel_draft_executor = Mock()
    launch_future = Mock()
    runner._parallel_draft_executor.submit.return_value = launch_future
    state = SimpleNamespace(draft_launch_future=None)

    runner._submit_optimistic_parallel_draft(state)

    runner._parallel_draft_executor.submit.assert_called_once_with(
        runner._launch_optimistic_parallel_draft,
        state,
    )
    assert state.draft_launch_future is launch_future


@pytest.mark.parametrize(
    ("target_arch", "draft_arch"),
    [
        ("Qwen3ForCausalLM", "Qwen3ForCausalLM"),
        ("LlamaForCausalLM", "Qwen3ForCausalLM"),
        ("Qwen3ForCausalLM", "LlamaForCausalLM"),
        ("Qwen3MoeForCausalLM", "Qwen3ForCausalLM"),
    ],
)
def test_parallel_draft_concurrent_submission_is_model_agnostic(
    target_arch: str,
    draft_arch: str,
):
    runner = object.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(architecture=target_arch)
    runner.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(architecture=draft_arch)
    )
    runner._parallel_draft_executor = Mock()
    runner._parallel_draft_workspace_slot = 1

    assert runner._supports_concurrent_parallel_draft_submission()


def test_parallel_draft_missing_launch_future_falls_back_after_target():
    runner = object.__new__(GPUModelRunner)
    runner._launch_optimistic_parallel_draft = Mock()
    state = SimpleNamespace(draft_launch_future=None)

    runner._launch_deferred_optimistic_parallel_draft(state)

    runner._launch_optimistic_parallel_draft.assert_called_once_with(state)


def test_parallel_draft_concurrently_submitted_model_is_not_launched_twice():
    runner = object.__new__(GPUModelRunner)
    runner._launch_optimistic_parallel_draft = Mock()
    state = SimpleNamespace(draft_launch_future=Mock())

    runner._launch_deferred_optimistic_parallel_draft(state)

    runner._launch_optimistic_parallel_draft.assert_not_called()


def test_parallel_draft_submission_exception_propagates_before_gpu_wait():
    runner = object.__new__(GPUModelRunner)
    launch_future = Mock()
    launch_future.result.side_effect = RuntimeError("draft submission failed")
    state = SimpleNamespace(
        draft_launch_future=launch_future,
        draft_done_event=None,
    )

    with pytest.raises(RuntimeError, match="draft submission failed"):
        runner._observe_optimistic_parallel_draft_latency(state)


def test_parallel_draft_shutdown_closes_executor():
    runner = object.__new__(GPUModelRunner)
    executor = Mock()
    runner._parallel_draft_executor = executor

    runner._shutdown_parallel_draft_executor()

    executor.shutdown.assert_called_once_with(
        wait=True,
        cancel_futures=False,
    )
    assert runner._parallel_draft_executor is None


def _make_parallel_draft_finish_runner():
    runner = _make_parallel_prob_runner()
    runner.speculative_config = SimpleNamespace(
        rejection_sample_method="standard",
        draft_sample_method="probabilistic",
    )
    random_params = SimpleNamespace(sampling_type=SamplingType.RANDOM_SEED)
    runner.requests = {
        "skip": SimpleNamespace(
            prompt_token_ids=[300],
            output_token_ids=[301, 302],
            sampling_params=SimpleNamespace(sampling_type=SamplingType.GREEDY),
        ),
        "draft-b": SimpleNamespace(
            prompt_token_ids=[200],
            output_token_ids=[201, 202],
            sampling_params=random_params,
        ),
        "draft-a": SimpleNamespace(
            prompt_token_ids=[100, 101],
            output_token_ids=[102, 103],
            sampling_params=random_params,
        ),
    }
    runner.input_batch = SimpleNamespace(req_ids=["skip", "draft-b", "draft-a"])
    runner._parallel_draft_kv_states = {
        "skip": DraftKVPathState([300]),
        "draft-b": DraftKVPathState([200, 201]),
        "draft-a": DraftKVPathState([100, 101, 102]),
    }
    runner._observe_optimistic_parallel_draft_latency = Mock()
    runner._optimistic_parallel_draft_state = Mock()

    maintenance = SimpleNamespace(
        req_ids=["skip"],
        source_row_indices=[0],
        token_ids=torch.tensor([301, 302]),
        positions=torch.tensor([1, 2]),
        common_attn_metadata=SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 2])),
    )
    extension = SimpleNamespace(
        req_ids=["draft-a", "draft-b"],
        source_row_indices=[2, 1],
        token_ids=torch.tensor([103, 202]),
        positions=torch.tensor([3, 2]),
        common_attn_metadata=SimpleNamespace(
            query_start_loc_cpu=torch.tensor([0, 1, 2])
        ),
    )
    draft_probs = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    state = SimpleNamespace(
        req_ids=["skip", "draft-b", "draft-a"],
        extension_req_ids=["draft-a", "draft-b"],
        maintenance=maintenance,
        extension=extension,
        draft_input_paths={
            "skip": [300, 301, 302],
            "draft-b": [200, 201, 202],
            "draft-a": [100, 101, 102, 103],
        },
        pending_token_ids={"skip": [], "draft-b": [], "draft-a": []},
        num_draft_tokens=2,
        draft_token_ids=torch.tensor([[11, 12], [21, 22]]),
        draft_probs=draft_probs,
        draft_done_event=Mock(),
    )
    return runner, state, draft_probs


def test_parallel_draft_compact_scatter_maps_tokens_probs_and_exact_kv_paths():
    runner, state, draft_probs = _make_parallel_draft_finish_runner()

    runner._finish_optimistic_parallel_draft(state, [[], [], []])

    assert runner._draft_token_req_ids == state.req_ids
    assert runner._draft_token_ids == [[], [21, 22], [11, 12]]
    state_a = runner._parallel_pending_draft_states["draft-a"]
    state_b = runner._parallel_pending_draft_states["draft-b"]
    assert state_a.token_ids == [11, 12]
    assert torch.equal(state_a.probs, draft_probs[0])
    assert state_b.token_ids == [21, 22]
    assert torch.equal(state_b.probs, draft_probs[1])
    assert runner._parallel_draft_kv_states["skip"].token_ids == [300, 301, 302]
    assert runner._parallel_draft_kv_states["draft-a"].token_ids == [
        100,
        101,
        102,
        103,
        11,
    ]
    assert runner._parallel_draft_kv_states["draft-b"].token_ids == [
        200,
        201,
        202,
        21,
    ]


def test_parallel_draft_finish_accepts_dynamic_single_round_output():
    runner, state, draft_probs = _make_parallel_draft_finish_runner()
    state.num_draft_tokens = 1
    state.draft_token_ids = torch.tensor([[11], [21]])
    state.draft_probs = draft_probs[:, :1]

    runner._finish_optimistic_parallel_draft(state, [[], [], []])

    assert runner._draft_token_ids == [[], [21], [11]]
    assert runner._parallel_draft_kv_states["draft-a"].token_ids == [
        100,
        101,
        102,
        103,
    ]
    assert runner._parallel_draft_kv_states["draft-b"].token_ids == [
        200,
        201,
        202,
    ]
    assert runner._parallel_pending_draft_states["draft-a"].probs.shape[0] == 1
    assert runner._parallel_pending_draft_states["draft-b"].probs.shape[0] == 1


def test_parallel_draft_finish_allows_catchup_then_extension_in_same_step():
    runner, state, _ = _make_parallel_draft_finish_runner()
    runner._parallel_draft_kv_states["draft-a"] = DraftKVPathState([100, 101])
    state.maintenance = SimpleNamespace(
        req_ids=["draft-a"],
        source_row_indices=[2],
        token_ids=torch.tensor([102, 103]),
        positions=torch.tensor([2, 3]),
        common_attn_metadata=SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 2])),
    )

    runner._finish_optimistic_parallel_draft(state, [[], [], []])

    assert runner._draft_token_ids == [[], [21, 22], [11, 12]]
    assert runner._parallel_draft_kv_states["draft-a"].token_ids == [
        100,
        101,
        102,
        103,
        11,
    ]


def test_parallel_draft_finish_uses_launch_path_after_target_bookkeeping():
    runner, state, _ = _make_parallel_draft_finish_runner()
    runner.requests["skip"].output_token_ids.append(399)
    runner.requests["draft-b"].output_token_ids.append(299)
    runner.requests["draft-a"].output_token_ids.append(199)

    runner._finish_optimistic_parallel_draft(state, [[399], [299], [199]])

    assert runner._draft_token_ids == [[], [21, 22], [11, 12]]
    assert runner._parallel_draft_kv_states["draft-b"].token_ids == [
        200,
        201,
        202,
        21,
    ]
    assert runner._parallel_draft_kv_states["draft-a"].token_ids == [
        100,
        101,
        102,
        103,
        11,
    ]


def test_parallel_draft_compact_scatter_finishes_maintenance_only():
    runner, state, _ = _make_parallel_draft_finish_runner()
    state.extension = None
    state.extension_req_ids = []
    state.draft_token_ids = None
    state.draft_probs = None

    runner._finish_optimistic_parallel_draft(state, [[], [], []])

    assert runner._draft_token_req_ids == state.req_ids
    assert runner._draft_token_ids == [[], [], []]
    assert runner._parallel_draft_kv_states["skip"].token_ids == [300, 301, 302]


def test_parallel_draft_fail_closed_finished_request_cannot_contaminate_reorder():
    runner, state, draft_probs = _make_parallel_draft_finish_runner()
    state.req_ids = ["draft-b", "draft-a"]
    state.maintenance = None
    state.extension.source_row_indices = [1, 0]
    state.pending_token_ids = {"draft-b": [], "draft-a": []}
    runner.input_batch.req_ids = ["draft-a", "draft-b"]
    runner.requests.pop("draft-a")

    runner._finish_optimistic_parallel_draft(state, [[], []])

    assert runner._draft_token_req_ids == ["draft-b", "draft-a"]
    assert runner._draft_token_ids == [[21, 22], []]
    assert "draft-a" not in runner._parallel_draft_kv_states
    assert "draft-a" not in runner._parallel_pending_draft_states
    state_b = runner._parallel_pending_draft_states["draft-b"]
    assert state_b.token_ids == [21, 22]
    assert torch.equal(state_b.probs, draft_probs[1])


def test_parallel_draft_fail_closed_missing_prob_row_isolated_per_request():
    runner, state, draft_probs = _make_parallel_draft_finish_runner()
    state.draft_probs = draft_probs[:1]

    runner._finish_optimistic_parallel_draft(state, [[], [], []])

    assert runner._draft_token_ids == [[], [], [11, 12]]
    assert "draft-b" not in runner._parallel_draft_kv_states
    assert "draft-b" not in runner._parallel_pending_draft_states
    state_a = runner._parallel_pending_draft_states["draft-a"]
    assert state_a.token_ids == [11, 12]
    assert torch.equal(state_a.probs, draft_probs[0])


def test_parallel_draft_fail_closed_missing_token_row_isolated_per_request():
    runner, state, draft_probs = _make_parallel_draft_finish_runner()
    state.draft_token_ids = state.draft_token_ids[:1]

    runner._finish_optimistic_parallel_draft(state, [[], [], []])

    assert runner._draft_token_ids == [[], [], [11, 12]]
    assert "draft-b" not in runner._parallel_draft_kv_states
    assert "draft-b" not in runner._parallel_pending_draft_states
    state_a = runner._parallel_pending_draft_states["draft-a"]
    assert state_a.token_ids == [11, 12]
    assert torch.equal(state_a.probs, draft_probs[0])


def test_parallel_draft_fail_closed_extension_id_disagreement():
    runner, state, _ = _make_parallel_draft_finish_runner()
    state.extension.req_ids = ["draft-b", "draft-a"]

    runner._finish_optimistic_parallel_draft(state, [[], [], []])

    assert runner._draft_token_ids == [[], [], []]
    assert "draft-a" not in runner._parallel_draft_kv_states
    assert "draft-b" not in runner._parallel_draft_kv_states
    assert "draft-a" not in runner._parallel_pending_draft_states
    assert "draft-b" not in runner._parallel_pending_draft_states


def test_parallel_draft_fail_closed_path_disagreement_isolated_per_request():
    runner, state, _ = _make_parallel_draft_finish_runner()
    runner._parallel_draft_kv_states["draft-a"] = DraftKVPathState([999])

    runner._finish_optimistic_parallel_draft(state, [[], [], []])

    assert runner._draft_token_ids == [[], [21, 22], []]
    assert "draft-a" not in runner._parallel_draft_kv_states
    assert "draft-a" not in runner._parallel_pending_draft_states
    assert runner._parallel_draft_kv_states["draft-b"].token_ids == [
        200,
        201,
        202,
        21,
    ]


@pytest.mark.parametrize(
    ("draft_done_ms", "deadline_missed", "has_launch_future"),
    [(8.0, False, True), (12.0, True, True), (8.0, False, False)],
)
def test_parallel_draft_latency_uses_gpu_event_timeline(
    draft_done_ms: float,
    deadline_missed: bool,
    has_launch_future: bool,
):
    class FakeEvent:
        def __init__(self, timestamp_ms: float):
            self.timestamp_ms = timestamp_ms
            self.synchronize = Mock()

        def elapsed_time(self, other: "FakeEvent") -> float:
            return other.timestamp_ms - self.timestamp_ms

    runner = object.__new__(GPUModelRunner)
    runner._parallel_draft_latency_controller = Mock()
    launch_future = Mock() if has_launch_future else None
    state = SimpleNamespace(
        num_draft_tokens=2,
        draft_launch_future=launch_future,
        target_start_event=FakeEvent(0.0),
        target_done_event=FakeEvent(10.0),
        draft_start_event=FakeEvent(2.0),
        draft_done_event=FakeEvent(draft_done_ms),
    )

    runner._observe_optimistic_parallel_draft_latency(state)

    if launch_future is not None:
        launch_future.result.assert_called_once_with()
    state.draft_done_event.synchronize.assert_called_once_with()
    state.target_done_event.synchronize.assert_called_once_with()
    runner._parallel_draft_latency_controller.observe.assert_called_once_with(
        target_ms=10.0,
        draft_start_delay_ms=2.0,
        draft_total_ms=draft_done_ms - 2.0,
        num_draft_tokens=2,
        deadline_missed=deadline_missed,
    )


def test_parallel_random_draft_probs_reconcile_without_pending_cap():
    runner = _make_parallel_prob_runner()
    old_probs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    new_probs = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100

    runner._reconcile_parallel_draft_prob_state(
        "req",
        old_pending_token_ids=[],
        generated_token_ids=[],
        draft_token_ids=[1, 2, 3],
        draft_probs=old_probs,
        requires_probs=True,
    )
    runner._reconcile_parallel_draft_prob_state(
        "req",
        old_pending_token_ids=[1, 2, 3],
        generated_token_ids=[1],
        draft_token_ids=[4, 5],
        draft_probs=new_probs,
        requires_probs=True,
    )

    state = runner._parallel_pending_draft_states["req"]
    assert state.token_ids == [2, 3, 4, 5]
    assert torch.equal(state.probs, torch.cat([old_probs[1:], new_probs]))


def test_parallel_random_draft_probs_clear_after_rejection():
    runner = _make_parallel_prob_runner()
    probs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    runner._reconcile_parallel_draft_prob_state(
        "req",
        old_pending_token_ids=[],
        generated_token_ids=[],
        draft_token_ids=[1, 2, 3],
        draft_probs=probs,
        requires_probs=True,
    )

    runner._reconcile_parallel_draft_prob_state(
        "req",
        old_pending_token_ids=[1, 2, 3],
        generated_token_ids=[1, 9],
        draft_token_ids=[4, 5],
        draft_probs=probs[:2],
        requires_probs=True,
    )

    assert "req" not in runner._parallel_pending_draft_states


def test_parallel_random_draft_probs_gather_mixed_rows():
    runner = _make_parallel_prob_runner()
    random_probs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    runner._reconcile_parallel_draft_prob_state(
        "random",
        old_pending_token_ids=[],
        generated_token_ids=[],
        draft_token_ids=[7, 8],
        draft_probs=random_probs,
        requires_probs=True,
    )
    runner.input_batch = SimpleNamespace(req_ids=["greedy", "random"])
    runner.requests = {
        "greedy": SimpleNamespace(
            sampling_params=SimpleNamespace(sampling_type=SamplingType.GREEDY)
        ),
        "random": SimpleNamespace(
            sampling_params=SimpleNamespace(sampling_type=SamplingType.RANDOM_SEED)
        ),
    }
    metadata = SpecDecodeMetadata.make_dummy(
        [[3], [7, 8]],
        device=torch.device("cpu"),
    )

    gathered = runner._get_spec_decode_draft_probs(metadata)

    assert gathered is not None
    assert torch.equal(gathered[0], torch.zeros(4))
    assert torch.equal(gathered[1:], random_probs)


def test_parallel_draft_compact_scatter_gathers_all_sampling_rows_and_rng():
    runner = _make_parallel_prob_runner()
    runner.device = torch.device("cpu")
    generator_a = torch.Generator().manual_seed(11)
    generator_b = torch.Generator().manual_seed(22)
    runner._parallel_draft_generators = {
        "draft-a": generator_a,
        "draft-b": generator_b,
    }
    runner.requests = {
        req_id: SimpleNamespace(
            sampling_params=SimpleNamespace(
                sampling_type=SamplingType.RANDOM_SEED,
                seed=123,
            )
        )
        for req_id in ("skip", "draft-b", "draft-a")
    }
    target_generator = torch.Generator().manual_seed(123)
    target_state = target_generator.get_state().clone()
    logitsprocs = Mock()
    runner.input_batch = SimpleNamespace(
        req_ids=["skip", "draft-b", "draft-a"],
        sampling_metadata=SamplingMetadata(
            temperature=torch.tensor([0.1, 0.2, 0.3]),
            all_greedy=False,
            all_random=True,
            top_p=torch.tensor([0.9, 0.8, 0.7]),
            top_k=torch.tensor([1, 2, 3]),
            generators={0: target_generator},
            max_num_logprobs=None,
            no_penalties=False,
            prompt_token_ids=torch.tensor([[10, 11], [20, 21], [30, 31]]),
            frequency_penalties=torch.tensor([1.0, 2.0, 3.0]),
            presence_penalties=torch.tensor([4.0, 5.0, 6.0]),
            repetition_penalties=torch.tensor([7.0, 8.0, 9.0]),
            output_token_ids=[[10], [20], [30]],
            allowed_token_ids_mask=torch.tensor(
                [[True, False], [False, True], [True, True]]
            ),
            bad_words_token_ids={0: [[10]], 1: [[20]], 2: [[30]]},
            logitsprocs=logitsprocs,
            logprob_token_ids={0: [10], 1: [20], 2: [30]},
            spec_token_ids=[[11], [21], [31]],
        ),
    )

    metadata = runner._build_parallel_draft_sampling_metadata(
        ["draft-a", "draft-b"],
        [2, 1],
    )

    assert metadata.all_greedy is False
    assert metadata.all_random is True
    assert torch.equal(metadata.temperature, torch.tensor([0.3, 0.2]))
    assert torch.equal(metadata.top_p, torch.tensor([0.7, 0.8]))
    assert torch.equal(metadata.top_k, torch.tensor([3, 2]))
    assert torch.equal(
        metadata.prompt_token_ids,
        torch.tensor([[30, 31], [20, 21]]),
    )
    assert torch.equal(metadata.frequency_penalties, torch.tensor([3.0, 2.0]))
    assert torch.equal(metadata.presence_penalties, torch.tensor([6.0, 5.0]))
    assert torch.equal(metadata.repetition_penalties, torch.tensor([9.0, 8.0]))
    assert torch.equal(
        metadata.allowed_token_ids_mask,
        torch.tensor([[True, True], [False, True]]),
    )
    assert metadata.output_token_ids == [[30], [20]]
    assert metadata.bad_words_token_ids == {0: [[30]], 1: [[20]]}
    assert metadata.logprob_token_ids == {0: [30], 1: [20]}
    assert metadata.spec_token_ids == [[31], [21]]
    assert metadata.logitsprocs is logitsprocs
    assert set(metadata.generators) == {0, 1}
    assert metadata.generators[0] is generator_a
    assert metadata.generators[1] is generator_b
    assert torch.equal(target_generator.get_state(), target_state)


def test_parallel_draft_sampling_accepts_padded_list_capacity():
    runner = _make_parallel_prob_runner()
    runner.device = torch.device("cpu")
    runner._parallel_draft_generators = {
        "draft-a": torch.Generator().manual_seed(1),
        "draft-b": torch.Generator().manual_seed(2),
    }
    runner.requests = {
        req_id: SimpleNamespace(
            sampling_params=SimpleNamespace(
                sampling_type=SamplingType.GREEDY,
                seed=None,
            )
        )
        for req_id in ("skip", "draft-b", "draft-a")
    }
    runner.input_batch = SimpleNamespace(
        req_ids=["skip", "draft-b", "draft-a"],
        sampling_metadata=SamplingMetadata(
            temperature=None,
            all_greedy=True,
            all_random=False,
            top_p=None,
            top_k=None,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=torch.zeros(3),
            presence_penalties=torch.zeros(3),
            repetition_penalties=torch.ones(3),
            output_token_ids=[],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=Mock(),
            spec_token_ids=[[1], [2], [3], [], []],
        ),
    )

    metadata = runner._build_parallel_draft_sampling_metadata(
        ["draft-a", "draft-b"],
        [2, 1],
    )

    assert metadata.spec_token_ids == [[3], [2]]


def test_parallel_draft_sampling_builds_generators_only_for_random_rows():
    runner = _make_parallel_prob_runner()
    runner.device = torch.device("cpu")
    greedy_generator = torch.Generator().manual_seed(1)
    random_generator = torch.Generator().manual_seed(2)
    runner._parallel_draft_generators = {
        "greedy": greedy_generator,
        "random": random_generator,
    }
    runner.requests = {
        "greedy": SimpleNamespace(
            sampling_params=SimpleNamespace(
                sampling_type=SamplingType.GREEDY,
                seed=None,
            )
        ),
        "random": SimpleNamespace(
            sampling_params=SimpleNamespace(
                sampling_type=SamplingType.RANDOM_SEED,
                seed=2,
            )
        ),
    }
    runner.input_batch = SimpleNamespace(
        req_ids=["greedy", "random"],
        sampling_metadata=SamplingMetadata(
            temperature=torch.tensor([0.0, 0.8]),
            all_greedy=False,
            all_random=False,
            top_p=torch.ones(2),
            top_k=torch.zeros(2, dtype=torch.int32),
            generators={1: torch.Generator().manual_seed(3)},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=torch.zeros(2),
            presence_penalties=torch.zeros(2),
            repetition_penalties=torch.ones(2),
            output_token_ids=[[], []],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=Mock(),
            spec_token_ids=[[], []],
        ),
    )

    metadata = runner._build_parallel_draft_sampling_metadata(
        ["greedy", "random"],
        [0, 1],
    )

    assert metadata.all_greedy is False
    assert metadata.all_random is False
    assert metadata.generators == {1: random_generator}


def test_parallel_draft_classifies_random_requests_consistently():
    runner = object.__new__(GPUModelRunner)
    runner.requests = {
        "no-params": SimpleNamespace(sampling_params=None),
        "greedy": SimpleNamespace(
            sampling_params=SimpleNamespace(sampling_type=SamplingType.GREEDY)
        ),
        "random": SimpleNamespace(
            sampling_params=SimpleNamespace(sampling_type=SamplingType.RANDOM_SEED)
        ),
    }

    assert not runner._request_uses_random_sampling("missing")
    assert not runner._request_uses_random_sampling("no-params")
    assert not runner._request_uses_random_sampling("greedy")
    assert runner._request_uses_random_sampling("random")


def test_empty_parallel_draft_output_keeps_request_rows_aligned():
    runner = object.__new__(GPUModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["request-a", "request-b"])
    runner._draft_token_ids = [[1], [2]]
    runner._draft_token_req_ids = ["stale"]

    runner._set_empty_parallel_draft_output()

    assert runner._draft_token_ids == [[], []]
    assert runner._draft_token_req_ids == ["request-a", "request-b"]
    assert runner._draft_token_req_ids is not runner.input_batch.req_ids


def test_seeded_parallel_draft_rng_does_not_depend_on_request_id():
    runner = _make_parallel_prob_runner()
    runner.device = torch.device("cpu")
    runner.model_config = SimpleNamespace(seed=0)
    runner._parallel_draft_generators = {}
    runner.requests = {
        req_id: SimpleNamespace(sampling_params=SimpleNamespace(seed=123))
        for req_id in ("request-a", "request-b")
    }

    generator_a = runner._get_parallel_draft_generator("request-a")
    generator_b = runner._get_parallel_draft_generator("request-b")

    assert torch.equal(
        torch.rand(16, generator=generator_a),
        torch.rand(16, generator=generator_b),
    )


def test_list_draft_token_ids_keep_snapshot_req_ids():
    runner = object.__new__(GPUModelRunner)
    runner._draft_token_ids = [[11], [22]]
    runner._draft_token_req_ids = ["launch-a", "launch-b"]
    runner.input_batch = SimpleNamespace(req_ids=["current-a", "current-b"])

    draft_token_ids, req_ids = runner._get_draft_token_ids_cpu()

    assert draft_token_ids == [[11], [22]]
    assert req_ids == ["launch-a", "launch-b"]


def test_apply_sparse_weight_patches_updates_only_selected_entries():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(6, dtype=torch.float32))

    runner = object.__new__(GPUModelRunner)
    runner.model = DummyModel()

    runner.apply_sparse_weight_patches(
        [
            SparseWeightPatch(
                name="weight",
                indices=torch.tensor([1, 4], dtype=torch.int32),
                values=torch.tensor([3.5, -2.0], dtype=torch.float32),
            )
        ]
    )

    expected = torch.tensor([0.0, 3.5, 0.0, 0.0, -2.0, 0.0], dtype=torch.float32)
    assert torch.equal(runner.get_model().weight.data, expected)


def test_apply_sparse_weight_patches_rejects_mismatched_lengths():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, dtype=torch.float32))

    runner = object.__new__(GPUModelRunner)
    runner.model = DummyModel()

    with pytest.raises(ValueError, match="matching lengths"):
        runner.apply_sparse_weight_patches(
            [
                SparseWeightPatch(
                    name="weight",
                    indices=torch.tensor([1, 2], dtype=torch.int32),
                    values=torch.tensor([1.0], dtype=torch.float32),
                )
            ]
        )


def test_apply_sparse_weight_patches_rejects_non_contiguous_param():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.arange(12, dtype=torch.float32).view(3, 4).t()
            )

    runner = object.__new__(GPUModelRunner)
    runner.model = DummyModel()

    with pytest.raises(NotImplementedError, match="contiguous params"):
        runner.apply_sparse_weight_patches(
            [
                SparseWeightPatch(
                    name="weight",
                    indices=torch.tensor([1], dtype=torch.int32),
                    values=torch.tensor([1.0], dtype=torch.float32),
                )
            ]
        )


def test_init_kv_cache_with_kv_sharing_invalid_target_layer_order(default_vllm_config):
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    error_msg = f"{layer_1} must come before the current layer"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            # initialization below will fail because target layer is invalid;
            # the target layer needs to come before layer 1
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
                kv_sharing_target_layer_name=layer_1,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
            ),
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_with_kv_sharing_target_layer_not_exist(default_vllm_config):
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    invalid_layer = "model.layers.0.cross_attn.attn"
    error_msg = f"{invalid_layer} is not a valid Attention layer in the model"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                # invalid layer: cross_attn.atn doesn't exist!
                kv_sharing_target_layer_name=invalid_layer,
            ),
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_with_kv_sharing_target_same_as_current(default_vllm_config):
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    error_msg = f"{layer_1} cannot be the same as the current layer"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            # initialization below will fail because target layer is invalid;
            # the target layer needs to come before layer 1
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                kv_sharing_target_layer_name=layer_1,
            ),
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_without_kv_sharing(default_vllm_config):
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        fwd_context = {
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
            ),
        }
        # suppress var not used error
        assert fwd_context is not None
    # Set high context length to test max context length estimation
    vllm_config.model_config.max_model_len = 3_000_000
    vllm_ctx = vllm_config.compilation_config.static_forward_context
    runner = GPUModelRunner(vllm_config, DEVICE_TYPE)
    kv_cache_spec = runner.get_kv_cache_spec()
    assert len(kv_cache_spec) == 2
    assert len(runner.shared_kv_cache_layers) == 0

    available_memory = 20 * GiB_bytes
    # page size for layer 0's kv_cache_spec is 32KB
    num_expected_blocks = 327680  # 20GB / 32KB / 2 (num layers)
    kv_cache_config = get_kv_cache_configs(
        vllm_config, [kv_cache_spec], [available_memory]
    )[0]
    assert kv_cache_config.num_blocks == num_expected_blocks
    assert len(kv_cache_config.kv_cache_tensors) == 2
    assert kv_cache_config.kv_cache_tensors[0].size == available_memory // 2
    assert kv_cache_config.kv_cache_tensors[1].size == available_memory // 2

    max_context_len = estimate_max_model_len(vllm_config, kv_cache_spec, 5 * GiB_bytes)
    # max context len with KV sharing should be 2x as large as without
    assert max_context_len == 1310720

    # important: override tensor size to prevent large mem alloc during test
    # this will only allocate 2 block worth of memory (2 * 32kb)
    kv_cache_config.num_blocks = 1
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        kv_cache_tensor.size = kv_cache_spec[
            kv_cache_tensor.shared_by[0]
        ].page_size_bytes

    runner.initialize_kv_cache(kv_cache_config)

    layer_0_kv = vllm_ctx[layer_0].kv_cache
    layer_1_kv = vllm_ctx[layer_1].kv_cache
    # check layer 1 kv cache does NOT share memory with layer 0
    assert id(layer_1_kv) != id(layer_0_kv)

    # check layer 1 added to kv cache group's layer names
    assert len(kv_cache_config.kv_cache_groups) == 1
    assert len(kv_cache_config.kv_cache_groups[0].layer_names) == 2
    assert kv_cache_config.kv_cache_groups[0].layer_names[0] == layer_0
    assert kv_cache_config.kv_cache_groups[0].layer_names[1] == layer_1


def test_init_kv_cache_with_kv_sharing_valid(default_vllm_config):
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        fwd_context = {
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                kv_sharing_target_layer_name="model.layers.0.self_attn.attn",
            ),
        }
        # suppress var not used error
        assert fwd_context is not None
    # Set high context length to test max context length estimation
    vllm_config.model_config.max_model_len = 3_000_000
    vllm_ctx = vllm_config.compilation_config.static_forward_context
    runner = GPUModelRunner(vllm_config, DEVICE_TYPE)
    kv_cache_spec = runner.get_kv_cache_spec()
    assert len(kv_cache_spec) == 1
    assert layer_0 in kv_cache_spec
    assert runner.shared_kv_cache_layers[layer_1] == layer_0

    available_memory = 20 * GiB_bytes
    # page size for layer 0's kv_cache_spec is 32KB
    # with KV sharing, we can allocate (available_mem//page_size//1) blocks
    # which is twice as many as without KV sharing
    num_expected_blocks = 655360  # 20GB / 32KB
    kv_cache_config = get_kv_cache_configs(
        vllm_config, [kv_cache_spec], [available_memory]
    )[0]
    assert kv_cache_config.num_blocks == num_expected_blocks
    assert len(kv_cache_config.kv_cache_tensors) == 1
    # Each layer now has twice the available memory for KV cache
    # compared to no KV sharing
    assert kv_cache_config.kv_cache_tensors[0].size == available_memory

    max_context_len = estimate_max_model_len(vllm_config, kv_cache_spec, 5 * GiB_bytes)
    # max context len with KV sharing should be 2x as large as without
    assert max_context_len == 2 * 1310720

    # important: override tensor size to prevent large mem alloc during test
    # this will only allocate 1 block worth of memory (32kb)
    kv_cache_config.num_blocks = 1
    kv_cache_config.kv_cache_tensors[0].size = kv_cache_spec[layer_0].page_size_bytes

    runner.initialize_kv_cache(kv_cache_config)
    kv_cache_config_after_init = runner.kv_cache_config

    layer_0_kv = vllm_ctx[layer_0].kv_cache
    layer_1_kv = vllm_ctx[layer_1].kv_cache
    # check layer 1 kv cache shares memory with layer 0
    assert id(layer_1_kv) == id(layer_0_kv)

    # check layer 1 added to kv cache group's layer names
    assert len(kv_cache_config_after_init.kv_cache_groups) == 1
    assert len(kv_cache_config_after_init.kv_cache_groups[0].layer_names) == 2
    assert kv_cache_config_after_init.kv_cache_groups[0].layer_names[0] == layer_0
    assert kv_cache_config_after_init.kv_cache_groups[0].layer_names[1] == layer_1


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Attention backend FLASHINFER is only supported on CUDA.",
)
def test_hybrid_attention_mamba_tensor_shapes():
    """
    The GPU model runner creates different views into the
    KVCacheTensors for the attention and mamba layers
    (via _reshape_kv_cache_tensors function). This test verifies
    that the views are compatible: writing a mamba block
    will not corrupt an attention block and vice versa
    """

    set_random_seed(42)

    update_environment_variables(
        {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
        }
    )
    from tests.utils import ensure_current_vllm_config

    with ensure_current_vllm_config():
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=1)
    torch.set_default_dtype(torch.float16)

    model_config = ModelConfig(
        model="ibm-granite/granite-4.0-tiny-preview",
        dtype="float16",
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    attention_config = AttentionConfig(backend=AttentionBackendEnum.FLASHINFER)
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        attention_config=attention_config,
    )

    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    layer_2 = "model.layers.2.mixer"
    layer_3 = "model.layers.3.mixer"
    layer_4 = "model.layers.4.mixer"
    layer_5 = "model.layers.5.mixer"

    with set_current_vllm_config(vllm_config):
        hf_config = vllm_config.model_config.hf_config
        fwd_context = {}
        for key in [layer_0, layer_1]:
            fwd_context[key] = Attention(
                num_heads=model_config.get_num_attention_heads(parallel_config),
                num_kv_heads=model_config.get_num_kv_heads(parallel_config),
                head_size=model_config.get_head_size(),
                scale=1.0,
                prefix=key,
            )
        for key in [layer_2, layer_3, layer_4, layer_5]:
            fwd_context[key] = MambaMixer2(
                hidden_size=hf_config.hidden_size,
                ssm_state_size=hf_config.mamba_d_state,
                conv_kernel_size=hf_config.mamba_d_conv,
                intermediate_size=hf_config.mamba_expand * hf_config.hidden_size,
                use_conv_bias=hf_config.mamba_conv_bias,
                use_bias=hf_config.mamba_proj_bias,
                n_groups=hf_config.mamba_n_groups,
                num_heads=hf_config.mamba_n_heads,
                head_dim=hf_config.mamba_d_head,
                rms_norm_eps=hf_config.rms_norm_eps,
                activation=hf_config.hidden_act,
                cache_config=cache_config,
                model_config=model_config,
                prefix=key,
            )
        # suppress var not used error
        assert fwd_context is not None
        vllm_ctx = vllm_config.compilation_config.static_forward_context

        runner = GPUModelRunner(vllm_config, DEVICE_TYPE)
        current_platform.update_block_size_for_backend(vllm_config)
        kv_cache_spec = runner.get_kv_cache_spec()

        available_memory = 5 * GiB_bytes
        kv_cache_config = get_kv_cache_configs(
            vllm_config, [kv_cache_spec], [available_memory]
        )[0]
        runner.initialize_kv_cache(kv_cache_config)

    # random partition of blocks
    # blocks0 will be assigned to attention layers
    # blocks1 will be assigned to mamba layers
    num_blocks = kv_cache_config.num_blocks
    ind = np.arange(num_blocks)
    np.random.shuffle(ind)
    blocks0, blocks1 = ind[: (num_blocks // 2)], ind[(num_blocks // 2) :]

    attn_shape = vllm_ctx[layer_0].kv_cache.shape
    conv_shape = vllm_ctx[layer_2].kv_cache[0].shape
    ssm_shape = vllm_ctx[layer_2].kv_cache[1].shape

    # assert we are using FlashInfer
    assert attn_shape[0] % num_blocks == 0
    block_split_ratio = attn_shape[0] // num_blocks

    # use small blocks for testing to avoid memory issues
    test_block_size = min(2, len(blocks0), len(blocks1))

    # use non-overlapping blocks to avoid data contamination
    # Split kernel blocks: first half for attention, second half for mamba
    mid_point = num_blocks // 2

    # attention uses kernel blocks from first half (mapped to logical blocks)
    kv_blocks_for_attention = np.array([0, 1])[:test_block_size]

    # mamba uses kernel blocks from second half
    kv_blocks_for_mamba = np.array([mid_point, mid_point + 1])[:test_block_size]

    # create small constant tensors for testing with corrected shapes
    # attention: [block_size, ...] starting from dimension 2
    attn_constant_shape = attn_shape[2:]
    conv_constant_shape = conv_shape[1:]
    ssm_constant_shape = ssm_shape[1:]

    attn_blocks_constant = torch.full(
        (test_block_size, *attn_constant_shape), device=DEVICE_TYPE, fill_value=3.33
    )
    conv_blocks_constant = torch.full(
        (test_block_size, *conv_constant_shape), device=DEVICE_TYPE, fill_value=6.66
    )
    ssm_blocks_constant = torch.full(
        (test_block_size, *ssm_constant_shape), device=DEVICE_TYPE, fill_value=9.99
    )

    # Fill attention blocks with constants using kv block indices
    kernel_blocks_for_attention = kv_blocks_for_attention * block_split_ratio

    for layer in [layer_0, layer_1]:
        # attention: kv_cache[kernel_block_idx, kv_idx, ...]
        for i, kernel_block in enumerate(kernel_blocks_for_attention):
            vllm_ctx[layer].kv_cache[kernel_block, :] = attn_blocks_constant[i]

    # fill mamba blocks with constants using kernel block indices
    for layer in [layer_2, layer_3, layer_4, layer_5]:
        # mamba: kv_cache[component][kernel_block_idx, ...]
        for i, kv_block in enumerate(kv_blocks_for_mamba):
            vllm_ctx[layer].kv_cache[0][kv_block, :] = conv_blocks_constant[i]
            vllm_ctx[layer].kv_cache[1][kv_block, :] = ssm_blocks_constant[i]

    # verify attention and mamba contents are correct
    for layer in [layer_0, layer_1]:
        for i, kernel_block in enumerate(kernel_blocks_for_attention):
            actual_kv = vllm_ctx[layer].kv_cache[kernel_block, :]
            expected = attn_blocks_constant[i]

            # Check K and V separately
            assert torch.equal(actual_kv[0], expected)
            assert torch.equal(actual_kv[1], expected)

    for layer in [layer_2, layer_3, layer_4, layer_5]:
        for i, kv_block in enumerate(kv_blocks_for_mamba):
            actual_conv = vllm_ctx[layer].kv_cache[0][kv_block, :]
            actual_ssm = vllm_ctx[layer].kv_cache[1][kv_block, :]
            expected_conv = conv_blocks_constant[i]
            expected_ssm = ssm_blocks_constant[i]

            assert torch.equal(actual_conv, expected_conv)
            assert torch.equal(actual_ssm, expected_ssm)

    for layer in [layer_2, layer_3, layer_4, layer_5]:
        for i, kv_block in enumerate(kv_blocks_for_mamba):
            actual_conv = vllm_ctx[layer].kv_cache[0][kv_block, :]
            actual_ssm = vllm_ctx[layer].kv_cache[1][kv_block, :]
            expected_conv = conv_blocks_constant[i]
            expected_ssm = ssm_blocks_constant[i]
            assert torch.equal(actual_conv, expected_conv)
            assert torch.equal(actual_ssm, expected_ssm)


def test_hybrid_block_table_initialization():
    """Test hybrid block table with different kernel and kvcache_manager block
    sizes."""
    from vllm.v1.worker.block_table import BlockTable

    # Test configuration: kvcache_manager block size = 32,
    # kernel block size = 16
    block_size = 32
    kernel_block_sizes = [16]
    max_num_reqs = 10
    max_num_blocks_per_req = 20
    max_num_batched_tokens = 512
    cp_kv_cache_interleave_size = 8

    block_table = BlockTable(
        block_size=block_size,
        max_num_reqs=max_num_reqs,
        max_num_blocks_per_req=max_num_blocks_per_req,
        max_num_batched_tokens=max_num_batched_tokens,
        pin_memory=False,
        device=torch.device(DEVICE_TYPE),
        kernel_block_size=kernel_block_sizes[0],
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )

    # Verify hybrid block configuration
    assert block_table.use_hybrid_blocks is True
    assert block_table.block_size == kernel_block_sizes[0]
    assert block_table.blocks_per_kv_block == (
        block_size // kernel_block_sizes[0]
    )  # Changed to use first element

    # Test block table conversion logic
    # One kvcache_manager block should map to multiple kernel blocks
    kvcache_manager_blocks = [0, 1, 2]

    # Verify that kvcache_manager blocks can be converted to kernel blocks
    # and that block table operations work correctly.
    req_index = 0
    block_table.append_row(kvcache_manager_blocks, req_index)
    # Get expected kernel blocks from the implementation for verification.
    expected_kernel_blocks = block_table.map_to_kernel_blocks(
        np.array(kvcache_manager_blocks),
        block_table.blocks_per_kv_block,
        block_table._kernel_block_arange,
    )
    # Verify block table state
    assert block_table.num_blocks_per_row[req_index] == len(expected_kernel_blocks)
    assert np.array_equal(
        block_table.block_table.np[req_index, : len(expected_kernel_blocks)],
        expected_kernel_blocks,
    )


def test_input_batch_with_kernel_block_sizes():
    """Test InputBatch initialization with kernel_block_sizes parameter."""
    max_num_reqs = 10
    max_model_len = 512
    max_num_batched_tokens = 512
    device = torch.device(DEVICE_TYPE)
    pin_memory = False
    vocab_size = 50272

    # Test with different kernel block sizes
    block_sizes = [32, 64]
    kernel_block_sizes = [16, 32]

    input_batch = InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        device=device,
        pin_memory=pin_memory,
        vocab_size=vocab_size,
        block_sizes=block_sizes,
        kernel_block_sizes=kernel_block_sizes,
    )

    # Verify that block tables were created with kernel block sizes
    assert len(input_batch.block_table.block_tables) == len(block_sizes)

    for i, (kv_size, kernel_size) in enumerate(zip(block_sizes, kernel_block_sizes)):
        block_table = input_batch.block_table.block_tables[i]
        if kv_size != kernel_size:
            assert block_table.use_hybrid_blocks is True
            assert block_table.block_size == kernel_size
        else:
            assert block_table.use_hybrid_blocks is False
            assert block_table.block_size == kernel_size


def test_hybrid_cache_integration(default_vllm_config, dist_init):
    """Test hybrid cache architecture integration with GPUModelRunner."""
    # Create a new model runner with hybrid cache configuration
    vllm_config = get_vllm_config()

    # Configure hybrid cache with different kvcache_manager block size
    vllm_config.cache_config.block_size = 32

    model_config = vllm_config.model_config
    num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = model_config.get_head_size()
    vllm_config.compilation_config.static_forward_context["layer.0"] = Attention(
        num_heads, head_size, 0.1
    )

    runner = GPUModelRunner(vllm_config, DEVICE_TYPE)

    # Initialize KV cache with configuration
    attn_spec = FullAttentionSpec(
        block_size=16,  # Use kernel block size directly
        num_kv_heads=runner.model_config.get_num_kv_heads(runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size, shared_by=["layer.0"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec)
        ],
    )
    runner.kv_cache_config = kv_cache_config

    # Initialize input batch with kernel block sizes
    runner.input_batch = InputBatch(
        max_num_reqs=runner.max_num_reqs,
        max_model_len=runner.max_model_len,
        max_num_batched_tokens=runner.max_num_tokens,
        device=runner.device,
        pin_memory=runner.pin_memory,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=[kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size],
        kernel_block_sizes=[16],
    )  # Use kernel block size

    runner.initialize_attn_backend(kv_cache_config)

    # Verify hybrid block table configuration
    block_table = runner.input_batch.block_table.block_tables[0]
    assert block_table.block_size == (
        kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
    )

    # Test request processing with hybrid blocks
    req_id = "hybrid_req_0"
    scheduler_output = _schedule_new_request(req_id)

    # Update states should work with hybrid blocks
    runner._update_states(scheduler_output)
    assert _is_req_scheduled(runner, req_id)
    assert _is_req_state_block_table_match(runner, req_id)


def test_is_uniform_decode() -> None:
    # Normal
    assert GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=16,
    )
    assert not GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=2,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=16,
    )
    assert not GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=15,
    )
    # Spec decoding
    assert GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=5,
        uniform_decode_query_len=5,
        num_tokens=30,
        num_reqs=6,
    )
    assert not GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=5,
        uniform_decode_query_len=4,
        num_tokens=30,
        num_reqs=6,
    )
    assert not GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=5,
        uniform_decode_query_len=5,
        num_tokens=30,
        num_reqs=7,
    )
    # Force uniform decode
    assert GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=16,
        force_uniform_decode=True,
    )
    assert GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=2,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=16,
        force_uniform_decode=True,
    )
    assert GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=15,
        force_uniform_decode=True,
    )
    assert not GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=16,
        force_uniform_decode=False,
    )
    assert not GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=2,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=16,
        force_uniform_decode=False,
    )
    assert not GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=15,
        force_uniform_decode=False,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Attention backend FLASHINFER is only supported on CUDA.",
)
def test_mamba_cache_raises_when_max_num_seqs_exceeds_blocks():
    """Test that a ValueError is raised when max_num_seqs exceeds the
    available Mamba cache blocks for hybrid models with FULL cudagraphs.

    See: https://github.com/vllm-project/vllm/issues/34094
    """
    set_random_seed(42)

    update_environment_variables(
        {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
        }
    )
    from tests.utils import ensure_current_vllm_config

    with ensure_current_vllm_config():
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=1)
    torch.set_default_dtype(torch.float16)

    model_config = ModelConfig(
        model="ibm-granite/granite-4.0-tiny-preview",
        dtype="float16",
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    attention_config = AttentionConfig(backend=AttentionBackendEnum.FLASHINFER)
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        attention_config=attention_config,
    )

    with set_current_vllm_config(vllm_config):
        hf_config = vllm_config.model_config.hf_config
        fwd_context = {}
        for key in ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]:
            fwd_context[key] = Attention(
                num_heads=model_config.get_num_attention_heads(parallel_config),
                num_kv_heads=model_config.get_num_kv_heads(parallel_config),
                head_size=model_config.get_head_size(),
                scale=1.0,
                prefix=key,
            )
        for key in [
            "model.layers.2.mixer",
            "model.layers.3.mixer",
            "model.layers.4.mixer",
            "model.layers.5.mixer",
        ]:
            fwd_context[key] = MambaMixer2(
                hidden_size=hf_config.hidden_size,
                ssm_state_size=hf_config.mamba_d_state,
                conv_kernel_size=hf_config.mamba_d_conv,
                intermediate_size=hf_config.mamba_expand * hf_config.hidden_size,
                use_conv_bias=hf_config.mamba_conv_bias,
                use_bias=hf_config.mamba_proj_bias,
                n_groups=hf_config.mamba_n_groups,
                num_heads=hf_config.mamba_n_heads,
                head_dim=hf_config.mamba_d_head,
                rms_norm_eps=hf_config.rms_norm_eps,
                activation=hf_config.hidden_act,
                cache_config=cache_config,
                model_config=model_config,
                prefix=key,
            )
        assert fwd_context is not None

        runner = GPUModelRunner(vllm_config, DEVICE_TYPE)
        current_platform.update_block_size_for_backend(vllm_config)
        kv_cache_spec = runner.get_kv_cache_spec()

        available_memory = 5 * GiB_bytes
        kv_cache_config = get_kv_cache_configs(
            vllm_config, [kv_cache_spec], [available_memory]
        )[0]
        num_blocks = kv_cache_config.num_blocks

        # Force max_num_seqs to exceed num_blocks so the check triggers.
        runner.max_num_reqs = num_blocks + 100

        with pytest.raises(ValueError, match="max_num_seqs"):
            runner.initialize_kv_cache(kv_cache_config)
