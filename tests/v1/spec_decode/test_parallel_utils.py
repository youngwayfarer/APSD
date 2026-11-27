# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.llm_base_proposer import (
    compute_probs_and_sample_next_token,
)
from vllm.v1.spec_decode.parallel_utils import (
    DraftKVPathState,
    DraftRoundDeadline,
    ParallelDraftLatencyController,
    advance_draft_kv_path,
    get_decoupled_suffix_slices,
    is_draft_extension_ready,
    plan_draft_catchup,
    supports_decoupled_parallel_draft,
)


def test_empty_draft_path_plans_the_full_input_path() -> None:
    state = DraftKVPathState()

    assert plan_draft_catchup(state, [1, 2]) == (0, [1, 2])


def test_draft_extension_ready_requires_prefix_before_input_tail() -> None:
    state = DraftKVPathState([1, 2, 3])

    assert is_draft_extension_ready(state, [1, 2, 3, 4])
    assert not is_draft_extension_ready(state, [1, 2, 9, 4])


def test_draft_extension_ready_when_one_token_behind() -> None:
    state = DraftKVPathState([1, 2])

    assert is_draft_extension_ready(state, [1, 2, 3])


def test_draft_extension_ready_ignores_stale_future_slots() -> None:
    state = DraftKVPathState([1, 2, 3, 99, 100])

    assert is_draft_extension_ready(state, [1, 2, 3, 4])


def test_draft_catchup_replays_from_first_divergence() -> None:
    state = DraftKVPathState([1, 2, 8, 9])

    start, token_ids = plan_draft_catchup(state, [1, 2, 3, 4])
    assert (start, token_ids) == (2, [3, 4])

    advance_draft_kv_path(state, start, token_ids)
    assert state.token_ids == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("old_pending", "generated", "drafted", "expected"),
    [
        ([], [], [4, 5], (0, 0)),
        ([1, 2, 3], [1], [4, 5], (1, 0)),
        ([1, 2, 3], [1, 2, 3], [4, 5], (3, 0)),
        ([1, 2, 3], [1, 2, 3, 4], [4, 5], (3, 1)),
        ([1, 2, 3], [1, 9], [4, 5], None),
        ([], [4], [7, 8], None),
    ],
)
def test_get_decoupled_suffix_slices(
    old_pending: list[int],
    generated: list[int],
    drafted: list[int],
    expected: tuple[int, int] | None,
) -> None:
    assert get_decoupled_suffix_slices(old_pending, generated, drafted) == expected


@pytest.mark.parametrize(
    ("unsupported_value", "override"),
    [
        (False, {"enabled": False}),
        (False, {"uses_draft_model": False}),
        (False, {"supports_mm_inputs": True}),
        (False, {"async_scheduling": True}),
        (False, {"use_ubatching": True}),
        (False, {"tensor_parallel_size": 2}),
        (False, {"pipeline_parallel_size": 2}),
        (False, {"data_parallel_size": 2}),
        (False, {"decode_context_parallel_size": 2}),
    ],
)
def test_decoupled_parallel_draft_static_capability_gates(
    unsupported_value: bool,
    override: dict[str, bool | int],
) -> None:
    arguments: dict[str, bool | int] = {
        "enabled": True,
        "uses_draft_model": True,
        "supports_mm_inputs": False,
        "async_scheduling": False,
        "use_ubatching": False,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "decode_context_parallel_size": 1,
    }
    arguments.update(override)

    assert supports_decoupled_parallel_draft(**arguments) is unsupported_value


def test_decoupled_parallel_draft_static_capability_accepts_supported_config() -> None:
    assert supports_decoupled_parallel_draft(
        enabled=True,
        uses_draft_model=True,
        supports_mm_inputs=False,
        async_scheduling=False,
        use_ubatching=False,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
    )


def test_probabilistic_draft_sampling_uses_supplied_generator() -> None:
    logits = torch.zeros((1, 32))

    def sample(global_seed: int) -> torch.Tensor:
        torch.manual_seed(global_seed)
        generator = torch.Generator().manual_seed(123)
        metadata = type(
            "Metadata",
            (),
            {
                "all_greedy": False,
                "all_random": True,
                "temperature": torch.tensor([0.8]),
                "generators": {0: generator},
            },
        )()
        token_ids, _ = compute_probs_and_sample_next_token(logits.clone(), metadata)
        return token_ids

    assert torch.equal(sample(1), sample(999))


def test_parallel_draft_latency_controller_starts_conservatively() -> None:
    controller = ParallelDraftLatencyController(
        max_draft_tokens=4,
        time_budget_ratio=0.8,
    )

    assert controller.select_num_draft_tokens() == 1


def test_draft_round_deadline_uses_current_round_without_averaging() -> None:
    deadline = DraftRoundDeadline(remaining_ms=10.0)

    assert deadline.observe_round(3.0)
    assert deadline.remaining_ms == pytest.approx(7.0)
    # The second decision uses 4ms, the current round's duration, rather than
    # averaging 3ms and 4ms. Only 3ms remains, so another 4ms round does not fit.
    assert not deadline.observe_round(4.0)
    assert deadline.remaining_ms == pytest.approx(3.0)


def test_draft_round_deadline_allows_exact_fit() -> None:
    deadline = DraftRoundDeadline(remaining_ms=6.0)

    assert deadline.observe_round(3.0)


def test_draft_round_deadline_invalid_timing_stops() -> None:
    deadline = DraftRoundDeadline(remaining_ms=6.0)

    assert not deadline.observe_round(0.0)
    assert deadline.remaining_ms == 0.0


def test_parallel_draft_latency_controller_promotes_with_headroom() -> None:
    controller = ParallelDraftLatencyController(
        max_draft_tokens=4,
        time_budget_ratio=0.8,
        ewma_alpha=1.0,
        promotion_interval=3,
    )

    for _ in range(3):
        controller.observe(
            target_ms=10.0,
            draft_start_delay_ms=1.0,
            draft_total_ms=3.0,
            num_draft_tokens=1,
            deadline_missed=False,
        )

    assert controller.select_num_draft_tokens() == 2


def test_parallel_draft_latency_controller_respects_predicted_budget() -> None:
    controller = ParallelDraftLatencyController(
        max_draft_tokens=4,
        time_budget_ratio=0.8,
        ewma_alpha=1.0,
        promotion_interval=1,
    )
    for _ in range(4):
        controller.observe(
            target_ms=10.0,
            draft_start_delay_ms=1.0,
            draft_total_ms=3.0,
            num_draft_tokens=1,
            deadline_missed=False,
        )

    assert controller.select_num_draft_tokens() == 2


def test_parallel_draft_latency_controller_demotes_after_miss() -> None:
    controller = ParallelDraftLatencyController(
        max_draft_tokens=4,
        time_budget_ratio=1.0,
        ewma_alpha=1.0,
        promotion_interval=1,
    )
    for _ in range(2):
        controller.observe(
            target_ms=10.0,
            draft_start_delay_ms=0.0,
            draft_total_ms=4.0,
            num_draft_tokens=1,
            deadline_missed=False,
        )
    assert controller.select_num_draft_tokens() == 2

    controller.observe(
        target_ms=10.0,
        draft_start_delay_ms=1.0,
        draft_total_ms=11.0,
        num_draft_tokens=2,
        deadline_missed=True,
    )

    assert controller.select_num_draft_tokens() == 1
    assert controller.deadline_misses == 1
