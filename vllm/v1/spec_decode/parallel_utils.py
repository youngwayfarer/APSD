# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass, field


@dataclass
class DraftKVPathState:
    token_ids: list[int] = field(default_factory=list)


def supports_decoupled_parallel_draft(
    *,
    enabled: bool,
    uses_draft_model: bool,
    supports_mm_inputs: bool,
    async_scheduling: bool,
    use_ubatching: bool,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    decode_context_parallel_size: int,
) -> bool:
    """Return whether static config supports decoupled parallel Draft."""
    return bool(
        enabled
        and uses_draft_model
        and not supports_mm_inputs
        and not async_scheduling
        and not use_ubatching
        and tensor_parallel_size == 1
        and pipeline_parallel_size == 1
        and data_parallel_size == 1
        and decode_context_parallel_size == 1
    )


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def is_draft_extension_ready(
    state: DraftKVPathState,
    draft_input_path: list[int],
) -> bool:
    required = draft_input_path[:-1]
    return state.token_ids[: len(required)] == required


def plan_draft_catchup(
    state: DraftKVPathState,
    draft_input_path: list[int],
) -> tuple[int, list[int]]:
    common = _common_prefix_len(state.token_ids, draft_input_path)
    return common, draft_input_path[common:]


def advance_draft_kv_path(
    state: DraftKVPathState,
    start_index: int,
    input_token_ids: list[int],
) -> None:
    if not 0 <= start_index <= len(state.token_ids):
        raise ValueError("start_index is outside the cached Draft path")
    state.token_ids[start_index:] = input_token_ids


@dataclass
class ParallelDraftLatencyController:
    """Choose a conservative Draft length from observed GPU timings."""

    max_draft_tokens: int
    time_budget_ratio: float
    ewma_alpha: float = 0.2
    promotion_interval: int = 3
    deadline_misses: int = field(init=False, default=0)
    _adaptive_limit: int = field(init=False, default=1)
    _headroom_successes: int = field(init=False, default=0)
    _target_ms: float | None = field(init=False, default=None)
    _draft_start_delay_ms: float | None = field(init=False, default=None)
    _draft_step_ms: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.max_draft_tokens < 1:
            raise ValueError("max_draft_tokens must be at least 1")
        if not 0.0 < self.time_budget_ratio <= 1.0:
            raise ValueError("time_budget_ratio must be in (0, 1]")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if self.promotion_interval < 1:
            raise ValueError("promotion_interval must be at least 1")

    def _update_ewma(self, old: float | None, new: float) -> float:
        if old is None:
            return new
        return self.ewma_alpha * new + (1.0 - self.ewma_alpha) * old

    def select_num_draft_tokens(self) -> int:
        """Return the next batch-wide Draft token budget."""
        if (
            self._target_ms is None
            or self._draft_start_delay_ms is None
            or self._draft_step_ms is None
        ):
            return 1

        available_ms = (
            self._target_ms * self.time_budget_ratio - self._draft_start_delay_ms
        )
        if available_ms <= 0.0 or self._draft_step_ms <= 0.0:
            return 1

        predicted_tokens = math.floor(available_ms / self._draft_step_ms)
        return max(
            1,
            min(
                self.max_draft_tokens,
                self._adaptive_limit,
                predicted_tokens,
            ),
        )

    def observe(
        self,
        *,
        target_ms: float,
        draft_start_delay_ms: float,
        draft_total_ms: float,
        num_draft_tokens: int,
        deadline_missed: bool,
    ) -> None:
        """Update latency estimates and promotion/demotion state."""
        if target_ms <= 0.0:
            raise ValueError("target_ms must be positive")
        if draft_start_delay_ms < 0.0:
            raise ValueError("draft_start_delay_ms must be non-negative")
        if draft_total_ms <= 0.0:
            raise ValueError("draft_total_ms must be positive")
        if not 1 <= num_draft_tokens <= self.max_draft_tokens:
            raise ValueError("num_draft_tokens is outside the configured range")

        self._target_ms = self._update_ewma(self._target_ms, target_ms)
        self._draft_start_delay_ms = self._update_ewma(
            self._draft_start_delay_ms,
            draft_start_delay_ms,
        )
        self._draft_step_ms = self._update_ewma(
            self._draft_step_ms,
            draft_total_ms / num_draft_tokens,
        )

        if deadline_missed:
            self.deadline_misses += 1
            self._adaptive_limit = max(
                1,
                min(self._adaptive_limit, num_draft_tokens - 1),
            )
            self._headroom_successes = 0
            return

        has_headroom = (
            draft_start_delay_ms + draft_total_ms <= target_ms * self.time_budget_ratio
        )
        if not has_headroom:
            self._headroom_successes = 0
            return

        self._headroom_successes += 1
        if self._headroom_successes >= self.promotion_interval:
            self._adaptive_limit = min(
                self.max_draft_tokens,
                self._adaptive_limit + 1,
            )
            self._headroom_successes = 0


@dataclass
class DraftRoundDeadline:
    """Track one Target step's Draft time budget without smoothing.

    Each completed round is charged at its measured duration. The most recent
    round duration alone estimates whether another round fits.
    """

    remaining_ms: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.remaining_ms) or self.remaining_ms <= 0.0:
            raise ValueError("remaining_ms must be positive")

    def observe_round(self, round_ms: float) -> bool:
        """Charge one round and return whether another equal round fits."""
        if not math.isfinite(round_ms) or round_ms <= 0.0:
            # Invalid timing must fail closed to the already completed prefix.
            self.remaining_ms = 0.0
            return False
        self.remaining_ms -= round_ms
        return self.remaining_ms >= round_ms


def get_decoupled_suffix_slices(
    old_pending_token_ids: list[int],
    generated_token_ids: list[int],
    draft_token_ids: list[int],
) -> tuple[int, int] | None:
    """Return retained starts for old pending and new Draft state.

    The returned tuple is ``(old_start, draft_start)``. ``None`` means Target
    contradicted the optimistic path and both suffixes must be discarded.
    Values aligned with either token list, such as Draft probability rows, can
    apply the same slices.
    """
    if not generated_token_ids:
        return 0, 0

    compare_len = min(len(old_pending_token_ids), len(generated_token_ids))
    if old_pending_token_ids[:compare_len] != generated_token_ids[:compare_len]:
        return None

    if len(generated_token_ids) <= len(old_pending_token_ids):
        return len(generated_token_ids), 0

    extra_generated = generated_token_ids[len(old_pending_token_ids) :]
    if draft_token_ids[: len(extra_generated)] != extra_generated:
        return None
    return len(old_pending_token_ids), len(extra_generated)
