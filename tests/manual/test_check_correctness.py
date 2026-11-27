# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.manual import check_correctness as correctness_module
from tests.manual.check_correctness import (
    _find_kv_catchup_case,
    _sm_partition_kwargs,
)

SM_PROFILES = (
    (84, 0),
    (76, 8),
    (68, 16),
    (60, 24),
    (52, 32),
    (44, 40),
    (36, 48),
    (28, 56),
)


def test_sm_partition_kwargs_unset():
    assert _sm_partition_kwargs(None, None, None) == {}


def test_sm_partition_kwargs_64_20():
    assert _sm_partition_kwargs(64, 20, None) == {
        "parallel_spec_decode_target_sm_count": 64,
        "parallel_spec_decode_draft_sm_count": 20,
    }


def test_sm_partition_kwargs_dynamic_profiles():
    assert _sm_partition_kwargs(None, None, SM_PROFILES) == {
        "parallel_spec_decode_sm_profiles": SM_PROFILES,
        "scheduler_cls": correctness_module.CyclingSMProfileScheduler,
    }


@pytest.mark.parametrize(
    ("target_sms", "draft_sms"),
    [(64, None), (None, 20)],
)
def test_sm_partition_kwargs_rejects_unpaired_counts(
    target_sms: int | None,
    draft_sms: int | None,
):
    with pytest.raises(ValueError, match="must be set together"):
        _sm_partition_kwargs(target_sms, draft_sms, None)


def test_sm_partition_kwargs_rejects_fixed_and_dynamic_modes():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _sm_partition_kwargs(64, 20, SM_PROFILES)


def test_cycling_sm_profile_scheduler_uses_zero_only_without_draft():
    scheduler = correctness_module.CyclingSMProfileScheduler.__new__(
        correctness_module.CyclingSMProfileScheduler
    )
    scheduler.scheduler_config = type(
        "Config",
        (),
        {"parallel_spec_decode_sm_profiles": SM_PROFILES},
    )()
    scheduler._diagnostic_sm_profile_index = 0
    scheduler.current_step = 0

    assert scheduler._select_parallel_spec_decode_sm_profile_id(False) == 0
    assert [
        scheduler._select_parallel_spec_decode_sm_profile_id(True)
        for _ in range(8)
    ] == [1, 2, 3, 4, 5, 6, 7, 1]
    assert scheduler._select_parallel_spec_decode_sm_profile_id(False) == 0


def test_find_kv_catchup_case_detects_new_draft_rejection(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(
        correctness_module.TRACE,
        "reconcile",
        [
            {
                "old": [],
                "generated": [20],
                "drafted": [30],
                "result": [],
            },
            {
                "old": [],
                "generated": [21],
                "drafted": [40],
                "result": [],
            },
        ],
    )

    assert _find_kv_catchup_case([10]) == ([10, 20], 40)
