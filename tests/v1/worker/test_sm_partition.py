# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker import sm_partition as sm_partition_module
from vllm.v1.worker.sm_partition import (
    create_parallel_spec_decode_sm_partition,
)


def _mock_device_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=84),
    )


def _resource(sm_count: int):
    return SimpleNamespace(sm=SimpleNamespace(smCount=sm_count))


def test_sm_partition_rejects_requested_total_mismatch(
    monkeypatch: pytest.MonkeyPatch,
):
    _mock_device_properties(monkeypatch)
    split = Mock()
    monkeypatch.setattr(sm_partition_module, "_split_device_green_ctx", split)

    with pytest.raises(
        RuntimeError,
        match=r"requested 64 \+ 16 SMs but device has 84",
    ):
        create_parallel_spec_decode_sm_partition(
            torch.device("cuda:0"),
            target_sm_count=64,
            draft_sm_count=16,
        )

    split.assert_not_called()


def test_sm_partition_maps_one_split_to_target_and_draft(
    monkeypatch: pytest.MonkeyPatch,
):
    _mock_device_properties(monkeypatch)
    target_stream = Mock(name="target_stream")
    draft_stream = Mock(name="draft_stream")
    target_context = Mock(name="target_context")
    draft_context = Mock(name="draft_context")
    resources = [_resource(64), _resource(20)]
    split = Mock(
        return_value=(
            [target_stream, draft_stream],
            resources,
            [target_context, draft_context],
        )
    )
    monkeypatch.setattr(sm_partition_module, "_split_device_green_ctx", split)
    device = torch.device("cuda:0")

    partition = create_parallel_spec_decode_sm_partition(
        device,
        target_sm_count=64,
        draft_sm_count=20,
    )

    split.assert_called_once_with(device, num_groups=1, min_count=64)
    assert partition.target_stream is target_stream
    assert partition.draft_stream is draft_stream
    assert partition.target_sm_count == 64
    assert partition.draft_sm_count == 20
    assert partition.resources == (resources[0], resources[1])
    assert partition.green_contexts == (target_context, draft_context)


@pytest.mark.parametrize(
    ("streams", "resources", "message"),
    [
        ([Mock(), Mock()], [_resource(60), _resource(24)], "returned SM counts"),
        ([Mock()], [_resource(64)], "exactly two"),
        (
            [Mock(), Mock(), Mock()],
            [_resource(64), _resource(16), _resource(4)],
            "exactly two",
        ),
    ],
)
def test_sm_partition_rejects_unexpected_split_result(
    monkeypatch: pytest.MonkeyPatch,
    streams,
    resources,
    message: str,
):
    _mock_device_properties(monkeypatch)
    contexts = [Mock() for _ in resources]
    monkeypatch.setattr(
        sm_partition_module,
        "_split_device_green_ctx",
        Mock(return_value=(streams, resources, contexts)),
    )

    with pytest.raises(RuntimeError, match=message):
        create_parallel_spec_decode_sm_partition(
            torch.device("cuda:0"),
            target_sm_count=64,
            draft_sm_count=20,
        )


def test_sm_partition_uses_cuda_12_4_compatible_stream_creation(
    monkeypatch: pytest.MonkeyPatch,
):
    target_resource = _resource(64)
    draft_resource = _resource(20)
    target_stream = Mock(name="target_stream")
    draft_stream = Mock(name="draft_stream")

    driver = SimpleNamespace(
        CUgreenCtxCreate_flags=SimpleNamespace(CU_GREEN_CTX_DEFAULT_STREAM=1),
        CUstream_flags=SimpleNamespace(CU_STREAM_NON_BLOCKING=2),
        cuDevResourceGenerateDesc=Mock(side_effect=["target_desc", "draft_desc"]),
        cuGreenCtxCreate=Mock(side_effect=["target_green", "draft_green"]),
        cuCtxFromGreenCtx=Mock(side_effect=["target_context", "draft_context"]),
        cuCtxPushCurrent=Mock(),
        cuCtxPopCurrent=Mock(),
        cuStreamCreate=Mock(side_effect=["target_raw_stream", "draft_raw_stream"]),
    )
    check_cuda_errors = lambda result: result
    get_cudevice = Mock(return_value="cuda_device")
    get_device_resource = Mock(return_value="full_resource")
    split_resource = Mock(return_value=([target_resource], draft_resource))
    monkeypatch.setattr(
        sm_partition_module,
        "_load_green_context_dependencies",
        lambda: (
            driver,
            check_cuda_errors,
            get_cudevice,
            get_device_resource,
            split_resource,
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_stream_from_external",
        Mock(side_effect=[target_stream, draft_stream]),
    )
    device = torch.device("cuda:0")

    streams, resources, green_contexts = sm_partition_module._split_device_green_ctx(
        device,
        num_groups=1,
        min_count=64,
    )

    assert streams == [target_stream, draft_stream]
    assert resources == [target_resource, draft_resource]
    assert green_contexts == ["target_green", "draft_green"]
    get_cudevice.assert_called_once_with(device)
    get_device_resource.assert_called_once_with("cuda_device")
    split_resource.assert_called_once_with("full_resource", 1, 64)
    assert driver.cuCtxPushCurrent.call_args_list == [
        (("target_context",),),
        (("draft_context",),),
    ]
    assert driver.cuCtxPopCurrent.call_count == 2
    assert driver.cuStreamCreate.call_count == 2
    assert torch.cuda.get_stream_from_external.call_args_list == [
        (("target_raw_stream", device),),
        (("draft_raw_stream", device),),
    ]


def _mock_profile_pool_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_stream_index: int | None = None,
):
    calls: list[tuple[str, object]] = []
    stream_index = 0

    def create_stream(_flags):
        nonlocal stream_index
        stream_index += 1
        if stream_index == fail_stream_index:
            raise RuntimeError("stream creation failed")
        return f"raw_stream_{stream_index}"

    driver = SimpleNamespace(
        CUgreenCtxCreate_flags=SimpleNamespace(CU_GREEN_CTX_DEFAULT_STREAM=1),
        CUstream_flags=SimpleNamespace(CU_STREAM_NON_BLOCKING=2),
        cuDevResourceGenerateDesc=Mock(
            side_effect=lambda resources, count: (
                f"descriptor_{resources[0].sm.smCount}_{count}"
            )
        ),
        cuGreenCtxCreate=Mock(
            side_effect=lambda descriptor, _device, _flags: f"green_{descriptor}"
        ),
        cuCtxFromGreenCtx=Mock(
            side_effect=lambda green_context: f"context_{green_context}"
        ),
        cuCtxPushCurrent=Mock(),
        cuCtxPopCurrent=Mock(),
        cuStreamCreate=Mock(side_effect=create_stream),
        cuStreamSynchronize=Mock(
            side_effect=lambda stream: calls.append(("sync", stream))
        ),
        cuStreamDestroy=Mock(
            side_effect=lambda stream: calls.append(("stream", stream))
        ),
        cuGreenCtxDestroy=Mock(
            side_effect=lambda context: calls.append(("context", context))
        ),
    )
    full_resource = _resource(84)

    def split_resource(resource, num_groups, min_count):
        assert resource is full_resource
        assert num_groups == 1
        return [_resource(min_count)], _resource(84 - min_count)

    dependencies = (
        driver,
        lambda result: result,
        Mock(return_value="cuda_device"),
        Mock(return_value=full_resource),
        Mock(side_effect=split_resource),
    )
    monkeypatch.setattr(
        sm_partition_module,
        "_load_green_context_dependencies",
        lambda: dependencies,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_stream_from_external",
        Mock(side_effect=lambda raw_stream, _device: Mock(name=raw_stream)),
    )
    _mock_device_properties(monkeypatch)
    return driver, calls


def test_sm_profile_pool_preserves_ids_and_independent_splits(
    monkeypatch: pytest.MonkeyPatch,
):
    driver, _ = _mock_profile_pool_dependencies(monkeypatch)
    device = torch.device("cuda:0")

    pool = sm_partition_module.create_parallel_spec_decode_sm_profile_pool(
        device,
        ((84, 0), (76, 8), (60, 24)),
    )

    assert [profile.profile_id for profile in pool.profiles] == [0, 1, 2]
    assert [
        (profile.target_sm_count, profile.draft_sm_count)
        for profile in pool.profiles
    ] == [(84, 0), (76, 8), (60, 24)]
    assert pool.profiles[0].draft_stream is None
    assert pool.profiles[1].draft_stream is not None
    assert pool.profiles[2].draft_stream is not None
    assert driver.cuGreenCtxCreate.call_count == 5


@pytest.mark.parametrize(
    ("profiles", "message"),
    [
        (((76, 0), (68, 16)), "profile 0"),
        (((84, 0), (64, 16)), "profile 1"),
    ],
)
def test_sm_profile_pool_rejects_device_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    profiles,
    message: str,
):
    driver, _ = _mock_profile_pool_dependencies(monkeypatch)

    with pytest.raises(RuntimeError, match=message):
        sm_partition_module.create_parallel_spec_decode_sm_profile_pool(
            torch.device("cuda:0"),
            profiles,
        )

    driver.cuGreenCtxCreate.assert_not_called()


def test_sm_profile_pool_rejects_unexpected_split_counts(
    monkeypatch: pytest.MonkeyPatch,
):
    driver, _ = _mock_profile_pool_dependencies(monkeypatch)
    dependencies = sm_partition_module._load_green_context_dependencies()
    split_resource = dependencies[-1]
    split_resource.side_effect = lambda *_args: ([_resource(72)], _resource(12))

    with pytest.raises(RuntimeError, match=r"profile 1.*\[72, 12\]"):
        sm_partition_module.create_parallel_spec_decode_sm_profile_pool(
            torch.device("cuda:0"),
            ((84, 0), (76, 8)),
        )

    assert driver.cuGreenCtxCreate.call_count == 1


@pytest.mark.parametrize("profile_id", [-1, 3])
def test_sm_profile_pool_rejects_invalid_profile_id(
    monkeypatch: pytest.MonkeyPatch,
    profile_id: int,
):
    _mock_profile_pool_dependencies(monkeypatch)
    pool = sm_partition_module.create_parallel_spec_decode_sm_profile_pool(
        torch.device("cuda:0"),
        ((84, 0), (76, 8), (68, 16)),
    )

    with pytest.raises(RuntimeError, match="outside"):
        pool.get(profile_id)


def test_sm_profile_pool_cleans_up_partial_initialization_in_reverse(
    monkeypatch: pytest.MonkeyPatch,
):
    driver, calls = _mock_profile_pool_dependencies(
        monkeypatch,
        fail_stream_index=3,
    )

    with pytest.raises(RuntimeError, match="stream creation failed"):
        sm_partition_module.create_parallel_spec_decode_sm_profile_pool(
            torch.device("cuda:0"),
            ((84, 0), (76, 8)),
        )

    assert [item for item in calls if item[0] == "stream"] == [
        ("stream", "raw_stream_2"),
        ("stream", "raw_stream_1"),
    ]
    created_contexts = [
        f"green_{call.args[0]}" for call in driver.cuGreenCtxCreate.call_args_list
    ]
    assert [item[1] for item in calls if item[0] == "context"] == list(
        reversed(created_contexts)
    )


def test_sm_profile_pool_close_is_reverse_order_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
):
    driver, calls = _mock_profile_pool_dependencies(monkeypatch)
    pool = sm_partition_module.create_parallel_spec_decode_sm_profile_pool(
        torch.device("cuda:0"),
        ((84, 0), (76, 8)),
    )

    pool.close()
    first_close_calls = calls.copy()
    pool.close()

    assert calls == first_close_calls
    assert [item for item in calls if item[0] == "stream"] == [
        ("stream", "raw_stream_3"),
        ("stream", "raw_stream_2"),
        ("stream", "raw_stream_1"),
    ]
    created_contexts = [
        f"green_{call.args[0]}" for call in driver.cuGreenCtxCreate.call_args_list
    ]
    assert [item[1] for item in calls if item[0] == "context"] == list(
        reversed(created_contexts)
    )
    assert pool.profiles == ()
