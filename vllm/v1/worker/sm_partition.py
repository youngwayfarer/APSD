# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


class _GreenContextResourceOwner:
    def __init__(self, driver: Any, check_cuda_errors: Callable[[Any], Any]):
        self._driver = driver
        self._check_cuda_errors = check_cuda_errors
        self._raw_streams: list[Any] = []
        self._green_contexts: list[Any] = []
        self._closed = False

    def add_raw_stream(self, raw_stream: Any) -> None:
        self._raw_streams.append(raw_stream)

    def add_green_context(self, green_context: Any) -> None:
        self._green_contexts.append(green_context)

    def close(self, release_wrappers: Callable[[], None] | None = None) -> None:
        if self._closed:
            return
        self._closed = True

        first_error: BaseException | None = None

        def run(operation: Callable[[], None]) -> None:
            nonlocal first_error
            try:
                operation()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc

        for raw_stream in reversed(self._raw_streams):
            run(
                lambda raw_stream=raw_stream: self._check_cuda_errors(
                    self._driver.cuStreamSynchronize(raw_stream)
                )
            )
        for raw_stream in reversed(self._raw_streams):
            run(
                lambda raw_stream=raw_stream: self._check_cuda_errors(
                    self._driver.cuStreamDestroy(raw_stream)
                )
            )
        self._raw_streams.clear()

        if release_wrappers is not None:
            run(release_wrappers)

        for green_context in reversed(self._green_contexts):
            run(
                lambda green_context=green_context: self._check_cuda_errors(
                    self._driver.cuGreenCtxDestroy(green_context)
                )
            )
        self._green_contexts.clear()

        if first_error is not None:
            raise first_error


class _NoopResourceOwner:
    def close(self, release_wrappers: Callable[[], None] | None = None) -> None:
        if release_wrappers is not None:
            release_wrappers()


@dataclass(frozen=True)
class _GreenContextStream:
    stream: torch.Stream
    raw_stream: Any
    resource: Any
    green_context: Any


@dataclass
class _GreenContextSplit:
    states: tuple[_GreenContextStream, ...]
    owner: _GreenContextResourceOwner

    def __iter__(self) -> Iterator[list[Any]]:
        yield [state.stream for state in self.states]
        yield [state.resource for state in self.states]
        yield [state.green_context for state in self.states]


@dataclass
class ParallelSpecDecodeSMPartition:
    target_stream: torch.Stream
    draft_stream: torch.Stream
    target_sm_count: int
    draft_sm_count: int
    resources: tuple[Any, Any]
    green_contexts: tuple[Any, Any]
    raw_streams: tuple[Any, ...] = ()
    _owner: _GreenContextResourceOwner | _NoopResourceOwner = field(
        default_factory=_NoopResourceOwner,
        repr=False,
        compare=False,
    )

    def close(self) -> None:
        self._owner.close()


@dataclass(frozen=True)
class ParallelSpecDecodeSMProfile:
    profile_id: int
    target_sm_count: int
    draft_sm_count: int
    target_stream: torch.Stream
    draft_stream: torch.Stream | None
    raw_streams: tuple[Any, ...]
    resources: tuple[Any, ...]
    green_contexts: tuple[Any, ...]


class ParallelSpecDecodeSMProfilePool:
    def __init__(
        self,
        profiles: Sequence[ParallelSpecDecodeSMProfile],
        owner: _GreenContextResourceOwner,
    ):
        self._profiles = tuple(profiles)
        self._owner = owner

    @property
    def profiles(self) -> tuple[ParallelSpecDecodeSMProfile, ...]:
        return self._profiles

    def get(self, profile_id: int) -> ParallelSpecDecodeSMProfile:
        if profile_id < 0 or profile_id >= len(self._profiles):
            raise RuntimeError(
                f"SM profile ID {profile_id} is outside the configured range"
            )
        return self._profiles[profile_id]

    def close(self) -> None:
        self._owner.close(self._release_profiles)

    def _release_profiles(self) -> None:
        self._profiles = ()


def _load_green_context_dependencies():
    try:
        import cuda.bindings.driver as driver
        from flashinfer.cuda_utils import checkCudaErrors
        from flashinfer.green_ctx import (
            get_cudevice,
            get_device_resource,
            split_resource,
        )
    except ImportError as exc:
        raise RuntimeError(
            "SM partitioning requires FlashInfer Green Context support "
            "and compatible cuda-python bindings"
        ) from exc
    return (
        driver,
        checkCudaErrors,
        get_cudevice,
        get_device_resource,
        split_resource,
    )


def _create_green_context_stream(
    *,
    driver: Any,
    check_cuda_errors: Callable[[Any], Any],
    cu_device: Any,
    device: torch.device,
    resource: Any,
    owner: _GreenContextResourceOwner,
) -> _GreenContextStream:
    descriptor = check_cuda_errors(driver.cuDevResourceGenerateDesc([resource], 1))
    green_context = check_cuda_errors(
        driver.cuGreenCtxCreate(
            descriptor,
            cu_device,
            driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
        )
    )
    owner.add_green_context(green_context)
    context = check_cuda_errors(driver.cuCtxFromGreenCtx(green_context))
    check_cuda_errors(driver.cuCtxPushCurrent(context))
    try:
        raw_stream = check_cuda_errors(
            driver.cuStreamCreate(driver.CUstream_flags.CU_STREAM_NON_BLOCKING)
        )
        owner.add_raw_stream(raw_stream)
    finally:
        check_cuda_errors(driver.cuCtxPopCurrent())
    stream = torch.cuda.get_stream_from_external(raw_stream, device)
    return _GreenContextStream(
        stream=stream,
        raw_stream=raw_stream,
        resource=resource,
        green_context=green_context,
    )


def _split_device_green_ctx(
    device: torch.device,
    *,
    num_groups: int,
    min_count: int,
):
    (
        driver,
        check_cuda_errors,
        get_cudevice,
        get_device_resource,
        split_resource,
    ) = _load_green_context_dependencies()
    cu_device = get_cudevice(device)
    device_resource = get_device_resource(cu_device)
    groups, remaining = split_resource(
        device_resource,
        num_groups,
        min_count,
    )
    resources = [*groups, remaining]
    owner = _GreenContextResourceOwner(driver, check_cuda_errors)
    try:
        states = tuple(
            _create_green_context_stream(
                driver=driver,
                check_cuda_errors=check_cuda_errors,
                cu_device=cu_device,
                device=device,
                resource=resource,
                owner=owner,
            )
            for resource in resources
        )
    except BaseException:
        owner.close()
        raise
    return _GreenContextSplit(states, owner)


def create_parallel_spec_decode_sm_partition(
    device: torch.device,
    target_sm_count: int,
    draft_sm_count: int,
) -> ParallelSpecDecodeSMPartition:
    device_sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    requested_total = target_sm_count + draft_sm_count
    if requested_total != device_sm_count:
        raise RuntimeError(
            "Parallel speculative decoding requested "
            f"{target_sm_count} + {draft_sm_count} SMs but device has "
            f"{device_sm_count}"
        )

    split = _split_device_green_ctx(
        device,
        num_groups=1,
        min_count=target_sm_count,
    )
    streams, resources, green_contexts = split
    if len(streams) != 2 or len(resources) != 2:
        owner = getattr(split, "owner", _NoopResourceOwner())
        owner.close()
        raise RuntimeError(
            "Green Context split must return exactly two resource partitions"
        )

    actual_counts = [resource.sm.smCount for resource in resources]
    expected_counts = [target_sm_count, draft_sm_count]
    if actual_counts != expected_counts:
        owner = getattr(split, "owner", _NoopResourceOwner())
        owner.close()
        raise RuntimeError(
            f"Green Context returned SM counts {actual_counts}, "
            f"expected {expected_counts}"
        )

    states = getattr(split, "states", ())
    return ParallelSpecDecodeSMPartition(
        target_stream=streams[0],
        draft_stream=streams[1],
        target_sm_count=actual_counts[0],
        draft_sm_count=actual_counts[1],
        resources=(resources[0], resources[1]),
        green_contexts=(green_contexts[0], green_contexts[1]),
        raw_streams=tuple(state.raw_stream for state in states),
        _owner=getattr(split, "owner", _NoopResourceOwner()),
    )


def create_parallel_spec_decode_sm_profile_pool(
    device: torch.device,
    profile_specs: Sequence[tuple[int, int]],
) -> ParallelSpecDecodeSMProfilePool:
    device_sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    for profile_id, (target_sm_count, draft_sm_count) in enumerate(profile_specs):
        if profile_id == 0 and (
            target_sm_count != device_sm_count or draft_sm_count != 0
        ):
            raise RuntimeError(
                f"SM profile 0 must be {device_sm_count}/0, "
                f"got {target_sm_count}/{draft_sm_count}"
            )
        if target_sm_count + draft_sm_count != device_sm_count:
            raise RuntimeError(
                f"SM profile {profile_id} requested "
                f"{target_sm_count} + {draft_sm_count} SMs but device has "
                f"{device_sm_count}"
            )

    (
        driver,
        check_cuda_errors,
        get_cudevice,
        get_device_resource,
        split_resource,
    ) = _load_green_context_dependencies()
    cu_device = get_cudevice(device)
    full_resource = get_device_resource(cu_device)
    owner = _GreenContextResourceOwner(driver, check_cuda_errors)
    profiles: list[ParallelSpecDecodeSMProfile] = []

    try:
        for profile_id, (target_sm_count, draft_sm_count) in enumerate(
            profile_specs
        ):
            if draft_sm_count == 0:
                resources = [full_resource]
            else:
                groups, remaining = split_resource(
                    full_resource,
                    1,
                    target_sm_count,
                )
                resources = [*groups, remaining]
                if len(resources) != 2:
                    raise RuntimeError(
                        f"SM profile {profile_id} split must return exactly "
                        "two resource partitions"
                    )
                actual_counts = [
                    resource.sm.smCount for resource in resources
                ]
                expected_counts = [target_sm_count, draft_sm_count]
                if actual_counts != expected_counts:
                    raise RuntimeError(
                        f"SM profile {profile_id} returned SM counts "
                        f"{actual_counts}, expected {expected_counts}"
                    )

            states = tuple(
                _create_green_context_stream(
                    driver=driver,
                    check_cuda_errors=check_cuda_errors,
                    cu_device=cu_device,
                    device=device,
                    resource=resource,
                    owner=owner,
                )
                for resource in resources
            )
            profiles.append(
                ParallelSpecDecodeSMProfile(
                    profile_id=profile_id,
                    target_sm_count=target_sm_count,
                    draft_sm_count=draft_sm_count,
                    target_stream=states[0].stream,
                    draft_stream=(
                        states[1].stream if draft_sm_count > 0 else None
                    ),
                    raw_streams=tuple(state.raw_stream for state in states),
                    resources=tuple(state.resource for state in states),
                    green_contexts=tuple(
                        state.green_context for state in states
                    ),
                )
            )
    except BaseException:
        owner.close()
        raise

    return ParallelSpecDecodeSMProfilePool(profiles, owner)
