# SPDX-License-Identifier: Apache-2.0
"""Validate a CUDA Green Context SM profile pool on a real GPU."""

from __future__ import annotations

import argparse
import gc
import json
from concurrent.futures import ThreadPoolExecutor

import torch

from vllm.v1.worker.sm_partition import (
    ParallelSpecDecodeSMProfile,
    create_parallel_spec_decode_sm_profile_pool,
)

DEFAULT_SM_PROFILES = (
    (84, 0),
    (76, 8),
    (68, 16),
    (60, 24),
    (52, 32),
    (44, 40),
    (36, 48),
    (28, 56),
)


def _parse_sm_profile(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "SM profile must use TARGET,DRAFT format"
        )
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "SM profile counts must be integers"
        ) from exc


def _initialize_target_blas_handles(
    profiles: tuple[ParallelSpecDecodeSMProfile, ...],
) -> dict[int, int]:
    handles = {}
    for profile in profiles:
        with torch.cuda.stream(profile.target_stream):
            handles[profile.profile_id] = torch.cuda.current_blas_handle()
    return handles


def _initialize_draft_blas_handles(
    profiles: tuple[ParallelSpecDecodeSMProfile, ...],
) -> dict[int, int]:
    handles = {}
    for profile in profiles:
        if profile.draft_stream is not None:
            with torch.cuda.stream(profile.draft_stream):
                handles[profile.profile_id] = torch.cuda.current_blas_handle()
    return handles


def _validate_profile(
    profile: ParallelSpecDecodeSMProfile,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    expected_dependent: torch.Tensor,
    iterations: int,
) -> dict[str, float | int | None]:
    ready = torch.cuda.Event()
    target_start = torch.cuda.Event(enable_timing=True)
    target_end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(profile.target_stream):
        shared = lhs + rhs
        ready.record()
        target_start.record()
        for _ in range(iterations):
            target_output = lhs @ rhs
        target_end.record()

    draft_start = None
    draft_end = None
    dependent = None
    draft_output = None
    if profile.draft_stream is not None:
        draft_start = torch.cuda.Event(enable_timing=True)
        draft_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(profile.draft_stream):
            profile.draft_stream.wait_event(ready)
            dependent = shared * 2
            draft_start.record()
            for _ in range(iterations):
                draft_output = lhs @ rhs
            draft_end.record()

    torch.cuda.synchronize()
    if dependent is not None:
        torch.testing.assert_close(dependent, expected_dependent)
        torch.testing.assert_close(target_output, draft_output)

    return {
        "profile_id": profile.profile_id,
        "target_duration_ms": target_start.elapsed_time(target_end),
        "draft_duration_ms": (
            None
            if draft_start is None or draft_end is None
            else draft_start.elapsed_time(draft_end)
        ),
    }


def _profile_switch_sequence(num_profiles: int) -> list[int]:
    last_profile_id = num_profiles - 1
    return [
        0,
        min(1, last_profile_id),
        min(3, last_profile_id),
        last_profile_id,
        0,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sm-profile",
        action="append",
        type=_parse_sm_profile,
        default=None,
        metavar="TARGET,DRAFT",
    )
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--switch-repetitions", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available for the SM pool preflight")

    requested_profiles = tuple(args.sm_profile or DEFAULT_SM_PROFILES)
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    lhs = torch.randn(
        args.matrix_size,
        args.matrix_size,
        device=device,
        dtype=torch.bfloat16,
    )
    rhs = torch.randn_like(lhs)
    expected_dependent = (lhs + rhs) * 2
    torch.cuda.synchronize()

    pool = create_parallel_spec_decode_sm_profile_pool(
        device,
        requested_profiles,
    )
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="sm-profile-draft",
    )
    teardown_succeeded = False
    try:
        profiles = pool.profiles
        target_blas_handles = _initialize_target_blas_handles(profiles)
        draft_blas_handles = executor.submit(
            _initialize_draft_blas_handles,
            profiles,
        ).result()

        profile_results = [
            _validate_profile(
                profile,
                lhs,
                rhs,
                expected_dependent,
                args.iterations,
            )
            for profile in profiles
        ]

        switch_sequence = _profile_switch_sequence(len(profiles))
        executed_switches = switch_sequence * args.switch_repetitions
        switch_events = []
        for profile_id in executed_switches:
            profile = pool.get(profile_id)
            event = torch.cuda.Event()
            with torch.cuda.stream(profile.target_stream):
                switched_output = lhs + profile_id
                event.record()
            event.synchronize()
            torch.testing.assert_close(switched_output, lhs + profile_id)
            switch_events.append(event)

        actual_profiles = [
            {
                "profile_id": profile.profile_id,
                "target_sm_count": profile.target_sm_count,
                "draft_sm_count": profile.draft_sm_count,
            }
            for profile in profiles
        ]
        target_streams = {
            profile.profile_id: profile.target_stream.cuda_stream
            for profile in profiles
        }
        draft_streams = {
            profile.profile_id: profile.draft_stream.cuda_stream
            for profile in profiles
            if profile.draft_stream is not None
        }
        raw_stream_handles = [
            int(raw_stream)
            for profile in profiles
            for raw_stream in profile.raw_streams
        ]
        del profile, profiles, switch_events
        gc.collect()
    finally:
        executor.shutdown(wait=True)
        pool.close()
        teardown_succeeded = True

    summary = {
        "device": properties.name,
        "device_sm_count": properties.multi_processor_count,
        "requested_profiles": [
            {
                "profile_id": profile_id,
                "target_sm_count": target_sm_count,
                "draft_sm_count": draft_sm_count,
            }
            for profile_id, (target_sm_count, draft_sm_count) in enumerate(
                requested_profiles
            )
        ],
        "actual_profiles": actual_profiles,
        "target_streams": target_streams,
        "draft_streams": draft_streams,
        "raw_stream_count": len(raw_stream_handles),
        "unique_raw_stream_count": len(set(raw_stream_handles)),
        "target_blas_handles": target_blas_handles,
        "draft_blas_handles": draft_blas_handles,
        "profile_results": profile_results,
        "switch_sequence": executed_switches,
        "matrix_size": args.matrix_size,
        "iterations": args.iterations,
        "teardown_succeeded": teardown_succeeded,
    }
    print("SM_PROFILE_POOL_PREFLIGHT=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
