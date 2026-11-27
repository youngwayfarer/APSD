# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections import defaultdict

import pytest

import vllm.forward_context as forward_context_module
from vllm.config import CUDAGraphMode, DeviceConfig, VllmConfig
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
    override_forward_context,
    set_forward_context,
)
from vllm.v1.worker.ubatching import UBatchContext


def _make_forward_context(marker: str) -> ForwardContext:
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        additional_kwargs={"marker": marker},
    )


def test_forward_context_is_isolated_between_threads():
    contexts = [_make_forward_context("target"), _make_forward_context("draft")]
    entered = threading.Barrier(2)
    checked = threading.Barrier(2)
    observed: list[ForwardContext | None] = [None, None]

    def run(index: int) -> None:
        with override_forward_context(contexts[index]):
            entered.wait()
            observed[index] = get_forward_context()
            checked.wait()
        assert not is_forward_context_available()

    threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert observed == contexts
    assert not is_forward_context_available()


def test_forward_context_nested_override_restores_outer_context():
    outer = _make_forward_context("outer")
    inner = _make_forward_context("inner")

    with override_forward_context(outer):
        assert get_forward_context() is outer
        with override_forward_context(inner):
            assert get_forward_context() is inner
        assert get_forward_context() is outer

    assert not is_forward_context_available()


def test_forward_context_override_restores_after_exception():
    outer = _make_forward_context("outer")
    inner = _make_forward_context("inner")

    with override_forward_context(outer):
        with (
            pytest.raises(RuntimeError, match="forward failed"),
            override_forward_context(inner),
        ):
            raise RuntimeError("forward failed")
        assert get_forward_context() is outer

    assert not is_forward_context_available()


def test_ubatch_restore_sets_current_thread_forward_context():
    context = _make_forward_context("ubatch")
    ubatch = object.__new__(UBatchContext)
    ubatch.forward_context = context

    with override_forward_context(None):
        ubatch._restore_context()
        assert get_forward_context() is context

    assert not is_forward_context_available()


def test_batchsize_logging_uses_per_forward_start_time(monkeypatch):
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    target_entered = threading.Event()
    draft_entered = threading.Event()
    release_draft = threading.Event()
    failures: list[BaseException] = []
    times = {
        "target": iter((1.0, 11.0)),
        "draft": iter((5.0, 7.0)),
    }

    monkeypatch.setattr(forward_context_module, "track_batchsize", True)
    monkeypatch.setattr(forward_context_module, "last_logging_time", 1000.0)
    monkeypatch.setattr(forward_context_module, "batchsize_logging_interval", 1000.0)
    monkeypatch.setattr(
        forward_context_module,
        "batchsize_forward_time",
        defaultdict(list),
    )
    monkeypatch.setattr(
        forward_context_module.time,
        "perf_counter",
        lambda: next(times[threading.current_thread().name]),
    )

    def run_target() -> None:
        try:
            with set_forward_context({}, vllm_config, num_tokens=1):
                target_entered.set()
                assert draft_entered.wait(timeout=5)
            release_draft.set()
        except BaseException as exc:
            failures.append(exc)
            release_draft.set()

    def run_draft() -> None:
        try:
            assert target_entered.wait(timeout=5)
            with set_forward_context({}, vllm_config, num_tokens=2):
                draft_entered.set()
                assert release_draft.wait(timeout=5)
        except BaseException as exc:
            failures.append(exc)

    threads = [
        threading.Thread(target=run_target, name="target"),
        threading.Thread(target=run_draft, name="draft"),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert not failures
    assert forward_context_module.batchsize_forward_time == {
        1: [pytest.approx(10_000.0)],
        2: [pytest.approx(2_000.0)],
    }
