# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import pytest
import torch

from vllm.v1.worker import workspace


def test_workspace_slot_override_is_isolated_between_threads():
    entered = threading.Barrier(2)
    checked = threading.Barrier(2)
    observed = [-1, -1]

    def run(slot: int) -> None:
        with workspace.use_workspace_slot(slot):
            entered.wait()
            observed[slot - 1] = workspace.current_workspace_slot()
            checked.wait()

    threads = [threading.Thread(target=run, args=(slot,)) for slot in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert observed == [1, 2]


def test_workspace_slots_have_independent_buffers():
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_ubatches=2)

    with workspace.use_workspace_slot(0):
        target_workspace = manager.get_simultaneous(((16,), torch.float32))[0]
    with workspace.use_workspace_slot(1):
        draft_workspace = manager.get_simultaneous(((16,), torch.float32))[0]

    assert target_workspace.data_ptr() != draft_workspace.data_ptr()
    target_workspace.fill_(1)
    draft_workspace.fill_(2)
    assert target_workspace.tolist() == [1.0] * 16
    assert draft_workspace.tolist() == [2.0] * 16


def test_workspace_slot_override_restores_dbo_default(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.worker.workspace.dbo_current_ubatch_id",
        lambda: 1,
    )

    assert workspace.current_workspace_slot() == 1
    with workspace.use_workspace_slot(2):
        assert workspace.current_workspace_slot() == 2
    assert workspace.current_workspace_slot() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_workspace_slots_isolate_concurrent_cuda_stream_writes():
    manager = workspace.WorkspaceManager(torch.device("cuda"), num_ubatches=2)
    with workspace.use_workspace_slot(0):
        target_workspace = manager.get_simultaneous(((4096,), torch.int32))[0]
    with workspace.use_workspace_slot(1):
        draft_workspace = manager.get_simultaneous(((4096,), torch.int32))[0]

    target_stream = torch.cuda.Stream()
    draft_stream = torch.cuda.Stream()
    with torch.cuda.stream(target_stream):
        target_workspace.fill_(17)
    with torch.cuda.stream(draft_stream):
        draft_workspace.fill_(29)
    torch.cuda.synchronize()

    assert torch.all(target_workspace == 17)
    assert torch.all(draft_workspace == 29)
