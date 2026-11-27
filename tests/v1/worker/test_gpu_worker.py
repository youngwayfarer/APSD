# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.v1.worker import gpu_worker as gpu_worker_module
from vllm.v1.worker.gpu_worker import Worker


def test_compile_warmup_keeps_ordinary_v2_runner_compatible(monkeypatch):
    compilation_config = SimpleNamespace(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        compilation_time=0.0,
        encoder_compilation_time=0.0,
    )
    model_runner = SimpleNamespace(
        lora_config=None,
        maybe_remove_all_loras=Mock(),
    )
    worker = object.__new__(Worker)
    worker.vllm_config = SimpleNamespace(compilation_config=compilation_config)
    worker.compilation_config = compilation_config
    worker.model_runner = model_runner
    worker.model_config = SimpleNamespace(enforce_eager=True, seed=0)
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=1)
    worker.use_v2_model_runner = True
    worker.execute_model = Mock()
    worker.sample_tokens = Mock()

    monkeypatch.setattr(gpu_worker_module, "kernel_warmup", Mock())
    monkeypatch.setattr(gpu_worker_module, "warmup_kernels", Mock())
    monkeypatch.setattr(gpu_worker_module, "set_random_seed", Mock())
    monkeypatch.setattr(
        "vllm.triton_utils.jit_monitor.activate",
        Mock(),
    )

    Worker.compile_or_warm_up_model(worker)

    model_runner.maybe_remove_all_loras.assert_called_once_with(None)
    gpu_worker_module.warmup_kernels.assert_called_once_with(
        model_runner,
        worker.execute_model,
        worker.sample_tokens,
    )
