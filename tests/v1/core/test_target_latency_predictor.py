# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.v1.core.sched.target_latency_predictor import (
    FEATURE_COLUMNS,
    TargetLatencyPredictor,
    TargetStepWorkload,
    _LatencyMLP,
)

pytestmark = pytest.mark.cpu_test


def test_target_step_workload_features():
    workload = TargetStepWorkload(
        prefill_chunks=((20, 10), (5, 30)),
        verify_chunks=((3, 200),),
        decode_context_lens=(100, 300),
    )

    assert workload.batch_type == "prefill_verify_decode"
    assert workload.feature_dict() == {
        "sum_prefill_tokens": 25.0,
        "sum_verify_tokens": 3.0,
        "num_verify_seq": 1.0,
        "num_prefill_chunk": 2.0,
        "num_decode": 2.0,
        "sum_prefill_context_len": 40.0,
        "sum_verify_context_len": 200.0,
        "sum_decode_context_len": 400.0,
        "max_prefill_context_len": 30.0,
        "max_verify_context_len": 200.0,
        "max_decode_context_len": 300.0,
        "max_prefill_chunk": 20.0,
        "max_verify_token": 3.0,
        "prefill_cross_interaction": 350.0,
        "prefill_self_interaction": 425.0,
        "verify_cross_interaction": 600.0,
        "verify_self_interaction": 9.0,
    }


class _SMRatioModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sm_ratio_index = FEATURE_COLUMNS.index("sm_ratio")
        return inputs[:, sm_ratio_index]


def test_predictor_batches_sm_levels_on_cpu():
    predictor = TargetLatencyPredictor(
        model=_SMRatioModel(),
        feature_mean=torch.zeros(len(FEATURE_COLUMNS)),
        feature_std=torch.ones(len(FEATURE_COLUMNS)),
        log1p_features=frozenset(),
    )
    workload = TargetStepWorkload(decode_context_lens=(100,))

    predictions = predictor.predict(workload, [84, 42], 84)

    assert predictions == pytest.approx(
        [
            torch.expm1(torch.tensor(1.0)).item(),
            torch.expm1(torch.tensor(0.5)).item(),
        ]
    )


def test_target_latency_checkpoint_matches_online_schema(tmp_path):
    feature_count = len(FEATURE_COLUMNS)
    model = _LatencyMLP(feature_count, [4], 0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias.fill_(1.0)
    torch.save(
        {
            "config": {
                "feature_columns": list(FEATURE_COLUMNS),
                "student_hidden_dims": [4],
                "target_transform": "log1p",
                "log1p_features": [],
            },
            "model_state_dict": model.state_dict(),
            "feature_mean": torch.zeros(feature_count),
            "feature_std": torch.ones(feature_count),
        },
        tmp_path / TargetLatencyPredictor.CHECKPOINT_FILENAME,
    )
    predictor = TargetLatencyPredictor.load(tmp_path)

    predictions = predictor.predict(
        TargetStepWorkload(decode_context_lens=(128, 256)),
        [84, 76, 68, 60, 52, 44, 36, 28],
        84,
    )

    assert len(predictions) == 8
    assert all(latency_ms > 0 for latency_ms in predictions)
