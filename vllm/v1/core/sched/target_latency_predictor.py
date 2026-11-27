# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU target-step latency prediction for dynamic Draft/Target SM allocation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

WORKLOAD_FEATURE_COLUMNS = (
    "sum_prefill_tokens",
    "sum_verify_tokens",
    "num_verify_seq",
    "num_prefill_chunk",
    "num_decode",
    "sum_prefill_context_len",
    "sum_verify_context_len",
    "sum_decode_context_len",
    "max_prefill_context_len",
    "max_verify_context_len",
    "max_decode_context_len",
    "max_prefill_chunk",
    "max_verify_token",
    "prefill_cross_interaction",
    "prefill_self_interaction",
    "verify_cross_interaction",
    "verify_self_interaction",
)
BATCH_TYPES = (
    "decode",
    "prefill",
    "verify",
    "prefill_decode",
    "verify_decode",
    "prefill_verify",
    "prefill_verify_decode",
)
FEATURE_COLUMNS = (
    *WORKLOAD_FEATURE_COLUMNS,
    "sm_ratio",
    *(f"batch_type_{batch_type}" for batch_type in BATCH_TYPES),
)


@dataclass(frozen=True)
class TargetStepWorkload:
    """Target work issued in one scheduler step.

    Verify token counts exclude the ordinary one-token query, matching the
    feature definition used to train the predictor.
    """

    prefill_chunks: tuple[tuple[int, int], ...] = ()
    verify_chunks: tuple[tuple[int, int], ...] = ()
    decode_context_lens: tuple[int, ...] = ()

    @property
    def batch_type(self) -> str:
        roles: list[str] = []
        if self.prefill_chunks:
            roles.append("prefill")
        if self.verify_chunks:
            roles.append("verify")
        if self.decode_context_lens:
            roles.append("decode")
        batch_type = "_".join(roles)
        if batch_type not in BATCH_TYPES:
            raise ValueError(f"unsupported or empty Target batch type: {batch_type!r}")
        return batch_type

    def feature_dict(self) -> dict[str, float]:
        prefill_tokens = [tokens for tokens, _ in self.prefill_chunks]
        prefill_contexts = [context for _, context in self.prefill_chunks]
        verify_tokens = [tokens for tokens, _ in self.verify_chunks]
        verify_contexts = [context for _, context in self.verify_chunks]
        decode_contexts = list(self.decode_context_lens)
        return {
            "sum_prefill_tokens": float(sum(prefill_tokens)),
            "sum_verify_tokens": float(sum(verify_tokens)),
            "num_verify_seq": float(len(verify_tokens)),
            "num_prefill_chunk": float(len(prefill_tokens)),
            "num_decode": float(len(decode_contexts)),
            "sum_prefill_context_len": float(sum(prefill_contexts)),
            "sum_verify_context_len": float(sum(verify_contexts)),
            "sum_decode_context_len": float(sum(decode_contexts)),
            "max_prefill_context_len": float(max(prefill_contexts, default=0)),
            "max_verify_context_len": float(max(verify_contexts, default=0)),
            "max_decode_context_len": float(max(decode_contexts, default=0)),
            "max_prefill_chunk": float(max(prefill_tokens, default=0)),
            "max_verify_token": float(max(verify_tokens, default=0)),
            "prefill_cross_interaction": float(
                sum(tokens * context for tokens, context in self.prefill_chunks)
            ),
            "prefill_self_interaction": float(
                sum(tokens * tokens for tokens in prefill_tokens)
            ),
            "verify_cross_interaction": float(
                sum(tokens * context for tokens, context in self.verify_chunks)
            ),
            "verify_self_interaction": float(
                sum(tokens * tokens for tokens in verify_tokens)
            ),
        }


class _LatencyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for index, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0 and index < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(-1)


class TargetLatencyPredictor:
    """Loaded MLP and normalization state used by the CPU scheduler."""

    CHECKPOINT_FILENAME = "latency_mlp_distill.pt"

    def __init__(
        self,
        model: nn.Module,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        log1p_features: frozenset[str],
    ) -> None:
        self.model = model.eval()
        self.feature_mean = feature_mean.to(device="cpu", dtype=torch.float32)
        self.feature_std = feature_std.to(device="cpu", dtype=torch.float32)
        self.log1p_features = log1p_features

    @classmethod
    def load(cls, model_path: str | Path) -> TargetLatencyPredictor:
        path = Path(model_path).expanduser()
        if path.is_dir():
            path = path / cls.CHECKPOINT_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Target latency predictor not found: {path}")

        # The repository-owned checkpoint contains numpy normalization arrays,
        # which require weights_only=False with current PyTorch versions.
        checkpoint: dict[str, Any] = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        config = checkpoint["config"]
        feature_columns = tuple(config["feature_columns"])
        if feature_columns != FEATURE_COLUMNS:
            raise ValueError(
                "Target latency predictor feature schema does not match the "
                f"online scheduler schema: {feature_columns}"
            )
        if config.get("target_transform") != "log1p":
            raise ValueError("Target latency predictor must use log1p target transform")

        model = _LatencyMLP(
            input_dim=len(feature_columns),
            hidden_dims=[int(dim) for dim in config["student_hidden_dims"]],
            dropout=float(config.get("dropout", 0.0)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(
            model=model,
            feature_mean=torch.as_tensor(checkpoint["feature_mean"]),
            feature_std=torch.as_tensor(checkpoint["feature_std"]),
            log1p_features=frozenset(config["log1p_features"]),
        )

    def predict(
        self,
        workload: TargetStepWorkload,
        target_sm_counts: list[int] | tuple[int, ...],
        total_sm_count: int,
    ) -> list[float]:
        if total_sm_count <= 0:
            raise ValueError("total_sm_count must be positive")
        if not target_sm_counts:
            raise ValueError("target_sm_counts must not be empty")

        workload_features = workload.feature_dict()
        batch_type = workload.batch_type
        rows: list[list[float]] = []
        for target_sm_count in target_sm_counts:
            if not 0 < target_sm_count <= total_sm_count:
                raise ValueError(
                    "Target SM count must be in (0, total_sm_count], got "
                    f"{target_sm_count}/{total_sm_count}"
                )
            values = {
                **workload_features,
                "sm_ratio": target_sm_count / total_sm_count,
                **{
                    f"batch_type_{candidate}": float(candidate == batch_type)
                    for candidate in BATCH_TYPES
                },
            }
            rows.append([values[column] for column in FEATURE_COLUMNS])

        inputs = torch.tensor(rows, dtype=torch.float32)
        for index, column in enumerate(FEATURE_COLUMNS):
            if column in self.log1p_features:
                inputs[:, index] = torch.log1p(torch.clamp_min(inputs[:, index], 0))
        inputs = (inputs - self.feature_mean) / self.feature_std
        with torch.inference_mode():
            log_latency_ms = self.model(inputs)
        return torch.expm1(log_latency_ms).tolist()
