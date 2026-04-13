#!/usr/bin/env python3
"""Bidirectional LSTM sequence regressor with FiLM conditioning."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model import FiLMLayer


class LSTMBlock(nn.Module):
    """
    One recurrent processing block with optional FiLM conditioning.

    The block runs a single bidirectional or unidirectional LSTM layer, then
    projects its hidden state back to ``d_model`` so the stacked latent shape
    matches the transformer path.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float,
        global_input_dim: int,
        bidirectional: bool,
        film_clamp: float,
    ) -> None:
        super().__init__()

        self.bidirectional = bidirectional
        self.has_film = global_input_dim > 0
        hidden_multiplier = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.output_proj = nn.Linear(hidden_multiplier * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        if self.has_film:
            self.film = FiLMLayer(global_input_dim, d_model, clamp_gamma=film_clamp)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize non-LSTM projection layers while keeping FiLM custom init."""
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(
        self,
        x: Tensor,
        global_features: Optional[Tensor] = None,
        sequence_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Run one recurrent block.

        Args:
            x: Latent sequence tensor with shape ``[batch, seq_len, d_model]``
            global_features: Optional global conditioning tensor with shape
                ``[batch, global_dim]``
            sequence_mask: Optional padding mask with shape ``[batch, seq_len]``
                where ``True`` marks padding timesteps

        Returns:
            Tensor with shape ``[batch, seq_len, d_model]``
        """
        if sequence_mask is not None:
            valid_lengths = (~sequence_mask).sum(dim=1, dtype=torch.int64).clamp_min(1)
            packed = pack_padded_sequence(
                x,
                valid_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.lstm(packed)
            lstm_output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=x.size(1),
            )
        else:
            lstm_output, _ = self.lstm(x)

        x = self.output_proj(lstm_output)
        x = self.norm(x)

        if self.has_film and global_features is not None:
            x = self.film(x, global_features)

        return x


class LSTMPredictionModel(nn.Module):
    """
    LSTM model for atmospheric profile regression.

    Features:
    - Input projection to ``d_model``
    - Optional FiLM global conditioning at input and after each recurrent block
    - Stacked recurrent blocks with projection back to ``d_model``
    - Final regression head matching the transformer output contract
    """

    def __init__(
        self,
        input_dim: int,
        global_input_dim: int,
        output_dim: int,
        d_model: int,
        num_lstm_layers: int,
        bidirectional: bool,
        dropout: float,
        max_sequence_length: int,
        film_clamp: float,
        output_head_divisor: int,
        output_head_dropout_factor: float,
    ) -> None:
        super().__init__()

        if int(num_lstm_layers) <= 0:
            raise ValueError("num_lstm_layers must be a positive integer.")
        if not isinstance(bidirectional, bool):
            raise ValueError("bidirectional must be a boolean.")
        if not (0.0 <= float(dropout) <= 1.0):
            raise ValueError("dropout must be in [0, 1].")
        if int(max_sequence_length) <= 0:
            raise ValueError("max_sequence_length must be a positive integer.")
        if float(film_clamp) <= 0:
            raise ValueError("film_clamp must be > 0.")
        if int(output_head_divisor) <= 0:
            raise ValueError("output_head_divisor must be a positive integer.")
        if not (0.0 <= float(output_head_dropout_factor) <= 1.0):
            raise ValueError("output_head_dropout_factor must be in [0, 1].")

        self.has_global_features = global_input_dim > 0
        self.max_sequence_length = int(max_sequence_length)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        if self.has_global_features:
            self.initial_film = FiLMLayer(global_input_dim, d_model, clamp_gamma=film_clamp)
        else:
            self.initial_film = None

        self.blocks = nn.ModuleList(
            [
                LSTMBlock(
                    d_model=d_model,
                    dropout=dropout,
                    global_input_dim=global_input_dim,
                    bidirectional=bidirectional,
                    film_clamp=film_clamp,
                )
                for _ in range(num_lstm_layers)
            ]
        )
        self.inter_layer_dropout = nn.Dropout(dropout if num_lstm_layers > 1 else 0.0)
        self.final_norm = nn.LayerNorm(d_model)

        intermediate_dim = d_model // output_head_divisor
        if intermediate_dim < 1:
            raise ValueError(
                "output_head_divisor is too large for d_model; "
                f"got d_model={d_model}, output_head_divisor={output_head_divisor}."
            )
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout * output_head_dropout_factor),
            nn.Linear(intermediate_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize shared projection and normalization layers."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name == "initial_film.projection" or name.endswith(".film.projection"):
                    continue
                if name == "output_proj.4":
                    nn.init.trunc_normal_(module.weight, std=0.01)
                else:
                    nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        sequence: Tensor,
        global_features: Optional[Tensor] = None,
        sequence_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the LSTM model.

        Args:
            sequence: Input tensor with shape ``[batch, seq_len, input_dim]``
            global_features: Optional global tensor with shape ``[batch, global_dim]``
            sequence_mask: Optional padding mask with shape ``[batch, seq_len]``

        Returns:
            Output predictions with shape ``[batch, seq_len, output_dim]``
        """
        if sequence.ndim != 3:
            raise ValueError(
                f"Expected sequence shape [B, L, C], got {tuple(sequence.shape)}."
            )
        if sequence.size(1) > self.max_sequence_length:
            raise ValueError(
                f"Sequence length overflow: {sequence.size(1)} > "
                f"max_sequence_length={self.max_sequence_length}."
            )
        if sequence_mask is not None:
            if sequence_mask.ndim != 2:
                raise ValueError(
                    f"Expected sequence_mask shape [B, L], got {tuple(sequence_mask.shape)}."
                )
            if sequence_mask.shape[:2] != sequence.shape[:2]:
                raise ValueError(
                    f"sequence_mask shape {tuple(sequence_mask.shape)} does not match "
                    f"sequence batch/length {tuple(sequence.shape[:2])}."
                )
        if self.has_global_features:
            if global_features is None:
                raise ValueError("Model expects global_features, but none were provided.")
            if global_features.ndim != 2 or global_features.size(0) != sequence.size(0):
                raise ValueError(
                    f"Expected global_features shape [B, G] with B={sequence.size(0)}, "
                    f"got {tuple(global_features.shape)}."
                )

        # shape: (batch_size, sequence_length, d_model)
        x = self.input_proj(sequence)

        if self.initial_film is not None and global_features is not None:
            x = self.initial_film(x, global_features)

        for block_index, block in enumerate(self.blocks):
            x = block(x, global_features=global_features, sequence_mask=sequence_mask)
            if block_index < len(self.blocks) - 1:
                x = self.inter_layer_dropout(x)

        x = self.final_norm(x)
        return self.output_proj(x)


__all__ = ["LSTMPredictionModel"]
