#!/usr/bin/env python3
"""
model.py - Sequence model factory with transformer support.

Architecture:
- Transformer encoder with sinusoidal positional encoding
- FiLM (Feature-wise Linear Modulation) for global context conditioning
- Support for torch.compile and torch.export

PADDING CONVENTION:
- Mask values: True = padding position, False = valid position
- This follows PyTorch's convention for src_key_padding_mask
- Padding positions are excluded from attention and loss computation

DATA ASSUMPTIONS:
- Input tensors are expected to have consistent dtype and device within a forward pass
- Sequence lengths can vary but must not exceed max_sequence_length
- Global features, if present, must match batch size
- The configured padding value is assumed to never occur naturally in data
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils import get_precision_config, validate_config

logger = logging.getLogger(__name__)


class SinePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding - stateless for multi-GPU safety."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding and hard-fail on sequence length overflow."""
        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length overflow: {seq_len} > {self.pe.size(1)}."
            )
        return x + self.pe[:, :seq_len, :].to(x.dtype)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.

    Modulates features using global context through learned scale and shift.
    Uses improved initialization for better training stability.
    """

    def __init__(
            self,
            context_dim: int,
            feature_dim: int,
            clamp_gamma: float = 2.0,
    ) -> None:
        """
        Initialize FiLM layer.

        Args:
            context_dim: Dimension of context vector
            feature_dim: Dimension of features to modulate
            clamp_gamma: Maximum magnitude for scale/shift
        """
        super().__init__()
        if float(clamp_gamma) <= 0.0:
            raise ValueError("clamp_gamma must be > 0.")

        # Project context to scale and shift parameters
        self.projection = nn.Linear(context_dim, feature_dim * 2)

        # Small-scale initialization for near-identity FiLM at training start
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)

        # Keep the clamp as a tensor buffer so export and dtype/device moves stay aligned.
        self.register_buffer(
            'clamp_gamma',
            torch.tensor(clamp_gamma, dtype=torch.float32),
            persistent=False
        )

    def forward(self, features: Tensor, context: Tensor) -> Tensor:
        """
        Apply FiLM modulation to features.

        Args:
            features: Feature tensor with shape ``[batch, seq_len, feature_dim]``
            context: Global context tensor with shape ``[batch, context_dim]``

        Returns:
            Modulated feature tensor with shape ``[batch, seq_len, feature_dim]``
        """
        # Get scale and shift parameters
        gamma_beta = self.projection(context)
        delta_gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        # Clamp both FiLM branches to the explicitly configured range.
        clamp_val = self.clamp_gamma.to(features.dtype)
        delta_gamma = torch.clamp(delta_gamma, -clamp_val, clamp_val)
        beta = torch.clamp(beta, -clamp_val, clamp_val)

        # Expand for sequence dimension
        delta_gamma = delta_gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # Apply FiLM transformation: y = x * (1 + γ) + β
        return features * (1.0 + delta_gamma) + beta


class DecomposedTransformerEncoderLayer(nn.Module):
    """
    Custom transformer encoder layer built from primitives.

    Fully compatible with torch.export and torch.compile.
    Implements the checked-in pre-norm GELU architecture directly.

    ATTENTION MASKING:
    - Uses src_key_padding_mask where True = padding position
    - Padding positions are prevented from attending or being attended to
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            batch_first: bool = True,
            use_qk_norm: bool = False,
            qkv_bias: bool = True,
            ffn_type: str = "gelu",
    ) -> None:
        """
        Initialize transformer encoder layer.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability for feedforward and residual
            attention_dropout: Dropout probability for attention
            batch_first: If True, expect batch dimension first
            use_qk_norm: Apply LayerNorm to per-head Q and K before SDPA
                (Henry et al. 2020). Stabilises attention logits.
            qkv_bias: Include bias on Q/K/V/out projections.
            ffn_type: "gelu" for the original Linear-GELU-Linear FFN,
                "swiglu" for the gated variant used in LLaMA / PaLM.
        """
        super().__init__()
        if not batch_first:
            raise ValueError("batch_first=False is not supported.")
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        ffn_type_norm = str(ffn_type).lower()
        if ffn_type_norm not in {"gelu", "swiglu"}:
            raise ValueError(f"ffn_type must be 'gelu' or 'swiglu' (got '{ffn_type}').")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.attention_dropout = float(attention_dropout)
        self.use_qk_norm = bool(use_qk_norm)
        self.ffn_type = ffn_type_norm

        # Explicit Q/K/V/out projections so we can plug in QK-Norm and SDPA
        # directly (replacing nn.MultiheadAttention).
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=qkv_bias)

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

        # Feed-forward network. Both variants land back in ``d_model`` and use
        # ``self.dropout`` between the activation and the down-projection so
        # the existing dropout knob keeps working unchanged.
        if self.ffn_type == "gelu":
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.activation = nn.GELU()
        else:
            self.swiglu_in = nn.Linear(d_model, 2 * dim_feedforward)
            self.swiglu_out = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for attention and feed-forward sub-layers."""
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        if self.ffn_type == "gelu":
            nn.init.xavier_uniform_(self.linear1.weight, gain=math.sqrt(2))
            nn.init.xavier_uniform_(self.linear2.weight)
            nn.init.zeros_(self.linear1.bias)
            nn.init.zeros_(self.linear2.bias)
        else:
            # SwiGLU: gate path uses SiLU (x * sigmoid(x)); xavier with the
            # default unit gain matches Shazeer's GLU-variants ablation.
            nn.init.xavier_uniform_(self.swiglu_in.weight)
            nn.init.xavier_uniform_(self.swiglu_out.weight)
            nn.init.zeros_(self.swiglu_in.bias)
            nn.init.zeros_(self.swiglu_out.bias)

    def _attention(
            self,
            x: Tensor,
            src_key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        """Run the QKV projection, optional QK-Norm, and SDPA attention."""
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # SDPA boolean masks: True = participate, False = masked out. The
        # incoming ``src_key_padding_mask`` uses True = padding (PyTorch
        # convention), so invert and broadcast over heads and queries.
        attn_mask: Optional[Tensor] = None
        if src_key_padding_mask is not None:
            attn_mask = (~src_key_padding_mask).view(bsz, 1, 1, seq_len)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(attn_out)

    def _feed_forward(self, x: Tensor) -> Tensor:
        """Apply the configured FFN variant."""
        if self.ffn_type == "gelu":
            return self.linear2(self.dropout(self.activation(self.linear1(x))))
        # SwiGLU: gate = SiLU(W1 x), up = W2 x, out = W3(gate * up).
        gate, up = self.swiglu_in(x).chunk(2, dim=-1)
        return self.swiglu_out(self.dropout(F.silu(gate) * up))

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through transformer encoder layer.

        Args:
            src: Source sequence (batch, seq_len, d_model)
            src_mask: Reserved for compatibility (must be None for SDPA path)
            src_key_padding_mask: Padding mask where True = padding position

        Returns:
            Transformed sequence
        """
        if src_mask is not None:
            raise ValueError(
                "DecomposedTransformerEncoderLayer no longer accepts src_mask; "
                "use src_key_padding_mask for padding."
            )

        x = src

        x2 = self.norm1(x)
        x2 = self._attention(x2, src_key_padding_mask)
        x = x + self.dropout1(x2)

        x2 = self.norm2(x)
        x2 = self._feed_forward(x2)
        x = x + self.dropout2(x2)

        return x


class TransformerBlock(nn.Module):
    """
    Combined transformer + optional FiLM block.

    Avoids dynamic control flow for export compatibility.
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float,
            attention_dropout: float,
            global_input_dim: int,
            film_clamp: float = 2.0,
            use_qk_norm: bool = False,
            qkv_bias: bool = True,
            ffn_type: str = "gelu",
    ) -> None:
        """
        Initialize transformer block with optional FiLM.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
            global_input_dim: Dimension of global features (0 for no FiLM)
            film_clamp: Clamping value for FiLM parameters
            use_qk_norm: Forwarded to the encoder layer.
            qkv_bias: Forwarded to the encoder layer.
            ffn_type: Forwarded to the encoder layer.
        """
        super().__init__()

        self.has_film = global_input_dim > 0

        # Transformer layer with separate attention dropout
        self.transformer = DecomposedTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attention_dropout=attention_dropout,
            batch_first=True,
            use_qk_norm=use_qk_norm,
            qkv_bias=qkv_bias,
            ffn_type=ffn_type,
        )

        # Optional FiLM layer
        if self.has_film:
            self.film = FiLMLayer(global_input_dim, d_model, clamp_gamma=film_clamp)

    def forward(
            self,
            x: Tensor,
            global_features: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input sequence
            global_features: Optional global context
            src_key_padding_mask: Padding mask (True = padding)

        Returns:
            Transformed sequence
        """
        # Always pass through transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Apply FiLM if it exists and global features are provided
        if self.has_film and global_features is not None:
            x = self.film(x, global_features)

        return x


class PredictionModel(nn.Module):
    """
    Transformer model for atmospheric profile regression.

    Features:
    - Deep transformer architecture with FiLM conditioning
    - Sinusoidal positional encoding
    - Export-friendly implementation
    - Proper padding mask handling (no output overwriting)
    - Pre-norm transformer blocks with explicit LayerNorm at input, within
      each residual block, and at the final encoder output
    """

    def __init__(
            self,
            input_dim: int,
            global_input_dim: int,
            output_dim: int,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            attention_dropout: float,
            max_sequence_length: int,
            film_clamp: float,
            output_head_divisor: int,
            output_head_dropout_factor: float,
            use_qk_norm: bool = False,
            qkv_bias: bool = True,
            ffn_type: str = "gelu",
    ) -> None:
        """
        Initialize prediction model.

        Args:
            input_dim: Dimension of input features
            global_input_dim: Dimension of global features (0 for none)
            output_dim: Dimension of output predictions
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
            max_sequence_length: Maximum allowed sequence length
            film_clamp: Clamping value for FiLM parameters
            output_head_divisor: Divisor for intermediate output projection dimension
            output_head_dropout_factor: Factor applied to dropout for lighter pre-output dropout
        """
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        if int(num_encoder_layers) <= 0:
            raise ValueError("num_encoder_layers must be a positive integer.")
        if int(dim_feedforward) <= 0:
            raise ValueError("dim_feedforward must be a positive integer.")
        if not (0.0 <= float(dropout) <= 1.0):
            raise ValueError("dropout must be in [0, 1].")
        if not (0.0 <= float(attention_dropout) <= 1.0):
            raise ValueError("attention_dropout must be in [0, 1].")
        if int(max_sequence_length) <= 0:
            raise ValueError("max_sequence_length must be a positive integer.")
        if float(film_clamp) <= 0:
            raise ValueError("film_clamp must be > 0.")
        if int(output_head_divisor) <= 0:
            raise ValueError("output_head_divisor must be a positive integer.")
        if not (0.0 <= float(output_head_dropout_factor) <= 1.0):
            raise ValueError("output_head_dropout_factor must be in [0, 1].")

        self.d_model = d_model
        self.has_global_features = global_input_dim > 0
        self.max_sequence_length = int(max_sequence_length)


        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # Positional encoding
        self.pos_encoder = SinePositionalEncoding(
            d_model=d_model, max_len=self.max_sequence_length
        )

        # Initial FiLM if we have global features
        if self.has_global_features:
            self.initial_film = FiLMLayer(global_input_dim, d_model, clamp_gamma=film_clamp)
        else:
            self.initial_film = None

        # Build transformer blocks with integrated FiLM
        self.blocks = nn.ModuleList()
        for _ in range(num_encoder_layers):
            self.blocks.append(
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    global_input_dim=global_input_dim,
                    film_clamp=film_clamp,
                    use_qk_norm=use_qk_norm,
                    qkv_bias=qkv_bias,
                    ffn_type=ffn_type,
                )
            )

        # Final normalization keeps the encoder explicitly pre-norm.
        self.final_norm = nn.LayerNorm(d_model)

        # - Intermediate layer with activation and dropout
        # - Final layer without activation (standard for regression)
        # - Lighter dropout before final projection
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

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights"""
        # Submodules whose linears are already initialized inside their owning
        # module (FiLM gates, the encoder layer's QKV / out / FFN projections).
        transformer_owned = {
            "linear1", "linear2",          # GELU FFN
            "swiglu_in", "swiglu_out",     # SwiGLU FFN
            "q_proj", "k_proj", "v_proj", "out_proj",  # Attention
        }
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name == "initial_film.projection" or name.endswith(".film.projection"):
                    continue
                if ".transformer." in name:
                    leaf = name.rsplit(".transformer.", 1)[1]
                    if leaf in transformer_owned:
                        continue

                # Check if this is the final output layer
                if name == "output_proj.4":
                    # Very small initialization for final regression layer
                    nn.init.trunc_normal_(module.weight, std=0.01)
                elif "output_proj" in name:
                    # Intermediate output layers
                    nn.init.trunc_normal_(module.weight, std=0.02)
                else:
                    # Standard initialization for other layers
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
        Forward pass through the model.

        Args:
            sequence: Input sequence (batch, seq_len, input_dim)
            global_features: Optional global features (batch, global_dim)
            sequence_mask: Padding mask where True = padding position

        Returns:
            Output predictions with shape ``[batch, seq_len, output_dim]``

        Note: Outputs at padding positions are NOT overwritten.
              The loss function handles masking these positions.
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
                raise ValueError(
                    "Model expects global_features, but none were provided."
                )
            if global_features.ndim != 2 or global_features.size(0) != sequence.size(0):
                raise ValueError(
                    f"Expected global_features shape [B, G] with B={sequence.size(0)}, "
                    f"got {tuple(global_features.shape)}."
                )

        # Project input features to model dimension
        x = self.input_proj(sequence)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply initial FiLM if it exists
        if self.initial_film is not None and global_features is not None:
            x = self.initial_film(x, global_features)

        # The transformer remains explicitly pre-norm; each block applies
        # LayerNorm before attention and feed-forward sublayers.
        for block in self.blocks:
            x = block(x, global_features, sequence_mask)

        # Apply final normalization (important for stability)
        x = self.final_norm(x)

        # Project to output dimension
        output = self.output_proj(x)

        return output


def create_prediction_model(
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        compile_model: bool = True,
) -> nn.Module:
    """
    Create a prediction model from configuration.

    Args:
        config: Configuration dictionary
        device: Device to place model on
        compile_model: Whether to apply torch.compile

    Returns:
        Initialized prediction model
    """
    validate_config(config)

    data_spec = config["data_specification"]
    model_params = config["model_hyperparameters"]
    model_type = str(model_params["model_type"]).lower()
    precision = get_precision_config(config)

    if device is None:
        device = torch.device("cpu")

    common_kwargs = {
        "input_dim": len(data_spec["input_variables"]),
        "global_input_dim": len(data_spec["global_variables"]),
        "output_dim": len(data_spec["target_variables"]),
        "d_model": int(model_params["d_model"]),
        "dropout": float(model_params["dropout"]),
        "max_sequence_length": int(model_params["max_sequence_length"]),
        "film_clamp": float(model_params["film_clamp"]),
        "output_head_divisor": int(model_params["output_head_divisor"]),
        "output_head_dropout_factor": float(model_params["output_head_dropout_factor"]),
    }

    if model_type == "transformer":
        transformer_params = model_params["transformer"]
        model = PredictionModel(
            nhead=int(transformer_params["nhead"]),
            num_encoder_layers=int(transformer_params["num_layers"]),
            dim_feedforward=int(transformer_params["dim_feedforward"]),
            attention_dropout=float(transformer_params["attention_dropout"]),
            use_qk_norm=bool(transformer_params.get("use_qk_norm", False)),
            qkv_bias=bool(transformer_params.get("qkv_bias", True)),
            ffn_type=str(transformer_params.get("ffn_type", "gelu")).lower(),
            **common_kwargs,
        )
    elif model_type == "lstm":
        from lstm_model import LSTMPredictionModel

        lstm_params = model_params["lstm"]
        model = LSTMPredictionModel(
            num_lstm_layers=int(lstm_params["num_layers"]),
            bidirectional=bool(lstm_params["bidirectional"]),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'.")

    model.to(device=device, dtype=precision["model_dtype"])

    # Conditionally compile the model
    if compile_model:
        misc = config["miscellaneous_settings"]
        compile_enabled = bool(misc["torch_compile"])
        compile_mode = str(misc["compile_mode"])

        # Check for torch.compile capability
        has_compile = hasattr(torch, "compile")

        if compile_enabled:
            if not has_compile:
                raise RuntimeError("torch.compile is enabled, but this PyTorch build does not support it.")
            if device.type != "cuda":
                raise RuntimeError("torch.compile is enabled, but selected device is not CUDA.")
            try:
                logger.info(f"Attempting torch.compile with mode='{compile_mode}'")

                # Compilation kwargs
                compile_kwargs = {"mode": compile_mode}

                # Use fullgraph for max optimization if no dynamic control flow
                if compile_mode == "max-autotune":
                    compile_kwargs["fullgraph"] = True

                model = torch.compile(model, **compile_kwargs)
                logger.info("Model compiled successfully")

            except Exception as e:
                raise RuntimeError(f"torch.compile failed: {e}") from e

    return model

__all__ = ["FiLMLayer", "PredictionModel", "create_prediction_model"]
