"""
Transformer Neural Network Brain for the Autonomous Herding Dog.
Processes heterogeneous world-aligned sensory modalities (Vision, Hearing, Smell, Proprioception)
with positional + modality-aware self-attention and a spatially-preserving readout
to output agile 2D holonomic acceleration commands (ax, ay) and bark triggers.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from config import (
    TRANSFORMER_D_MODEL,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_DIM_FEEDFORWARD,
)


class HerdingTransformerBrain(nn.Module):
    """
    Multi-Modal Sensory Transformer with Spatial Readout for Holonomic Shepherding.
    Sequence layout (17 tokens):
      [0..8]   9 Vision rays (left-to-right across FOV)
      [9..12]  4 Hearing quadrants (Front, Right, Behind, Left)
      [13..15] 3 Olfactory smell tokens (Herd Centroid, Furthest Straggler, Active Waypoint)
      [16]     Proprioception / Self token
    """

    NUM_TOKENS = 17

    def __init__(
        self,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dim_ff: int = TRANSFORMER_DIM_FEEDFORWARD,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Modality Input Projections
        self.proj_vision = nn.Linear(8, d_model)
        self.proj_hearing = nn.Linear(5, d_model)
        self.proj_smell = nn.Linear(4, d_model)
        self.proj_self = nn.Linear(8, d_model)

        # 2. Modality Type Embeddings (Vision=0, Hearing=1, Smell=2, Self=3)
        self.modality_emb = nn.Embedding(4, d_model)

        # 3. Per-token positional embeddings
        self.pos_emb = nn.Parameter(torch.empty(1, self.NUM_TOKENS, d_model))

        # 4. Transformer Encoder with Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Structured Multi-Modal Readout Head (8 * d_model -> 24 -> 3)
        readout_dim = d_model * 8
        self.action_head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, 24),
            nn.GELU(),
            nn.Linear(24, 3),  # [ax, ay, bark_prob]
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize linear/attention layers, unit LayerNorm scale, strong pos_emb, and neutral 2D bias."""
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

        for name, p in self.named_parameters():
            if name == "pos_emb":
                nn.init.normal_(p, mean=0.0, std=0.50)
            elif "self_attn" in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "self_attn" in name and p.dim() == 1:
                nn.init.constant_(p, 0.0)

        # Start neutral in 2D (ax=0, ay=0) and quiet (bark=-1.5)
        final_linear = self.action_head[-1]
        nn.init.normal_(final_linear.weight, mean=0.0, std=0.15)
        final_linear.bias.data.copy_(torch.tensor([0.0, 0.0, -1.5]))

    def forward(
        self,
        vision_tokens: torch.Tensor,   # Shape: [Batch, 9, 8]
        hearing_tokens: torch.Tensor,  # Shape: [Batch, 4, 5]
        smell_tokens: torch.Tensor,    # Shape: [Batch, 3, 4]
        self_token: torch.Tensor,      # Shape: [Batch, 1, 8]
    ) -> torch.Tensor:
        v_emb = self.proj_vision(vision_tokens) + self.modality_emb.weight[0]
        h_emb = self.proj_hearing(hearing_tokens) + self.modality_emb.weight[1]
        s_emb = self.proj_smell(smell_tokens) + self.modality_emb.weight[2]
        p_emb = self.proj_self(self_token) + self.modality_emb.weight[3]

        seq_tokens = torch.cat([v_emb, h_emb, s_emb, p_emb], dim=1) + self.pos_emb
        out = self.transformer(seq_tokens)

        v_left = out[:, 0:4, :].mean(dim=1)
        v_center = out[:, 4, :]
        v_right = out[:, 5:9, :].mean(dim=1)
        h_back = out[:, 11, :]
        s_herd = out[:, 13, :]
        s_strag = out[:, 14, :]
        s_wp = out[:, 15, :]
        p_self = out[:, 16, :]

        readout = torch.cat([v_left, v_center, v_right, h_back, s_herd, s_strag, s_wp, p_self], dim=-1)

        raw_actions = self.action_head(readout)
        ax = torch.tanh(raw_actions[:, 0])
        ay = torch.tanh(raw_actions[:, 1])
        bark = torch.sigmoid(raw_actions[:, 2])

        return torch.stack([ax, ay, bark], dim=-1)

    @torch.no_grad()
    def decide_action(
        self,
        vision_data: list[list[float]],
        hearing_data: list[list[float]],
        smell_data: list[list[float]],
        self_data: list[float],
    ) -> tuple[float, float, bool]:
        """
        Fast CPU inference for in-game decision making without gradient tracking.
        Returns:
            ax: float [-1.0, 1.0]
            ay: float [-1.0, 1.0]
            should_bark: bool
        """
        v_tensor = torch.as_tensor(vision_data, dtype=torch.float32).unsqueeze(0)
        h_tensor = torch.as_tensor(hearing_data, dtype=torch.float32).unsqueeze(0)
        s_tensor = torch.as_tensor(smell_data, dtype=torch.float32).unsqueeze(0)
        p_tensor = torch.as_tensor(self_data, dtype=torch.float32).view(1, 1, -1)

        actions = self.forward(v_tensor, h_tensor, s_tensor, p_tensor)[0]
        ax = float(actions[0].item())
        ay = float(actions[1].item())
        bark_prob = float(actions[2].item())

        return ax, ay, bark_prob > 0.72

    def get_flat_genome(self) -> np.ndarray:
        """Extracts all model parameters as a flattened 1D NumPy array."""
        params = [p.data.cpu().numpy().ravel() for p in self.parameters()]
        return np.concatenate(params)

    def set_flat_genome(self, flat_genome: np.ndarray):
        """Loads flattened 1D parameters directly into the neural network."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            arr = flat_genome[offset : offset + numel].reshape(p.shape)
            p.data.copy_(torch.from_numpy(arr))
            offset += numel
