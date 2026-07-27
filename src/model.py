"""CrossModalFusionNet — 5-stream cross-modal transformer fusion.

Streams: RGB, ELA, Noise, DCT (EfficientNet-B0) + ViT-B/16 (RGB).
Fusion:  transformer encoder over the modality dimension, so each
         forensic stream attends to all the others.

Extracted from RealTimeMultiModal_Fixed.ipynb, sections 10 and 10b.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from transformers import ViTConfig, ViTModel

__all__ = ["CrossModalFusionNet", "ModalityTransformerEncoder", "ViTFeatureExtractor"]

NUM_STREAMS = 5


class ViTFeatureExtractor(nn.Module):
    """ViT-B/16 returning the CLS token, shape (B, 768)."""

    def __init__(self, unfreeze_last_blocks: int = 2, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        else:
            # Same architecture, random init — weights arrive via load_state_dict.
            # Avoids a ~330 MB download on every cold start at inference time.
            self.vit = ViTModel(ViTConfig())

        for p in self.vit.parameters():
            p.requires_grad = False
        if unfreeze_last_blocks > 0:
            for p in self.vit.encoder.layer[-unfreeze_last_blocks:].parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=x).last_hidden_state[:, 0]


class ModalityTransformerEncoder(nn.Module):
    """Transformer encoder over the modality dimension.

    Input:  (B, N, D) — N modalities, D feature dim
    Output: (B, D)    — mean-pooled attended features
    """

    def __init__(
        self,
        d_model: int = 1280,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_modalities: int = NUM_STREAMS,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Learnable per-modality embedding so the encoder can tell streams apart.
        self.modality_pos = nn.Parameter(torch.randn(1, n_modalities, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.modality_pos
        return self.transformer(x).mean(dim=1)


class CrossModalFusionNet(nn.Module):
    """Five-stream fusion network with cross-modal transformer attention.

    Args:
        num_classes: 2 for the CASIA2 binary model, 3 for the fine-tuned
            model (Authentic / Manipulated / AI-Generated).
        pretrained: True when training from ImageNet initialisation.
            Set False for inference — the checkpoint supplies all weights,
            so downloading backbone weights first is wasted bandwidth.
        freeze_until: number of leading EfficientNet blocks to freeze.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = False,
        freeze_until: int = 1,
        d_model: int = 1280,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None

        self.rgb_bb = efficientnet_b0(weights=weights)
        self.ela_bb = efficientnet_b0(weights=weights)
        self.noise_bb = efficientnet_b0(weights=weights)
        self.dct_bb = efficientnet_b0(weights=weights)

        for backbone in (self.rgb_bb, self.ela_bb, self.noise_bb, self.dct_bb):
            backbone.classifier = nn.Identity()
            for i, block in enumerate(backbone.features.children()):
                if i < freeze_until:
                    for p in block.parameters():
                        p.requires_grad = False

        self.vit = ViTFeatureExtractor(unfreeze_last_blocks=2, pretrained=pretrained)
        self.vit_proj = nn.Linear(768, d_model)

        self.modality_transformer = ModalityTransformerEncoder(
            d_model=d_model, n_heads=8, n_layers=2, dropout=0.1
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        ela: torch.Tensor,
        noise: torch.Tensor,
        dct: torch.Tensor,
    ) -> torch.Tensor:
        feats = torch.stack(
            [
                self.rgb_bb(rgb),
                self.ela_bb(ela),
                self.noise_bb(noise),
                self.dct_bb(dct),
                self.vit_proj(self.vit(rgb)),
            ],
            dim=1,
        )  # (B, 5, D)
        return self.classifier(self.modality_transformer(feats))