from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn


NUM_STYLES = 3


def suggest_reconstruction_weight(trial: Any, low: float = 10.0, high: float = 200.0) -> float:
    """Suggest lambda_rec for Optuna; 100 is only an initial reference value."""
    return trial.suggest_float("lambda_rec", low, high, log=True)


class ConditionalUNetGenerator(nn.Module):
    """U-Net generator with a learned style embedding injected at the bottleneck."""

    def __init__(self, style_count: int = NUM_STYLES, style_dim: int = 64, base_channels: int = 64):
        super().__init__()
        self.style_embedding = nn.Embedding(style_count, style_dim)
        self.encoder1 = self._down(3, base_channels, normalize=False)
        self.encoder2 = self._down(base_channels, base_channels * 2)
        self.encoder3 = self._down(base_channels * 2, base_channels * 4)
        self.encoder4 = self._down(base_channels * 4, base_channels * 8)
        self.encoder5 = self._down(base_channels * 8, base_channels * 8)
        self.style_projection = nn.Sequential(
            nn.Linear(style_dim, base_channels * 8),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = self._up(base_channels * 16, base_channels * 8)
        self.decoder2 = self._up(base_channels * 16, base_channels * 4)
        self.decoder3 = self._up(base_channels * 8, base_channels * 2)
        self.decoder4 = self._up(base_channels * 4, base_channels)
        self.output = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    @staticmethod
    def _down(in_channels: int, out_channels: int, normalize: bool = True) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    @staticmethod
    def _up(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, photograph: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        skip1 = self.encoder1(photograph)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)
        bottleneck = self.encoder5(skip4)
        style_vector = self.style_projection(self.style_embedding(style))
        style_map = style_vector[:, :, None, None].expand(-1, -1, bottleneck.size(2), bottleneck.size(3))
        fused = torch.cat([bottleneck, style_map], dim=1)
        decoded1 = self.decoder1(fused)
        decoded2 = self.decoder2(torch.cat([decoded1, skip4], dim=1))
        decoded3 = self.decoder3(torch.cat([decoded2, skip3], dim=1))
        decoded4 = self.decoder4(torch.cat([decoded3, skip2], dim=1))
        return self.output(torch.cat([decoded4, skip1], dim=1))


class ConditionalPatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator conditioned on photo, sketch, and learned style embedding."""

    def __init__(self, style_count: int = NUM_STYLES, style_dim: int = 64, base_channels: int = 64):
        super().__init__()
        self.style_embedding = nn.Embedding(style_count, style_dim)
        self.style_projection = nn.Sequential(
            nn.Linear(style_dim, 16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.network = nn.Sequential(
            self._block(3 + 3 + 16, base_channels, normalize=False),
            self._block(base_channels, base_channels * 2),
            self._block(base_channels * 2, base_channels * 4),
            self._block(base_channels * 4, base_channels * 8, stride=1),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int, normalize: bool = True, stride: int = 2) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, photograph: torch.Tensor, sketch: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        style_vector = self.style_projection(self.style_embedding(style))
        style_map = style_vector[:, :, None, None].expand(-1, -1, photograph.size(2), photograph.size(3))
        conditioned_input = torch.cat([photograph, sketch, style_map], dim=1)
        return self.network(conditioned_input)


def generator_loss(
    discriminator_fake: torch.Tensor,
    generated_sketch: torch.Tensor,
    target_sketch: torch.Tensor,
    lambda_rec: float,
) -> torch.Tensor:
    if lambda_rec < 0.0:
        raise ValueError(f"lambda_rec must be non-negative, got {lambda_rec}.")
    adversarial = nn.functional.binary_cross_entropy_with_logits(
        discriminator_fake, torch.ones_like(discriminator_fake)
    )
    reconstruction = nn.functional.l1_loss(generated_sketch, target_sketch)
    return adversarial + lambda_rec * reconstruction


def discriminator_loss(
    discriminator_real: torch.Tensor,
    discriminator_fake: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        nn.functional.binary_cross_entropy_with_logits(discriminator_real, torch.ones_like(discriminator_real))
        + nn.functional.binary_cross_entropy_with_logits(discriminator_fake.detach(), torch.zeros_like(discriminator_fake))
    )


def conditional_gan_losses(
    discriminator_real: torch.Tensor,
    discriminator_fake: torch.Tensor,
    generated_sketch: torch.Tensor,
    target_sketch: torch.Tensor,
    lambda_rec: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return generator and discriminator BCE-with-logits plus paired L1 losses."""
    return (
        generator_loss(discriminator_fake, generated_sketch, target_sketch, lambda_rec),
        discriminator_loss(discriminator_real, discriminator_fake),
    )
