"""network.py
================
Core neural‑network components for **CPU‑Net**.

This module defines two groups of models that power the CycleGAN
translation between simulated and measured HPGe detector pulses:

* **PositionalUNet** – a 1‑D U‑Net enhanced with layer‑wise positional
  encodings and a *re‑parameterisation* bottleneck (à la Variational
  Auto‑Encoder) to increase stochastic capacity.  It serves as both the
  *REN* (sim→data) and *IREN* (data→sim) generators in the pipeline.

* **RNN** – a single‑layer bidirectional GRU discriminator augmented
  with a Bahdanau‐style attention head; used for both source‑domain and
  target‑domain adversaries.

All code is *pure‑PyTorch* (≥1.13) and keeps external dependencies to a
minimum so that portable execution on clusters and workstations is
straight‑forward.

Example
-------
>>> from network import PositionalUNet, RNN
>>> gen = PositionalUNet().cuda()
>>> disc = RNN().cuda()
>>> dummy = torch.randn(8, 1, 800, device="cuda")
>>> out   = gen(dummy)           # (8, 1, 800)
>>> score = disc(out)            # (8, 1)

Notes
-----
* The `SEQ_LEN` constant is imported from :pymod:`dataset` to stay
  consistent with the length used during preprocessing.
* Convolution kernel sizes were optimised for HPGe pulse shapes; feel
  free to expose them as hyper‑parameters for other detectors.
* Gradient checkpointing is not enabled by default – memory is rarely a
  bottleneck on A100s for these models, but can be added via
  ``torch.utils.checkpoint`` if needed.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import SEQ_LEN  # waveform length after alignment/padding

# -----------------------------------------------------------------------------
# Helper blocks
# -----------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive 1‑D convolutions **Conv→BN→LeakyReLU** ×2.

    Parameters
    ----------
    in_channels : int
        Number of input feature maps.
    out_channels : int
        Number of output feature maps.
    mid_channels : int, optional
        Hidden channel count.  If *None*, defaults to ``out_channels``.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 – simple forward
        return self.double_conv(x)


class Down(nn.Module):
    """Down‑sampling block = **MaxPool/2** → :class:`DoubleConv`."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Up‑sampling block with either bilinear **Upsample** or `ConvTranspose1d`.

    Parameters
    ----------
    in_channels : int
        Channel count of the *concatenated* feature maps
        (skip‑connection ⊕ upsampled tensor).
    out_channels : int
        Desired output channels after the fusion convolution.
    bilinear : bool, default ``True``
        If *True* use ``nn.Upsample``; otherwise learnable transpose‑conv.
    """

    def __init__(self, in_channels: int, out_channels: int, *, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad if necessary so that both tensors align
        diff = x2.size(2) - x1.size(2)
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    """Final *1×1* convolution to map features → single waveform channel."""

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 – simple forward
        return self.conv(x)


class PositionalEncoding(nn.Module):
    """Add deterministic sin/cos positional encodings to a 1‑D feature map.

    This mirrors the formulation from the original *Transformer* paper but
    is adapted to channel‑first tensors (``N,C,L``).

    Parameters
    ----------
    d_model : int
        Channel dimension of the tensor the encoding will be added to.
    start : int, default ``0``
        Starting offset in the pre‑computed positional table.  Useful for
        the *decoding* path where the spatial resolution differs.
    dropout : float, default ``0.1``
        Drop‑out probability applied *after* adding the encoding.
    max_len : int, default ``10000``
        Maximum sequence length for which to pre‑compute the table.
    factor : float, default ``1.0``
        Multiplicative factor applied to the encoding before addition –
        allows progressively blending‐in positional information.
    """

    def __init__(self, d_model: int, *, start: int = 0, dropout: float = 0.1,
                 max_len: int = 10000, factor: float = 1.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.factor = factor
        self.start = start

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(1, 2))  # shape (1, C, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 – simple forward
        x = x + self.factor * self.pe[:, :, self.start:self.start + x.size(2)]
        return self.dropout(x)

# -----------------------------------------------------------------------------
# Generator – Positional UNet
# -----------------------------------------------------------------------------

class PositionalUNet(nn.Module):
    """U‑Net‑like generator with positional encodings and VAE bottleneck."""

    def __init__(self, *, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear
        mult = 40  # base channel multiplier
        fac = 2 if bilinear else 1

        # Contracting path
        self.inc   = DoubleConv(1, mult)
        self.down1 = Down(mult,   mult * 2)
        self.down2 = Down(mult*2, mult * 4)
        self.down3 = Down(mult*4, mult * 8)
        self.down4 = Down(mult*8, mult * 16 // fac)

        # VAE‑style re‑parameterisation at the bottleneck
        self.fc_mean = nn.Conv1d(mult * 16 // fac, mult * 16 // fac, 1)
        self.fc_var  = nn.Conv1d(mult * 16 // fac, mult * 16 // fac, 1)

        # Expanding path
        self.up1  = Up(mult*16, mult * 8 // fac, bilinear)
        self.up2  = Up(mult*8,  mult * 4 // fac, bilinear)
        self.up3  = Up(mult*4,  mult * 2 // fac, bilinear)
        self.up4  = Up(mult*2,  mult // fac,     bilinear)
        self.outc = OutConv(mult // fac, 1)

        # Positional encodings (distinct for each spatial scale)
        self.pe1 = PositionalEncoding(mult)
        self.pe2 = PositionalEncoding(mult*2)
        self.pe3 = PositionalEncoding(mult*4)
        self.pe4 = PositionalEncoding(mult*8)
        self.pe5 = PositionalEncoding(mult*16//fac)
        self.pe6 = PositionalEncoding(mult*8//fac,  start=mult*4)
        self.pe7 = PositionalEncoding(mult*4//fac,  start=mult*2)
        self.pe8 = PositionalEncoding(mult*2//fac,  start=mult*2)

    # ---------------------------------------------------------------------
    # Helper
    # ---------------------------------------------------------------------

    @staticmethod
    def _reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE re‑parameterisation trick: *z = μ + σ·ϵ*."""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 – simple forward
        # Contracting ----------------------------------------------------------------
        x1 = self.pe1(self.inc(x))
        x2 = self.pe2(self.down1(x1))
        x3 = self.pe3(self.down2(x2))
        x4 = self.pe4(self.down3(x3))
        x5 = self.down4(x4)

        # Re‑parameterisation ---------------------------------------------------------
        x5 = self.pe5(self._reparametrize(self.fc_mean(x5), self.fc_var(x5)))

        # Expanding ------------------------------------------------------------------
        x = self.pe6(self.up1(x5, x4))
        x = self.pe7(self.up2(x,  x3))
        x = self.pe8(self.up3(x,  x2))
        x = self.up4(x, x1)

        return self.outc(x)

# -----------------------------------------------------------------------------
# Discriminator – Attention‑coupled GRU
# -----------------------------------------------------------------------------

class RNN(nn.Module):
    """Bidirectional GRU discriminator with attention head.

    Parameters
    ----------
    get_attention : bool, default ``False``
        If *True*, :py:meth:`forward` will *return the attention weights*
        instead of a real/fake score – handy for visualisation.
    """

    def __init__(self, *, get_attention: bool = False):
        super().__init__()
        self.get_attention = get_attention

        # Hyper‑parameters
        self.seg       = 1            # down‑sample factor (segment length)
        self.emb_dim   = 64           # lookup‑table dimension
        self.emb_tick  = 1 / 1000.0   # quantisation step for amplitudes
        self.seq_len   = SEQ_LEN // self.seg
        feed_dim       = 128

        # Layers ------------------------------------------------------------------
        self.embedding = nn.Embedding(int(1 / self.emb_tick), self.emb_dim)
        self.gru = nn.GRU(
            input_size=self.emb_dim,
            hidden_size=feed_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        feed_dim *= 2  # bidirectional doubles hidden size

        self.att_lin = nn.Linear(feed_dim // 2, feed_dim // 2, bias=False)
        self.fcnet   = nn.Linear(feed_dim, 1)

    # -------------------------------------------------------------------------
    # Forward helpers
    # -------------------------------------------------------------------------

    def _calculate_attention(self, output: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Cosine‑similarity attention scores *α* ∈ [0,1] for each timestep."""
        inner = torch.einsum("ijl,il->ij", output, hidden)
        denom = torch.linalg.norm(output, dim=-1) * torch.linalg.norm(hidden, dim=-1, keepdim=True)
        return torch.softmax(inner / (denom + 1e-8), dim=-1)

    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 – simple forward
        # --- preprocessing: normalise and embed ----------------------------------
        x = x.view(-1, self.seq_len)
        x = (x - x.min(dim=-1, keepdim=True).values) / (x.max(dim=-1, keepdim=True).values - x.min(dim=-1, keepdim=True).values)
        x = (x / self.emb_tick).long()
        x = self.embedding(x)

        # --- GRU -----------------------------------------------------------------
        out_seq, hidden = self.gru(x)
        hidden = hidden.transpose(0, 1).reshape(x.size(0), -1)  # concat bidirectional

        # --- Attention -----------------------------------------------------------
        attn = self._calculate_attention(out_seq, hidden)
        if self.get_attention:
            return attn  # shape (N, L)

        context = torch.sum(attn.unsqueeze(-1) * out_seq, dim=1)
        score   = self.fcnet(torch.cat([context, hidden], dim=-1))
        return torch.sigmoid(score)
