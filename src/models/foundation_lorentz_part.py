"""
Foundation LorentzParT — Extended Multi-Task Foundation Model
=============================================================
Extends the original 3-task model (classification + regression + reconstruction)
to a full foundation model with 5 tasks:

  1. Classification  : Jet tagging (10 classes, JetClass)
  2. Regression      : Jet mass prediction
  3. Reconstruction  : Masked particle autoencoder (pre-training)
  4. Generative      : Conditional VAE — sample new particle 4-momenta
  5. SuperResolution : Low-res → high-res jet upsampling

Architecture:
  Input [pT, deta, dphi, E] → multivector (16D)
        ↓
  FoundationLorentzParTEncoder  (EquiLinear + ParticleAttentionBlocks + EquiLinear)
        ↓
  embeddings (B,N,D)  +  equivariant_features (B,N,16)
        ↓
  Task dispatch → Classification | Regression | Reconstruction | Generative | SuperRes

Author: Ranjeet Gupta
Based on: Thanh Nguyen GSoC 2025 LorentzParT
"""

from typing import List, Tuple, Dict, Optional, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F

try:
    from lgatr.layers import EquiLinear
except ImportError:
    raise ImportError("Install lgatr: pip install lgatr")

try:
    from lgatr.interface import extract_vector
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Shared building blocks (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

class InteractionEmbedding(nn.Module):
    """
    Pairwise particle interaction → per-head attention bias.

    Computes (Δη, Δφ, ΔR, log(pT_i·pT_j)) between all particle pairs,
    maps to num_heads scalars, and adds them to attention logits.
    """
    def __init__(
        self,
        num_interaction_features: int = 4,
        pair_embed_dims: List[int] = [64, 64, 64],
    ):
        super().__init__()
        layers = []
        in_dim = num_interaction_features
        for out_dim in pair_embed_dims:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.pair_embed = nn.Sequential(*layers)

    def forward(self, U: Tensor) -> Tensor:
        """U: (B, N, N, F) → (B*num_heads, N, N)"""
        x = self.pair_embed(U)             # (B, N, N, num_heads)
        B, N, _, H = x.shape
        return x.permute(0, 3, 1, 2).reshape(B * H, N, N)


class ParticleAttentionBlock(nn.Module):
    """
    Pre-norm transformer block with interaction-biased multi-head attention.
    Identical to original implementation.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.k_proj   = nn.Linear(embed_dim, embed_dim)
        self.v_proj   = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm1    = nn.LayerNorm(embed_dim)
        self.norm2    = nn.LayerNorm(embed_dim)
        self.ffn      = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_factor),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, padding_mask: Tensor,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        residual = x
        x = self.norm1(x)

        def _split(proj):
            return proj(x).reshape(B, N, H, D).permute(0, 2, 1, 3).reshape(B*H, N, D)

        q, k, v = _split(self.q_proj), _split(self.k_proj), _split(self.v_proj)
        scores = torch.bmm(q, k.transpose(1, 2)) / (D ** 0.5)

        if attn_bias is not None:
            scores = scores + attn_bias

        if padding_mask is not None:
            key_mask = padding_mask.unsqueeze(1).unsqueeze(2)       # (B,1,1,N)
            key_mask = key_mask.expand(-1, H, 1, -1).reshape(B*H, 1, N)
            scores = scores.masked_fill(~key_mask, float('-inf'))

        w = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        w = self.dropout(w)
        self.last_attn_weights = w.detach()  # store for interpretability

        out = torch.bmm(w, v).reshape(B, H, N, D).permute(0, 2, 1, 3).reshape(B, N, C)
        x = residual + self.dropout(self.out_proj(out))

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


class FoundationLorentzParTEncoder(nn.Module):
    """
    Lorentz-equivariant particle encoder.
    EquiLinear → proj → N×ParticleAttentionBlock → proj_back → EquiLinear
    (Unchanged from original except added norm_out for stable fine-tuning.)
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 8,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        pair_embed_dims: List[int] = [64, 64, 64],
    ):
        super().__init__()

        self.equilinear_in = EquiLinear(
            in_mv_channels=1, out_mv_channels=1,
            in_s_channels=in_s_channels, out_s_channels=out_s_channels,
        )
        self.proj = nn.Linear(16, embed_dim)

        self.interaction_embed = InteractionEmbedding(
            num_interaction_features=4,
            pair_embed_dims=pair_embed_dims + [num_heads],
        )

        self.encoder = nn.ModuleList([
            ParticleAttentionBlock(embed_dim, num_heads, dropout, expansion_factor)
            for _ in range(num_layers)
        ])

        self.norm_out   = nn.LayerNorm(embed_dim)   # <-- added for stability
        self.proj_out   = nn.Linear(embed_dim, 16)
        self.equilinear_out = EquiLinear(
            in_mv_channels=1, out_mv_channels=1,
            in_s_channels=None, out_s_channels=None,
        )

    def forward(self, x: Tensor, padding_mask: Tensor, U: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x            : (B, N, 16)
        padding_mask : (B, N)      True = valid
        U            : (B, N, N, 4)
        Returns      : embeddings (B,N,D), equivariant_features (B,N,16)
        """
        B, N, F = x.shape
        attn_bias = self.interaction_embed(U)

        x_mv = x.view(B, N, 1, F)
        x_mv, _ = self.equilinear_in(x_mv)
        x_mv = x_mv.view(B, N, 16)

        x_embed = self.proj(x_mv)
        for layer in self.encoder:
            x_embed = layer(x_embed, padding_mask, attn_bias)
        x_embed = self.norm_out(x_embed)

        x_out = self.proj_out(x_embed).view(B, N, 1, 16)
        x_equi, _ = self.equilinear_out(x_out)
        x_equi = x_equi.view(B, N, 16)

        return x_embed, x_equi


# ─────────────────────────────────────────────────────────────────────────────
# Task heads
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Jet tagging: class-token attention + mean-pool skip → MLP.
    Skip connection prevents class-token collapse (critical fix).
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.class_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.class_token, std=0.02)
        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.class_attn    = nn.MultiheadAttention(embed_dim, 8, dropout=dropout, batch_first=True)
        self.combine       = nn.Linear(embed_dim * 2, embed_dim)
        self.combine_norm  = nn.LayerNorm(embed_dim)
        self.classifier    = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        B = x.shape[0]
        if padding_mask is not None:
            m = padding_mask.unsqueeze(-1).float()
            mean_embed = (x * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            mean_embed = x.mean(1)

        x_norm = self.pre_attn_norm(x)
        cls    = self.class_token.expand(B, -1, -1)
        attn_mask = ~padding_mask if padding_mask is not None else None
        cls_out, _ = self.class_attn(cls, x_norm, x_norm, key_padding_mask=attn_mask)
        cls_out    = cls_out.squeeze(1)

        fused  = self.combine_norm(self.combine(torch.cat([cls_out, mean_embed], dim=-1)))
        return self.classifier(fused)


class RegressionHead(nn.Module):
    """Predicts continuous jet properties (mass, pT, …)."""
    def __init__(
        self,
        embed_dim: int = 128,
        num_targets: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.aggregation = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim), nn.GELU(),
        )
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        if padding_mask is not None:
            m = padding_mask.unsqueeze(-1).float()
            x_agg = (x * m).sum(1) / m.sum(1, keepdim=True).squeeze(-1).clamp(min=1)
        else:
            x_agg = x.mean(1)
        return self.regressor(self.aggregation(x_agg))


class ReconstructionHead(nn.Module):
    """
    Masked Particle Autoencoder head (MPA pre-training).
    Given context embeddings, reconstructs the 4-momentum of masked particles.
    Combines transformer embeddings + Lorentz-equivariant features.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        equi_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_combine = nn.Sequential(
            nn.Linear(embed_dim + equi_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 4),  # pT, deta, dphi, E
        )

    def forward(self, x_embed: Tensor, x_equi: Tensor, mask_indices: Tensor) -> Tensor:
        """mask_indices: (B,) index of masked particle → returns (B, 4)"""
        B  = x_embed.shape[0]
        bi = torch.arange(B, device=x_embed.device)
        combined = torch.cat([x_embed[bi, mask_indices], x_equi[bi, mask_indices]], dim=-1)
        return self.reconstructor(self.feature_combine(combined))


class ConditionalVAEHead(nn.Module):
    """
    Conditional VAE generative head.

    Generates particle 4-momenta conditioned on the jet-level context embedding.

    Pre-training / fine-tuning for generation:
      • Encoder   q(z | particle, jet_ctx)  →  μ, log σ²
      • Decoder   p(x | z,        jet_ctx)  →  (pT, deta, dphi, E)
      • Loss = reconstruction MSE + β·KL  (β-VAE)

    Physics motivation: the jet context acts as a "prior" encoding the overall
    jet kinematics, while z captures particle-level deviations.
    """
    def __init__(
        self,
        embed_dim:    int   = 128,
        latent_dim:   int   = 32,
        particle_dim: int   = 4,
        hidden_dim:   int   = 256,
        dropout:      float = 0.1,
        beta:         float = 1.0,
    ):
        super().__init__()
        self.latent_dim   = latent_dim
        self.particle_dim = particle_dim
        self.beta         = beta

        # Encoder: q(z | particle_features, jet_context)
        self.enc_net = nn.Sequential(
            nn.Linear(particle_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        )
        self.mu_head     = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder: p(x | z, jet_context)
        self.dec_net = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, particle_dim),
        )

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            return mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return mu

    def forward(
        self,
        x_embed:      Tensor,  # (B, N, embed_dim)
        padding_mask: Tensor,  # (B, N)
        raw_particles: Tensor, # (B, N, 4)  normalized [pT, deta, dphi, E]
    ) -> Dict[str, Tensor]:
        """
        Returns dict:
          'recon'      : (B, N, 4)           per-particle reconstructions
          'mu'         : (B, N, latent_dim)
          'logvar'     : (B, N, latent_dim)
          'loss'       : scalar              recon_loss + β·kl_loss
          'recon_loss' : scalar
          'kl_loss'    : scalar
        """
        B, N, _ = x_embed.shape

        # Jet-level context = masked mean pool
        if padding_mask is not None:
            m = padding_mask.unsqueeze(-1).float()
            jet_ctx = (x_embed * m).sum(1) / m.sum(1).clamp(min=1)   # (B, D)
        else:
            jet_ctx = x_embed.mean(1)

        ctx_exp  = jet_ctx.unsqueeze(1).expand(-1, N, -1)           # (B, N, D)
        BN       = B * N
        p_flat   = raw_particles.reshape(BN, self.particle_dim)
        ctx_flat = ctx_exp.reshape(BN, -1)

        h          = self.enc_net(torch.cat([p_flat, ctx_flat], dim=-1))
        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z          = self._reparameterize(mu, logvar)
        recon      = self.dec_net(torch.cat([z, ctx_flat], dim=-1))

        recon  = recon.view(B, N, self.particle_dim)
        mu     = mu.view(B, N, self.latent_dim)
        logvar = logvar.view(B, N, self.latent_dim)

        # Loss over valid particles only
        mask_flat = padding_mask.reshape(BN).float() if padding_mask is not None \
                    else torch.ones(BN, device=x_embed.device)

        recon_loss = (F.mse_loss(recon.reshape(BN, -1), p_flat, reduction='none')
                      .mean(-1) * mask_flat).sum() / mask_flat.sum().clamp(min=1)
        kl_loss    = (-0.5 * (1 + logvar.reshape(BN, -1)
                               - mu.reshape(BN, -1).pow(2)
                               - logvar.reshape(BN, -1).exp()).sum(-1) * mask_flat
                      ).sum() / mask_flat.sum().clamp(min=1)

        return {
            "recon":      recon,
            "mu":         mu,
            "logvar":     logvar,
            "loss":       recon_loss + self.beta * kl_loss,
            "recon_loss": recon_loss,
            "kl_loss":    kl_loss,
        }

    @torch.no_grad()
    def generate(self, jet_context: Tensor, num_particles: int = 30) -> Tensor:
        """
        Sample new particles from the prior p(z) ~ N(0,I).
        jet_context : (B, embed_dim)
        Returns     : (B, num_particles, 4)
        """
        B   = jet_context.size(0)
        z   = torch.randn(B, num_particles, self.latent_dim, device=jet_context.device)
        ctx = jet_context.unsqueeze(1).expand(-1, num_particles, -1)
        return self.dec_net(torch.cat([z, ctx], dim=-1))


class SuperResolutionHead(nn.Module):
    """
    Jet super-resolution head: low-res → high-res particle upsampling.

    Task: given a jet with N_low particles, predict N_high > N_low particles.

    Strategy:
      1. Mean-pool low-res embeddings → jet context (for skip connection)
      2. N_high learnable seed queries represent target high-res positions
      3. Cross-attention: seed queries attend to low-res particle embeddings
      4. Project to 4-momentum predictions (pT, deta, dphi, E)

    Physics motivation: simulates the task of recovering fine-grained
    calorimeter deposits from coarser tracker-only information.
    """
    def __init__(
        self,
        embed_dim:  int   = 128,
        n_low:      int   = 30,
        n_high:     int   = 128,
        num_heads:  int   = 8,
        hidden_dim: int   = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_high          = n_high
        self.upsample_queries = nn.Parameter(torch.randn(1, n_high, embed_dim) * 0.02)
        self.cross_attn      = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm            = nn.LayerNorm(embed_dim)
        self.head            = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 4),   # pT, deta, dphi, E
        )

    def forward(self, x_embed: Tensor, padding_mask: Tensor) -> Tensor:
        """
        x_embed      : (B, N_low, embed_dim)
        padding_mask : (B, N_low)  True = valid
        Returns      : (B, n_high, 4)
        """
        B       = x_embed.size(0)
        queries = self.upsample_queries.expand(B, -1, -1)      # (B, n_high, D)
        kp_mask = ~padding_mask if padding_mask is not None else None
        attended, _ = self.cross_attn(queries, x_embed, x_embed,
                                       key_padding_mask=kp_mask)
        return self.head(self.norm(attended))                   # (B, n_high, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Full Foundation Model
# ─────────────────────────────────────────────────────────────────────────────

class FoundationLorentzParT(nn.Module):
    """
    Foundation Model for End-to-End Event Reconstruction in High-Energy Physics.

    5 tasks dispatched from a single shared encoder:
      'reconstruction'  — MPA pre-training (requires mask_indices)
      'classification'  — jet tagging
      'regression'      — jet mass / property prediction
      'generative'      — CVAE particle generation (requires raw_particles)
      'superresolution' — low-res → high-res upsampling
      'all'             — classification + regression + reconstruction simultaneously

    Parameters
    ----------
    max_num_particles    : max particles per jet
    num_particle_features: raw feature count (4: pT, deta, dphi, E)
    num_classes          : jet classes (10 for JetClass)
    num_regression_targets : regression output count
    embed_dim            : transformer width
    num_heads            : attention heads
    num_layers           : transformer depth
    dropout              : dropout rate
    expansion_factor     : FFN expansion
    pair_embed_dims      : pairwise MLP hidden dims (excl. final num_heads entry)
    latent_dim           : VAE latent dimension
    n_low / n_high       : super-resolution particle counts
    in_s_channels        : EquiLinear scalar input channels
    out_s_channels       : EquiLinear scalar output channels
    vae_beta             : β for KL weight in CVAE loss
    """

    def __init__(
        self,
        max_num_particles:     int            = 128,
        num_particle_features: int            = 4,
        num_classes:           int            = 10,
        num_regression_targets:int            = 1,
        embed_dim:             int            = 128,
        num_heads:             int            = 8,
        num_layers:            int            = 8,
        dropout:               float          = 0.1,
        expansion_factor:      int            = 4,
        pair_embed_dims:       List[int]      = [64, 64, 64],
        latent_dim:            int            = 32,
        n_low:                 int            = 30,
        n_high:                int            = 128,
        in_s_channels:         Optional[int]  = None,
        out_s_channels:        Optional[int]  = None,
        vae_beta:              float          = 1.0,
    ):
        super().__init__()

        self.max_num_particles     = max_num_particles
        self.num_particle_features = num_particle_features
        self.num_classes           = num_classes
        self.num_regression_targets= num_regression_targets

        # ── Shared encoder ────────────────────────────────────────────────
        self.encoder = FoundationLorentzParTEncoder(
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
            in_s_channels=in_s_channels, out_s_channels=out_s_channels,
            dropout=dropout, expansion_factor=expansion_factor,
            pair_embed_dims=pair_embed_dims,
        )

        # ── Task heads ────────────────────────────────────────────────────
        self.classification_head = ClassificationHead(
            embed_dim=embed_dim, num_classes=num_classes, dropout=dropout)

        self.regression_head = RegressionHead(
            embed_dim=embed_dim, num_targets=num_regression_targets,
            dropout=dropout)

        self.reconstruction_head = ReconstructionHead(
            embed_dim=embed_dim, equi_dim=16, dropout=dropout)

        self.generative_head = ConditionalVAEHead(
            embed_dim=embed_dim, latent_dim=latent_dim,
            particle_dim=num_particle_features, dropout=dropout, beta=vae_beta)

        self.superresolution_head = SuperResolutionHead(
            embed_dim=embed_dim, n_low=n_low, n_high=n_high,
            num_heads=num_heads, dropout=dropout)

    # ── Encoder control ───────────────────────────────────────────────────

    def freeze_encoder(self, freeze: bool = True):
        """Freeze/unfreeze encoder. Used for fine-tuning strategies."""
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    def unfreeze_top_k_layers(self, k: int):
        """
        Partial fine-tuning: freeze all but the last k transformer layers.
        Also unfreezes final norm, projection, and output EquiLinear.
        """
        self.freeze_encoder(freeze=True)
        for layer in list(self.encoder.encoder)[-k:]:
            for p in layer.parameters():
                p.requires_grad = True
        for mod in [self.encoder.norm_out, self.encoder.proj_out,
                    self.encoder.equilinear_out]:
            for p in mod.parameters():
                p.requires_grad = True

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x:             Tensor,
        padding_mask:  Tensor,
        U:             Tensor,
        task:          str             = "all",
        mask_indices:  Optional[Tensor] = None,
        raw_particles: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        x             : (B, N, 16)   multivector particle features
        padding_mask  : (B, N)       True = valid particle
        U             : (B, N, N, 4) pairwise interaction features
        task          : 'all' | 'classification' | 'regression' |
                        'reconstruction' | 'generative' | 'superresolution'
        mask_indices  : (B,)  required for 'reconstruction' and 'all'
        raw_particles : (B, N, 4)  normalized [pT, deta, dphi, E]
                        required for 'generative'

        Returns
        -------
        Dict[str, Tensor]
          'all'             → {classification, regression, reconstruction}
          'generative'      → {recon, mu, logvar, loss, recon_loss, kl_loss}
          'superresolution' → {high_res: (B, n_high, 4)}
          others            → single-key dict matching task name
        """
        embeddings, equivariant_features = self.encoder(x, padding_mask, U)
        outputs: Dict[str, Tensor] = {}

        if task in ("all", "classification"):
            outputs["classification"] = self.classification_head(embeddings, padding_mask)

        if task in ("all", "regression"):
            outputs["regression"] = self.regression_head(embeddings, padding_mask)

        if task in ("all", "reconstruction"):
            if mask_indices is None:
                raise ValueError("mask_indices required for reconstruction task")
            outputs["reconstruction"] = self.reconstruction_head(
                embeddings, equivariant_features, mask_indices)

        if task == "generative":
            if raw_particles is None:
                raise ValueError("raw_particles required for generative task")
            outputs = self.generative_head(embeddings, padding_mask, raw_particles)

        if task == "superresolution":
            outputs["high_res"] = self.superresolution_head(embeddings, padding_mask)

        if task not in ("all", "classification", "regression",
                         "reconstruction", "generative", "superresolution"):
            raise ValueError(
                f"Unknown task: {task!r}. Choose from: "
                "all, classification, regression, reconstruction, generative, superresolution"
            )

        return outputs

    # ── Generation convenience ────────────────────────────────────────────

    @torch.no_grad()
    def generate_particles(
        self,
        x: Tensor, padding_mask: Tensor, U: Tensor,
        num_particles: int = 30,
    ) -> Tensor:
        """
        Encode a jet then sample new particles from the CVAE prior.
        Returns (B, num_particles, 4).
        """
        embeddings, _ = self.encoder(x, padding_mask, U)
        if padding_mask is not None:
            m   = padding_mask.unsqueeze(-1).float()
            ctx = (embeddings * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            ctx = embeddings.mean(1)
        return self.generative_head.generate(ctx, num_particles)

    # ── Utilities ─────────────────────────────────────────────────────────

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_parameter_summary(self):
        components = {
            "Encoder (shared)":          self.encoder,
            "Classification Head":        self.classification_head,
            "Regression Head":            self.regression_head,
            "Reconstruction Head (MPA)":  self.reconstruction_head,
            "Generative Head (CVAE)":     self.generative_head,
            "Super-Resolution Head":      self.superresolution_head,
        }
        print("=" * 62)
        print(f"  {'Component':<42} {'Parameters':>14}")
        print("=" * 62)
        for name, mod in components.items():
            n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            print(f"  {name:<42} {n:>14,}")
        print("-" * 62)
        print(f"  {'TOTAL':<42} {self.get_num_parameters():>14,}")
        print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = FoundationLorentzParT(
        num_classes=10, embed_dim=128, num_heads=8, num_layers=4,
        n_low=30, n_high=128, latent_dim=32,
    )
    model.print_parameter_summary()

    B, N = 4, 128
    x        = torch.randn(B, N, 16)
    pmask    = torch.ones(B, N, dtype=torch.bool);  pmask[:, 100:] = False
    U        = torch.randn(B, N, N, 4)
    mask_idx = torch.randint(0, 100, (B,))
    raw      = torch.randn(B, N, 4)

    tests = [
        ("all",             dict(mask_indices=mask_idx)),
        ("classification",  {}),
        ("regression",      {}),
        ("reconstruction",  dict(mask_indices=mask_idx)),
        ("generative",      dict(raw_particles=raw)),
        ("superresolution", {}),
    ]
    print("\nOutput shapes:")
    for task, kwargs in tests:
        out = model(x, pmask, U, task=task, **kwargs)
        shapes = {k: tuple(v.shape) for k, v in out.items() if isinstance(v, Tensor)}
        print(f"  {task:<20}: {shapes}")