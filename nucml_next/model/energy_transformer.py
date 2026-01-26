"""
Energy Transformer for Cross-Section Prediction
================================================

Transformer that learns to predict smooth cross-section curves σ(E).

Why Transformer for Energy Sequences?
    - Cross sections are CONTINUOUS functions of energy
    - Resonances have smooth shapes (Breit-Wigner, etc.)
    - Self-attention captures long-range correlations
    - Positional encoding handles logarithmic energy scale

This solves the smoothness problem that plagues tree-based models!

Architecture:
    Input: Energy sequence [E_1, E_2, ..., E_n] + isotope embedding from GNN
    Output: Cross-section sequence [σ(E_1), σ(E_2), ..., σ(E_n)]
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for energy sequences.

    Adapted for logarithmic energy scale (eV to MeV spans ~7 orders of magnitude).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Embedding dimension (must be even)
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EnergyTransformer(nn.Module):
    """
    Transformer for predicting smooth cross-section curves.

    The model learns to:
    1. Encode energy values with physical meaning
    2. Attend to relevant energy regions (resonances, thresholds)
    3. Generate smooth predictions (unlike trees!)

    Educational Benefit:
        Students will see smooth resonance curves that match physics,
        not jagged staircase predictions.
    """

    def __init__(
        self,
        isotope_embedding_dim: int = 32,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
    ):
        """
        Initialize Energy Transformer.

        Args:
            isotope_embedding_dim: Dimension of GNN isotope embeddings
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            max_seq_len: Maximum energy sequence length
        """
        super().__init__()

        self.isotope_embedding_dim = isotope_embedding_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Energy encoder: maps scalar energy to d_model dimensions
        self.energy_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Isotope context encoder: incorporates GNN embedding
        self.isotope_encoder = nn.Sequential(
            nn.Linear(isotope_embedding_dim, d_model),
            nn.ReLU(),
        )

        # Positional encoding (for sequence position)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # GELU works well for smooth functions
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-section predictor head
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),  # Ensures σ(E) > 0
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        energies: torch.Tensor,
        isotope_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: predict cross sections for energy sequence.

        Args:
            energies: Energy values [batch, seq_len] or [batch, seq_len, 1]
            isotope_embedding: GNN embedding for isotope [batch, isotope_embedding_dim]
            mask: Attention mask [batch, seq_len] (optional)

        Returns:
            Cross sections [batch, seq_len, 1]
        """
        batch_size, seq_len = energies.shape[0], energies.shape[1]

        # Ensure energies are [batch, seq_len, 1]
        if energies.dim() == 2:
            energies = energies.unsqueeze(-1)

        # Encode energies
        # We use log scale because energies span many orders of magnitude
        log_energies = torch.log10(energies + 1e-10)
        energy_features = self.energy_encoder(log_energies)  # [batch, seq_len, d_model]

        # Encode isotope context
        isotope_features = self.isotope_encoder(isotope_embedding)  # [batch, d_model]
        isotope_features = isotope_features.unsqueeze(1)  # [batch, 1, d_model]

        # Broadcast isotope features to all energy points
        isotope_features = isotope_features.expand(-1, seq_len, -1)  # [batch, seq_len, d_model]

        # Combine energy and isotope information
        x = energy_features + isotope_features  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)  # [batch, seq_len, d_model]

        # Predict cross sections
        cross_sections = self.predictor(x)  # [batch, seq_len, 1]

        return cross_sections

    def predict_with_uncertainty(
        self,
        energies: torch.Tensor,
        isotope_embedding: torch.Tensor,
        num_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty using Monte Carlo dropout.

        Args:
            energies: Energy values [batch, seq_len]
            isotope_embedding: Isotope embedding [batch, isotope_embedding_dim]
            num_samples: Number of MC samples

        Returns:
            (mean_prediction, std_prediction)

        Educational Use:
            Show students that uncertainty is higher in:
            - Resonance regions (complex physics)
            - Extrapolation regions (beyond training data)
        """
        self.train()  # Enable dropout

        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(energies, isotope_embedding)
                predictions.append(pred)

        predictions = torch.stack(predictions)  # [num_samples, batch, seq_len, 1]

        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        self.eval()  # Restore eval mode

        return mean_pred, std_pred


class EnergyTransformerWithThreshold(EnergyTransformer):
    """
    Enhanced transformer that explicitly handles reaction thresholds.

    For reactions like (n,2n), cross section MUST be zero below threshold.
    This variant enforces that constraint in the architecture.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with threshold awareness."""
        super().__init__(*args, **kwargs)

        # Threshold predictor (learns E_threshold from isotope)
        self.threshold_predictor = nn.Sequential(
            nn.Linear(self.isotope_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # Threshold must be positive
        )

    def forward(
        self,
        energies: torch.Tensor,
        isotope_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with threshold enforcement.

        Args:
            energies: Energy values [batch, seq_len]
            isotope_embedding: Isotope embedding [batch, isotope_embedding_dim]
            mask: Attention mask

        Returns:
            Cross sections [batch, seq_len, 1] with threshold applied
        """
        # Get base predictions
        cross_sections = super().forward(energies, isotope_embedding, mask)

        # Predict threshold energy
        threshold = self.threshold_predictor(isotope_embedding)  # [batch, 1]

        # Apply threshold: σ(E) = 0 for E < E_threshold
        # Use smooth Heaviside function for differentiability
        energies_2d = energies.unsqueeze(-1) if energies.dim() == 2 else energies
        threshold_mask = torch.sigmoid(10.0 * (energies_2d - threshold.unsqueeze(1)))

        cross_sections = cross_sections * threshold_mask

        return cross_sections

    def get_threshold(self, isotope_embedding: torch.Tensor) -> torch.Tensor:
        """
        Get predicted threshold energy for isotope.

        Args:
            isotope_embedding: Isotope embedding [batch, isotope_embedding_dim]

        Returns:
            Threshold energies [batch, 1]
        """
        return self.threshold_predictor(isotope_embedding)
