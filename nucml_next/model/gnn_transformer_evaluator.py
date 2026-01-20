"""
GNN-Transformer Evaluator
=========================

Integrated model combining:
1. NuclideGNN: Learns isotope topology from Chart of Nuclides
2. EnergyTransformer: Predicts smooth cross-section curves σ(E)

This is the "solution" to the problems shown in Notebook 00!

Architecture Flow:
    Graph → GNN → Isotope Embeddings → Transformer → Cross Sections
          (topology)                  (smooth curves)

Benefits:
    ✓ Smooth predictions (no staircase effect)
    ✓ Physics-aware (learned from graph structure)
    ✓ Better extrapolation (embeddings capture relationships)
    ✓ Can enforce constraints (with physics-informed loss)
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt

from nucml_next.model.nuclide_gnn import NuclideGNN
from nucml_next.model.energy_transformer import EnergyTransformer, EnergyTransformerWithThreshold


class GNNTransformerEvaluator(nn.Module):
    """
    Combined GNN-Transformer model for nuclear data evaluation.

    This is the "hero" model that solves the limitations of classical ML.
    """

    def __init__(
        self,
        # GNN parameters
        node_features: int = 7,
        gnn_hidden_dim: int = 64,
        gnn_embedding_dim: int = 32,
        gnn_num_layers: int = 3,
        gnn_type: str = 'GCN',
        # Transformer parameters
        transformer_d_model: int = 128,
        transformer_nhead: int = 8,
        transformer_num_layers: int = 4,
        transformer_dim_feedforward: int = 512,
        # General parameters
        dropout: float = 0.1,
        use_threshold_model: bool = False,
    ):
        """
        Initialize GNN-Transformer evaluator.

        Args:
            node_features: Number of isotope features
            gnn_hidden_dim: GNN hidden dimension
            gnn_embedding_dim: GNN embedding dimension (isotope embedding)
            gnn_num_layers: Number of GNN layers
            gnn_type: 'GCN' or 'GAT'
            transformer_d_model: Transformer hidden dimension
            transformer_nhead: Number of attention heads
            transformer_num_layers: Number of transformer layers
            transformer_dim_feedforward: Transformer FFN dimension
            dropout: Dropout rate
            use_threshold_model: Use threshold-aware transformer
        """
        super().__init__()

        self.gnn_embedding_dim = gnn_embedding_dim

        # GNN for isotope embeddings
        self.gnn = NuclideGNN(
            node_features=node_features,
            hidden_dim=gnn_hidden_dim,
            embedding_dim=gnn_embedding_dim,
            num_layers=gnn_num_layers,
            gnn_type=gnn_type,
            dropout=dropout,
        )

        # Transformer for energy-dependent predictions
        if use_threshold_model:
            self.transformer = EnergyTransformerWithThreshold(
                isotope_embedding_dim=gnn_embedding_dim,
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                num_layers=transformer_num_layers,
                dim_feedforward=transformer_dim_feedforward,
                dropout=dropout,
            )
        else:
            self.transformer = EnergyTransformer(
                isotope_embedding_dim=gnn_embedding_dim,
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                num_layers=transformer_num_layers,
                dim_feedforward=transformer_dim_feedforward,
                dropout=dropout,
            )

        self.use_threshold_model = use_threshold_model

    def forward(
        self,
        graph_data: Data,
        isotope_indices: torch.Tensor,
        energies: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: predict cross sections.

        Args:
            graph_data: PyG Data object with nuclear graph
            isotope_indices: Indices of target isotopes [batch]
            energies: Energy values [batch, seq_len]

        Returns:
            Cross sections [batch, seq_len, 1]
        """
        # Step 1: Get isotope embeddings from GNN
        isotope_embeddings = self.gnn(graph_data)  # [num_nodes, gnn_embedding_dim]

        # Step 2: Select embeddings for target isotopes
        batch_embeddings = isotope_embeddings[isotope_indices]  # [batch, gnn_embedding_dim]

        # Step 3: Predict cross sections via Transformer
        cross_sections = self.transformer(energies, batch_embeddings)  # [batch, seq_len, 1]

        return cross_sections

    def predict_isotope(
        self,
        graph_data: Data,
        isotope_idx: int,
        energies: np.ndarray,
    ) -> np.ndarray:
        """
        Predict cross section for a single isotope.

        Args:
            graph_data: PyG Data object
            isotope_idx: Isotope node index
            energies: Energy array [num_energies]

        Returns:
            Cross sections [num_energies]
        """
        self.eval()

        with torch.no_grad():
            # Prepare inputs
            isotope_indices = torch.tensor([isotope_idx], dtype=torch.long)
            energies_tensor = torch.tensor(energies, dtype=torch.float32).unsqueeze(0)

            # Predict
            predictions = self.forward(graph_data, isotope_indices, energies_tensor)

            # Convert to numpy
            predictions_np = predictions.squeeze().cpu().numpy()

        return predictions_np

    def train_model(
        self,
        train_data: List[Tuple[Data, int, torch.Tensor, torch.Tensor]],
        val_data: Optional[List[Tuple[Data, int, torch.Tensor, torch.Tensor]]] = None,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = 'cpu',
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the GNN-Transformer model.

        Args:
            train_data: List of (graph, isotope_idx, energies, targets)
            val_data: Validation data (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: L2 regularization
            device: 'cpu' or 'cuda'
            verbose: Print progress

        Returns:
            Dictionary with training history
        """
        self.to(device)

        # Optimizer
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
        )

        # Loss function (MSE in log space for numerical stability)
        def loss_fn(pred, target):
            log_pred = torch.log10(pred + 1e-10)
            log_target = torch.log10(target + 1e-10)
            return nn.functional.mse_loss(log_pred, log_target)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
        }

        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_losses = []

            for graph, isotope_idx, energies, targets in train_data:
                # Move to device
                graph = graph.to(device)
                isotope_indices = torch.tensor([isotope_idx], dtype=torch.long).to(device)
                energies = energies.unsqueeze(0).to(device)
                targets = targets.unsqueeze(0).unsqueeze(-1).to(device)

                # Forward pass
                predictions = self.forward(graph, isotope_indices, energies)

                # Compute loss
                loss = loss_fn(predictions, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            if val_data is not None:
                self.eval()
                val_losses = []

                with torch.no_grad():
                    for graph, isotope_idx, energies, targets in val_data:
                        graph = graph.to(device)
                        isotope_indices = torch.tensor([isotope_idx], dtype=torch.long).to(device)
                        energies = energies.unsqueeze(0).to(device)
                        targets = targets.unsqueeze(0).unsqueeze(-1).to(device)

                        predictions = self.forward(graph, isotope_indices, energies)
                        loss = loss_fn(predictions, targets)
                        val_losses.append(loss.item())

                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)

                # Update learning rate
                scheduler.step(avg_val_loss)

                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.6f} - "
                          f"Val Loss: {avg_val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")

        return history

    def plot_training_history(self, history: Dict[str, List[float]]):
        """
        Plot training curves.

        Args:
            history: Training history from train_model()
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history and len(history['val_loss']) > 0:
            ax.plot(history['val_loss'], label='Val Loss', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss (MSE in log space)', fontsize=12, fontweight='bold')
        ax.set_title('GNN-Transformer Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.show()

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'gnn_embedding_dim': self.gnn_embedding_dim,
            'use_threshold_model': self.use_threshold_model,
        }, filepath)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath: str, device: str = 'cpu'):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(device)
        print(f"✓ Model loaded from {filepath}")


class GNNTransformerTrainer:
    """
    Training utility for GNN-Transformer models.

    Provides convenience methods for data preparation and training.
    """

    def __init__(self, model: GNNTransformerEvaluator):
        """
        Initialize trainer.

        Args:
            model: GNN-Transformer model
        """
        self.model = model

    def prepare_training_data(
        self,
        dataset,
        isotope_filter: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Tuple[Data, int, torch.Tensor, torch.Tensor]]:
        """
        Prepare training data from NucmlDataset.

        Args:
            dataset: NucmlDataset instance
            isotope_filter: List of (Z, A) tuples to include (None = all)

        Returns:
            List of (graph, isotope_idx, energies, targets)
        """
        # Build global graph
        graph_data = dataset.graph_builder.build_global_graph()

        # Prepare data samples
        training_samples = []

        # Get unique isotopes
        isotopes = dataset.df[['Z', 'A']].drop_duplicates()

        for _, row in isotopes.iterrows():
            Z, A = row['Z'], row['A']

            # Filter if needed
            if isotope_filter and (Z, A) not in isotope_filter:
                continue

            # Get isotope index
            isotope_idx = dataset.graph_builder.isotope_to_idx.get((Z, A))
            if isotope_idx is None:
                continue

            # Get cross-section data for this isotope
            isotope_data = dataset.df[(dataset.df['Z'] == Z) & (dataset.df['A'] == A)]

            # For each reaction type
            for mt_code in isotope_data['MT'].unique():
                reaction_data = isotope_data[isotope_data['MT'] == mt_code]

                energies = torch.tensor(reaction_data['Energy'].values, dtype=torch.float32)
                targets = torch.tensor(reaction_data['CrossSection'].values, dtype=torch.float32)

                training_samples.append((graph_data, isotope_idx, energies, targets))

        print(f"✓ Prepared {len(training_samples)} training samples")
        return training_samples
