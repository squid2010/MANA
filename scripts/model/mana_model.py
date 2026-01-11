import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum


class RadialBasisFunction(nn.Module):
    """
    Module to compute radial basis functions (RBFs) for given distances.
    Uses Gaussian RBFs centered at specified points with given widths.
    """

    def __init__(self, num_rbf, cutoff=5.0):
        """
        num_rbf: Number of radial basis functions.
        cutoff: Cutoff distance for the RBFs.
        """

        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.gamma = nn.Parameter(torch.ones(num_rbf), requires_grad=False)

    def forward(self, distances):
        """
        Defines the forward pass to compute RBFs.
        distances: Tensor of shape (num_edges,) containing distances.
        Returns: Tensor of shape (num_edges, num_rbf) containing RBF values.
        """

        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff**2)


class PaiNNLayer(nn.Module):
    """
    A single layer of the PaiNN architecture.
    """

    def __init__(self, hidden_dim, num_rbf):
        """
        Initializes the PaiNN layer.
        hidden_dim: Dimension of the hidden features.
        num_rbf: Number of radial basis functions.
        """

        super().__init__()

        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

        self.update_net = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(self, s, v, edge_index, edge_attr, rbf):
        """
        PaiNN message passing with strict E(3)-equivariance.

        s: (N, F) scalar features
        v: (N, F, 3) vector features
        edge_index: (2, E)
        edge_attr: (E, 4) = (distance, dx, dy, dz)
        rbf: (E, num_rbf)
        """

        row, col = edge_index
        directions = edge_attr[:, 1:4]  # (E, 3)

        phi_ss, phi_vv, phi_sv = self.filter_net(rbf).chunk(3, dim=-1)

        m_s = phi_ss * s[col]
        m_v = phi_vv.unsqueeze(-1) * v[col] + phi_sv.unsqueeze(
            -1
        ) * directions.unsqueeze(1) * s[col].unsqueeze(-1)

        m_s = scatter_sum(m_s, row, dim=0, dim_size=s.size(0))
        m_v = scatter_sum(m_v, row, dim=0, dim_size=v.size(0))

        v_norm = torch.norm(m_v, dim=-1)
        delta_s, alpha, beta = self.update_net(
            torch.cat([s, m_s, v_norm], dim=-1)
        ).chunk(3, dim=-1)

        s = s + delta_s
        v = alpha.unsqueeze(-1) * v + beta.unsqueeze(-1) * m_v

        return s, v


class LambdaMaxHead(nn.Module):
    """
    Predicts absorption maximum (lambda_max) from molecular embedding
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_mol):
        """
        h_mol: (num_molecules, hidden_dim) molecular embeddings
        returns: (num_molecules, 1) lambda_max
        """
        return self.net(h_mol)


class PhiDeltaHead(nn.Module):
    """
    Predicts a singlet oxygen from molecular embedding
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Yield between 0 and 1
        )

    def forward(self, h_mol):
        """
        h_mol: (num_molecules, hidden_dim) molecular embeddings
        returns: (num_molecules, 1) singlet oxygen yield
        """
        return self.net(h_mol)


class MANA(nn.Module):
    def __init__(
        self,
        num_atom_types,
        hidden_dim=128,
        num_layers=4,
        num_rbf=20,
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.rbf = RadialBasisFunction(num_rbf)

        self.layers = nn.ModuleList(
            [PaiNNLayer(hidden_dim, num_rbf) for _ in range(num_layers)]
        )
        self.lambda_head = LambdaMaxHead(hidden_dim)
        self.phi_head = PhiDeltaHead(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        """
        Defines the forward pass of the MANA
        data: A batch of molecular graphs with attributes:
            - x: Atomic numbers (num_atoms,)
            - edge_index: Edge indices (2, num_edges)
            - batch: Batch indices for atoms (num_atoms,)
            - pos: Atomic positions (num_atoms, 3)
        Returns: Dictionary with predictions:
            - lambda_max: Predicted absorption maximum (num_molecules, 1)
            - phi: Predicted singlet oxygen yield (num_molecules, 1)
        """
        z, pos, edge_index, batch = (
                    data.x,
                    data.pos,
                    data.edge_index,
                    data.batch,
                )

        row, col = edge_index
        diff = pos[col] - pos[row]
        dist = torch.norm(diff, dim=1)
        
        edge_attr = torch.cat(
            [dist.unsqueeze(-1), diff / (dist.unsqueeze(-1) + 1e-8)], dim=1
        )
        rbf = self.rbf(dist)

        s = self.embedding(z)
        v = torch.zeros(s.size(0), s.size(1), 3, device=s.device)

        for layer in self.layers:
            s, v = layer(s, v, edge_index, edge_attr, rbf)

        # Molecular embeddings by summing atomic features
        h_mol = scatter_sum(s, batch, dim=0)
        h_mol /= torch.bincount(batch).unsqueeze(-1)

        lambda_max = self.lambda_head(h_mol).squeeze(-1)
        phi_delta = self.phi_head(h_mol).squeeze(-1)

        return {
                    "lambda": lambda_max,
                    "phi": phi_delta,
                }

    def loss_fn(self, preds, batch):
        """
        Defines the loss function for training.
        preds: Tuple of model predictions (lambda_max, phi_delta)
        batch: Batch of data with ground truth values:
            - batch.lambda_max : (B,) or NaN
            - batch.phi_delta  : (B,) or NaN
        Returns:
            - total_loss: Weighted sum of individual losses.
            - loss_lambda: Absorption maximum loss.
            - loss_phi: Singlet oxygen yield loss.
        """
        loss = 0
        metrics = {}
        
        if hasattr(batch, "lambda_max"):
            mask = torch.isfinite(batch.lambda_max)
            if mask.any():
                loss_lambda = F.huber_loss(
                    preds["lambda"][mask], 
                    batch.lambda_max[mask],
                    delta = 20.0,
                )
                loss += loss_lambda
                metrics["loss_lambda"] = loss_lambda.item()
                
        if hasattr(batch, "phi_delta"):
            mask = torch.isfinite(batch.phi_delta)
            # if mask.sum() > 1:
            #     phi_pred = preds["phi"][mask]
            #     phi_true = batch.phi_delta[mask]
        
            #     diff_pred = phi_pred.unsqueeze(1) - phi_pred.unsqueeze(0)
            #     diff_true = phi_true.unsqueeze(1) - phi_true.unsqueeze(0)
        
            #     loss_phi = F.relu(-diff_pred * diff_true).mean()
            #     loss += loss_phi
            #     metrics["loss_phi"] = loss_phi.item()
            if mask.any():
                loss_phi = F.mse_loss(
                    preds["phi"][mask],
                    batch.phi_delta[mask],
                )
                loss += loss_phi
                metrics["loss_phi"] = loss_phi.item()
                
        return loss, metrics
