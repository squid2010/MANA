import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_sum(src, index, dim=-1, dim_size=None):
    """
    Native PyTorch implementation of scatter_sum to avoid torch_scatter dependency
    which often hangs on macOS (Apple Silicon).
    """
    if dim_size is None:
        dim_size = index.max().item() + 1

    # Create the output tensor of zeros
    size = list(src.size())
    size[dim] = dim_size
    out = torch.zeros(size, dtype=src.dtype, device=src.device)

    # index_add_ expects the index to have the same number of dimensions as src?
    # No, index_add_ expects a 1D index tensor.
    # We just need to ensure shapes match for the operation.
    return out.index_add_(dim, index, src)


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
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Softplus ensures output >= 0, no upper bound (unlike Sigmoid's [0,1])
        self.activation = nn.Softplus()

    def forward(self, h_mol):
        """
        h_mol: (num_molecules, hidden_dim) molecular embeddings
        returns: (num_molecules, 1) singlet oxygen yield (non-negative, can exceed 1.0)
        """
        return self.activation(self.net(h_mol))


class MANA(nn.Module):
    def __init__(
        self,
        num_atom_types,
        hidden_dim=128,
        num_layers=4,
        num_rbf=20,
        tasks=None,
        lambda_mean=500.0,
        lambda_std=100.0,
    ):
        super().__init__()
        if tasks is None:
            tasks = ["lambda", "phi"]
        self.tasks = tasks

        self.embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.rbf = RadialBasisFunction(num_rbf)
        self.layers = nn.ModuleList(
            [PaiNNLayer(hidden_dim, num_rbf) for _ in range(num_layers)]
        )

        # Takes only the molecule embedding (128)
        self.lambda_head = LambdaMaxHead(hidden_dim)

        solvent_dim = 64
        # Encodes Dielectric Constant (1 float) -> Vector (64)
        self.solvent_encoder = nn.Sequential(
            nn.Linear(1, solvent_dim), nn.SiLU(), nn.Linear(solvent_dim, solvent_dim)
        )

        # Phi Head takes (Mol_Emb + Solv_Emb) = 128 + 64
        self.phi_head = PhiDeltaHead(hidden_dim + solvent_dim)

        self._init_weights()

        self.register_buffer("lambda_mean", torch.tensor(lambda_mean))
        self.register_buffer("lambda_std", torch.tensor(lambda_std))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        z, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # 1. Run Backbone
        dist = edge_attr[:, 0]
        rbf = self.rbf(dist)
        s = self.embedding(z)
        v = torch.zeros(s.size(0), s.size(1), 3, device=s.device)

        for layer in self.layers:
            s, v = layer(s, v, edge_index, edge_attr, rbf)

        # Molecular Embedding
        h_mol = scatter_sum(s, batch, dim=0)
        h_mol = h_mol / (torch.bincount(batch).unsqueeze(-1).float() + 1e-9)

        results = {}

        # 2. Lambda Head (Standard)
        if "lambda" in self.tasks:
            results["lambda"] = self.lambda_head(h_mol).squeeze(-1)

        # 3. Phi Head (Solvent Aware)
        if "phi" in self.tasks:
            # Expecting data.dielectric to be shape (Batch_Size, 1)
            if not hasattr(data, "dielectric"):
                raise ValueError("Model expects 'data.dielectric' attribute!")

            h_solv = self.solvent_encoder(data.dielectric)

            # Concatenate [Molecule, Solvent]
            h_combined = torch.cat([h_mol, h_solv], dim=1)
            results["phi"] = self.phi_head(h_combined).squeeze(-1)

        return results

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

        if "lambda" in self.tasks and hasattr(batch, "lambda_max"):
            mask = torch.isfinite(batch.lambda_max.squeeze())
            if mask.any():
                pred_norm = (preds["lambda"][mask] - self.lambda_mean) / self.lambda_std
                target_norm = (
                    batch.lambda_max[mask] - self.lambda_mean
                ) / self.lambda_std

                loss_lambda = F.huber_loss(pred_norm, target_norm, delta=1.0)
                loss += loss_lambda
                metrics["loss_lambda"] = loss_lambda.item()

        if "phi" in self.tasks and hasattr(batch, "phi_delta"):
            mask = torch.isfinite(batch.phi_delta.squeeze())
            # if mask.sum() > 1:
            #     phi_pred = preds["phi"][mask]
            #     phi_true = batch.phi_delta[mask]

            #     diff_pred = phi_pred.unsqueeze(1) - phi_pred.unsqueeze(0)
            #     diff_true = phi_true.unsqueeze(1) - phi_true.unsqueeze(0)

            #     loss_phi = F.relu(-diff_pred * diff_true).mean()
            #     loss += loss_phi
            #     metrics["loss_phi"] = loss_phi.item()
            if mask.any():
                # Use Huber loss for robustness to outliers in phi values
                loss_phi = F.huber_loss(
                    preds["phi"][mask], batch.phi_delta[mask], delta=0.5
                )
                loss += loss_phi
                metrics["loss_phi"] = loss_phi.item()

        return loss, metrics

    def freeze_backbone(self):
        """
        Freeze the backbone layers (embedding, RBF, PaiNN layers, lambda_head).
        Only phi_head and solvent_encoder remain trainable.
        """
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.rbf.parameters():
            param.requires_grad = False
        for param in self.layers.parameters():
            param.requires_grad = False
        for param in self.lambda_head.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen (embedding + RBF + PaiNN layers + lambda_head)")
