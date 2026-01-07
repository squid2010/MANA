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


class DipoleHead(nn.Module):
    """
    Equivariant head to predict dipole moments from scalar and vector features.
    """

    def __init__(self, hidden_dim):
        """
        hidden_dim: Dimension of the hidden features.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, v, batch):
        """
        v: Vector features of shape (num_atoms, hidden_dim, 3)
        batch: Tensor of shape (num_atoms,) indicating molecule indices.
        Returns: Tensor of shape (num_molecules, 3) containing predicted dipole moments.
        """
        mu_atom = torch.einsum("nfk,f->nk", v, self.weight)
        return scatter_sum(mu_atom, batch, dim=0)


class EnergyHead(nn.Module):
    """
    Head to predict energy for a single electronic state.
    """

    def __init__(self, hidden_dim):
        """
        hidden_dim: Dimension of the hidden features.
        dropout: Dropout rate for regularization.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s, batch):
        """
        s: Scalar features of shape (num_atoms, hidden_dim)
        batch: Tensor of shape (num_atoms,) indicating molecule indices.
        Returns: Tensor of shape (num_molecules, 1) containing predicted energies.
        """
        return scatter_sum(self.net(s), batch, dim=0)


class NonAdiabaticCouplingHead(nn.Module):
    """
    Equivariant head to predict non-adiabatic coupling vectors between electronic states.
    """

    def __init__(self, hidden_dim, num_pairs):
        """
        hidden_dim: Dimension of the hidden features.
        num_coupling_pairs: Number of state pairs for coupling (e.g., 3 for S0-S1, S1-S2, S2-S3).
        """
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_pairs, hidden_dim))

    def forward(self, v, batch, batch_size):
        """
        v: Vector features of shape (num_atoms, hidden_dim, 3)
        batch: Tensor of shape (num_atoms,) indicating molecule indices.
        batch_size: Number of molecules in the batch.
        Returns: Tensor of shape (num_molecules, num_coupling_pairs, 3) containing coupling vectors.
        """
        couplings = []
        for w in self.weights:
            c_atom = torch.einsum("nfk,f->nk", v, w)
            couplings.append(scatter_sum(c_atom, batch, dim=0, dim_size=batch_size))
        return torch.stack(couplings, dim=1)


class AbsorptionHead(nn.Module):
    """
    Predicts absorption maximum (lambda_max) from excited state energy gaps
    """

    def __init__(self, num_states, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states - 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, energies):
        """
        energies: (num_molecules, num_states)
        returns: (num_molecules, 1) lambda_max
        """
        # Compute energy gaps relative to ground state
        gaps = energies[:, 1:] - energies[:, :1]  # (num_molecules, num_states - 1)
        lambda_max = self.net(gaps)  # (num_molecules, 1)
        return lambda_max


class SingletOxygenYieldHead(nn.Module):
    """
    Predicts a singlet oxygen yield from excited state energies + NAC magnitudes
    """

    def __init__(self, num_states, num_couplings, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states - 1 + num_couplings, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Yield between 0 and 1
        )

    def forward(self, energies, nac):
        """
        energies: (num_molecules, num_states)
        nac: (num_molecules, num_coupling_pairs, 3)
        returns: (num_molecules, 1) singlet oxygen yield
        """
        # Energy gaps
        gaps = energies[:, 1:] - energies[:, :1]  # (num_molecules, num_states - 1)

        # NAC magnitudes
        nac_mag = torch.norm(nac, dim=-1)  # (num_molecules, num_coupling_pairs)

        # Concatenate features
        return self.net(torch.cat([gaps, nac_mag], dim=1))


class MANA(nn.Module):
    def __init__(
        self,
        num_atom_types,
        num_states,
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

        self.energy_heads = nn.ModuleList(
            [EnergyHead(hidden_dim) for _ in range(num_states)]
        )

        self.dipole_head = DipoleHead(hidden_dim)
        self.nac_head = NonAdiabaticCouplingHead(hidden_dim, num_states - 1)
        self.absorption_head = AbsorptionHead(num_states)
        self.yield_head = SingletOxygenYieldHead(num_states, num_states - 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, data):
        """
        Defines the forward pass of the MANA
        data: A batch of molecular graphs with attributes:
            - x: Atomic numbers (num_atoms,)
            - edge_index: Edge indices (2, num_edges)
            - edge_attr: Edge attributes (num_edges, 4) = (distance, direction_x, direction_y, direction_z)
            - batch: Batch indices for atoms (num_atoms,)
            - pos: Atomic positions (num_atoms, 3) - needed for gradient computation
        Returns:
            - energies: Predicted energies (num_molecules, num_singlet_states + 1)
            - dipoles: Predicted dipole moments (num_molecules, 3)
            - nac: Predicted non-adiabatic couplings (num_molecules, num_coupling_pairs, 3)
            - lambda_max: Predicted absorption maximum (num_molecules, 1)
            - phi: Predicted singlet oxygen yield (num_molecules, 1)
        """
        z, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        batch_size = batch.max().item() + 1

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

        energies = torch.cat([h(s, batch) for h in self.energy_heads], dim=1)
        dipoles = self.dipole_head(v, batch)
        nac = self.nac_head(v, batch, batch_size)
        lambda_max = self.absorption_head(energies)
        phi = self.yield_head(energies, nac)

        return energies, dipoles, nac, lambda_max, phi

    def loss_fn(self, preds, batch, weights):
        """
        Defines the loss function for training.
        preds: Tuple of model predictions (energies, dipoles, nac, lambda_max, phi)
        batch: Batch of data with ground truth values.
        weights: Dictionary of weights for each loss component.
        Returns:
            - total_loss: Weighted sum of individual losses.
            - loss_E: Energy loss.
            - loss_nac: NAC loss.
            - loss_lambda: Absorption maximum loss.
            - loss_phi: Singlet oxygen yield loss.
        """
        energies, dipoles, nac_pred, lambda_pred, phi_pred = preds

        # --------------------------------------------------
        # Energies
        E0_pred = energies[:, 0]
        Ei_pred = energies[:, 1:]

        # Reshape ground truth energies which are flattened by DataLoader
        if batch.energies.ndim == 1:
            energies_true = batch.energies.view(energies.size(0), -1)
        else:
            energies_true = batch.energies

        E0_true = energies_true[:, 0]
        Ei_true = energies_true[:, 1:]
        loss_E0 = F.mse_loss(E0_pred, E0_true)
        loss_Ei = F.mse_loss(Ei_pred, Ei_true)
        loss_E = loss_E0 + loss_Ei

        # --------------------------------------------------
        # NACs
        nac_true_mag = torch.norm(batch.nac, dim=(-1, -2))
        if nac_true_mag.ndim == 1:
            nac_true_mag = nac_true_mag.view(energies.size(0), -1)

        nac_pred_mag = torch.norm(nac_pred, dim=-1)  # (B, n_pairs)
        loss_nac = F.mse_loss(nac_pred_mag, nac_true_mag)

        # --------------------------------------------------
        # Spectroscopic targets
        loss_lambda = F.mse_loss(lambda_pred.squeeze(-1), batch.lambda_max)
        loss_phi = F.mse_loss(phi_pred.squeeze(-1), batch.phi_delta)

        # --------------------------------------------------
        # Total loss
        total_loss = (
            loss_E
            + weights["lambda"] * loss_lambda
            + weights["phi"] * loss_phi
            + weights["nac_reg"] * loss_nac
        )

        return total_loss, loss_E, loss_nac, loss_lambda, loss_phi
