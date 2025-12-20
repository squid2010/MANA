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
        self.centers = nn.Parameter(torch.linspace(0, cutoff, num_rbf), requires_grad=False)
        self.widths = nn.Parameter(torch.ones(num_rbf), requires_grad=False)
        
    def forward(self, distances):
        """
        Defines the forward pass to compute RBFs.
        distances: Tensor of shape (num_edges,) containing distances.
        Returns: Tensor of shape (num_edges, num_rbf) containing RBF values.
        """
        
        diff = distances.unsqueeze(-1) - self.centers
        rbf = torch.exp(- (diff ** 2) / (2 * self.widths ** 2))
        return rbf
        
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
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim)
        )
        
    def forward(self, s, v, edge_index, edge_attr, rbf):
        """
        Defines the forward pass of the PaiNN layer.
        s: Scalar features of shape (num_atoms, hidden_dim)
        v: Vector features of shape (num_atoms, hidden_dim, 3)
        edge_index: Edge indices of shape (2, num_edges)
        edge_attr: Edge attributes of shape (num_edges, 4) = (distance, direction_x, direction_y, direction_z)
        rbf: Radial basis functions of shape (num_edges, num_rbf)
        Returns: Updated scalar and vector features.
        """
        
        row, col = edge_index
        unit_direction_vectors = edge_attr[:, 1:4]  # Direction vectors
        
        # Filters
        filter = self.filter_net(rbf)
        filter_s, filter_v = filter.chunk(2, dim=-1)
        
        # Messages
        message_s = filter_s * s[col]
        message_v = (
            filter_v.unsqueeze(-1) * v[col] + 
            filter_v * s[col].unsqueeze(-1) * unit_direction_vectors.unsqueeze(1)
        )
        
        # Aggregate
        message_s = scatter_sum(message_s, row, dim=0, dim_size=s.size(0))
        message_v = scatter_sum(message_v, row, dim=0, dim_size=v.size(0))
        
        # equivariant update
        v_norm =  torch.norm(message_v, dim=-1)
        update_in = torch.cat([s, message_s, v_norm], dim=-1)
        
        delta_s, alpha, beta = self.update_net(update_in).chunk(3, dim=-1)
        
        s = s + delta_s
        v = alpha.unsqueeze(-1) * v + beta.unsqueeze(-1) * message_v
        
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
        self.linear = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, v, batch):
        """
        v: Vector features of shape (num_atoms, hidden_dim, 3)
        batch: Tensor of shape (num_atoms,) indicating molecule indices.
        Returns: Tensor of shape (num_molecules, 3) containing predicted dipole moments.
        """
        weights = self.linear.weight.squeeze(0)  # Shape: (hidden_dim,)
        mu_atom = torch.einsum("nfk,f->nk", v, weights)  # Shape: (num_atoms, 3)
        mu = scatter_sum(mu_atom, batch, dim=0)  # Shape: (num_molecules, 3)
        return mu
    
class MANA(nn.Module):
    def __init__(self, num_atom_types, num_singlet_states, hidden_dim=128, num_layers=4, num_rbf=20):
        """
        MANA model for predicting molecular properties.
        num_atom_types: Number of unique atom types.
        num_singlet_states: Number of singlet states to consider.
        hidden_dim: Dimension of the hidden features.
        num_layers: Number of PaiNN layers.
        num_rbf: Number of radial basis functions.
        """
        super().__init__()
        
        self.embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.rbf = RadialBasisFunction(num_rbf)
        
        self.layers = nn.ModuleList([PaiNNLayer(hidden_dim, num_rbf) for _ in range(num_layers)])
        
        # +1 state = T1
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_singlet_states + 1)
        )
        
        self.dipole_head = DipoleHead(hidden_dim)