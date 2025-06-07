import torch
import numpy as np
from scipy.linalg import sqrtm, inv
from sklearn.neighbors import radius_neighbors_graph
import networkx as nx
from collections import defaultdict

class SPD_Hypergraph:
    def __init__(self, epsilon=0.5):
        """
        Initialize the SPD Hypergraph.
        
        Args:
            epsilon (float): Threshold for the ε-neighborhoods approach.
        """
        self.epsilon = epsilon
        self.edge_index = None
        self.S_matrices = None  # Covariance matrices
        self.hyperedges = None  # Dictionary mapping hyperedge index to node indices
    
    def _compute_airm_distance(self, S1, S2):
        """
        Compute the Affine Invariant Riemannian Metric (AIRM) distance between two SPD matrices.
        
        Args:
            S1, S2: SPD matrices
            
        Returns:
            float: The AIRM distance between S1 and S2
        """
        # Ensure matrices are numpy arrays
        S1 = S1.detach().cpu().numpy() if torch.is_tensor(S1) else S1
        S2 = S2.detach().cpu().numpy() if torch.is_tensor(S2) else S2
        S1 = S1.astype(float)
        S2 = S2.astype(float)
        
        # AIRM distance: ||log(S1^(-1/2) S2 S1^(-1/2))||_F
        S1_sqrt_inv = inv(sqrtm(S1))
        M = S1_sqrt_inv @ S2 @ S1_sqrt_inv
        eigvals = np.linalg.eigvalsh(M)
        # Taking log of eigenvalues and computing Frobenius norm
        log_eigs = np.log(eigvals)
        airm_dist = np.sqrt(np.sum(log_eigs**2))
        
        return airm_dist
    
    def construct_hypergraph(self, x):
        """
        Construct a hypergraph from time series data.
        
        Args:
            x (torch.Tensor): Time series data of shape [batch_size, channels, time_points]
            
        Returns:
            tuple: (features, edge_index) where features are node features and 
                  edge_index is a tensor of shape [2, num_edges] containing [node_indices, hyperedge_indices]
        """
        batch_size, channels, time_points = x.shape
        
        # Compute spatial covariance matrices for each sample in the batch
        S_matrices = []
        for i in range(batch_size):
            # x[i] is of shape [channels, time_points]
            # Spatial covariance matrix: (1/T) * X * X^T
            S = (1/time_points) * x[i] @ x[i].transpose(-1, -2)
          
          
            # Add small diagonal term for numerical stability
            S = S + 1e-5 * torch.eye(channels, device=x.device)

            S_matrices.append(S)
        
        self.S_matrices = S_matrices
        self.features = torch.stack(self.S_matrices)
        self.features = self.features.reshape(self.features.shape[0], self.features.shape[1]*self.features.shape[2])
        
        
        # Compute pairwise AIRM distances between all SPD matrices
        distances = np.zeros((batch_size, batch_size))
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                dist = self._compute_airm_distance(S_matrices[i], S_matrices[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Create adjacency matrix using ε-neighborhoods approach
        adjacency = (distances**2 < self.epsilon).astype(np.float32)
        
        # Convert to NetworkX graph to find connected components (which will be our hyperedges)
        G = nx.from_numpy_array(adjacency)
        
        # Find connected components - each will be a hyperedge
        self.hyperedges = {}
        for hyperedge_idx, component in enumerate(nx.connected_components(G)):
            self.hyperedges[hyperedge_idx] = list(component)
        
        # Create edge_index in the format [node_indices, hyperedge_indices]
        node_indices = []
        hyperedge_indices = []
        
        for he_idx, nodes in self.hyperedges.items():
            for node in nodes:
                node_indices.append(node)
                hyperedge_indices.append(he_idx)
        
        # Convert to torch tensors
        self.edge_index = torch.tensor([node_indices, hyperedge_indices], dtype=torch.long)
        
        return self.features, self.edge_index
    
    def get_hypergraph(self):
        """
        Return the constructed hypergraph.
        
        Returns:
            tuple: (features, edge_index)
        """
        if self.features is None or self.edge_index is None:
            raise ValueError("Hypergraph not constructed yet. Call construct_hypergraph first.")
        
        return self.features, self.edge_index