import torch

class TS_Hypergraph:
    def __init__(self, window_size, sampling_rate=0.0016):
        """
        Initialize the Time Series Hypergraph constructor.
        
        Args:
            window_size (int): Size of the sliding window forming hyperedges.
            sampling_rate (int): Step size for sliding window.
        """
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.edge_index = None
        self.features = None
    
    def construct_hypergraph(self, x):
        """
        Constructs a hypergraph from time series data.
        
        Args:
            x (torch.Tensor): Input time series data of shape (batch_size, features, sequence_length).
        """
        batch_size, feat_dim, seq_length = x.shape
        num_windows = (batch_size*seq_length) // self.window_size
        #imputation

        node_indices = []
        hyperedge_indices = []

        for window_idx in range(num_windows):
            start_idx = window_idx * self.window_size
            end_idx = start_idx + self.window_size
            window_nodes = torch.arange(start_idx, end_idx)
            
            hyperedge_idx = window_idx
            node_indices.extend(window_nodes.tolist())
            hyperedge_indices.extend([hyperedge_idx] * self.window_size)
        
        self.edge_index = torch.tensor([node_indices, hyperedge_indices], dtype=torch.long)

        self.features = x.reshape(batch_size * seq_length, feat_dim)

    
    def get_hypergraph(self):
        """
        Returns the constructed hypergraph features and edge index.
        
        Returns:
            tuple: (features, edge_index)
        """
        if self.features is None or self.edge_index is None:
            raise ValueError("Hypergraph has not been constructed yet. Call construct_hypergraph first.")

        #print("fe ", self.features.size())
        #print("edge_index ", self.edge_index.size())

        return self.features, self.edge_index
