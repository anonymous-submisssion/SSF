# Author: Abeer Mostafa

import torch
import numpy as np
import pywt

class Wavelet_Hypergraph:
    def __init__(self, band_size):
        """
        Initializes a hypergraph using wavelet transform signal decomposition.
        """
        self.band_size = band_size
        

    def construct_hypergraph(self, x):
        """
        Constructs a hypergraph from the input time-series data using wavelet transform.

        Args:
            x (np.ndarray): Time-series data of shape [B, T, C]
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (features, edge_index)
        """
        batch_size, seq_length, feat_dim = x.shape
        x = x.reshape(batch_size * seq_length, feat_dim)
        time_features = x.detach().cpu().numpy()
        # print("input shape: ", x.shape)
        x = x.detach().cpu().numpy()
        # Decompose the time series using wavelet transform
        # Break down the data into different frequency components using wavelet transform
        coeffs = pywt.wavedec(x, 'db4', level=3)
        
        # Reconstruct the smoothed version of the time series from the decomposition
        coeff_1 = np.array(coeffs[0])
        coeff_2 = np.array(coeffs[1])
        coeff_3 = np.array(coeffs[2])
        smooth = []
        for i in range(coeffs[-1].shape[0]):
            approx = pywt.upcoef('a', coeffs[-1][i] , 'db4', level=3)
            smooth.append(approx)
        smooth = np.array(smooth)
        # freq_features = np.concatenate((coeff_1, coeff_2, coeff_3, smooth), axis=1)
        freq_features = coeff_1
        
        con_coeff = np.concatenate((time_features, freq_features), axis=1)
        # print("conc shape", con_coeff.shape)
        num_windows = con_coeff.shape[0] // self.band_size

        # Construct hyperedge indices
        node_indices = []
        hyperedge_indices = []
        
        for window_idx in range(num_windows):
            start_idx = window_idx * self.band_size
            end_idx = start_idx + self.band_size
            window_nodes = torch.arange(start_idx, end_idx)
            
            hyperedge_idx = window_idx
            node_indices.extend(window_nodes.tolist())
            hyperedge_indices.extend([hyperedge_idx] * self.band_size)
        
        # Create the edge_index tensor
        edge_index = torch.tensor([node_indices, hyperedge_indices], dtype=torch.long)
        
        self.features = con_coeff
        self.edge_index = edge_index

    def get_hypergraph(self):
        """
        Returns the constructed hypergraph.
        """
        # print("feat size: ", torch.Tensor(self.features).size())
        # print("edge_index size: ", self.edge_index.size())
        return torch.Tensor(self.features), self.edge_index