#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
# https://github.com/IuliaDuta/sheaf_hypergraph_networks
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch_scatter import scatter, scatter_mean, scatter_add
from torch_geometric.utils import softmax
import torch_sparse

from layers.sheaf_layers import *
from utils import sheaf_utils
from sheaf_models.sheaf_builder import *
from sheaf_models.hgcn_sheaf_laplacians import *




def plot_tensor_heatmap(tensor, title="Tensor Heatmap", cmap="viridis", 
                        figsize=(10, 8), output_file="tensor_heatmap.png"):
    """
    Plot a PyTorch tensor as a heatmap and save it to a PNG file.
    
    Args:
        tensor (torch.Tensor): The PyTorch tensor to visualize (2D)
        title (str): Title for the plot
        cmap (str): Colormap to use for the heatmap
        figsize (tuple): Figure size (width, height) in inches
        output_file (str): Filename to save the heatmap
    """
    # Convert tensor to numpy array for plotting
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Make sure tensor is 2D
    if len(tensor.shape) > 2:
        print(f"Warning: tensor has shape {tensor.shape}, flattening to 2D")
        # Reshape to 2D if it has more dimensions
        tensor = tensor.reshape(tensor.shape[0], -1)
    
    tensor_np = tensor.detach().numpy()
    
    # Create the figure and axis
    plt.figure(figsize=figsize)
    
    # Create heatmap
    ax = sns.heatmap(tensor_np, cmap=cmap, annot=False, cbar=True)
    
    # Set title and labels
    plt.title(title, fontsize=16)
    plt.xlabel("Columns", fontsize=12)
    plt.ylabel("Rows", fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_file}")
    
    # Show the plot
    plt.show()
    
    return plt.gcf()  # Return the figure object


class SheafHyperGNN(nn.Module):
    """
        This is a Hypergraph Sheaf Model with 
        the dxd blocks in H_BIG associated to each pair (node, hyperedge)
        being **diagonal**


    """
    def __init__(self, args, sheaf_type):
        super(SheafHyperGNN, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.num_features = args.num_features
        self.MLP_hidden = args.MLP_hidden 
        self.d = args.heads  # dimension of the stalks
        self.init_hedge = args.init_hedge # how to initialise hyperedge attributes: avg or rand
        self.norm_type = args.sheaf_normtype #type of laplacian normalisation degree_norm or block_norm
        self.act = args.sheaf_act # type of nonlinearity used when predicting the dxd blocks
        self.left_proj = args.sheaf_left_proj # multiply with (I x W_1) to the left
        self.args = args
        self.norm = args.AllSet_input_norm
        self.dynamic_sheaf = args.dynamic_sheaf # if True, theb sheaf changes from one layer to another
        self.residual = args.residual_HCHA

        self.hyperedge_attr = None
        if args.cuda in [0, 1]:
            self.device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.lin = MLP(in_channels=self.num_features, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.MLP_hidden*self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=False)
        
        
        # define the model and sheaf generator according to the type of sheaf wanted
        # The diuffusion does not change, however tha implementation for diag and ortho is more efficient
        if sheaf_type == 'SheafHyperGNNDiag':
            ModelSheaf, ModelConv = SheafBuilderDiag, HyperDiffusionDiagSheafConv
        elif sheaf_type == 'SheafHyperGNNOrtho':
            ModelSheaf, ModelConv = SheafBuilderOrtho, HyperDiffusionOrthoSheafConv
        elif sheaf_type == 'SheafHyperGNNGeneral':
            ModelSheaf, ModelConv = SheafBuilderGeneral, HyperDiffusionGeneralSheafConv
        elif sheaf_type == 'SheafHyperGNNLowRank':
            ModelSheaf, ModelConv = SheafBuilderLowRank, HyperDiffusionGeneralSheafConv
        
        self.convs = nn.ModuleList()
        # Sheaf Diffusion layers
        self.convs.append(ModelConv(self.MLP_hidden, self.MLP_hidden, d=self.d, device=self.device, 
                                        norm_type=self.norm_type, left_proj=self.left_proj, 
                                        norm=self.norm, residual=self.residual))
                                        
        # Model to generate the reduction maps
        self.sheaf_builder = nn.ModuleList()
        self.sheaf_builder.append(ModelSheaf(args))

        for _ in range(self.num_layers-1):
            # Sheaf Diffusion layers
            self.convs.append(ModelConv(self.MLP_hidden, self.MLP_hidden, d=self.d, device=self.device, 
                                        norm_type=self.norm_type, left_proj=self.left_proj, 
                                        norm=self.norm, residual=self.residual))
            # Model to generate the reduction maps if the sheaf changes from one layer to another
            if self.dynamic_sheaf:
                self.sheaf_builder.append(ModelSheaf(args))
                
        # self.lin2 = Linear(self.MLP_hidden*self.d, args.num_classes, bias=False)
        self.lin2 = Linear(54, 4968, bias=True)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for sheaf_builder in self.sheaf_builder:
            sheaf_builder.reset_parameters()

        self.lin.reset_parameters()
        self.lin2.reset_parameters()    


    def init_hyperedge_attr(self, type, num_edges=None, x=None, hyperedge_index=None):
        #initialize hyperedge attributes either random or as the average of the nodes
        if type == 'rand':
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == 'avg':
            hyperedge_attr = scatter_mean(x[hyperedge_index[0]],hyperedge_index[1], dim=0)
        else:
            hyperedge_attr = None
        return hyperedge_attr

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        num_nodes = data.x.shape[0] #data.edge_index[0].max().item() + 1
        num_edges = data.edge_index[1].max().item() + 1

        #if we are at the first epoch, initialise the attribute, otherwise use the previous ones
        if self.hyperedge_attr is None:
            self.hyperedge_attr = self.init_hyperedge_attr(self.init_hedge, num_edges=num_edges, x=x, hyperedge_index=edge_index)

        # expand the input N x num_features -> Nd x num_features such that we can apply the propagation
        x = self.lin(x)
        x = x.view((x.shape[0]*self.d, self.MLP_hidden)) # (N * d) x num_features

        hyperedge_attr = self.lin(self.hyperedge_attr)
        hyperedge_attr = hyperedge_attr.view((hyperedge_attr.shape[0]*self.d, self.MLP_hidden))


        for i, conv in enumerate(self.convs[:-1]):
            # infer the sheaf as a sparse incidence matrix Nd x Ed, with each block being diagonal
            if i == 0 or self.dynamic_sheaf:
                h_sheaf_index, h_sheaf_attributes = self.sheaf_builder[i](x, hyperedge_attr, edge_index)
            # Sheaf Laplacian Diffusion
            x = F.elu(conv(x, hyperedge_index=h_sheaf_index, alpha=h_sheaf_attributes, num_nodes=num_nodes, num_edges=num_edges))
            x = F.dropout(x, p=self.dropout, training=self.training)

        #infer the sheaf as a sparse incidence matrix Nd x Ed, with each block being diagonal
        if len(self.convs) == 1 or self.dynamic_sheaf:
            h_sheaf_index, h_sheaf_attributes = self.sheaf_builder[-1](x, hyperedge_attr, edge_index) 
        # Sheaf Laplacian Diffusion
        x = self.convs[-1](x,  hyperedge_index=h_sheaf_index, alpha=h_sheaf_attributes, num_nodes=num_nodes, num_edges=num_edges)
        x = x.view(num_nodes, -1) # Nd x out_channels -> Nx(d*out_channels)

        x = self.lin2(x) # Nx(d*out_channels)-> N x num_classes
        return x

class SheafHyperGCN(nn.Module):
    # replace hyperedge with edges amax(F_v<e(x_v)) ~ amin(F_v<e(x_v))
    def __init__(self, V, num_features, num_layers, num_classses, args, sheaf_type):
        super(SheafHyperGCN, self).__init__()
        d, l, self.c = num_features, num_layers, num_classses
        cuda = args.cuda  # and torch.cuda.is_available()

        self.num_nodes = V
        h = [args.MLP_hidden]
        for i in range(l-1):
            power = l - i + 2
            if (getattr(args, 'dname', None) is not None) and args.dname == 'citeseer':
                power = l - i + 4
            h.append(2**power)
        h.append(self.c)

        reapproximate = False # for HyperGCN we take care of this via dynamic_sheaf

        self.MLP_hidden = args.MLP_hidden
        self.d = args.heads
        self.horizon = args.horizon
        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.num_features = args.num_features
        self.MLP_hidden = args.MLP_hidden 
        self.d = args.heads # dimension of the stalks
        self.init_hedge = args.init_hedge # how to initialise hyperedge attributes: avg or rand
        self.norm_type = args.sheaf_normtype #type of laplacian normalisation degree_norm or block_norm
        self.act = args.sheaf_act # type of nonlinearity used when predicting the dxd blocks
        self.left_proj = args.sheaf_left_proj # multiply with (I x W_1) to the left
        self.args = args
        self.norm = args.AllSet_input_norm
        self.dynamic_sheaf = args.dynamic_sheaf # if True, theb sheaf changes from one layer to another
        self.sheaf_type = sheaf_type #'DiagSheafs', 'OrthoSheafs', 'GeneralSheafs' or 'LowRankSheafs'

        self.hyperedge_attr = None
        self.residual = args.residual_HCHA
  
        # sheaf_type = 'OrthoSheafs'
        if sheaf_type == 'DiagSheafs':
            ModelSheaf, self.Laplacian = HGCNSheafBuilderDiag, SheafLaplacianDiag
        elif sheaf_type == 'OrthoSheafs':
            ModelSheaf, self.Laplacian= HGCNSheafBuilderOrtho, SheafLaplacianOrtho
        elif sheaf_type == 'GeneralSheafs':
            ModelSheaf, self.Laplacian = HGCNSheafBuilderGeneral, SheafLaplacianGeneral
        elif sheaf_type == 'LowRankSheafs':
            ModelSheaf, self.Laplacian = HGCNSheafBuilderLowRank, SheafLaplacianGeneral

        if self.left_proj:
            self.lin_left_proj = nn.ModuleList([
                MLP(in_channels=self.d, 
                        hidden_channels=self.d,
                        out_channels=self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm) for i in range(l)])

        self.lin = MLP(in_channels=self.num_features, 
                        hidden_channels=self.MLP_hidden,
                        out_channels=self.MLP_hidden*self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=False)

        self.sheaf_builder = nn.ModuleList()
        self.sheaf_builder.append(ModelSheaf(args, args.MLP_hidden))

        # self.lin2 = Linear(h[-1]*self.d, args.num_classes, bias=False)
        
        
        self.lin2 = Linear(self.d * self.c, num_features, bias=True) 
        
        if self.dynamic_sheaf:
            for i in range(1,l):
                    self.sheaf_builder.append(ModelSheaf(args, h[i]))

        self.layers = nn.ModuleList([sheaf_utils.HyperGraphConvolution(
            h[i], h[i+1], reapproximate, cuda) for i in range(l)])
        self.do, self.l = args.dropout, num_layers
        self.m = args.HyperGCN_mediators

        self.spectral_k = 5
        self.use_spectral_filter = False

    def spectral_filter(self, eigenvalues, filter_type='heat', params=None):
        """
        Apply spectral filtering to the eigenvalues of the Laplacian matrix.
        
        Args:
            eigenvalues (torch.Tensor): Eigenvalues of the Laplacian matrix
            filter_type (str): Type of spectral filter to apply. Options:
                            'heat' - Heat kernel filter
                            'wave' - Wave kernel filter
                            'chebyshev' - Chebyshev polynomial filter
                            'threshold' - Simple thresholding filter
                            'band' - Band-pass filter
            params (dict): Parameters for the specific filter type
            
        Returns:
            torch.Tensor: Filtered eigenvalues
        """
        if params is None:
            params = {}
        
        # Default parameters
        alpha = params.get('alpha', 1.0)  # Diffusion time for heat kernel
        beta = params.get('beta', 1.0)    # Wave scale parameter
        order = params.get('order', 3)     # Order for Chebyshev polynomials
        threshold = params.get('threshold', 0.01)  # Threshold value
        low_cutoff = params.get('low_cutoff', 0.1)  # Low frequency cutoff
        high_cutoff = params.get('high_cutoff', 2.0)  # High frequency cutoff
        
        if filter_type == 'heat':
            # Heat kernel filter: exp(-α*λ)
            # Acts as a low-pass filter, attenuating high frequencies
            return torch.exp(-alpha * eigenvalues)
            
        elif filter_type == 'wave':
            # Wave kernel filter: exp(-β*(λ-1)²)
            # Band-pass filter centered at λ=1
            return torch.exp(-beta * (eigenvalues - 1.0)**2)
            
        elif filter_type == 'chebyshev':
            # Chebyshev polynomial approximation
            # Implements polynomial filters of specified order
            # For simplicity, we'll implement a basic version
            filtered_vals = torch.zeros_like(eigenvalues)
            T_prev = torch.ones_like(eigenvalues)  # T_0(x) = 1
            T_curr = eigenvalues.clone()          # T_1(x) = x
            
            # Start with the first two Chebyshev polynomials
            filtered_vals = params.get('c0', 0.5) * T_prev + params.get('c1', 0.5) * T_curr
            
            # Add higher order terms if needed
            for k in range(2, order + 1):
                # Recurrence relation: T_k(x) = 2x*T_{k-1}(x) - T_{k-2}(x)
                T_next = 2 * eigenvalues * T_curr - T_prev
                coef = params.get(f'c{k}', 1.0 / (k + 1))  # Default coefficient
                filtered_vals += coef * T_next
                T_prev, T_curr = T_curr, T_next
                
            return filtered_vals

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.left_proj:
            for lin_layer in self.lin_left_proj:
                lin_layer.reset_parameters()
        self.lin.reset_parameters()
        self.lin2.reset_parameters()
        for sheaf_builder in self.sheaf_builder:
            sheaf_builder.reset_parameters()

    def init_hyperedge_attr(self, type, num_edges=None, x=None, hyperedge_index=None):
        #initialize hyperedge attributes either random or as the average of the nodes
        if type == 'rand':
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == 'avg':
            hyperedge_attr = scatter_mean(x[hyperedge_index[0]],hyperedge_index[1], dim=0)
        else:
            hyperedge_attr = None
        return hyperedge_attr

    def normalise(self, A, hyperedge_index, num_nodes, d):
        if self.args.sheaf_normtype == 'degree_norm':
            # compute D^-1
            D = scatter_add(hyperedge_index.new_ones(hyperedge_index.size(1)), hyperedge_index[0], dim=0, dim_size=num_nodes*d) 
            D = torch.pow(D, -1.0)
            D[D == float("inf")] = 0
            D = sheaf_utils.sparse_diagonal(D, (D.shape[0], D.shape[0]))
            D = D.coalesce()

            # compute D^-1A
            A = torch_sparse.spspmm(D.indices(), D.values(), A.indices(), A.values(), D.shape[0],D.shape[1], A.shape[1])
            A = torch.sparse_coo_tensor(A[0], A[1], size=(num_nodes*d, num_nodes*d)).to(D.device)
        elif self.args.sheaf_normtype == 'sym_degree_norm':
            # compute D^-0.5
            D = scatter_add(hyperedge_index.new_ones(hyperedge_index.size(1)), hyperedge_index[0], dim=0, dim_size=num_nodes*d) 
            D = torch.pow(D, -0.5)
            D[D == float("inf")] = 0
            D = sheaf_utils.sparse_diagonal(D, (D.shape[0], D.shape[0]))
            D = D.coalesce()

            # compute D^-0.5AD^-0.5
            A = torch_sparse.spspmm(D.indices(), D.values(), A.indices(), A.values(), D.shape[0],D.shape[1], A.shape[1], coalesced=True)
            A = torch_sparse.spspmm(A[0], A[1], D.indices(), D.values(), D.shape[0],D.shape[1], D.shape[1], coalesced=True)
            A = torch.sparse_coo_tensor(A[0], A[1], size=(num_nodes*d, num_nodes*d)).to(D.device)
           
        elif self.args.sheaf_normtype == 'block_norm':
            # D computed based on the block diagonal
            D = A.to_dense().view((num_nodes, d, num_nodes, d))
            D = torch.permute(D, (0,2,1,3)) #num_nodes x num_nodes x d x d
            D = torch.diagonal(D, dim1=0, dim2=1) # d x d x num_nodes (the block diagonal ones)
            D = torch.permute(D, (2,0,1)) #num_nodes x d x d

            if self.sheaf_type in ["GeneralSheafs", "LowRankSheafs"]:
                D = sheaf_utils.batched_sym_matrix_pow(D, -1.0) #num_nodes x d x d
            else:
                D = torch.pow(D, -1.0)
                D[D == float("inf")] = 0
            D = torch.block_diag(*torch.unbind(D,0))
            D = D.to_sparse()

            # compute D^-1A
            A = torch.sparse.mm(D, A) # this is laplacian delta
            if self.sheaf_type in ["GeneralSheafs", "LowRankSheafs"]:
                A = A.to_dense().clamp(-1,1).to_sparse()
        
        elif self.args.sheaf_normtype == 'sym_block_norm':
            # D computed based on the block diagonal
            D = A.to_dense().view((num_nodes, d, num_nodes, d))
            D = torch.permute(D, (0,2,1,3)) #num_nodes x num_nodes x d x d
            D = torch.diagonal(D, dim1=0, dim2=1) # d x d x num_nodes
            D = torch.permute(D, (2,0,1)) #num_nodes x d x d

            # compute D^-1
            if self.sheaf_type in ["GeneralSheafs", "LowRankSheafs"]:
                D = sheaf_utils.batched_sym_matrix_pow(D, -0.5) #num_nodes x d x d
            else:
                D = torch.pow(D, -0.5)
                D[D == float("inf")] = 0
            D = torch.block_diag(*torch.unbind(D,0))
            D = D.to_sparse()

            # compute D^-0.5AD^-0.5
            A = torch.sparse.mm(D, A) 
            A = torch.sparse.mm(A, D) 
            if self.sheaf_type in ["GeneralSheafs", "LowRankSheafs"]:
                A = A.to_dense().clamp(-1,1).to_sparse()
        return A

    def forward(self, data):
        """
        an l-layer GCN
        """
        do, l, m = self.do, self.l, self.m
        H = data.x

        num_nodes = data.x.shape[0]
        num_edges = data.edge_index[1].max().item() + 1

        # print("num_nodes ", num_nodes)
        # print("num_edges ", num_edges)
        # print("data size ", H.size())
        edge_index= data.edge_index

        if self.hyperedge_attr is None:
            self.hyperedge_attr = self.init_hyperedge_attr(self.init_hedge, num_edges=num_edges, x=H, hyperedge_index=edge_index)
        

        H = self.lin(H)
        hyperedge_attr = self.lin(self.hyperedge_attr)

        H = H.view((H.shape[0]*self.d, self.MLP_hidden)) # (N * d) x num_features
        hyperedge_attr = hyperedge_attr.view((hyperedge_attr.shape[0]*self.d, self.MLP_hidden))
        for i, hidden in enumerate(self.layers):
            if i == 0 or self.dynamic_sheaf:
                # compute the sheaf
                sheaf = self.sheaf_builder[i](H, hyperedge_attr, edge_index) # N x E x d x d

                # build the laplacian based on edges amax(F_v<e(x_v)) ~ amin(F_v<e(x_v))
                # with nondiagonal terms -F_v<e(x_v)^T F_w<e(x_w)
                # and diagonal terms \sum_e F_v<e(x_v)^T F_v<e(x_v)
                h_sheaf_index, h_sheaf_attributes = self.Laplacian(H, m, self.d, edge_index, sheaf)
        
                A = torch.sparse.FloatTensor(h_sheaf_index, h_sheaf_attributes, (num_nodes*self.d,num_nodes*self.d))
                
                A = A.coalesce()
                A = self.normalise(A, h_sheaf_index, num_nodes, self.d)

                # Convert sparse to dense for eigendecomposition
                A_dense = A.to_dense()
                
                eye_diag = torch.ones((num_nodes*self.d))
                L = torch.diag(eye_diag).to(A.device) - A_dense
                
                # Perform eigendecomposition
                eigenvalues, eigenvectors = torch.linalg.eigh(L)
                
                # Sort eigenvalues and eigenvectors (ascending order for eigenvalues)
                idx = eigenvalues.argsort()
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Store eigendecomposition for later use
                self.eigenvalues = eigenvalues
                self.eigenvectors = eigenvectors

                k = min(self.spectral_k, eigenvalues.shape[0]) if hasattr(self, 'spectral_k') else eigenvalues.shape[0]
                
                if hasattr(self, 'use_spectral_filter') and self.use_spectral_filter:
                    filtered_eigenvalues = self.spectral_filter(eigenvalues[:k])
                    # Reconstruct filtered Laplacian
                    eigenvectors_k = eigenvectors[:, :k]
                    filtered_L = eigenvectors_k @ torch.diag(filtered_eigenvalues) @ eigenvectors_k.t()
                    
                    A = torch.diag(eye_diag).to(A.device) - filtered_L
                else:
                    # Continue with original A if no filtering is applied
                    A = torch.sparse.FloatTensor(h_sheaf_index, h_sheaf_attributes, (num_nodes*self.d, num_nodes*self.d))
                    A = A.coalesce()
                    A = self.normalise(A, h_sheaf_index, num_nodes, self.d)
                    A = sheaf_utils.sparse_diagonal(eye_diag, (num_nodes*self.d, num_nodes*self.d)).to(A.device) - A # I - A

            if self.left_proj:
                H = H.t().reshape(-1, self.d)
                H = self.lin_left_proj[i](H)
                H = H.reshape(-1, num_nodes * self.d).t()
            
            H = F.relu(hidden(A, H, m))
            if i < l - 1:
                H = F.dropout(H, do, training=self.training)

        H = H.view(self.num_nodes, -1) # Nd x out_channels -> Nx(d*out_channels)

        H = H.reshape(num_nodes, self.d * self.c) 

        H = self.lin2(H) 

        # one hour: 12 (12*5min)
        # 30 min: 6
        # 15 min: 3

        H = H.reshape(num_nodes, self.horizon, self.num_features//self.horizon)  # METR-LA: 207, PEMS-BAY:325, NAVER-seoul: 774

        return H

