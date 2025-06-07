import torch
import pickle
import scipy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool
from torch_geometric.nn import GraphNorm, JumpingKnowledge
from argparse import Namespace

from models.FFT_Hypergraph import FFT_Hypergraph
from models.TS_Hypergraph import TS_Hypergraph
from models.Wavelet_Hypergraph import Wavelet_Hypergraph
from sheaf_models.sheaf_models import *



def transform_adjacency_matrix():

    sensor_ids, sensor_id_to_ind, adj_matrix = load_graph_data("./Datasets/sensor_graph/adj_mx.pkl")

    adj_array = np.array(adj_matrix)
    binary_adj = (adj_array != 0).astype(int)
    # Create edge_index
    # Find non-zero elements (edges)
    rows, cols = np.where(binary_adj == 1)
    
    # Create edge_index tensor
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    
    return edge_index

def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.device = configs.device

        # self.feat = configs.new_len  # Length    PS: batch_size x seq_len should be > features

        self.feat = self.configs.num_sensors * self.configs.data_dim * self.configs.horizon
        self.num_class = 12
        self.top_k = 3
        self.length = 8
        self.fft_hypergraph = FFT_Hypergraph(self.top_k)
        self.ts_hypergraph = TS_Hypergraph(window_size=self.length)
        self.wavelet_hypergraph = Wavelet_Hypergraph(band_size=self.length)

        args_dict = {
                'horizon': self.configs.horizon,
                'num_features': self.feat,     # number of node features
                'num_classes': self.num_class,       # number of classes
                'All_num_layers': 4,    # number of layers
                'dropout': 0.3,         # dropout rate
                'MLP_hidden': 256,      # dimension of hidden state (for most of the layers)
                'AllSet_input_norm': True,  # normalising the input at each layer
                'residual_HCHA': True, # using or not a residual connectoon per sheaf layer

                'heads': 6,             # dimension of reduction map (d)
                'init_hedge': 'avg',    # how to compute hedge features when needed. options: 'avg'or 'rand'
                'sheaf_normtype': 'sym_degree_norm',  # the type of normalisation for the sheaf Laplacian. options: 'degree_norm', 'block_norm', 'sym_degree_norm', 'sym_block_norm'
                'sheaf_act': 'tanh',    # non-linear activation used on tpop of the d x d restriction maps. options: 'sigmoid', 'tanh', 'none'
                'sheaf_left_proj': False,   # multiply to the left with IxW or not
                'dynamic_sheaf': False, # infering a differetn sheaf at each layer or use ta shared one

                'sheaf_pred_block': 'cp_decomp', # indicated the type of model used to predict the restriction maps. options: 'MLP_var1', 'MLP_var3' or 'cp_decomp'
                'sheaf_dropout': True, # use dropout in the sheaf layer or not
                'sheaf_special_head': False,    # if True, add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
                'rank': 2,              # only for LowRank type of sheaf. mention the rank of the reduction matrix

                'HyperGCN_mediators': True, #only for the Non-Linear sheaf laplacian. Indicates if mediators are used when computing the non-linear Laplacian (same as in HyperGCN)
                'cuda': 0

            }

        self.sheaf_args = Namespace(**args_dict)
        # self.sheaf_model = SheafHyperGNN(self.sheaf_args, sheaf_type='SheafHyperGNNDiag').to(self.device)
        num_nodes = self.configs.batch_size*self.length
        self.sheaf_model = SheafHyperGCN(V=num_nodes,
                         num_features=self.sheaf_args.num_features,
                         num_layers=self.sheaf_args.All_num_layers,
                         num_classses=self.sheaf_args.num_classes,
                         args=self.sheaf_args, sheaf_type= 'DiagSheafs'
                         ).to(self.device)
        self.sheaf_projection = nn.Linear(self.length, 1)
        self.gcn_model = ImprovedGCN()

    def forecast(self, x):
        # print("Data size: ", x.size())
        features = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

        edge_index = transform_adjacency_matrix()
        #print("Edge index size: ", edge_index.size())

        hypergraph_data = Data(x = features, edge_index = edge_index).to(self.device)
        
        output = self.sheaf_model(hypergraph_data)
        #print("model output size: ", output.size())

        return output 

    def classification(self, x):
        
        print("Data size: ", x.size())
        features = x.reshape(x.shape[0]*x.shape[1], x.shape[2]*x.shape[3])

        edge_index = transform_adjacency_matrix()
        print("Edge index size: ", edge_index.size())
        hypergraph_data = Data(x = features, edge_index = edge_index).to(self.device)
        
        output = self.sheaf_model(hypergraph_data)
        print("model output size: ", output.size())
        output = output.mean(dim=1)

        return output

    def forward(self, x):

        if self.task_name == 'forecast':
            out = self.forecast(x)
            return out 

        if self.task_name == 'classification':
            out = self.classification(x)
            return out 
        return None

class ImprovedGCN(torch.nn.Module):
    def __init__(self, in_channels=4644, hidden_channels=512, out_channels=4644, 
                 num_layers=3, dropout=0.2, residual=True, output_shape=(300, 3, 774*2)):
        super(ImprovedGCN, self).__init__()
        
        # Save output shape for reshaping in forward pass
        self.output_shape = output_shape
        self.residual = residual
        
        # Input layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(GraphNorm(hidden_channels))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(GraphNorm(hidden_channels))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Last GCN layer
        self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
        self.batch_norms.append(GraphNorm(hidden_channels))
        self.dropouts.append(nn.Dropout(dropout))
        
        # JumpingKnowledge aggregation
        self.jk = JumpingKnowledge(mode="max")
        
        # Final projection layer
        final_output_size = output_shape[1] * output_shape[2]
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, final_output_size)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Initial feature transformation
        identity = x
        
        # Apply multiple GCN layers with residual connections
        layer_outputs = []
        
        for i, (conv, norm, dropout) in enumerate(zip(self.conv_layers, self.batch_norms, self.dropouts)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = dropout(x)
            layer_outputs.append(x)
            
            # Add residual connection except for the first layer (dimension mismatch)
            if self.residual and i > 0:
                x = x + layer_outputs[i-1]
        
        # Apply jumping knowledge
        x = self.jk(layer_outputs)
        
        # Final projection
        x = self.lin(x)

        # Reshape to desired output dimensions
        x = x.reshape(self.output_shape)
        
        return x


# For optional use: Edge feature version with attention
class EdgeFeatureGCN(torch.nn.Module):
    def __init__(self, in_channels=1242, edge_features=1, hidden_channels=512, 
                 out_channels=1242, num_layers=3, dropout=0.2, output_shape=(300, 3, 414)):
        super(EdgeFeatureGCN, self).__init__()
        
        self.output_shape = output_shape
        
        # Import needed here since this is a specialized version
        from torch_geometric.nn import GATConv
        
        # Initial edge embedding (if edge features exist)
        self.edge_embedding = nn.Linear(edge_features, hidden_channels) if edge_features > 0 else None
        
        # Layer structure
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=4, concat=False, dropout=dropout))
        self.norms.append(GraphNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=4, concat=False, dropout=dropout))
            self.norms.append(GraphNorm(hidden_channels))
            
        # Output layer
        self.convs.append(GATConv(hidden_channels, hidden_channels, heads=4, concat=False, dropout=dropout))
        self.norms.append(GraphNorm(hidden_channels))
        
        # Final projection
        final_output_size = output_shape[0] * output_shape[1] * output_shape[2]
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, final_output_size)
        )
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Process edge features if available
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        if edge_attr is not None and self.edge_embedding is not None:
            edge_attr = self.edge_embedding(edge_attr)
        
        # Apply layers
        for conv, norm in zip(self.convs, self.norms):
            if hasattr(conv, 'supports_edge_attr') and conv.supports_edge_attr and edge_attr is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        
        # Global pooling if batch information is available
        if hasattr(data, 'batch') and data.batch is not None:
            x = global_mean_pool(x, data.batch)
            
        # Output projection
        x = self.output_layer(x)
        x = x.reshape(self.output_shape)
        
        return x