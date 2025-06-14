# Based on the public repo: https://github.com/IuliaDuta/sheaf_hypergraph_networks


import numpy as np
import time
import torch, math, scipy.sparse as sp
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter




class HyperGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, a, b, reapproximate=True, cuda=None):
        super(HyperGraphConvolution, self).__init__()
        self.a, self.b = a, b
        self.reapproximate = reapproximate

        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        self.reset_parameters()
        


    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)



    def forward(self, structure, H, m=True):
        W, b = self.W, self.bias
        HW = torch.mm(H, W)
        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()    
            A = Laplacian(n, structure, X, m)
        else: A = structure

        A = A.to(H.device)
        A = Variable(A)

        AHW = SparseMM.apply(A, HW)     
        return AHW + b



    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.a) + ' -> ' \
               + str(self.b) + ')'


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2



def Laplacian(V, E, X, m):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    V: number of vertices
    E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
    X: features on the vertices
    m: True gives Laplacian with mediators, while False gives without

    A: adjacency matrix of the graph approximation
    returns: 
    updated data with 'graph' as a key and its value the approximated hypergraph 
    """
    
    edges, weights = [], {}
    rv = np.random.rand(X.shape[1])

    time1 = time.time()
    for k in E.keys():
        hyperedge = list(E[k])
        
        p = np.dot(X[hyperedge], rv)   #projection onto a random vector rv
        s, i = np.argmax(p), np.argmin(p)
        Se, Ie = hyperedge[s], hyperedge[i]

        # two stars with mediators
        c = max(1, 2*len(hyperedge) - 3)    # normalisation constant
        if m:
            
            # connect the supremum (Se) with the infimum (Ie)
            edges.extend([[Se, Ie], [Ie, Se]])
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/c)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/c)
            
            # connect the supremum (Se) and the infimum (Ie) with each mediator
            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    edges.extend([[Se,mediator], [Ie,mediator], [mediator,Se], [mediator,Ie]])
                    weights = update(Se, Ie, mediator, weights, c)
        else:
            edges.extend([[Se,Ie], [Ie,Se]])
            e = len(hyperedge)
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/e)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/e)    
    time2=time.time()
    print(time2-time1)

    return adjacency(edges, weights, V)



def update(Se, Ie, mediator, weights, c):
    """
    updates the weight on {Se,mediator} and {Ie,mediator}
    """    
    
    if (Se,mediator) not in weights:
        weights[(Se,mediator)] = 0
    weights[(Se,mediator)] += float(1/c)

    if (Ie,mediator) not in weights:
        weights[(Ie,mediator)] = 0
    weights[(Ie,mediator)] += float(1/c)

    if (mediator,Se) not in weights:
        weights[(mediator,Se)] = 0
    weights[(mediator,Se)] += float(1/c)

    if (mediator,Ie) not in weights:
        weights[(mediator,Ie)] = 0
    weights[(mediator,Ie)] += float(1/c)

    return weights



def adjacency(edges, weights, n):
    """
    computes an sparse adjacency matrix

    arguments:
    edges: list of pairs
    weights: dictionary of edge weights (key: tuple representing edge, value: weight on the edge)
    n: number of nodes

    returns: a scipy.sparse adjacency matrix with unit weight self loops for edges with the given weights
    """
    
    dictionary = {tuple(item): index for index, item in enumerate(edges)}
    edges = [list(itm) for itm in dictionary.keys()]   
    organised = []

    for e in edges:
        i,j = e[0],e[1]
        w = weights[(i,j)]
        organised.append(w)

    edges, weights = np.array(edges), np.array(organised)
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + sp.eye(n)

    A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    A = ssm2tst(A)
    return A



def symnormalise(M):
    """
    symmetrically normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2} 
    where D is the diagonal node-degree matrix
    """
    
    d = np.array(M.sum(1))
    
    dhi = np.power(d, -1/2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)    # D half inverse i.e. D^{-1/2}
    
    return (DHI.dot(M)).dot(DHI) 



def ssm2tst(M):
    """
    converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)

    arguments:
    M: scipy sparse matrix

    returns:
    a torch sparse tensor of M
    """
    
    M = M.tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)


def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    
    d = np.array(M.sum(1))
    
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    di = np.nan_to_num(di)
    DI = sp.diags(di)    # D inverse i.e. D^{-1}
    
    return DI.dot(M)

import pdb
def get_color_coded_background(color, i):
    return "\033[4{}m {:.4f} \033[0m".format(color+1, i)

def print_a_colored_ndarray(map, d, row_sep=""):
    map = np.round(map,3)
    n,m = map.shape 
    n = n // d
    m = m // d
    color_range_row = np.arange(m)[np.newaxis,...].repeat(d,axis=1)
    color_range_col = np.arange(n)[...,np.newaxis].repeat(d,axis=0)
    color_range = color_range_row + color_range_col

    back_map_modified = np.vectorize(get_color_coded_background)(color_range, map)
    n, m = back_map_modified.shape
    fmt_str = "\n".join([row_sep.join(["{}"]*m)]*n)
    print(fmt_str.format(*back_map_modified.ravel()))

def batched_sym_matrix_pow(matrices: torch.Tensor, p: float) -> torch.Tensor:
        r"""
        Power of a matrix using Eigen Decomposition.
        Args:
            matrices: A batch of matrices.
            p: Power.
            positive_definite: If positive definite
        Returns:
            Power of each matrix in the batch.
        """
        # vals, vecs = torch.linalg.eigh(matrices)
        # SVD is much faster than  vals, vecs = torch.linalg.eigh(matrices) for large batches.
        vecs, vals, _ = torch.linalg.svd(matrices)
        good = vals > vals.max(-1, True).values * vals.size(-1) * torch.finfo(vals.dtype).eps
        vals = vals.pow(p).where(good, torch.zeros((), device=matrices.device, dtype=matrices.dtype))
        matrix_power = (vecs * vals.unsqueeze(-2)) @ torch.transpose(vecs, -2, -1)
        return matrix_power

def sparse_diagonal(diag, shape):
    r,c = shape
    assert r == c
    indexes = torch.arange(r).to(diag.device)
    indexes = torch.stack([indexes, indexes], dim=0)
    return torch.sparse.FloatTensor(indexes, diag)


def generate_indices_general(indexes, d):
    d_range = torch.arange(d)
    d_range_edges = d_range.repeat(d).view(-1,1) #0,1..d,0,1..d..   d*d elems
    d_range_nodes = d_range.repeat_interleave(d).view(-1,1) #0,0..0,1,1..1..d,d..d  d*d elems
    indexes = indexes.unsqueeze(1) 

    large_indexes_0 = d * indexes[0] + d_range_nodes
    large_indexes_0 = large_indexes_0.permute((1,0)).reshape(1,-1)
    large_indexes_1 = d * indexes[1] + d_range_edges
    large_indexes_1 = large_indexes_1.permute((1,0)).reshape(1,-1)
    large_indexes = torch.concat((large_indexes_0, large_indexes_1), 0)

    return large_indexes