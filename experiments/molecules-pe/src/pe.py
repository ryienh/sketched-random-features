from typing import List, Callable

import torch
from torch import nn
from torch_geometric.utils import unbatch

from src.gin import GIN
from src.gine import GINE
from src.mlp import MLP
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops,get_laplacian,remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from src.schema import Schema
from torch_geometric.nn.conv import MessagePassing
from scipy.special import comb
import math

# This code is adapted from:
# https://github.com/Graph-COM/SPE/blob/master/src/stable_expressive_pe.py 

class SwiGLU(nn.Module):
    def __init__(self, input_dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        swish_part = self.fc1(x) * torch.sigmoid(self.fc1(x))  
        gate = torch.sigmoid(self.fc2(x))  # Sigmoid
        return swish_part * gate 
    
'''
    Basic graph filtering used in all experiments of our paper.
'''
def filter(S, W, k):
    # S is laplacian and W is NxN e or NxM x_m
    out = W
    w_list = []
    w_list.append(out.unsqueeze(-1))
    for i in range(k-1): 
        out = S @ out # NxN or NxM
        w_list.append(out.unsqueeze(-1)) 
    return torch.cat(w_list, dim=-1) #NxMxK

'''
    Optional filter from BernNet: https://github.com/ivam-he/BernNet
'''
def bern_filter(S, W, k):
    out = W
    w_list = []
    w_list.append(out.unsqueeze(-1))
    for i in range(1, k): 
        L = (1/(2**k)) * math.comb(k, i) * torch.linalg.matrix_power(
                                    (2*(torch.eye(S.shape[0]).to(S.device)) - S), k) @ S
        out = L @ W # NxN or NxM
        w_list.append(out.unsqueeze(-1)) 
    return torch.cat(w_list, dim=-1)

class PEARLPositionalEncoder(nn.Module):
    def __init__(self, sample_aggr: nn.Module, basis, k=16, mlp_nlayers=1, mlp_hid=16, pearl_act='relu', mlp_out=16) -> None:
        super().__init__()
        self.mlp_nlayers = mlp_nlayers
        if mlp_nlayers > 0:
            if mlp_nlayers == 1:
                assert(mlp_hid == mlp_out)
            self.mlp_nlayers = mlp_nlayers
            self.layers = nn.ModuleList([nn.Linear(k if i==0 else mlp_hid, 
                                        mlp_hid if i<mlp_nlayers-1 else mlp_out, bias=True) for i in range(mlp_nlayers)])
            self.norms = nn.ModuleList([nn.BatchNorm1d(mlp_hid if i<mlp_nlayers-1 else mlp_out,track_running_stats=True) for i in range(mlp_nlayers)])
        if pearl_act == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif pearl_act == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = SwiGLU(mlp_hid) ## edit if you want more than 1 mlp layers!!
        self.sample_aggr = sample_aggr
        self.k = k
        self.basis = basis

    def forward(
        self, Lap, W, edge_index: torch.Tensor, batch: torch.Tensor, final=True
    ) -> torch.Tensor:
        """
        :param Lap: Laplacian
        :param W: B*[NxM] or BxNxN
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        W_list = []
        # for loop N times for each Nx1 e
        if isinstance(W[0], int):
            # split into N*B*[Nx1]
            j = 0
            for lap, w in zip(Lap, W):
                for i in range(w):
                    e_i = torch.zeros(w).to(device)
                    e_i[i] = 1
                    output = filter(lap, e_i, self.k)  #can also use bern_filter(lap, e_i, self.k) 
                    W_list.append(output)             # [NxMxK]*B
                if j == 0:
                    out = self.sample_aggr(W_list, edge_index, self.basis)
                else:
                    out += self.sample_aggr(W_list, edge_index, self.basis)
                j += 1
            return out
        else:
            for lap, w in zip(Lap, W):
                output = filter(lap, w, self.k)   # output [NxMxK]
                if self.mlp_nlayers > 0:
                    for layer, bn in zip(self.layers, self.norms):
                        output = output.transpose(0, 1)
                        output = layer(output)
                        output = bn(output.transpose(1,2)).transpose(1,2)
                        output = self.activation(output)
                        output = output.transpose(0, 1)
                W_list.append(output)             # [NxMxK]*B
            return self.sample_aggr(W_list, edge_index, self.basis, final=final)   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.sample_aggr.out_dims


class GINSampleAggregator(nn.Module):
    gin: GIN

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP], bn: bool
    ) -> None:
        super().__init__()
        self.gin = GIN(n_layers, in_dims, hidden_dims, out_dims, create_mlp, bn, laplacian=None)
        self.mlp = create_mlp(out_dims, out_dims, use_bias=True)
        self.running_sum = 0

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor, basis, mean=False, final=True) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """ 
        if not basis:
            W = torch.cat(W_list, dim=0)  
            PE = self.gin(W, edge_index)  
            if mean:
                PE = (PE).mean(dim=1) # sum or mean along M dimension
            else:
                PE = (PE).sum(dim=1)
                self.running_sum += PE
            if final:
                PE = self.running_sum
                self.running_sum = 0
            return PE              
        else:
            n_max = max(W.size(0) for W in W_list)
            W_pad_list = []     # [N_i, N_max, M] * B
            mask = [] # node masking, [N_i, N_max] * B
            for W in W_list:
                zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
                W_pad = torch.cat([W, zeros], dim=1)   # [N_i, N_max, M]
                W_pad_list.append(W_pad)
                mask.append((torch.arange(n_max, device=W.device) < W.size(0)).tile((W.size(0), 1))) # [N_i, N_max]
            W = torch.cat(W_pad_list, dim=0)   # [N_sum, N_max, M]
            mask = torch.cat(mask, dim=0)   # [N_sum, N_max]
            PE = self.gin(W, edge_index, mask=mask)       # [N_sum, N_max, D_pe]
            PE = (PE * mask.unsqueeze(-1)).sum(dim=1)
            return PE

    @property
    def out_dims(self) -> int:
        return self.gin.out_dims

def GetSampleAggregator(cfg: Schema, create_mlp: Callable[[int, int], MLP], device):
    if cfg.sample_aggr_model_name == 'gin':
        return GINSampleAggregator(cfg.n_sample_aggr_layers, cfg.pearl_mlp_out, cfg.sample_aggr_hidden_dims, cfg.pe_dims,
                                         create_mlp, cfg.batch_norm)
    else:
        raise Exception ("sample_aggr function not implemented!")
