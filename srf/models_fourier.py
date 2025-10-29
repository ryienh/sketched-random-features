import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn.norm import batch_norm


# Example MLP used by GIN layers. You should have this defined somewhere.
# If not, here's a simple example:
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0):
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class GINF(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout=0.0,
        kernel_based_feature_expension=False,
    ):
        """
        A GIN model with optional kernel-based random feature expansion.

        Args:
            input_size (int): Input feature dimension.
            hidden_size (int): Hidden dimension for the MLPs inside GINConv.
            num_layers (int): Number of GIN layers.
            output_size (int): Dimension of the final output (e.g., number of classes).
            dropout (float): Dropout rate.
            kernel_based_feature_expension (bool): If True, apply kernel-based random feature
                                                   expansion at each layer.
        """
        super(GINF, self).__init__()

        self.kernel_based_feature_expension = kernel_based_feature_expension
        self.num_layers = num_layers
        self.dropout = dropout

        if self.kernel_based_feature_expension:
            # Hyperparameters for random Fourier features
            self.m = 8
            self.sigma = 1.0

            # Random features for the first layer (input dimension)
            self.register_buffer("W_in", torch.randn(input_size, self.m) / self.sigma)
            self.register_buffer("b_in", 2 * math.pi * torch.rand(self.m))

            # Random features for subsequent layers (hidden dimension)
            self.register_buffer(
                "W_hidden", torch.randn(hidden_size, self.m) / self.sigma
            )
            self.register_buffer("b_hidden", 2 * math.pi * torch.rand(self.m))

            # Adjust input dimensions for MLPs
            first_layer_input_dim = input_size + self.m
            hidden_layer_input_dim = hidden_size + self.m
        else:
            first_layer_input_dim = input_size
            hidden_layer_input_dim = hidden_size

        # Create the GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer from input to hidden
        mlp = MLP(
            first_layer_input_dim,
            hidden_size,
            hidden_size,
            num_layers=2,
            dropout=dropout,
        )
        self.convs.append(GINConv(mlp))
        self.bns.append(nn.BatchNorm1d(hidden_size))

        # Additional hidden layers
        for _ in range(num_layers - 1):
            mlp = MLP(
                hidden_layer_input_dim,
                hidden_size,
                hidden_size,
                num_layers=2,
                dropout=dropout,
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_size))

        # A final linear classifier layer
        self.linear = nn.Linear(hidden_size, output_size)

    def apply_random_features(self, x, W, b):
        # x: [num_nodes, dimension]
        # W: [dimension, m]
        # b: [m]
        # Output: [num_nodes, m]
        # Use random Fourier features approximation:
        # sqrt(1/m) * cos(xW + b)
        return (1.0 / math.sqrt(self.m)) * torch.cos(x @ W + b)

    def forward(self, x, edge_index, batch):
        # x: [num_nodes, input_size]
        # edge_index: [2, num_edges]
        # batch: [num_nodes], mapping each node to its graph index

        for i in range(self.num_layers):
            # Apply kernel-based feature expansion at each layer
            if self.kernel_based_feature_expension:
                if i == 0:
                    # First layer uses W_in, b_in
                    x_rff = self.apply_random_features(x, self.W_in, self.b_in)
                else:
                    # Subsequent layers use W_hidden, b_hidden
                    x_rff = self.apply_random_features(x, self.W_hidden, self.b_hidden)
                x = torch.cat([x, x_rff], dim=-1)

            x = self.convs[i](x, edge_index)
            # x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling to get graph-level representation
        x = global_add_pool(x, batch)

        # Final classification layer
        x = self.linear(x)
        return x
