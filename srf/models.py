import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GINConv, global_mean_pool, GCNConv  # type: ignore
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GINEGlobalRandom(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        k,
        dropout,
        method,
        use_batch_norm,
        edge_dim,
        dataset_type,
    ):
        """
        GINE model that repeatedly injects a random embedding at each layer.
        """
        super().__init__()
        assert method in ["rbf", "laplace", "linear", "random", "baseline", "ablation"]
        assert not (method == "baseline" and k != 0), "Baseline requires that k"
        self.method = method
        self.k = k
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_batch_norm = use_batch_norm
        self.edge_dim = edge_dim
        self.dataset_type = dataset_type

        # Add atom encoder for molecular graphs
        if dataset_type.lower() == "zinc":
            self.atom_encoder = nn.Embedding(21, 75)  # ZINC has 21 atom types
            self.bond_encoder = (
                nn.Embedding(4, 50) if edge_dim > 0 else None
            )  # ZINC has 4 bond types
            # self.atom_encoder = AtomEncoder(hidden_size)
            # self.bond_encoder = BondEncoder(16) if edge_dim > 0 else None
        elif dataset_type.lower() == "ogb":
            self.atom_encoder = AtomEncoder(hidden_size)
            self.bond_encoder = BondEncoder(16) if edge_dim > 0 else None
        elif dataset_type.lower().startswith("peptides"):
            self.atom_encoder = AtomEncoder(hidden_size)
            self.bond_encoder = BondEncoder(16) if edge_dim > 0 else None
        else:
            self.atom_encoder = None
            self.bond_encoder = None

        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Use hidden_size as in_dim if using atom encoders
        in_dim = input_size
        if dataset_type.lower() in ["ogb", "peptides"]:
            in_dim = hidden_size
        elif dataset_type.lower() in ["zinc"]:
            in_dim = 75

        edge_dim_size = 50 if dataset_type.lower() in ["zinc"] else 16

        for idx in range(num_layers):

            if idx == 0:
                mlp = nn.Sequential(
                    nn.Linear(in_dim + k, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                conv = (
                    GINEConv(
                        mlp, train_eps=True, edge_dim=edge_dim_size
                    )  # Use 16 for edge features
                    if self.edge_dim > 0
                    else GINConv(mlp, train_eps=True)
                )
                self.gin_layers.append(conv)
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_size + k, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                conv = (
                    GINEConv(
                        mlp, train_eps=True, edge_dim=edge_dim_size
                    )  # Use 16 for edge features
                    if self.edge_dim > 0
                    else GINConv(mlp, train_eps=True)
                )
                self.gin_layers.append(conv)

            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))

        self.final_lin = (
            nn.Linear(hidden_size, num_classes)
            if dataset_type.lower() not in ["zinc"]
            else nn.Sequential(
                nn.Linear(hidden_size, 50),
                nn.ReLU(),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, 1),
            )
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Encode atoms and bonds if using molecular graphs
        if self.atom_encoder is not None:
            if self.dataset_type != "zinc":
                x = self.atom_encoder(x.squeeze())  # Add squeeze() here
            else:
                x = self.atom_encoder(x.squeeze())
        if self.bond_encoder is not None and edge_attr is not None:
            if self.dataset_type != "zinc":
                edge_attr = self.bond_encoder(edge_attr.squeeze())
            else:
                edge_attr = self.bond_encoder(edge_attr)

        # Keep track of the node embedding as x_l
        x_l = x

        # 1) get random feats
        feats = None
        if self.method == "rbf":
            feats = data.rbf_feats
        elif self.method == "linear":
            feats = data.linear_feats
        elif self.method == "laplace":
            feats = data.laplace_feats
        elif self.method == "random":
            feats = data.random_feats
        elif self.method == "ablation":
            feats = data.laplace_feats_no_sketch

        # For each GIN layer, concat feats and apply conv
        for layer_idx, conv in enumerate(self.gin_layers):
            # 2) Concat feats with node attribute matrix
            x_cat = torch.cat([x_l, feats], dim=-1) if feats is not None else x_l

            # 3) GINConv layer
            h = conv(x_cat, edge_index, edge_attr)

            # Apply batch normalization if enabled
            if self.use_batch_norm:
                h = self.batch_norms[layer_idx](h)

            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # 4) Update x_l -> h
            x_l = h

        # Global mean pool for graph-level representation
        h_pool = global_mean_pool(x_l, batch)  # [num_graphs_in_batch, hidden_size]

        out = self.final_lin(h_pool)  # [num_graphs_in_batch, num_classes]
        return out

    def forward_by_layer(self, data, curr_embeddings, layer_idx):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # 1) get random feats
        feats = None
        if self.method == "rbf":
            feats = data.rbf_feats
        elif self.method == "linear":
            feats = data.linear_feats
        elif self.method == "laplace":
            feats = data.laplace_feats
        elif self.method == "random":
            feats = data.random_feats

        # 2) Concat feats with node attribute matrix
        x_cat = (
            torch.cat([curr_embeddings, feats], dim=-1)
            if feats is not None
            else curr_embeddings
        )

        # 3) GINConv layer
        h = self.gin_layers[layer_idx](x_cat, edge_index, edge_attr)

        # Apply batch normalization if enabled
        if self.use_batch_norm:
            h = self.batch_norms[layer_idx](h)

        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # print(torch.norm(h, p=2, dim=1).sum().item())

        # 4) return the hidden representation
        return h


class GINELaplaceVariant(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        k,
        dropout=0.0,
        variant="standard",  # "standard", "var_{factor}", "nonlinear_{type}"
        use_batch_norm=False,
        edge_dim=None,
        dataset_type="other",
    ):
        """
        GINE model that uses different variants of Laplace embeddings.

        :param input_size: Dimension of input node features
        :param hidden_size: Dimension of hidden layers
        :param num_layers: Number of GIN layers
        :param num_classes: Number of output classes
        :param k: Dimension of the sketch features to use
        :param dropout: Dropout probability
        :param variant: Which Laplace variant to use:
                        - "standard": Default Laplace features
                        - "var_X.X": Variance-tuned Laplace with factor X.X
                        - "nonlinear_rbf": Nonlinear RBF combination
                        - "nonlinear_laplace": Nonlinear Laplace combination
                        - "nonlinear_tanh": Simple tanh transformation
                        - "nonlinear_mlp": MLP-style transformation
        :param use_batch_norm: Whether to use batch normalization
        :param edge_dim: Dimension of edge features (if any)
        :param dataset_type: Type of dataset ("zinc", "ogb", "peptides", or "other")
        """
        super().__init__()
        self.variant = variant
        self.k = k
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_batch_norm = use_batch_norm
        self.edge_dim = edge_dim
        self.dataset_type = dataset_type

        # Add atom encoder for molecular graphs
        if dataset_type.lower() == "zinc":
            self.atom_encoder = nn.Embedding(21, 75)  # ZINC has 21 atom types
            self.bond_encoder = (
                nn.Embedding(4, 50) if edge_dim > 0 else None
            )  # ZINC has 4 bond types
        elif dataset_type.lower() == "ogb":
            self.atom_encoder = AtomEncoder(hidden_size)
            self.bond_encoder = BondEncoder(16) if edge_dim > 0 else None
        elif dataset_type.lower().startswith("peptides"):
            self.atom_encoder = AtomEncoder(hidden_size)
            self.bond_encoder = BondEncoder(16) if edge_dim > 0 else None
        else:
            self.atom_encoder = None
            self.bond_encoder = None

        # Create GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Use hidden_size as in_dim if using atom encoders
        in_dim = input_size
        if dataset_type.lower() in ["ogb", "peptides"]:
            in_dim = hidden_size
        elif dataset_type.lower() in ["zinc"]:
            in_dim = 75

        edge_dim_size = 50 if dataset_type.lower() in ["zinc"] else 16

        # Create GIN layers
        for idx in range(num_layers):
            if idx == 0:
                mlp = nn.Sequential(
                    nn.Linear(in_dim + k, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_size + k, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )

            conv = (
                GINEConv(mlp, train_eps=True, edge_dim=edge_dim_size)
                if self.edge_dim > 0
                else GINConv(mlp, train_eps=True)
            )
            self.convs.append(conv)

            if use_batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_size))
            else:
                self.bns.append(nn.Identity())

        # Global pooling function
        self.global_pool = global_mean_pool

        # Prediction head
        if dataset_type.lower() == "zinc":
            self.pred_head = nn.Sequential(
                nn.Linear(hidden_size, 50),
                nn.ReLU(),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, 1),
            )
        else:
            self.pred_head = nn.Linear(hidden_size, num_classes)

    def forward(self, data):
        # Extract the graph structure
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr if hasattr(data, "edge_attr") else None,
            data.batch,
        )

        # Choose the appropriate feature set based on variant
        laplace_feats = None
        if self.variant == "standard":
            # Standard Laplace features
            laplace_feats = data.laplace_feats
        elif self.variant.startswith("var_"):
            # Variance-tuned variant, extract factor from name
            factor = float(self.variant.split("_")[1])
            laplace_feats = data.laplace_feats_tune_var[factor]
        elif self.variant.startswith("nonlinear_"):
            # Nonlinear combination variant, extract type from name
            nl_type = self.variant.split("_")[1]
            laplace_feats = data.laplace_nonlinear_combine[nl_type]
        elif self.variant == "raw_gaussian":
            # Raw Gaussian sketch variant: [Φ || G₁Φ || G₂Φ || ... || GₖΦ]
            laplace_feats = data.laplace_raw_gaussian
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        # Encode atoms and bonds if using molecular graphs
        if self.atom_encoder is not None:
            x = self.atom_encoder(x.squeeze())
        if self.bond_encoder is not None and edge_attr is not None:
            edge_attr = self.bond_encoder(
                edge_attr.squeeze() if self.dataset_type != "zinc" else edge_attr
            )

        # Process through GNN layers
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # Concatenate the laplace features with node embeddings
            h_cat = torch.cat([h, laplace_feats], dim=1) if self.k > 0 else h

            # Apply GIN layer
            h_new = conv(h_cat, edge_index, edge_attr=edge_attr)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            # Residual connections after first layer
            if i > 0:
                h = h_new + h
            else:
                h = h_new

        # Global pooling and prediction head
        h = self.global_pool(h, batch)
        out = self.pred_head(h)

        if self.dataset_type == "zinc":
            out = out.view(-1)  # Flatten for ZINC regression

        return out


class GCNGlobalRandom(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        aux_feats_dim,
        dropout,
        method,
        use_batch_norm,
        edge_dim,  # Not used here but kept for a consistent signature
        dataset_type,
    ):
        """
        GCN model that repeatedly injects a random embedding (or other auxiliary features)
        at each layer. Edge features are ignored.
        """
        super().__init__()
        assert method in ["rbf", "laplace", "linear", "random", "baseline", "ablation"]
        assert not (
            method == "baseline" and aux_feats_dim != 0
        ), "Baseline requires that k=0"
        self.method = method
        self.aux_feats_dim = aux_feats_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_batch_norm = use_batch_norm

        # Optionally add atom encoder for molecular datasets
        if dataset_type.lower() == "zinc":
            # ZINC has 21 atom types
            self.atom_encoder = nn.Embedding(21, hidden_size)
        elif dataset_type.lower() == "ogb":
            from ogb.graphproppred.mol_encoder import AtomEncoder

            self.atom_encoder = AtomEncoder(hidden_size)
        else:
            self.atom_encoder = None

        # Create layers
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # If using an atom encoder, input dimension is `hidden_size`,
        # otherwise it is `input_size`.
        in_dim = (
            hidden_size if (dataset_type.lower() in ["zinc", "ogb"]) else input_size
        )

        # Build each GCN layer
        for idx in range(num_layers):
            if idx == 0:
                conv = GCNConv(in_dim + aux_feats_dim, hidden_size)
            else:
                conv = GCNConv(hidden_size + aux_feats_dim, hidden_size)
            self.gcn_layers.append(conv)

            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))

        # Final linear layer for classification/regression
        self.final_lin = nn.Linear(hidden_size, num_classes)

    def forward(self, data):
        # Unpack data
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Atom encoder if using a molecular dataset
        if self.atom_encoder is not None:
            x = self.atom_encoder(x.squeeze())

        # x_l will store the current node embeddings
        x_l = x

        # Pick up the auxiliary features based on the chosen method
        feats = None
        if self.method == "rbf":
            feats = data.rbf_feats
        elif self.method == "laplace":
            feats = data.laplace_feats
        elif self.method == "linear":
            feats = data.linear_feats
        elif self.method == "random":
            feats = data.random_feats
        elif self.method == "ablation":
            feats = data.laplace_feats_no_sketch
        # (If method == "baseline" or feats not present, feats remains None)

        # Pass through each GCN layer
        for layer_idx, conv in enumerate(self.gcn_layers):
            # Optionally concatenate random/aux features
            x_cat = torch.cat([x_l, feats], dim=-1) if feats is not None else x_l

            # GCN forward pass (no edge features)
            h = conv(x_cat, edge_index)

            # Batch normalization if requested
            if self.use_batch_norm:
                h = self.batch_norms[layer_idx](h)

            # Nonlinear activation and dropout
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # Update x_l
            x_l = h

        # Global mean pool for graph-level embeddings
        h_pool = global_mean_pool(x_l, batch)  # [num_graphs, hidden_size]

        # Final classification/regression layer
        out = self.final_lin(h_pool)  # [num_graphs, num_classes]
        return out

    def forward_by_layer(self, data, curr_embeddings, layer_idx):
        """
        Optional method for extracting the intermediate node embeddings
        after a given layer.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Pick up the auxiliary features
        feats = None
        if self.method == "rbf":
            feats = data.rbf_feats
        elif self.method == "laplace":
            feats = data.laplace_feats
        elif self.method == "linear":
            feats = data.linear_feats
        elif self.method == "random":
            feats = data.random_feats
        elif self.method == "ablation":
            feats = data.laplace_feats_no_sketch

        # Concat feats with current embeddings
        x_cat = (
            torch.cat([curr_embeddings, feats], dim=-1)
            if feats is not None
            else curr_embeddings
        )

        # GCN forward pass
        h = self.gcn_layers[layer_idx](x_cat, edge_index)

        # Batch normalization if requested
        if self.use_batch_norm:
            h = self.batch_norms[layer_idx](h)

        # Nonlinear activation and dropout
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return h
