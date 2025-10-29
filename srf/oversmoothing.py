import torch
import torch_geometric
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
from models import GCNGlobalRandom
from preprocess import AddFeaturesTransform
import scienceplots


def create_grid_dataset(grid_size, feat_dim):
    num_nodes = grid_size * grid_size
    edge_index = torch_geometric.utils.erdos_renyi_graph(
        num_nodes=num_nodes, edge_prob=0.05
    )
    x = torch.randn((num_nodes, feat_dim))
    data = Data(x=x, edge_index=edge_index)
    return data


def compute_dirichlet_energy(node_embeddings, edge_index):
    energy = 0
    for node_id in range(node_embeddings.shape[0]):
        x_i = node_embeddings[node_id]
        _, neighbors, _, _ = torch_geometric.utils.k_hop_subgraph(
            node_idx=node_id, num_hops=1, edge_index=edge_index
        )
        neighbors = neighbors[1, : neighbors.shape[1] // 2]
        for neighbor_id in neighbors:
            x_j = node_embeddings[neighbor_id]
            the_norm = torch.norm((x_i - x_j), p=2, dim=0).item()
            energy += the_norm**2
    energy /= node_embeddings.shape[0]
    return energy


def evaluate_model(model, data, device, num_layers=100):
    model.eval()
    energies = []
    data = data.to(device)

    with torch.no_grad():
        current_embeddings = data.x
        energies.append(compute_dirichlet_energy(current_embeddings, data.edge_index))

        for layer_idx in range(num_layers):
            current_embeddings = model.forward_by_layer(
                data, current_embeddings, layer_idx
            )
            current_embeddings = F.normalize(current_embeddings, p=2, dim=1)
            energy = compute_dirichlet_energy(current_embeddings, data.edge_index)
            energies.append(energy)

    return energies


def plot_oversmoothing_figures(all_energies, k_energies):
    plt.style.use(["science", "no-latex"])
    # Change figure size to make plots more rectangular (wider) - now only 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Increase font sizes
    LABEL_SIZE = 22  # Increased from 20
    TITLE_SIZE = 26  # New size for titles
    TICK_SIZE = 20  # Increased from 20
    LEGEND_SIZE = 15  # Increased from 16
    LINE_WIDTH = 4.0

    # Define colors
    colors = {
        "baseline": "#1f77b4",
        "random": "#ff7f0e",
        "rbf": "#2ca02c",
        "laplace": "#d62728",
        "linear": "#9467bd",
        "ablation": "#8c564b",
    }

    # Define method labels with math symbols - fixed to be compatible with matplotlib
    method_labels = {
        "baseline": "baseline",
        "linear": r"$(\mathcal{E}_{\text{linear}}, \mathcal{S}_{\mathrm{AG}}^{(8)})$",
        "rbf": r"$(\mathcal{E}_{\text{RBF}}, \mathcal{S}_{\mathrm{AG}}^{(8)})$",
        "laplace": r"$(\mathcal{E}_{\mathcal{L}}, \mathcal{S}_{\mathrm{AG}}^{(8)})$",
        "ablation": r"$(\mathcal{E}_{\mathcal{L}}, \mathcal{S}_{\mathrm{id}})$ (ablation)",
    }

    # Plot 1: Methods comparison (Dirichlet) - Oversmoothing
    for method, energies in all_energies.items():
        layers = np.arange(1, len(energies) + 1)
        axes[0].loglog(
            layers,
            np.array(energies) + 1e-10,
            label=method_labels.get(method, method),
            color=colors[method],
            linewidth=LINE_WIDTH,
        )

    # Add title to first plot with increased size
    axes[0].set_title("Oversmoothing", fontsize=TITLE_SIZE, fontweight="bold")
    axes[0].set_xlabel("Layer $l$", fontsize=LABEL_SIZE, fontweight="bold")
    axes[0].set_ylabel(
        "Dirichlet energy",
        fontsize=LABEL_SIZE,
        fontweight="bold",
    )
    axes[0].tick_params(axis="both", labelsize=TICK_SIZE)

    # my code
    handles, labels = axes[0].get_legend_handles_labels()
    new_order = [0, 4, 3, 2, 1]  # Example indices for reordering
    handles = [handles[i] for i in new_order]
    labels = [labels[i] for i in new_order]
    axes[0].legend(
        handles, labels, fontsize=LEGEND_SIZE, frameon=True, loc="lower left"
    )
    axes[0].grid(True, which="both", ls="-", alpha=0.2)
    axes[0].text(
        0.07,
        0.93,
        "(a)",
        transform=axes[0].transAxes,
        fontsize=LABEL_SIZE,
        fontweight="bold",
    )

    # Plot 2: K-values comparison (Dirichlet) - Oversmoothing
    k_colors = plt.cm.viridis(np.linspace(0, 1, len(k_energies)))
    for (k, energies), color in zip(k_energies.items(), k_colors):
        layers = np.arange(1, len(energies) + 1)
        axes[1].loglog(
            layers,
            np.array(energies) + 1e-10,
            label=f"k={k}",
            color=color,
            linewidth=LINE_WIDTH,
        )

    # Add title to second plot with increased size
    axes[1].set_title("Oversmoothing", fontsize=TITLE_SIZE, fontweight="bold")
    axes[1].set_xlabel("Layer $l$", fontsize=LABEL_SIZE, fontweight="bold")
    axes[1].set_ylabel(
        "Dirichlet energy",
        fontsize=LABEL_SIZE,
        fontweight="bold",
    )
    axes[1].tick_params(axis="both", labelsize=TICK_SIZE)

    # Move legend to top right for the second plot
    axes[1].legend(fontsize=LEGEND_SIZE, frameon=True, loc="upper right")
    axes[1].grid(True, which="both", ls="-", alpha=0.2)
    axes[1].text(
        0.07,
        0.93,
        "(b)",
        transform=axes[1].transAxes,
        fontsize=LABEL_SIZE,
        fontweight="bold",
    )

    # Add more space between subplots
    plt.subplots_adjust(wspace=0.3)
    plt.savefig("oversmoothing_analysis.pdf", bbox_inches="tight", dpi=800)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    HIDDEN_SIZE = 512
    INPUT_SIZE = 32
    NUM_LAYERS = 100
    NUM_CLASSES = 1
    D_out = 32
    DROPOUT = 0.0
    EDGE_DIM = 0
    USE_BATCH_NORM = False

    # Create dataset
    data = create_grid_dataset(grid_size=10, feat_dim=INPUT_SIZE)

    # Methods plot
    methods = ["baseline", "rbf", "laplace", "linear", "ablation"]
    all_energies = {}

    transform = AddFeaturesTransform(D_out=D_out, k=1)
    data_methods = transform(data.clone())

    for method in methods:
        print(f"Evaluating method: {method}")
        dout_ = D_out if method != "baseline" else 0
        model = GCNGlobalRandom(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            aux_feats_dim=dout_,
            dropout=DROPOUT,
            method=method,
            use_batch_norm=USE_BATCH_NORM,
            edge_dim=EDGE_DIM,
            dataset_type="synthetic",
        ).to(device)

        energies = evaluate_model(model, data_methods, device, num_layers=NUM_LAYERS)
        all_energies[method] = energies

    # K-values plot (Laplace only)
    k_values = [1, 2, 4, 8, 16, 32]
    k_energies = {}

    for k in k_values:
        print(f"Evaluating k={k}")
        transform_k = AddFeaturesTransform(D_out=D_out, k=k)
        data_k = transform_k(data.clone())

        model = GCNGlobalRandom(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            aux_feats_dim=D_out,
            dropout=DROPOUT,
            method="laplace",
            use_batch_norm=USE_BATCH_NORM,
            edge_dim=EDGE_DIM,
            dataset_type="synthetic",
        ).to(device)

        energies = evaluate_model(model, data_k, device, num_layers=NUM_LAYERS)
        k_energies[k] = energies

    # Create plots (oversmoothing only)
    plot_oversmoothing_figures(all_energies, k_energies)

    return all_energies, k_energies


if __name__ == "__main__":
    main()
