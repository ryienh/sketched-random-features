# dataset.py

import torch
import torch_geometric  # type: ignore
from torch_geometric.datasets import ZINC, QM9, GNNBenchmarkDataset, LRGBDataset, TUDataset  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.utils import smiles2graph
from torch_geometric.transforms import BaseTransform

from preprocess import AddFeaturesTransform


class CastToFloat(BaseTransform):
    def __call__(self, data):
        # data.x = data.x.float()
        # if data.edge_attr is not None:
        #     data.edge_attr = data.edge_attr.float()
        if data.edge_attr is not None and len(data.edge_attr.shape) == 1:
            data.edge_attr = data.edge_attr.unsqueeze(-1)
        return data


def get_loaders(
    dataset_name: str,
    batch_size: int,
    transform_args: dict,
    seed: int = 42,
    shuffle_train: bool = True,
):
    """
    Returns train/val/test DataLoaders along with metadata like input_size and num_classes.

    Parameters:
    -----------
    - dataset_name: str
        One of ['qm9', 'csl', ...].
    - batch_size: int
        Batch size for the DataLoaders.
    - transform_args: dict
        Arguments for instantiating your transform(s). For example:
        {
            "K": 4,
            "width": 4,
            "R_global_dim": 1000,
        }
    - seed: int
        Random seed to ensure reproducibility.
    - shuffle_train: bool
        Whether to shuffle train data.

    Returns:
    --------
    - train_loader, val_loader, test_loader (torch_geometric.loader.DataLoader)
    - input_size (int)
    - num_classes (int)
    """

    torch.manual_seed(seed)
    torch_geometric.seed_everything(seed)

    # Create transform
    transform = AddFeaturesTransform(
        D_out=transform_args["width"],
        k=transform_args["K"],
    )

    print(f"Using D_out = {transform_args['width']} and K = {transform_args['K']}")

    if dataset_name.lower() == "zinc":
        float_transform = CastToFloat()
        train_dataset = ZINC(
            root="data/zinc",
            subset=True,
            split="train",
            transform=None,
            pre_transform=transform,
        )

        val_dataset = ZINC(
            root="data/zinc",
            subset=True,
            split="val",
            transform=None,
            pre_transform=transform,
        )

        test_dataset = ZINC(
            root="data/zinc",
            subset=True,
            split="test",
            transform=None,
            pre_transform=transform,
        )

        input_size = train_dataset.num_node_features
        edge_dim = train_dataset.num_edge_features
        num_classes = 1

    elif dataset_name.lower() == "qm9":
        dataset = QM9(root="data/qm9", pre_transform=transform).shuffle()

        n = len(dataset)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)

        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size : train_size + val_size]
        test_dataset = dataset[train_size + val_size :]

        # normalize targets
        mean = train_dataset.data.y.mean(dim=0, keepdim=True)  # type: ignore
        std = train_dataset.data.y.std(dim=0, keepdim=True)  # type: ignore
        train_dataset.data.y = (train_dataset.data.y - mean) / std  # type: ignore
        val_dataset.data.y = (val_dataset.data.y - mean) / std  # type: ignore
        test_dataset.data.y = (test_dataset.data.y - mean) / std  # type: ignore

        input_size = dataset.num_node_features  # type: ignore
        edge_dim = dataset.num_edge_features  # type: ignore
        num_classes = 1  # MSE regression target

    elif dataset_name.lower() == "csl":
        train_dataset = GNNBenchmarkDataset(
            root="data/csl", name="CSL", split="train", pre_transform=transform
        )
        val_dataset = GNNBenchmarkDataset(
            root="data/csl", name="CSL", split="val", pre_transform=transform
        )
        test_dataset = GNNBenchmarkDataset(
            root="data/csl", name="CSL", split="test", pre_transform=transform
        )
        if shuffle_train:
            train_dataset = train_dataset.shuffle()

        input_size = 4  # FIXME: fix hardcode
        edge_dim = 0  # train_dataset.num_edge_features  # type: ignore
        num_classes = 10  # Graph classification

    elif dataset_name.lower() == "ogbg-ppa":
        dataset = PygGraphPropPredDataset(
            name=dataset_name.lower(), pre_transform=transform
        )

        split_idx = dataset.get_idx_split()

        train_dataset = dataset[split_idx["train"]]
        val_dataset = dataset[split_idx["valid"]]
        test_dataset = dataset[split_idx["test"]]

        input_size = dataset.num_node_features
        edge_dim = dataset.num_edge_features  # type: ignore
        num_classes = 37

    elif dataset_name.lower() == "ogbg-molhiv":
        float_transform = CastToFloat()
        dataset = PygGraphPropPredDataset(
            name=dataset_name.lower(),
            pre_transform=transform,
            transform=float_transform,
        )

        split_idx = dataset.get_idx_split()

        train_dataset = dataset[split_idx["train"]]
        val_dataset = dataset[split_idx["valid"]]
        test_dataset = dataset[split_idx["test"]]

        input_size = dataset.num_node_features
        edge_dim = dataset.num_edge_features  # type: ignore
        num_classes = 1

    elif dataset_name.lower() == "pcqm4mv2":
        dataset = PygPCQM4Mv2Dataset(
            "data", smiles2graph=smiles2graph, pre_transform=transform
        )

        split_idx = dataset.get_idx_split()

        train_dataset = dataset[split_idx["train"]]  # type: ignore
        val_dataset = dataset[split_idx["valid"]]  # type: ignore
        test_dataset = dataset[split_idx["test-dev"]]  # type: ignore

        input_size = dataset.num_node_features
        edge_dim = dataset.num_edge_features  # type: ignore
        num_classes = 1

    elif dataset_name.lower() == "peptides-func":
        train_dataset = LRGBDataset(
            root="data", name="Peptides-func", split="train", pre_transform=transform
        )
        val_dataset = LRGBDataset(
            root="data", name="Peptides-func", split="val", pre_transform=transform
        )
        test_dataset = LRGBDataset(
            root="data", name="Peptides-func", split="test", pre_transform=transform
        )

        input_size = train_dataset.num_node_features
        edge_dim = train_dataset.num_edge_features
        num_classes = 10

    elif dataset_name.lower() == "peptides-struct":
        train_dataset = LRGBDataset(
            root="data", name="Peptides-struct", split="train", pre_transform=transform
        )
        val_dataset = LRGBDataset(
            root="data", name="Peptides-struct", split="val", pre_transform=transform
        )
        test_dataset = LRGBDataset(
            root="data", name="Peptides-struct", split="test", pre_transform=transform
        )

        input_size = train_dataset.num_node_features
        edge_dim = train_dataset.num_edge_features
        num_classes = 11

    elif dataset_name.lower() == "reddit-m":
        dataset = TUDataset(
            root="data", name="REDDIT-MULTI-5K", pre_transform=transform
        ).shuffle()  # Shuffle before splitting into folds

        # For k-fold cross validation, we don't want fixed train/val/test splits
        # Instead, return the full dataset and num_folds
        num_folds = 10
        fold_size = len(dataset) // num_folds

        input_size = dataset.num_node_features
        edge_dim = dataset.num_edge_features
        num_classes = dataset.num_classes

        # Instead of creating fixed loaders, we'll create a function to get loaders for a specific fold
        def get_fold_loaders(fold_idx, batch_size, shuffle_train=True):
            # Calculate indices for this fold
            val_start = fold_idx * fold_size
            val_end = (fold_idx + 1) * fold_size

            # Split dataset for this fold
            val_indices = list(range(val_start, val_end))
            train_indices = list(range(0, val_start)) + list(
                range(val_end, len(dataset))
            )

            # Create fold-specific datasets
            train_dataset = dataset[train_indices]
            val_dataset = dataset[val_indices]

            # Create loaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=shuffle_train
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            return train_loader, val_loader

        # Return special values to indicate cross-validation is needed
        return get_fold_loaders, num_folds, input_size, edge_dim, num_classes

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train  # type: ignore
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # type: ignore
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # type: ignore

    return train_loader, val_loader, test_loader, input_size, edge_dim, num_classes
