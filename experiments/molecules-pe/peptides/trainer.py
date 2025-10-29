import logging
import os
import random
import uuid
from typing import TextIO, Optional, List, Dict, Any

import tqdm
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# from torch_geometric.loader import DataLoader
from src.data_utils.dataloader import DataLoader
from torch_geometric.utils import get_laplacian, to_dense_adj

from src.data_utils.peptides import PeptidesStructuralDataset

# models
from root import root
from src.mlp import MLP
from src.model import PEARL_GNN_Model, construct_model
from src.schema import Schema

from collections import defaultdict

# Ours
from sketch import AddFeaturesTransform


def print_parameter_count_by_module(model):
    module_params = defaultdict(int)
    total_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            module_name = name.split(".")[0]  # Get the top-level module name
            num_params = param.numel()
            module_params[module_name] += num_params
            total_params += num_params

    print("\nParameter count by module:\n")
    for module, num_params in module_params.items():
        print(f"{module}: {num_params} parameters")

    print(f"\nTotal parameters: {total_params}")
    return total_params


def print_parameter_count(model):
    total_params = 0
    print("Detailed parameter count:\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            print(f"{name}: {num_params} parameters")
            total_params += num_params
    print(f"\nTotal parameters: {total_params}")

    return total_params


def custom_collate_fn(data_list):
    """
    Custom collate function to handle batching of graphs with varying-sized Laplacians.
    """
    """for i, data in enumerate(data_list):
        print(f"Data {i}:")
        print(f"  Node features (x) shape: {data.x.shape if data.x is not None else 'None'}")
        print(f"  Edge index shape: {data.edge_index.shape if data.edge_index is not None else 'None'}")
        print(f"  Edge attributes shape: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")
        print(f"  Laplacian shape: {data.Lap.shape if hasattr(data, 'Lap') else 'None'}")"""
    batch = Batch.from_data_list(data_list)

    # Gather Laplacians into a list since they have different sizes
    batch.Lap = [data.Lap for data in data_list]

    return batch


class Trainer:
    cfg: Schema
    model: PEARL_GNN_Model
    train_loader: DataLoader
    val_loader: DataLoader
    optimizer: optim.Adam
    criterion: nn.L1Loss
    metric: nn.L1Loss
    logger: logging.Logger
    val_writer: TextIO
    curr_epoch: int
    curr_batch: int

    def __init__(self, cfg: Schema, gpu_id: Optional[int], splits=10) -> None:
        set_seed(cfg.seed)
        self.seed = cfg.seed

        self.cfg = cfg
        self.splits = splits
        self.splits = 1
        cfg.out_dirpath = root(cfg.out_dirpath)

        self.method = cfg.pe_method

        processed_suffix = "_pe" + str(cfg.pe_dims) if cfg.pe_method != "none" else ""
        transform = self.get_lap
        print("Loading peptides dataset...")

        # OURS
        pre_transform = AddFeaturesTransform(
            D_out=cfg.sketch_width,
            k=cfg.sketch_k,
            normalize=False,
            # R_global=R_global,
        )

        dataset = PeptidesStructuralDataset(
            root="dataset", transform=self.get_lap, pre_transform=pre_transform
        )
        split_idx = dataset.get_idx_split()
        train_dataset, val_dataset, test_dataset = (
            dataset[split_idx["train"]],
            dataset[split_idx["val"]],
            dataset[split_idx["test"]],
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=4
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=4
        )

        # construct model after loading dataset
        kwargs = {}
        kwargs["deg"] = None
        kwargs["device"] = "cuda"  # f"cuda:{gpu_id}"
        kwargs["residual"] = cfg.residual
        kwargs["bn"] = cfg.batch_norm
        kwargs["sn"] = cfg.graph_norm
        kwargs["feature_type"] = "discrete"
        self.device = torch.device(f"cuda:{gpu_id}")
        # self.model = construct_model(cfg, self.create_mlp, **kwargs)
        self.model = construct_model(
            cfg, (self.create_mlp, self.create_mlp_ln), peptides=True, **kwargs
        )
        self.model.to("cpu" if gpu_id is None else f"cuda:{gpu_id}")

        # Construct auxiliary training objects
        param_groups = self.get_param_groups()
        self.optimizer = optim.Adam(
            param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = LambdaLR(self.optimizer, self.lr_lambda)
        self.criterion = nn.L1Loss(reduction="mean")
        self.metric = nn.L1Loss(reduction="mean")

        # Set up logger and writer
        name = str(uuid.uuid4())
        # self.logger = logging.getLogger(name)
        # self.logger.setLevel(logging.INFO)

        os.makedirs(cfg.out_dirpath, exist_ok=True)
        handler = logging.FileHandler(os.path.join(cfg.out_dirpath, "train_logs.txt"))
        handler.setLevel(logging.INFO)

        # formatter = logging.Formatter(
        #     "%(asctime)s - %(levelname)s - %(message)s", "%Y/%m/%d %H:%M:%S"
        # )
        # handler.setFormatter(formatter)
        # self.logger.addHandler(handler)

        self.val_writer = open(os.path.join(cfg.out_dirpath, "evaluate_logs.txt"), "a")

        # Set up WandB
        self.wandb = cfg.wandb

        # Miscellaneous
        self.curr_epoch = 1
        self.curr_batch = 1

        # PEARL setup
        self.basis = cfg.basis
        self.num_samples = cfg.num_samples

    def train(self, seed="", name="RAND_200_TEST") -> None:
        print(self.seed, "SEED")
        # self.logger.info("Configuration:\n" + OmegaConf.to_yaml(self.cfg))
        print(
            f"Total parameters: {sum(param.numel() for param in self.model.parameters())}"
        )

        print_parameter_count_by_module(self.model)
        # print_parameter_count(self.model)

        # self.logger.info(
        #     f"Total parameters: {sum(param.numel() for param in self.model.parameters())}"
        # )
        # self.logger.info(f"Total training steps: {self.n_total_steps}")
        # self.logger.info(
        #     "Optimizer groups:\n"
        #     + "\n".join(group["name"] for group in self.optimizer.param_groups)
        #     + "\n"
        # )

        best_val_loss, best_test_loss = 999.0, 999.0
        for self.curr_epoch in range(1, self.cfg.n_epochs + 1):
            print("training")
            train_loss = self.train_epoch()
            print(f"Train loss is {train_loss}")
            print("validating")
            val_loss = self.evaluate(self.val_loader)
            test_loss = self.evaluate(self.test_loader)
            print(f"Val and test loss is {val_loss}, {test_loss}")
            # self.scheduler.step(eval_loss)
            # lr = self.scheduler.get_last_lr()[0]
            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            if self.wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "eval_loss": val_loss,
                        "test_loss": test_loss,
                        "lr": lr,
                    }
                )
            if val_loss < best_val_loss:
                if self.wandb:
                    best_val_loss, best_test_loss = val_loss, test_loss
                    wandb.run.summary["best_val_loss" + str(seed)] = best_val_loss
                    wandb.run.summary["best_test_loss" + str(seed)] = best_test_loss
                    wandb.run.summary["best_training_loss" + str(seed)] = train_loss
                if self.curr_epoch > 800:
                    new_name = "./" + name + ".pth"
                    torch.save(self.model.state_dict(), new_name)

        return best_test_loss

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        for batch in tqdm.tqdm(self.train_loader, ncols=100):
            total_loss += self.train_batch(batch)

        return total_loss / len(self.train_loader.dataset)

    def train_batch(self, batch: Batch) -> float:
        batch.to(device(self.model))
        self.optimizer.zero_grad()
        for k in range(self.splits):
            if self.method in ["pearl", "both"]:
                W_list = []
                for i in range(len(batch.Lap)):
                    if self.basis:
                        W = torch.eye(batch.Lap[i].shape[0]).to(self.device)
                    else:
                        W = torch.randn(
                            batch.Lap[i].shape[0], self.num_samples // self.splits
                        ).to(
                            self.device
                        )  # BxNxM
                    if len(W.shape) < 2:
                        print("TRAIN BATCH, i, LAP|W: ", i, batch.Lap[i].shape, W.shape)
                    W_list.append(W)
                self.model.forward(batch, W_list, final=False)
                if k == self.splits - 1:
                    y_pred = self.model(batch, W_list, final=True)  # [B]
            else:
                W_list = []
                # print("just sketch for training")
                self.model.forward(batch, W_list, final=False)
                if k == self.splits - 1:
                    y_pred = self.model(batch, W_list, final=True)  # [B]

        loss = self.criterion(y_pred, batch.y)  # [1]
        loss.backward()
        self.optimizer.step()

        loss = loss.item()
        self.scheduler.step()

        return loss * batch.y.size(0)
        # return loss

    def evaluate(self, eval_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in tqdm.tqdm(eval_loader, ncols=100):
            total_loss += self.evaluate_batch(batch)
            # self.logger.info(
            #     f"Evaluating... Epoch: {self.curr_epoch}, batch: {self.curr_batch}"
            # )
        total_loss /= len(eval_loader.dataset)  # FIXME I JUST ADDED THIS

        self.val_writer.write(f"Epoch: {self.curr_epoch}\t Loss: {total_loss}\n")
        self.val_writer.flush()
        return total_loss

    def evaluate_batch(self, batch: Batch) -> float:
        batch.to(device(self.model))
        for k in range(self.splits):
            if self.method in ["pearl", "both"]:
                W_list = []
                for i in range(len(batch.Lap)):
                    if self.basis:
                        W = torch.eye(batch.Lap[i].shape[0]).to(self.device)
                    else:
                        W = torch.randn(
                            batch.Lap[i].shape[0], self.num_samples // self.splits
                        ).to(
                            self.device
                        )  # BxNxM
                    if len(W.shape) < 2:
                        print("TRAIN BATCH, i, LAP|W: ", i, batch.Lap[i].shape, W.shape)
                    W_list.append(W)
                with torch.no_grad():
                    self.model.forward(batch, W_list, final=False)
                if k == self.splits - 1:
                    with torch.no_grad():
                        y_pred = self.model(batch, W_list, final=True)
            else:
                # print("just sketch for evaluation")
                W_list = []
                with torch.no_grad():
                    self.model.forward(batch, W_list, final=False)
                    if k == self.splits - 1:
                        # with torch.no_grad():
                        y_pred = self.model(batch, W_list, final=True)
        return self.metric(y_pred, batch.y).item() * batch.y.size(0)

    @property
    def n_total_steps(self) -> int:
        return len(self.train_loader) * self.cfg.n_epochs

    def create_mlp(self, in_dims: int, out_dims: int, LAYERS=None) -> MLP:
        if LAYERS is not None:
            return MLP(
                LAYERS,
                in_dims,
                self.cfg.mlp_hidden_dims,
                out_dims,
                self.cfg.mlp_use_bn,
                self.cfg.mlp_activation,
                self.cfg.mlp_dropout_prob,
            )
        return MLP(
            self.cfg.n_mlp_layers,
            in_dims,
            self.cfg.mlp_hidden_dims,
            out_dims,
            self.cfg.mlp_use_bn,
            self.cfg.mlp_activation,
            self.cfg.mlp_dropout_prob,
        )

    def create_mlp_ln(self, in_dims: int, out_dims: int, use_bias=True) -> MLP:
        return MLP(
            self.cfg.n_mlp_layers,
            in_dims,
            self.cfg.mlp_hidden_dims,
            out_dims,
            self.cfg.mlp_use_ln,
            self.cfg.pearl_act,
            self.cfg.mlp_dropout_prob,
            norm_type="layer",
            NEW_BATCH_NORM=True,
            use_bias=use_bias,
        )

    def get_lap(self, instance: Data) -> Data:
        n = instance.num_nodes
        L_edge_index, L_values = get_laplacian(
            instance.edge_index, normalization="sym", num_nodes=n
        )  # [2, X], [X]
        L = to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=n).squeeze(
            dim=0
        )
        instance.Lap = L
        return instance

    def get_param_groups(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "params": [param],
                "weight_decay": 0.0 if "bias" in name else self.cfg.weight_decay,
            }
            for name, param in self.model.named_parameters()
        ]

    def lr_lambda(self, curr_step: int) -> float:
        """
        Based on https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/optimization.py#L79
        """
        if curr_step < self.cfg.n_warmup_steps:
            return curr_step / max(1, self.cfg.n_warmup_steps)
        else:
            return max(
                0.0,
                (self.n_total_steps - curr_step)
                / max(1, self.n_total_steps - self.cfg.n_warmup_steps),
            )


def set_seed(seed: int) -> None:
    """
    Based on https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_utils.py#L83
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device
