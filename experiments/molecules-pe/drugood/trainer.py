import logging
import os
import random
import uuid
from typing import TextIO, Optional, List, Dict, Any

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torchmetrics.classification import BinaryAUROC

from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
from dataset import DrugOOD
from src.data_utils.dataloader import DataLoader
from torch_geometric.utils import get_laplacian, to_dense_adj
import time

# models
from root import root
from src.mlp import MLP
from src.model import PEARL_GNN_Model, GINEBaseModel, construct_model
from src.schema import Schema

# Ours
from sketch import AddFeaturesTransform


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

    def __init__(self, cfg: Schema, gpu_id: Optional[int]) -> None:
        set_seed(cfg.seed)

        # Initialize configuration
        self.cfg = cfg
        cfg.out_dirpath = root(cfg.out_dirpath)

        # Construct data loaders
        ## dataset preprocessing (before saved in disk) and loading
        # OURS
        pre_transform = AddFeaturesTransform(
            D_out=cfg.sketch_width,
            k=cfg.sketch_k,
            # R_global=R_global,
        )
        processed_suffix = (
            "_pearl" + str(cfg.pe_dims) if cfg.pe_method != "none" else ""
        )
        transform = self.get_lap
        curator = "lbap_core_ic50_" + cfg.dataset
        data_root = root(
            os.path.join("data/drugood", curator)
        )  # use your own data root
        train_dataset = DrugOOD(
            data_root,
            curator=curator,
            split="train",
            pre_transform=pre_transform,
            transform=transform,
            processed_suffix=processed_suffix,
        )
        iid_val_dataset = DrugOOD(
            data_root,
            curator=curator,
            split="iid_val",
            pre_transform=pre_transform,
            transform=transform,
            processed_suffix=processed_suffix,
        )
        ood_val_dataset = DrugOOD(
            data_root,
            curator=curator,
            split="ood_val",
            pre_transform=pre_transform,
            transform=transform,
            processed_suffix=processed_suffix,
        )
        iid_test_dataset = DrugOOD(
            data_root,
            curator=curator,
            split="iid_test",
            pre_transform=pre_transform,
            transform=transform,
            processed_suffix=processed_suffix,
        )
        ood_test_dataset = DrugOOD(
            data_root,
            curator=curator,
            split="ood_test",
            pre_transform=pre_transform,
            transform=transform,
            processed_suffix=processed_suffix,
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=4
        )
        self.iid_val_loader = DataLoader(
            iid_val_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=4
        )
        self.ood_val_loader = DataLoader(
            ood_val_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=4
        )
        self.iid_test_loader = DataLoader(
            iid_test_dataset,
            batch_size=cfg.val_batch_size,
            shuffle=False,
            num_workers=4,
        )
        self.ood_test_loader = DataLoader(
            ood_test_dataset,
            batch_size=cfg.val_batch_size,
            shuffle=False,
            num_workers=4,
        )

        # construct model after loading dataset
        kwargs = {}
        kwargs["deg"] = None
        kwargs["device"] = f"cuda:{gpu_id}"
        kwargs["residual"] = cfg.residual
        kwargs["bn"] = cfg.batch_norm
        kwargs["sn"] = cfg.graph_norm
        kwargs["feature_type"] = "continuous"
        self.model = construct_model(
            cfg, (self.create_mlp, self.create_mlp_ln), **kwargs
        )
        print(self.model)
        self.model.to("cpu" if gpu_id is None else f"cuda:{gpu_id}")

        param_groups = self.get_param_groups()
        self.optimizer = optim.Adam(
            param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = LambdaLR(self.optimizer, self.lr_lambda)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.metric = BinaryAUROC()

        self.device = torch.device(f"cuda:{gpu_id}")

        # Set up logger and writer
        name = str(uuid.uuid4())
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        os.makedirs(cfg.out_dirpath, exist_ok=True)
        handler = logging.FileHandler(os.path.join(cfg.out_dirpath, "train_logs.txt"))
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", "%Y/%m/%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.val_writer = open(os.path.join(cfg.out_dirpath, "evaluate_logs.txt"), "a")

        # Set up WandB
        cfg.__dict__["num_params"] = sum(
            param.numel() for param in self.model.parameters()
        )
        print("num params: ", cfg.__dict__["num_params"])

        # Miscellaneous
        self.curr_epoch = 1
        self.curr_batch = 1
        self.basis = cfg.basis
        self.num_samples = cfg.num_samples
        self.seed = cfg.seed

    def train(self) -> None:
        self.logger.info("Configuration:\n" + OmegaConf.to_yaml(self.cfg))
        self.logger.info(
            f"Total parameters: {sum(param.numel() for param in self.model.parameters())}"
        )
        self.logger.info(f"Total training steps: {self.n_total_steps}")
        self.logger.info(
            "Optimizer groups:\n"
            + "\n".join(group["name"] for group in self.optimizer.param_groups)
            + "\n"
        )

        best_iid_val_loss, best_ood_val_loss, best_iid_test_loss, best_ood_test_loss = (
            0.0,
            0.0,
            0.0,
            0.0,
        )

        for self.curr_epoch in range(1, self.cfg.n_epochs + 1):
            start = time.time()

            train_loss = self.train_epoch()

            # memory_allocated = torch.cuda.max_memory_allocated("cuda:7") // (1024**2)
            # memory_reserved = torch.cuda.max_memory_reserved("cuda:7") // (1024**2)

            iid_val_loss = self.evaluate(self.iid_val_loader)
            ood_val_loss = self.evaluate(self.ood_val_loader)
            iid_test_loss = self.evaluate(self.iid_test_loader)
            ood_test_loss = self.evaluate(self.ood_test_loader)

            time_per_epoch = time.time() - start

            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            wandb.log(
                {
                    "train_loss": train_loss,
                    "iid_eval_loss": iid_val_loss,
                    "ood_val_loss": ood_val_loss,
                    "iid_test_loss": iid_test_loss,
                    "ood_test_loss": ood_test_loss,
                    "lr": lr,
                }
            )
            print(
                f"Seconds: {time_per_epoch:.4f}, "
                # f"Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved."
            )
            if iid_val_loss > best_iid_val_loss:
                best_iid_val_loss, best_iid_test_loss = iid_val_loss, iid_test_loss
                wandb.run.summary["best_iid_val_loss" + str(self.seed)] = (
                    best_iid_val_loss
                )
                wandb.run.summary["best_iid_test_loss" + str(self.seed)] = (
                    best_iid_test_loss
                )
            if ood_val_loss > best_ood_val_loss:
                best_ood_val_loss, best_ood_test_loss = ood_val_loss, ood_test_loss
                wandb.run.summary["best_ood_val_loss" + str(self.seed)] = (
                    best_ood_val_loss
                )
                wandb.run.summary["best_ood_test_loss" + str(self.seed)] = (
                    best_ood_test_loss
                )
        return best_ood_test_loss

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        for self.curr_batch, batch in enumerate(self.train_loader, 1):
            total_loss += self.train_batch(batch)

        return total_loss / len(self.train_loader.dataset)

    def train_batch(self, batch: Batch) -> float:
        batch.to(device(self.model))
        self.optimizer.zero_grad()
        W_list = []
        for i in range(len(batch.Lap)):
            if self.basis:
                W = torch.eye(batch.Lap[i].shape[0]).to(self.device)
            else:
                W = torch.randn(batch.Lap[i].shape[0], self.num_samples).to(
                    self.device
                )  # BxNxM
            if len(W.shape) < 2:
                print("TRAIN BATCH, i, LAP|W: ", i, batch.Lap[i].shape, W.shape)
            W_list.append(W)
        y_pred = self.model(batch, W_list)
        loss = self.criterion(y_pred, batch.y)
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        self.scheduler.step()
        return loss * batch.y.size(0)

    def evaluate(self, eval_loader: DataLoader) -> float:
        self.model.eval()

        for self.curr_batch, batch in enumerate(eval_loader, 1):
            self.evaluate_batch(batch)

        total_loss = self.metric.compute().item()
        self.metric.reset()
        self.val_writer.write(f"Epoch: {self.curr_epoch}\t Loss: {total_loss}\n")
        self.val_writer.flush()
        # wandb.log({"val_loss": total_loss})

        return total_loss

    def evaluate_batch(self, batch: Batch):
        batch.to(device(self.model))

        W_list = []
        for i in range(len(batch.Lap)):
            if self.basis:
                W = torch.eye(batch.Lap[i].shape[0]).to(self.device)
            else:
                W = torch.randn(batch.Lap[i].shape[0], self.num_samples).to(
                    self.device
                )  # BxNxM
            if len(W.shape) < 2:
                print("EVAL BATCH, i, LAP|W: ", i, batch.Lap[i].shape, W.shape)
            W_list.append(W)
        with torch.no_grad():
            y_pred = torch.nn.Sigmoid()(self.model(batch, W_list))
            self.metric.update(y_pred, batch.y)

    @property
    def n_total_steps(self) -> int:
        return len(self.train_loader) * self.cfg.n_epochs

    def create_mlp(self, in_dims: int, out_dims: int) -> MLP:
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
            self.cfg.mlp_activation,
            self.cfg.mlp_dropout_prob,
            norm_type="layer",
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
        return 1.0


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
