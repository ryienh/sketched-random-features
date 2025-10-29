import os
import torch
from core.config import cfg, update_cfg
from core.train import run
from core.pe import PE_GNN, Sketch_GNN
from core.transform import EVDTransform
from torch.utils.data import random_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import argparse

from torch_geometric.datasets import TUDataset
from sketch import AddFeaturesTransform

def stratified_split(dataset, seed, fold_idx, n_splits=10):
    assert 0 <= fold_idx < n_splits, "fold_idx must be from 0 to n_splits-1."
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = [data.y.item() for data in dataset]
    
    idx_list = []
    for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append((train_idx, test_idx))
    
    train_idx, test_idx = idx_list[fold_idx]
    
    # Create train and test datasets based on the indices
    train_dataset = dataset[torch.tensor(train_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]
    
    return train_dataset, test_dataset

def create_dataset(cfg, fold_idx): 
    torch.set_num_threads(cfg.num_workers)
    transform = transform_eval = None 
    # OURS
    pre_transform = AddFeaturesTransform(
        D_out=cfg.sketch_width,
        k=cfg.sketch_k,
        # R_global=R_global,
    )
    root = 'data/' + cfg.dataset
    dataset = TUDataset(root, name=cfg.dataset, transform=transform, pre_transform=pre_transform)
    labels = [data.y.item() for data in dataset]
    seed = 42  # Set seed
    train_dataset, test_dataset = stratified_split(dataset, seed, fold_idx)
    val_dataset = None
    return train_dataset, val_dataset, test_dataset

def create_model(cfg):

    model = None

    if cfg.model.method == "pearl":
        print("Using Pearl")
        model = PE_GNN(None, None,
                            n_hid_3d=cfg.model.hidden_size_3d, 
                            n_hid = cfg.model.hidden_size,
                            n_out=cfg.model.n_out, 
                            nl_pe_gnn=cfg.model.num_layers_pe, 
                            nl_gnn=cfg.model.num_layers)

    elif cfg.model.method == "sketch":
        print("Using SRF")
        model = Sketch_GNN(None, None,
                            n_hid_3d=cfg.model.hidden_size_3d, 
                            n_hid = cfg.model.hidden_size,
                            n_out=cfg.model.n_out, 
                            nl_pe_gnn=cfg.model.num_layers_pe, 
                            nl_gnn=cfg.model.num_layers,
                            sketch_type=cfg.model.sketch_type,
                            sketch_width=cfg.sketch_width)
    return model

def train_reddit(train_loader, model, optimizer, device, samples=30, splits=1):
    total_loss = 0
    N = 0 
    if model.n_out == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    for data in train_loader:
        r = 1+torch.randn(data.num_nodes, samples // splits, 1).to(device) #NxMx1
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        data.x = torch.ones(data.num_nodes,1).long().to(device)
        optimizer.zero_grad()
        out = model(data, r).squeeze()
        if model.n_out == 1:
            loss = criterion(out.float(), y.float()) # float if binary, long if multi  
        else:
            loss = criterion(out.float(), y.long())
        loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
        N += num_graphs
    return total_loss / N

@torch.no_grad()
def test_reddit(loader, model, evaluator, device, test_samples=200):
    N = 0
    total = 0
    correct = 0
    for data in loader:
        r = 1+torch.randn(data.num_nodes, test_samples, 1).to(device)
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        data.x = torch.ones(data.num_nodes,1).long().to(device)
        if model.n_out == 1:
            outputs = torch.sigmoid(model(data, r)).squeeze()
            predicted = (outputs > 0.5).squeeze()  
            correct += (predicted == data.y).sum().item()
            total += data.y.size(0)
        else:
            outputs = model(data, r)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == data.y).sum().item() 
            total += y.size(0)
    return correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="give_config")
    parser.add_argument('--config', type=str, default='zinc.yaml', help="Path to the config file")
    parser.add_argument('--gpu_id', type=str, default='0', help="GPU id")
    args = parser.parse_args()
    print('Arguments: ', args)
    cfg.merge_from_file('train/config/'+args.config)
    cfg = update_cfg(cfg)
    cfg.device = 'cuda:' + str(args.gpu_id)
    run(cfg, create_dataset, create_model, train_reddit, test_reddit, samples=cfg.num_samples, test_samples=cfg.test_samples, runs=cfg.runs)