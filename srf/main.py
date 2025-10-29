import torch
import torch.nn.functional as F
import torch_geometric  # type: ignore
import wandb
from ogb.graphproppred import Evaluator
import tqdm
from sklearn.metrics import average_precision_score

from models import GINEGlobalRandom  # type: ignore
from datasets import get_loaders

##############################################
# Constants / Config
##############################################

TARGET = 4  # Global target for QM9, e.g., property #4


##############################################
# Training/Evaluation Functions
##############################################
def train_regression(model, loader, optimizer, device, target_idx, ds_name) -> float:
    """One epoch of training for a regression task."""
    model.train()
    total_loss = 0.0

    for data in tqdm.tqdm(loader, ncols=50):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        loss = None
        if ds_name == "qm9":
            loss = F.mse_loss(out, data.y[:, target_idx].unsqueeze(1))
        elif ds_name == "peptides-struct":
            # Handle multi-target regression
            loss = F.mse_loss(out, data.y)
        elif ds_name == "zinc":
            loss = F.l1_loss(out.squeeze(), data.y)
        else:
            loss = F.mse_loss(out, data.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_regression(model, loader, device, target_idx, ds_name) -> float:
    """Evaluate MAE loss on validation or test loader for regression tasks."""
    model.eval()
    total_mae = 0.0

    for data in tqdm.tqdm(loader, ncols=50):
        data = data.to(device)
        out = model(data)

        mae = None
        if ds_name == "qm9":
            mae = F.l1_loss(out, data.y[:, target_idx].unsqueeze(1))
        elif ds_name == "peptides-struct":
            # Calculate MAE across all targets
            mae = F.l1_loss(out, data.y)
        elif ds_name == "zinc":
            mae = F.l1_loss(out.squeeze(), data.y)
        else:
            mae = F.l1_loss(out, data.y)

        total_mae += mae.item() * data.num_graphs

    return total_mae / len(loader.dataset)


def train_classification(
    model, loader, optimizer, device, ds_name
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    y_preds, y_trues = [], []

    # Use BCE for multi-label tasks
    criterion = torch.nn.BCEWithLogitsLoss()

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        if ds_name in ["ogbg-molhiv", "peptides-func"]:
            # Multi-label case
            loss = criterion(out, data.y.float())
            y_preds.append(out)
        else:
            # Multi-class case
            loss = torch.nn.CrossEntropyLoss()(out, data.y.squeeze())
            y_preds.append(torch.argmax(out, dim=-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        y_trues.append(data.y)

    avg_loss = total_loss / len(loader.dataset)

    # Calculate appropriate metric
    y_trues = torch.cat(y_trues, dim=0)
    y_preds = torch.cat(y_preds, dim=0)

    metric = None
    if ds_name == "ogbg-molhiv":
        evaluator = Evaluator(name=ds_name.lower())
        input_dict = {"y_true": y_trues, "y_pred": y_preds}
        metric = evaluator.eval(input_dict)["rocauc"]
    elif ds_name == "peptides-func":
        # Calculate mean AP for multi-label
        y_probs = torch.sigmoid(y_preds).detach().cpu().numpy()
        y_true = y_trues.cpu().numpy()
        metric = average_precision_score(y_true, y_probs, average="macro")
    else:
        accuracy = (y_preds == y_trues).float().mean().item()
        metric = accuracy

    return metric, avg_loss


@torch.no_grad()
def evaluate_classification(model, loader, device, ds_name) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    y_preds, y_trues = [], []

    # Use BCE for multi-label tasks
    criterion = torch.nn.BCEWithLogitsLoss()

    for data in loader:
        data = data.to(device)
        out = model(data)

        if ds_name in ["ogbg-molhiv", "peptides-func"]:
            loss = criterion(out, data.y.float())
            y_preds.append(out)
        else:
            loss = torch.nn.CrossEntropyLoss()(out, data.y.squeeze())
            y_preds.append(torch.argmax(out, dim=-1))

        total_loss += loss.item() * data.num_graphs
        y_trues.append(data.y)

    avg_loss = total_loss / len(loader.dataset)

    # Calculate appropriate metric
    y_trues = torch.cat(y_trues, dim=0)
    y_preds = torch.cat(y_preds, dim=0)

    metric = None
    if ds_name == "ogbg-molhiv":
        evaluator = Evaluator(name=ds_name.lower())
        input_dict = {"y_true": y_trues, "y_pred": y_preds}
        metric = evaluator.eval(input_dict)["rocauc"]
    elif ds_name == "peptides-func":
        # Calculate mean AP for multi-label
        y_probs = torch.sigmoid(y_preds).cpu().numpy()
        y_true = y_trues.cpu().numpy()
        metric = average_precision_score(y_true, y_probs, average="macro")
    else:
        accuracy = (y_preds == y_trues).float().mean().item()
        metric = accuracy

    return metric, avg_loss


##############################################
# Main script
##############################################
def main():
    # -----------------------------
    # Hyperparameters & Config
    # -----------------------------
    regression_ds = ["zinc", "qm9", "pcqm4mv2", "peptides-struct"]
    classification_ds = ["csl", "ogbg-ppa", "ogbg-molhiv", "peptides-func", "reddit-m"]

    config = {
        "dataset_name": "csl",  # valid dataset support: "csl", "qm9", "ogbg-ppa", "pcqm4mv2", "ogbg-molhiv", "zinc", "reddit-m"
        "target_idx": TARGET,
        "hidden_size": 250,
        "num_layers": 4,
        "dropout": 0.0,
        "use_batch_norm": False,
        "lr": 0.001,
        "epochs": 100,
        "batch_size": 128,
        "seed": 42,
        "wandb_mode": "disabled",  # set to "online" or "disabled"
        # Extra transform params
        "transform_args": {
            "K": 8,
            "width": 64,
            "R_global_dim": 1000,
        },
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    torch_geometric.seed_everything(config["seed"])

    # -----------------------------
    # Initialize Weights & Biases
    # -----------------------------
    wandb.init(  # type: ignore
        project="jl-gnn",
        config=config,
        mode=config["wandb_mode"],
    )

    # -----------------------------
    # Prepare Transform and Dataset, Dataloaders
    # -----------------------------
    train_loader, val_loader, test_loader, input_size, edge_dim, num_classes = (
        get_loaders(
            dataset_name=config["dataset_name"],
            batch_size=config["batch_size"],
            transform_args=config["transform_args"],
            seed=config["seed"],
        )
    )

    # -----------------------------
    # Create Five Models & Optimizers
    # -----------------------------
    if config["dataset_name"].lower() == "zinc":
        dataset_type = "zinc"
    elif "ogb" in config["dataset_name"].lower():
        dataset_type = "ogb"
    elif "peptides" in config["dataset_name"].lower():
        dataset_type = "peptides"
    else:
        dataset_type = "other"
    # 1) Baseline: no extra feats
    model_baseline = GINEGlobalRandom(
        input_size=input_size,
        hidden_size=config["hidden_size"] + 16,
        num_layers=config["num_layers"],
        num_classes=num_classes,
        k=0,
        dropout=config["dropout"],
        method="baseline",
        use_batch_norm=config["use_batch_norm"],
        edge_dim=edge_dim,
        dataset_type=dataset_type,
    ).to(device)

    # 2) Random: uses random feats
    model_random = GINEGlobalRandom(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=num_classes,
        k=config["transform_args"]["width"],
        dropout=config["dropout"],
        method="random",
        use_batch_norm=config["use_batch_norm"],
        edge_dim=edge_dim,
        dataset_type=dataset_type,
    ).to(device)

    # 3) Linear: uses linear feats
    model_linear = GINEGlobalRandom(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=num_classes,
        k=config["transform_args"]["width"],
        dropout=config["dropout"],
        method="linear",
        use_batch_norm=config["use_batch_norm"],
        edge_dim=edge_dim,
        dataset_type=dataset_type,
    ).to(device)

    # 4) RBF: uses RBF feats
    model_rbf = GINEGlobalRandom(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=num_classes,
        k=config["transform_args"]["width"],
        dropout=config["dropout"],
        method="rbf",
        use_batch_norm=config["use_batch_norm"],
        edge_dim=edge_dim,
        dataset_type=dataset_type,
    ).to(device)

    # 5) Laplacian: uses Laplacian feats
    model_laplacian = GINEGlobalRandom(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=num_classes,
        k=config["transform_args"]["width"],
        dropout=config["dropout"],
        method="laplace",
        use_batch_norm=config["use_batch_norm"],
        edge_dim=edge_dim,
        dataset_type=dataset_type,
    ).to(device)

    # 6) Ablation: uses laplacian feats without random features
    model_ablation = GINEGlobalRandom(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=num_classes,
        k=config["transform_args"]["width"],
        dropout=config["dropout"],
        method="ablation",
        use_batch_norm=config["use_batch_norm"],
        edge_dim=edge_dim,
        dataset_type=dataset_type,
    ).to(device)

    # Create optimizers for each
    optimizer_baseline = torch.optim.Adam(model_baseline.parameters(), lr=config["lr"])  # type: ignore
    optimizer_random = torch.optim.Adam(model_random.parameters(), lr=config["lr"])  # type: ignore
    optimizer_linear = torch.optim.Adam(model_linear.parameters(), lr=config["lr"])  # type: ignore
    optimizer_rbf = torch.optim.Adam(model_rbf.parameters(), lr=config["lr"])  # type: ignore
    optimizer_laplacian = torch.optim.Adam(  # type: ignore
        model_laplacian.parameters(), lr=config["lr"]
    )  # type: ignore
    optimizer_ablation = torch.optim.Adam(model_ablation.parameters(), lr=config["lr"])  # type: ignore

    # Create schedulers for each model
    scheduler_baseline = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_baseline, mode="min", factor=0.5, patience=20
    )
    scheduler_random = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_random, mode="min", factor=0.5, patience=20
    )
    scheduler_linear = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_linear, mode="min", factor=0.5, patience=20
    )
    scheduler_rbf = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_rbf, mode="min", factor=0.5, patience=20
    )
    scheduler_laplacian = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_laplacian, mode="min", factor=0.5, patience=20
    )
    scheduler_ablation = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ablation, mode="min", factor=0.5, patience=20
    )

    print(
        f"[Baseline]   #Params: {sum(p.numel() for p in model_baseline.parameters() if p.requires_grad)}"
    )
    print(
        f"[Random]     #Params: {sum(p.numel() for p in model_random.parameters()   if p.requires_grad)}"
    )
    print(
        f"[Linear]     #Params: {sum(p.numel() for p in model_linear.parameters()   if p.requires_grad)}"
    )
    print(
        f"[RBF]        #Params: {sum(p.numel() for p in model_rbf.parameters()      if p.requires_grad)}"
    )
    print(
        f"[Laplacian]  #Params: {sum(p.numel() for p in model_laplacian.parameters() if p.requires_grad)}"
    )
    # Add to param printing
    print(
        f"[Ablation]   #Params: {sum(p.numel() for p in model_ablation.parameters() if p.requires_grad)}"
    )

    # Track best states - for regression, we track minimal MSE; for classification, we track best accuracy
    if config["dataset_name"] in regression_ds:
        best_val_score_baseline = float("inf")
        best_val_score_random = float("inf")
        best_val_score_linear = float("inf")
        best_val_score_rbf = float("inf")
        best_val_score_laplacian = float("inf")
        best_val_score_ablation = float("inf")
    else:
        best_val_score_baseline = 0.0
        best_val_score_random = 0.0
        best_val_score_linear = 0.0
        best_val_score_rbf = 0.0
        best_val_score_laplacian = 0.0
        best_val_score_ablation = 0.0

    best_state_baseline = None
    best_state_random = None
    best_state_linear = None
    best_state_rbf = None
    best_state_laplacian = None
    best_state_ablation = None

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(1, config["epochs"] + 1):
        if config["dataset_name"] in regression_ds:
            # ------------------ Regression ------------------
            # Train
            train_loss_b = train_regression(
                model_baseline,
                train_loader,
                optimizer_baseline,
                device,
                config["target_idx"],
                config["dataset_name"],
            )
            train_loss_r = train_regression(
                model_random,
                train_loader,
                optimizer_random,
                device,
                config["target_idx"],
                config["dataset_name"],
            )
            train_loss_l = train_regression(
                model_linear,
                train_loader,
                optimizer_linear,
                device,
                config["target_idx"],
                config["dataset_name"],
            )
            train_loss_rf = train_regression(
                model_rbf,
                train_loader,
                optimizer_rbf,
                device,
                config["target_idx"],
                config["dataset_name"],
            )
            train_loss_lp = train_regression(
                model_laplacian,
                train_loader,
                optimizer_laplacian,
                device,
                config["target_idx"],
                config["dataset_name"],
            )
            train_loss_ab = train_regression(
                model_ablation,
                train_loader,
                optimizer_ablation,
                device,
                config["target_idx"],
                config["dataset_name"],
            )

            # Validate
            val_loss_b = evaluate_regression(
                model_baseline,
                val_loader,
                device,
                config["target_idx"],
                config["dataset_name"],
            )
            val_loss_r = evaluate_regression(
                model_random,
                val_loader,
                device,
                config["target_idx"],
                config["dataset_name"],
            )
            val_loss_l = evaluate_regression(
                model_linear,
                val_loader,
                device,
                config["target_idx"],
                config["dataset_name"],
            )
            val_loss_rf = evaluate_regression(
                model_rbf,
                val_loader,
                device,
                config["target_idx"],
                config["dataset_name"],
            )
            val_loss_lp = evaluate_regression(
                model_laplacian,
                val_loader,
                device,
                config["target_idx"],
                config["dataset_name"],
            )
            val_loss_ab = evaluate_regression(
                model_ablation,
                val_loader,
                device,
                config["target_idx"],
                config["dataset_name"],
            )

            # Step schedulers
            scheduler_baseline.step(val_loss_b)
            scheduler_random.step(val_loss_r)
            scheduler_linear.step(val_loss_l)
            scheduler_rbf.step(val_loss_rf)
            scheduler_laplacian.step(val_loss_lp)
            scheduler_ablation.step(val_loss_ab)

            # Check for best
            if val_loss_b < best_val_score_baseline:
                best_val_score_baseline = val_loss_b
                best_state_baseline = model_baseline.state_dict()
            if val_loss_r < best_val_score_random:
                best_val_score_random = val_loss_r
                best_state_random = model_random.state_dict()
            if val_loss_l < best_val_score_linear:
                best_val_score_linear = val_loss_l
                best_state_linear = model_linear.state_dict()
            if val_loss_rf < best_val_score_rbf:
                best_val_score_rbf = val_loss_rf
                best_state_rbf = model_rbf.state_dict()
            if val_loss_lp < best_val_score_laplacian:
                best_val_score_laplacian = val_loss_lp
                best_state_laplacian = model_laplacian.state_dict()
            if val_loss_ab < best_val_score_ablation:
                best_val_score_ablation = val_loss_ab
                best_state_ablation = model_ablation.state_dict()

            # Print & Log
            print(f"Epoch {epoch:03d} [Regression]")
            print(f"  Baseline   Train/Val MSE: {train_loss_b:.4f} / {val_loss_b:.4f}")
            print(f"  Random     Train/Val MSE: {train_loss_r:.4f} / {val_loss_r:.4f}")
            print(f"  Linear     Train/Val MSE: {train_loss_l:.4f} / {val_loss_l:.4f}")
            print(
                f"  RBF        Train/Val MSE: {train_loss_rf:.4f} / {val_loss_rf:.4f}"
            )
            print(
                f"  Laplacian  Train/Val MSE: {train_loss_lp:.4f} / {val_loss_lp:.4f}"
            )
            print(f"  Ablation  Train/Val MSE: {train_loss_ab:.4f} / {val_loss_ab:.4f}")

            wandb.log(  # type: ignore
                {
                    "epoch": epoch,
                    "train_loss_baseline": train_loss_b,
                    "val_loss_baseline": val_loss_b,
                    "train_loss_random": train_loss_r,
                    "val_loss_random": val_loss_r,
                    "train_loss_linear": train_loss_l,
                    "val_loss_linear": val_loss_l,
                    "train_loss_rbf": train_loss_rf,
                    "val_loss_rbf": val_loss_rf,
                    "train_loss_laplacian": train_loss_lp,
                    "val_loss_laplacian": val_loss_lp,
                    "train_loss_ablation": train_loss_ab,
                    "val_loss_ablation": val_loss_ab,
                },
                step=epoch,
            )

        else:
            # ------------------ Classification ------------------
            # Train
            train_acc_b, train_loss_b = train_classification(
                model_baseline,
                train_loader,
                optimizer_baseline,
                device,
                config["dataset_name"],
            )
            train_acc_r, train_loss_r = train_classification(
                model_random,
                train_loader,
                optimizer_random,
                device,
                config["dataset_name"],
            )
            train_acc_l, train_loss_l = train_classification(
                model_linear,
                train_loader,
                optimizer_linear,
                device,
                config["dataset_name"],
            )
            train_acc_rf, train_loss_rf = train_classification(
                model_rbf, train_loader, optimizer_rbf, device, config["dataset_name"]
            )
            train_acc_lp, train_loss_lp = train_classification(
                model_laplacian,
                train_loader,
                optimizer_laplacian,
                device,
                config["dataset_name"],
            )

            train_acc_ab, train_loss_ab = train_classification(
                model_ablation,
                train_loader,
                optimizer_ablation,
                device,
                config["dataset_name"],
            )

            # Validate
            val_acc_b, val_loss_b = evaluate_classification(
                model_baseline, val_loader, device, config["dataset_name"]
            )
            val_acc_r, val_loss_r = evaluate_classification(
                model_random, val_loader, device, config["dataset_name"]
            )
            val_acc_l, val_loss_l = evaluate_classification(
                model_linear, val_loader, device, config["dataset_name"]
            )
            val_acc_rf, val_loss_rf = evaluate_classification(
                model_rbf, val_loader, device, config["dataset_name"]
            )
            val_acc_lp, val_loss_lp = evaluate_classification(
                model_laplacian, val_loader, device, config["dataset_name"]
            )
            val_acc_ab, val_loss_ab = evaluate_classification(
                model_ablation, val_loader, device, config["dataset_name"]
            )

            # Check for best (tracking highest val_acc)
            if val_acc_b > best_val_score_baseline:
                best_val_score_baseline = val_acc_b
                best_state_baseline = model_baseline.state_dict()
            if val_acc_r > best_val_score_random:
                best_val_score_random = val_acc_r
                best_state_random = model_random.state_dict()
            if val_acc_l > best_val_score_linear:
                best_val_score_linear = val_acc_l
                best_state_linear = model_linear.state_dict()
            if val_acc_rf > best_val_score_rbf:
                best_val_score_rbf = val_acc_rf
                best_state_rbf = model_rbf.state_dict()
            if val_acc_lp > best_val_score_laplacian:
                best_val_score_laplacian = val_acc_lp
                best_state_laplacian = model_laplacian.state_dict()
            if val_acc_ab > best_val_score_ablation:
                best_val_score_ablation = val_acc_ab
                best_state_ablation = model_ablation.state_dict()

            # Print & Log
            print(f"Epoch {epoch:03d} [Classification]")
            print(
                f"  Baseline   TrainAcc/ValAcc: {train_acc_b:.4f} / {val_acc_b:.4f}, TrainLoss: {train_loss_b:.4f}"
            )
            print(
                f"  Random     TrainAcc/ValAcc: {train_acc_r:.4f} / {val_acc_r:.4f}, TrainLoss: {train_loss_r:.4f}"
            )
            print(
                f"  Linear     TrainAcc/ValAcc: {train_acc_l:.4f} / {val_acc_l:.4f}, TrainLoss: {train_loss_l:.4f}"
            )
            print(
                f"  RBF        TrainAcc/ValAcc: {train_acc_rf:.4f} / {val_acc_rf:.4f}, TrainLoss: {train_loss_rf:.4f}"
            )
            print(
                f"  Laplacian  TrainAcc/ValAcc: {train_acc_lp:.4f} / {val_acc_lp:.4f}, TrainLoss: {train_loss_lp:.4f}"
            )
            print(
                f"  Ablation  TrainAcc/ValAcc: {train_acc_ab:.4f} / {val_acc_ab:.4f}, TrainLoss: {train_loss_ab:.4f}"
            )

            wandb.log(  # type: ignore
                {
                    "epoch": epoch,
                    # Baseline logs
                    "train_acc_baseline": train_acc_b,
                    "val_acc_baseline": val_acc_b,
                    "train_loss_baseline": train_loss_b,
                    "val_loss_baseline": val_loss_b,
                    # Random logs
                    "train_acc_random": train_acc_r,
                    "val_acc_random": val_acc_r,
                    "train_loss_random": train_loss_r,
                    "val_loss_random": val_loss_r,
                    # Linear logs
                    "train_acc_linear": train_acc_l,
                    "val_acc_linear": val_acc_l,
                    "train_loss_linear": train_loss_l,
                    "val_loss_linear": val_loss_l,
                    # RBF logs
                    "train_acc_rbf": train_acc_rf,
                    "val_acc_rbf": val_acc_rf,
                    "train_loss_rbf": train_loss_rf,
                    "val_loss_rbf": val_loss_rf,
                    # Laplacian logs
                    "train_acc_laplacian": train_acc_lp,
                    "val_acc_laplacian": val_acc_lp,
                    "train_loss_laplacian": train_loss_lp,
                    "val_loss_laplacian": val_loss_lp,
                    # Ablation logs
                    "train_acc_ablation": train_acc_ab,
                    "val_acc_ablation": val_acc_ab,
                    "train_loss_ablation": train_loss_ab,
                    "val_loss_ablation": val_loss_ab,
                },
                step=epoch,
            )

    # -----------------------------
    # Final Test (Load Best States)
    # -----------------------------
    if best_state_baseline is not None:
        model_baseline.load_state_dict(best_state_baseline)
    if best_state_random is not None:
        model_random.load_state_dict(best_state_random)
    if best_state_linear is not None:
        model_linear.load_state_dict(best_state_linear)
    if best_state_rbf is not None:
        model_rbf.load_state_dict(best_state_rbf)
    if best_state_laplacian is not None:
        model_laplacian.load_state_dict(best_state_laplacian)
    if best_state_ablation is not None:
        model_ablation.load_state_dict(best_state_ablation)

    if config["dataset_name"] in regression_ds:
        # Regression Test
        test_loss_b = evaluate_regression(
            model_baseline,
            test_loader,
            device,
            config["target_idx"],
            config["dataset_name"],
        )
        test_loss_r = evaluate_regression(
            model_random,
            test_loader,
            device,
            config["target_idx"],
            config["dataset_name"],
        )
        test_loss_l = evaluate_regression(
            model_linear,
            test_loader,
            device,
            config["target_idx"],
            config["dataset_name"],
        )
        test_loss_rf = evaluate_regression(
            model_rbf, test_loader, device, config["target_idx"], config["dataset_name"]
        )
        test_loss_lp = evaluate_regression(
            model_laplacian,
            test_loader,
            device,
            config["target_idx"],
            config["dataset_name"],
        )
        test_loss_ab = evaluate_regression(
            model_ablation,
            test_loader,
            device,
            config["target_idx"],
            config["dataset_name"],
        )

        print("[Final Test] Regression MSE:")
        print(f"  Baseline:   {test_loss_b:.4f}")
        print(f"  Random:     {test_loss_r:.4f}")
        print(f"  Linear:     {test_loss_l:.4f}")
        print(f"  RBF:        {test_loss_rf:.4f}")
        print(f"  Laplacian:  {test_loss_lp:.4f}")

        wandb.summary["test_loss_baseline"] = test_loss_b  # type: ignore
        wandb.summary["test_loss_random"] = test_loss_r  # type: ignore
        wandb.summary["test_loss_linear"] = test_loss_l  # type: ignore
        wandb.summary["test_loss_rbf"] = test_loss_rf  # type: ignore
        wandb.summary["test_loss_laplacian"] = test_loss_lp  # type: ignore
        wandb.summary["test_loss_ablation"] = test_loss_ab  # type: ignore

    else:
        # Classification Test
        test_acc_b, test_loss_b = evaluate_classification(
            model_baseline, test_loader, device, ds_name=config["dataset_name"]
        )
        test_acc_r, test_loss_r = evaluate_classification(
            model_random, test_loader, device, ds_name=config["dataset_name"]
        )
        test_acc_l, test_loss_l = evaluate_classification(
            model_linear, test_loader, device, ds_name=config["dataset_name"]
        )
        test_acc_rf, test_loss_rf = evaluate_classification(
            model_rbf, test_loader, device, ds_name=config["dataset_name"]
        )
        test_acc_lp, test_loss_lp = evaluate_classification(
            model_laplacian, test_loader, device, ds_name=config["dataset_name"]
        )
        test_acc_ab, test_loss_ab = evaluate_classification(
            model_ablation, test_loader, device, ds_name=config["dataset_name"]
        )

        print("[Final Test] Classification Accuracy / Loss:")
        print(f"  Baseline   Acc: {test_acc_b:.4f},  Loss: {test_loss_b:.4f}")
        print(f"  Random     Acc: {test_acc_r:.4f},  Loss: {test_loss_r:.4f}")
        print(f"  Linear     Acc: {test_acc_l:.4f},  Loss: {test_loss_l:.4f}")
        print(f"  RBF        Acc: {test_acc_rf:.4f}, Loss: {test_loss_rf:.4f}")
        print(f"  Laplacian  Acc: {test_acc_lp:.4f}, Loss: {test_loss_lp:.4f}")
        print(f"  Ablation   Acc: {test_acc_ab:.4f}, Loss: {test_loss_ab:.4f}")

        wandb.summary["test_acc_baseline"] = test_acc_b  # type: ignore
        wandb.summary["test_loss_baseline"] = test_loss_b  # type: ignore
        wandb.summary["test_acc_random"] = test_acc_r  # type: ignore
        wandb.summary["test_loss_random"] = test_loss_r  # type: ignore
        wandb.summary["test_acc_linear"] = test_acc_l  # type: ignore
        wandb.summary["test_loss_linear"] = test_loss_l  # type: ignore
        wandb.summary["test_acc_rbf"] = test_acc_rf  # type: ignore
        wandb.summary["test_loss_rbf"] = test_loss_rf  # type: ignore
        wandb.summary["test_acc_laplacian"] = test_acc_lp  # type: ignore
        wandb.summary["test_loss_laplacian"] = test_loss_lp  # type: ignore
        wandb.summary["test_acc_ablation"] = test_acc_ab  # type: ignore
        wandb.summary["test_loss_ablation"] = test_loss_ab  # type: ignore

    wandb.finish()  # type: ignore


if __name__ == "__main__":
    main()
