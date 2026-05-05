"""Train a GNN across all k folds and print results.

Usage:
    python train.py dataset=cora model=gcn
    python train.py dataset=texas model=gat split.strategy=hpstrat
    python train.py dataset=amazon_computers model=sage split.strategy=random split.k=10
"""

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from rich import box
from rich.table import Table

from src.training.split import get_all_folds
from src.training.trainer import load_data, build_model, _train_loop, evaluate, get_device
from src.utils.logging import console

_METRIC_KEYS   = ["accuracy"]
_METRIC_LABELS = ["Accuracy"]


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.split.seed)
    np.random.seed(cfg.split.seed)

    strategy = cfg.split.strategy
    k        = cfg.split.k
    seed     = cfg.split.seed

    # ── Load dataset ──────────────────────────────────────────────────────
    data, num_classes = load_data(cfg.dataset)

    console.rule(
        f"[bold]{cfg.dataset.name}[/bold]  ·  "
        f"[bold]{cfg.model.name}[/bold]  ·  "
        f"{strategy}  ·  k={k}"
    )
    console.print(
        f"Nodes: {data.num_nodes}   "
        f"Features: {data.num_node_features}   "
        f"Classes: {num_classes}"
    )

    # ── Compute all k splits at once ──────────────────────────────────────
    all_folds = get_all_folds(data, data.y, strategy=strategy, k=k, seed=seed)

    device = get_device()
    data   = data.to(device)

    # ── Train each fold ───────────────────────────────────────────────────
    all_metrics: list[dict] = []

    for fold, (train_mask, val_mask, test_mask) in enumerate(all_folds):
        console.rule(f"Fold {fold + 1} / {k}", style="dim")
        console.print(
            f"  Train: {int(train_mask.sum())}   "
            f"Val: {int(val_mask.sum())}   "
            f"Test: {int(test_mask.sum())}"
        )

        model     = build_model(cfg, data.num_node_features, num_classes).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        best_state, best_epoch, elapsed = _train_loop(
            model, data,
            train_mask.to(device), val_mask.to(device),
            optimizer,
            max_epochs=cfg.training.max_epochs,
            patience=cfg.training.patience,
        )
        model.load_state_dict(best_state)

        metrics = evaluate(model, data, test_mask.to(device))
        all_metrics.append(metrics)

        console.print(
            f"  Best epoch: [yellow]{best_epoch}[/yellow]   Time: {elapsed:.1f}s"
        )
        console.print(
            f"  Acc: [bold green]{metrics['accuracy']:.4f}[/bold green]"
        )

    # ── Summary table ─────────────────────────────────────────────────────
    _print_summary(all_metrics, cfg.dataset.name, cfg.model.name, strategy, k)


def _print_summary(all_metrics, dataset, model_name, strategy, k):
    means = {key: np.mean([m[key] for m in all_metrics]) for key in _METRIC_KEYS}
    stds  = {key: np.std( [m[key] for m in all_metrics]) for key in _METRIC_KEYS}

    console.rule(
        f"[bold]Summary — {dataset} / {model_name} / {strategy} (k={k})[/bold]"
    )

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("", style="dim", min_width=8)
    for label in _METRIC_LABELS:
        table.add_column(label, justify="right", min_width=9)

    for i, m in enumerate(all_metrics):
        table.add_row(f"Fold {i + 1}", *[f"{m[key]:.4f}" for key in _METRIC_KEYS])

    table.add_section()
    table.add_row(
        "[bold]Mean[/bold]",
        *[f"[bold]{means[key]:.4f}[/bold]" for key in _METRIC_KEYS],
    )
    table.add_row(
        "Std",
        *[f"{stds[key]:.4f}" for key in _METRIC_KEYS],
    )

    console.print(table)


if __name__ == "__main__":
    main()
