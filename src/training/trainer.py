"""Model factory, dataset loader, and training primitives."""

import copy
import importlib
import os
import time
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
    MulticlassRecall,
)
from tqdm import tqdm

from src.training.models.appnp import APPNPModel
from src.training.models.gat import GAT
from src.training.models.gcn import GCN
from src.training.models.gprgnn import GPRGNN
from src.training.models.h2gcn import H2GCN
from src.training.models.mixhop import MixHop
from src.training.models.sage import GraphSAGE
from src.utils.logging import console


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_MODEL_REGISTRY = {
    "gcn":    GCN,
    "gat":    GAT,
    "sage":   GraphSAGE,
    "appnp":  APPNPModel,
    "mixhop": MixHop,
    "h2gcn":  H2GCN,
    "gprgnn": GPRGNN,
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_data(cfg_dataset: DictConfig) -> tuple[Any, int]:
    """Dynamically load a PyG dataset from config. Returns (data, num_classes)."""
    module_path, class_name = cfg_dataset.class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)

    kwargs = dict(cfg_dataset.kwargs)

    if "root" in kwargs:
        try:
            from hydra.utils import get_original_cwd
            cwd = get_original_cwd()
        except Exception:
            cwd = os.getcwd()
        root = kwargs["root"]
        if not os.path.isabs(root):
            kwargs["root"] = os.path.join(cwd, root)
        elif not os.path.exists(root):
            kwargs["root"] = os.path.join(cwd, "data", "pyg")

    dataset = cls(**kwargs)
    data    = dataset[0]

    if data.y.dim() > 1 and data.y.size(1) == 1:
        data.y = data.y.squeeze(1)
    if data.y.dtype != torch.long:
        data.y = data.y.long()

    return data, int(dataset.num_classes)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: DictConfig, in_channels: int, out_channels: int) -> torch.nn.Module:
    """Instantiate a model from config."""
    name = cfg.model.name
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_MODEL_REGISTRY)}")

    kwargs: dict = {
        "in_channels":     in_channels,
        "hidden_channels": cfg.training.hidden_channels,
        "out_channels":    out_channels,
        "dropout":         cfg.training.dropout,
        "num_layers":      cfg.training.num_layers,
    }
    if name == "gat":
        kwargs["heads"] = cfg.model.heads
    elif name == "appnp":
        kwargs["K"]     = cfg.model.K
        kwargs["alpha"] = cfg.model.alpha
    elif name == "mixhop":
        kwargs["powers"] = list(cfg.model.powers)
    elif name == "h2gcn":
        kwargs["K"] = cfg.model.K
    elif name == "gprgnn":
        kwargs["K"]     = cfg.model.K
        kwargs["alpha"] = cfg.model.alpha
        kwargs["Init"]  = cfg.model.Init

    return _MODEL_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Training primitives
# ---------------------------------------------------------------------------

def train_one_epoch(model, data, train_mask, optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def _compute_val_loss(model, data, val_mask) -> float:
    model.eval()
    out = model(data.x, data.edge_index)
    return float(F.cross_entropy(out[val_mask], data.y[val_mask]))


@torch.no_grad()
def evaluate(model, data, mask) -> dict:
    """Return classification metrics on nodes selected by mask.

    Keys: accuracy, auroc, macro_f1, weighted_f1, balanced_accuracy, auprc.
    """
    model.eval()
    out    = model(data.x, data.edge_index)
    logits = out[mask]
    y_true = data.y[mask]
    probs  = F.softmax(logits, dim=-1)
    preds  = logits.argmax(dim=-1)
    C      = logits.size(-1)
    dev    = logits.device

    return {
        "accuracy":          float(MulticlassAccuracy(C).to(dev)(preds, y_true)),
        "auroc":             float(MulticlassAUROC(C, average="macro").to(dev)(probs, y_true)),
        "macro_f1":          float(MulticlassF1Score(C, average="macro").to(dev)(preds, y_true)),
        "weighted_f1":       float(MulticlassF1Score(C, average="weighted").to(dev)(preds, y_true)),
        "balanced_accuracy": float(MulticlassRecall(C, average="macro").to(dev)(preds, y_true)),
        "auprc":             float(MulticlassAveragePrecision(C, average="macro").to(dev)(probs, y_true)),
    }


def _train_loop(model, data, train_mask, val_mask, optimizer, max_epochs, patience):
    """Training loop with early stopping. Returns (best_state, best_epoch, elapsed_secs)."""
    if val_mask.sum() == 0:
        patience = max_epochs

    best_val  = float("inf")
    best_ep   = 0
    counter   = 0
    best_state = copy.deepcopy(model.state_dict())

    t0 = time.perf_counter()
    for epoch in tqdm(range(max_epochs), desc="Training", leave=False, unit="ep"):
        train_one_epoch(model, data, train_mask, optimizer)
        val_loss = _compute_val_loss(model, data, val_mask)

        if val_loss < best_val:
            best_val   = val_loss
            best_ep    = epoch + 1
            counter    = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= patience:
                console.print(
                    f"  [dim]Early stop at epoch {epoch + 1} "
                    f"(best {best_ep}, val loss {best_val:.4f})[/dim]"
                )
                break

    return best_state, best_ep, time.perf_counter() - t0
