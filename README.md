# Not All Nodes Are Created Equal: Homophily-Aware Stratification for Stable GNN Evaluation

Accepted at the **Learning on Graphs (LoG) Conference 2026**.

Pre-print: [arXiv:2609.19210](https://arxiv.org/abs/2609.19210). 

The camera-ready version will be made available here once published.

This repository provides the training code and split implementations accompanying the paper. It supports **transductive node classification** on graphs using three k-fold split strategies, including a novel homophily-aware variant called HpStrat.

---

## Overview

The framework trains GNN models on benchmark graph datasets under three split strategies:

- `random` — plain k-fold, folds drawn uniformly at random
- `class_wise` — stratified k-fold, each fold preserves the global class distribution
- `hpstrat` — homophily-aware k-fold, stratified first by per-node homophily bin then by class label

Node homophily h(v) is the fraction of a node's neighbours that share its class label.

---

## Requirements

- Python **3.12**
- [`uv`](https://github.com/astral-sh/uv) for environment management

PyTorch is installed as **CPU** on macOS/Windows and **CUDA 11.8** on Linux. On Apple Silicon, MPS is used when available.

---

## Setup

```bash
uv sync
source .venv/bin/activate
```

---

## Sanity check

To verify all datasets download and load correctly, run:

```bash
python check_datasets.py
```

This will attempt to load all 15 datasets (downloading any that are missing) and print a summary table with node/edge/feature/class counts. Datasets that fail to load are reported individually.

---

## Usage

```bash
python train.py dataset=<dataset> model=<model> split.strategy=<strategy> split.k=<k>
```

**Examples:**

```bash
# Class-stratified (default)
python train.py dataset=cora model=gcn

# Homophily-aware
python train.py dataset=texas model=gat split.strategy=hpstrat

# Random
python train.py dataset=amazon_computers model=sage split.strategy=random
```

---

## Datasets

15 benchmark datasets covering homophilic and heterophilic graphs: Cora, CiteSeer, PubMed, Amazon Computers, Amazon Photo, Chameleon (filtered), Squirrel (filtered), Cornell, Wisconsin, Texas, Actor, Roman-empire, Amazon-ratings, Coauthor CS, Coauthor Physics.

Chameleon and Squirrel use the corrected versions from Platonov et al. 2023, which remove duplicated nodes present in the original splits.

Dataset configs are in `conf/dataset/`. Pass `dataset=<name>` where `<name>` matches the config filename (e.g. `dataset=coauthor_cs`).

---

## Models

GCN, GAT, GraphSAGE, APPNP, MixHop, H2GCN, GPRGNN.

Model configs are in `conf/model/`. Pass `model=<name>` (e.g. `model=h2gcn`).

---

## Configuration

All settings are managed via [Hydra](https://hydra.cc). The top-level config is `conf/config.yaml`.

Key parameters:

```yaml
split:
  strategy: class_wise   # class_wise | random | hpstrat
  k: 4                   # number of folds
  seed: 42
```

Hyperparameters (layers, hidden dim, dropout, lr, etc.) are in `conf/training/hypA.yaml`.

Any config key can be overridden on the command line:

```bash
python train.py training.max_epochs=500 training.lr=0.005
```

---

## Repository structure

```
.
├── conf/
│   ├── config.yaml               # top-level defaults
│   ├── dataset/                  # one yaml per dataset
│   ├── model/                    # one yaml per model
│   └── training/
│       └── hypA.yaml             # hyperparameter preset
├── src/
│   ├── datasets/
│   │   ├── transductive_registry.py  # 15 supported datasets
│   │   └── filtered_wikipedia.py     # loader for Platonov et al. 2023 graphs
│   ├── training/
│   │   ├── models/               # GCN, GAT, GraphSAGE, APPNP, MixHop, H2GCN, GPRGNN
│   │   ├── split.py              # get_all_folds — all split strategies
│   │   ├── hp_split.py           # HomophilyKFold — hpstrat splitter
│   │   └── trainer.py            # model factory, training loop, evaluate
│   └── utils/
│       └── logging.py            # shared Rich console
├── train.py                      # training entry point (Hydra)
├── data/                         # dataset downloads (gitignored)
└── pyproject.toml
```