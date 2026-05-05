"""Verify all 15 datasets download and load correctly.

Usage:
    python check_datasets.py
"""

import importlib
import os
import sys

import torch
from rich import box
from rich.table import Table

from src.utils.logging import console

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", "pyg")

DATASETS = [
    ("cora",             "torch_geometric.datasets.Planetoid",                        {"name": "Cora"}),
    ("citeseer",         "torch_geometric.datasets.Planetoid",                        {"name": "CiteSeer"}),
    ("pubmed",           "torch_geometric.datasets.Planetoid",                        {"name": "PubMed"}),
    ("amazon_computers", "torch_geometric.datasets.Amazon",                           {"name": "Computers"}),
    ("amazon_photo",     "torch_geometric.datasets.Amazon",                           {"name": "Photo"}),
    ("coauthor_cs",      "torch_geometric.datasets.Coauthor",                         {"name": "CS"}),
    ("coauthor_physics", "torch_geometric.datasets.Coauthor",                         {"name": "Physics"}),
    ("cornell",          "torch_geometric.datasets.WebKB",                            {"name": "Cornell"}),
    ("wisconsin",        "torch_geometric.datasets.WebKB",                            {"name": "Wisconsin"}),
    ("texas",            "torch_geometric.datasets.WebKB",                            {"name": "Texas"}),
    ("actor",            "torch_geometric.datasets.Actor",                            {}),
    ("roman_empire",     "torch_geometric.datasets.HeterophilousGraphDataset",        {"name": "Roman-empire"}),
    ("amazon_ratings",   "torch_geometric.datasets.HeterophilousGraphDataset",        {"name": "Amazon-ratings"}),
    ("chameleon",        "src.datasets.filtered_wikipedia.FilteredWikipediaNetwork",  {"name": "chameleon", "with_masks": False}),
    ("squirrel",         "src.datasets.filtered_wikipedia.FilteredWikipediaNetwork",  {"name": "squirrel",  "with_masks": False}),
]


def load_dataset(class_path, kwargs):
    module_path, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(root=DATA_ROOT, **kwargs)


def main():
    console.rule("[bold]Dataset download check[/bold]")
    console.print(f"Data root: [dim]{DATA_ROOT}[/dim]\n")

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Dataset",  min_width=18)
    table.add_column("Nodes",    justify="right", min_width=8)
    table.add_column("Edges",    justify="right", min_width=10)
    table.add_column("Features", justify="right", min_width=10)
    table.add_column("Classes",  justify="right", min_width=8)
    table.add_column("Status",   min_width=8)

    ok = failed = 0
    for name, class_path, kwargs in DATASETS:
        console.print(f"Loading [bold]{name}[/bold] ...", end=" ")
        try:
            dataset = load_dataset(class_path, kwargs)
            data    = dataset[0]
            if data.y.dim() > 1 and data.y.size(1) == 1:
                data.y = data.y.squeeze(1)
            nodes    = data.num_nodes
            edges    = data.num_edges
            features = data.num_node_features
            classes  = int(dataset.num_classes)
            table.add_row(name, f"{nodes:,}", f"{edges:,}", str(features), str(classes), "[green]OK[/green]")
            console.print("[green]OK[/green]")
            ok += 1
        except Exception as e:
            table.add_row(name, "-", "-", "-", "-", "[red]FAIL[/red]")
            console.print(f"[red]FAIL[/red] — {e}")
            failed += 1

    console.print()
    console.print(table)
    console.rule(f"[bold]{ok}[/bold] passed  ·  [bold]{failed}[/bold] failed")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
