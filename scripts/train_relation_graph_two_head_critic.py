from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.fs_utils import write_text_file
from idea_graph.relation_graph_critic_data import HashTextEmbeddingBackend, SentenceTransformerEmbeddingBackend
from idea_graph.relation_graph_two_head_data import build_relation_graph_two_head_dataset
from idea_graph.relation_graph_two_head_train import train_relation_graph_two_head_critic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the parallel two-head relation graph critic.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--g1-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--text-backend",
        choices=["sentence-transformer", "hash"],
        default="sentence-transformer",
    )
    parser.add_argument(
        "--text-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        json.dumps(
            {
                "event": "dataset_build_start",
                "dataset_dir": str(args.dataset_dir.resolve()),
                "g1_dataset_dir": str(args.g1_dataset_dir.resolve()),
                "text_backend": args.text_backend,
                "text_model_name": None if args.text_backend == "hash" else args.text_model_name,
            },
            indent=2,
        ),
        flush=True,
    )
    backend = (
        HashTextEmbeddingBackend(dim=args.embedding_dim)
        if args.text_backend == "hash"
        else SentenceTransformerEmbeddingBackend(args.text_model_name)
    )
    dataset = build_relation_graph_two_head_dataset(
        dataset_dir=args.dataset_dir,
        g1_dataset_dir=args.g1_dataset_dir,
        text_backend=backend,
    )
    print(
        json.dumps(
            {
                "event": "training_start",
                "dataset_dir": str(args.dataset_dir.resolve()),
                "g1_dataset_dir": str(args.g1_dataset_dir.resolve()),
                "text_backend": args.text_backend,
                "text_model_name": None if args.text_backend == "hash" else args.text_model_name,
                "edit_train_example_count": len(dataset.edit_train_examples),
                "edit_validation_example_count": len(dataset.edit_dev_examples),
                "commit_train_example_count": len(dataset.commit_train_examples),
                "commit_validation_example_count": len(dataset.commit_dev_examples),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "learning_rate": args.lr,
            },
            indent=2,
        ),
        flush=True,
    )
    artifacts = train_relation_graph_two_head_critic(
        dataset=dataset,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        text_backend_name=args.text_backend,
        text_model_name=None if args.text_backend == "hash" else args.text_model_name,
    )
    training_config = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "g1_dataset_dir": str(args.g1_dataset_dir.resolve()),
        "text_backend": args.text_backend,
        "text_model_name": None if args.text_backend == "hash" else args.text_model_name,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_text_file(args.output_dir / "training_config.json", json.dumps(training_config, indent=2))
    print(
        json.dumps(
            {
                "edit": artifacts.edit_metrics,
                "commit": artifacts.commit_metrics,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
