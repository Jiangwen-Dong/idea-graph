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
from idea_graph.relation_graph_critic_data import (
    HashTextEmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
    build_relation_graph_dataset,
)
from idea_graph.relation_graph_critic_train import train_relation_graph_critic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the offline relation-aware graph critic.")
    parser.add_argument("--candidate-dataset-dir", type=Path, required=True)
    parser.add_argument("--g1-dataset-dir", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
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
    backend = (
        HashTextEmbeddingBackend(dim=args.embedding_dim)
        if args.text_backend == "hash"
        else SentenceTransformerEmbeddingBackend(args.text_model_name)
    )
    dataset = build_relation_graph_dataset(
        candidate_dataset_dir=args.candidate_dataset_dir,
        g1_dataset_dir=args.g1_dataset_dir,
        partition_manifest_path=args.partition_manifest,
        text_backend=backend,
    )
    artifacts = train_relation_graph_critic(
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
        "candidate_dataset_dir": str(args.candidate_dataset_dir.resolve()),
        "g1_dataset_dir": str(args.g1_dataset_dir.resolve()),
        "partition_manifest": str(args.partition_manifest.resolve()),
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
                "all": artifacts.metrics_all,
                "edit_only": artifacts.metrics_edit_only,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
