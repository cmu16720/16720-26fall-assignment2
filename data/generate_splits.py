"""Generate and verify deterministic CIFAR-10 index splits."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.pa2_utils import SEED, balanced_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "cifar10_split_indices.npz"
    )
    args = parser.parse_args()
    from torchvision.datasets import CIFAR10

    training = CIFAR10(args.data_root, train=True, download=True)
    test = CIFAR10(args.data_root, train=False, download=True)
    train_idx = balanced_indices(training.targets, 2500, SEED, 0)
    val_idx = balanced_indices(training.targets, 500, SEED, 2500)
    test_idx = np.arange(len(test), dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        seed=np.array(SEED),
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output} sha256={digest}")


if __name__ == "__main__":
    main()
