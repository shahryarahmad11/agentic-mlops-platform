"""Generate a reproducible synthetic classification dataset."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True, help="number of samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/train.csv")
    args = parser.parse_args()

    features, target = make_classification(
        n_samples=args.rows,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=args.seed,
    )
    frame = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(5)])
    frame["target"] = target

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"Dataset generated: {frame.shape} -> {out}")


if __name__ == "__main__":
    main()
