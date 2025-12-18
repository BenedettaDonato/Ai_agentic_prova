"""Randomly sample repositories per category and write to CSV

Reads `repo_analysis_with_category.csv` (produced by pick_repositories.py),
randomly selects N samples per category (default 2) and writes them to
`tools/sampled_repositories.csv`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sample_repos(infile: Path, outfile: Path, per_category: int = 2, seed: int | None = 42) -> None:
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    df = pd.read_csv(infile)
    if "Category" not in df.columns:
        raise ValueError("Input CSV does not contain 'Category' column. Run pick_repositories.py first.")

    sampled = []
    categories = df["Category"].unique()
    for cat in categories:
        group = df[df["Category"] == cat]
        n = min(per_category, len(group))
        # Using sample with random_state for reproducibility
        sampled_group = group.sample(n=n, random_state=seed) if n > 0 else group.iloc[0:0]
        sampled.append(sampled_group)

    if sampled:
        out_df = pd.concat(sampled, ignore_index=True)
    else:
        out_df = df.iloc[0:0]

    original_cols = list(df.columns)
    out_df = out_df.reindex(columns=original_cols)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outfile, index=False)

    # Print summary
    print(f"Wrote {len(out_df)} sampled repositories to {outfile}")
    for cat in categories:
        cnt = len(out_df[out_df["Category"] == cat]) if "Category" in out_df.columns else 0
        print(f"  {cat}: {cnt}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", type=Path, default=Path(__file__).parents[1] / "repo_analysis_with_category.csv")
    p.add_argument("--output", "-o", type=Path, default=Path(__file__).parent / "sampled_repositories.csv")
    p.add_argument("--per-category", "-n", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    try:
        sample_repos(args.input, args.output, per_category=args.per_category, seed=args.seed)
    except Exception as e:
        print("Error:", e)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
