"""Pick and categorize repositories from repo_analysis.csv

Reads the CSV, computes total classes and methods, methods-per-class,
and assigns each repository into one of four categories using the
OR logic described by the user (Large/Highly interconnected, Medium,
Small, Trivial). Outputs a new CSV with the category and prints a
summary counts per category.

Usage: python tools/pick_repositories.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def categorize_row(row: pd.Series) -> str:
    """Return category name for a single repository row.

    The user asked to use OR across metrics. We'll evaluate the
    Large -> Medium -> Small -> Trivial precedence (largest match
    wins). Missing or zero values are handled conservatively.
    """
    loc = row.get("Lines of Code") or 0
    public_classes = row.get("Public Classes") or 0
    private_classes = row.get("Private Classes") or 0
    protected_classes = row.get("Protected Classes") or 0
    total_classes = public_classes + private_classes + protected_classes

    public_methods = row.get("Public Methods") or 0
    private_methods = row.get("Private Methods") or 0
    protected_methods = row.get("Protected Methods") or 0
    total_methods = public_methods + private_methods + protected_methods

    # avoid division by zero
    methods_per_class = float(total_methods) / total_classes if total_classes > 0 else 0.0

    avg_cc = row.get("Average Cyclomatic Complexity")
    if pd.isna(avg_cc):
        avg_cc = 0.0

    pct_cc_over_thresh = row.get("Percentage of Methods with Cyclomatic Complexity over Threshold")
    if pd.isna(pct_cc_over_thresh):
        pct_cc_over_thresh = 0.0

    # New categorization using 3-of-N rule. Evaluate highest category first.

    # Helper to count how many criteria are met for a given list of booleans
    def at_least_three(*conds: bool) -> bool:
        return sum(1 for c in conds if c) >= 3

    # Percentage of functions with CC >= 10 is provided in pct_cc_over_thresh
    pct_cc_ge_10 = pct_cc_over_thresh

    # Epic
    epic_conds = [
        loc > 80_000,  # interpreted as >80k (user listed >40k then >80k; choose the stricter)
        total_classes > 300,
        methods_per_class > 6,
        avg_cc > 4,
        pct_cc_ge_10 > 6,
    ]
    if at_least_three(*epic_conds):
        return "Epic"

    # Complex
    complex_conds = [
        loc > 10_000,
        total_classes > 100,
        methods_per_class > 4,
        avg_cc > 2.5,
        pct_cc_ge_10 > 4,
    ]
    if at_least_three(*complex_conds):
        return "Complex"

    # Moderate
    moderate_conds = [
        loc > 2_000,
        total_classes > 20,
        methods_per_class > 2,
        avg_cc > 1,
        pct_cc_ge_10 > 2,
    ]
    if at_least_three(*moderate_conds):
        return "Moderate"

    # Trivial
    trivial_conds = [
        loc > 0,
        total_classes > 0,
        methods_per_class > 0,
        avg_cc > 0,
        pct_cc_ge_10 > 0,
    ]
    if at_least_three(*trivial_conds):
        return "Trivial"

    # If nothing matches (e.g., all zeros), classify as Unknown/Empty
    return "Unknown"


def process_csv(infile: Path, outfile: Path) -> None:
    df = pd.read_csv(infile)

    # Normalize column names to expected capitalization/spacing if needed
    # We'll accept either the titled names present in the supplied CSV.

    # Compute totals
    for col in ["Public Classes", "Private Classes", "Protected Classes"]:
        if col not in df.columns:
            df[col] = 0

    for col in ["Public Methods", "Private Methods", "Protected Methods"]:
        if col not in df.columns:
            df[col] = 0

    df["Total Classes"] = (
        df["Public Classes"].fillna(0).astype(float)
        + df["Private Classes"].fillna(0).astype(float)
        + df["Protected Classes"].fillna(0).astype(float)
    )

    df["Total Methods"] = (
        df["Public Methods"].fillna(0).astype(float)
        + df["Private Methods"].fillna(0).astype(float)
        + df["Protected Methods"].fillna(0).astype(float)
    )

    # Methods per class (set to 0 when Total Classes == 0)
    df["Methods Per Class"] = np.where(
        df["Total Classes"] > 0,
        df["Total Methods"] / df["Total Classes"],
        0.0,
    )

    # Ensure numeric columns exist and missing values are handled
    numeric_cols = [
        "Lines of Code",
        "Average Cyclomatic Complexity",
        "Percentage of Methods with Cyclomatic Complexity over Threshold",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Categorize
    df["Category"] = df.apply(categorize_row, axis=1)

    # Save output
    df.to_csv(outfile, index=False)

    # Print summary counts
    counts = df["Category"].value_counts()
    print("Category counts:")
    for cat, cnt in counts.items():
        print(f"  {cat}: {cnt}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path(__file__).parents[1] / "repo_analysis.csv",
        help="Path to repo_analysis.csv",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).parents[1] / "repo_analysis_with_category.csv",
        help="Output CSV path",
    )
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        return 2

    process_csv(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
