"""Collate repository metrics for baseline and generated projects.

This script gathers the key metrics required to match the schema of
`repo_analysis.csv`, while also appending average and maximum similarity
values sourced from summary similarity CSV files. It produces two output
CSV files (`generated_metrics.csv` and `baseline_metrics.csv` by default)
containing a single row each.

Usage example:

	python tools/extract_metrics.py \
		--repository uppnrise/distributed-rate-limiter \
		--generated-metrics output/distributed-rate-limiter/metrics.csv \
		--baseline-metrics output/distributed-rate-limiter/metrics_baseline.csv \
		--generated-summary output/distributed-rate-limiter/comparison_report/summary_similarities.csv \
		--baseline-summary output/distributed-rate-limiter/comparison_report_baseline/summary_similarities.csv
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


OUTPUT_FIELDS: List[str] = [
	"Repository",
	"Repository Link",
	"Stars",
	"Public Classes",
	"Private Classes",
	"Protected Classes",
	"Total Classes",
	"Public Methods",
	"Private Methods",
	"Protected Methods",
	"Total Methods",
	"Lines of Code",
	"Average Cyclomatic Complexity",
	"Maximum Cyclomatic Complexity",
	"Percentage of Methods with Cyclomatic Complexity over Threshold",
	"Average Similarity",
	"Maximum Similarity",
	"Tokens Used",
]


METRIC_COLUMNS: Tuple[str, ...] = (
	"Public Classes",
	"Private Classes",
	"Protected Classes",
	"Public Methods",
	"Private Methods",
	"Protected Methods",
	"Lines of Code",
	"Average Cyclomatic Complexity",
	"Maximum Cyclomatic Complexity",
	"Percentage of Methods with Cyclomatic Complexity over Threshold",
	"Tokens Used",
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Collate baseline and generated metrics into CSV outputs."
	)
	parser.add_argument(
		"--repository",
		required=True,
		help="Repository in owner/name format (e.g. uppnrise/distributed-rate-limiter).",
	)
	parser.add_argument(
		"--generated-metrics",
		required=True,
		type=Path,
		help="Path to the generated project metrics CSV.",
	)
	parser.add_argument(
		"--baseline-metrics",
		required=True,
		type=Path,
		help="Path to the baseline project metrics CSV.",
	)
	parser.add_argument(
		"--generated-summary",
		required=True,
		type=Path,
		help="Path to the generated project summary similarities CSV.",
	)
	parser.add_argument(
		"--baseline-summary",
		required=True,
		type=Path,
		help="Path to the baseline project summary similarities CSV.",
	)
	parser.add_argument(
		"--generated-output",
		default="generated_metrics.csv",
		type=Path,
		help="Output path for the generated project metrics CSV.",
	)
	parser.add_argument(
		"--baseline-output",
		default="baseline_metrics.csv",
		type=Path,
		help="Output path for the baseline project metrics CSV.",
	)
	parser.add_argument(
		"--stars",
		default="",
		help="Optional stars value to include in the output (defaults to empty).",
	)
	return parser.parse_args()


def read_metrics(csv_path: Path) -> Dict[str, str]:
	"""Return the first row of the metrics CSV constrained to required columns."""

	with csv_path.open(newline="", encoding="utf-8") as fh:
		reader = csv.DictReader(fh)
		first_row = next(reader, None)

	if not first_row:
		raise ValueError(f"No rows found in metrics CSV: {csv_path}")

	missing = [column for column in METRIC_COLUMNS if column not in first_row]
	if missing:
		raise KeyError(
			f"Metrics CSV {csv_path} is missing required columns: {', '.join(missing)}"
		)

	return {column: first_row[column] for column in METRIC_COLUMNS}


def read_similarity_summary(csv_path: Path) -> Tuple[str, str]:
	"""Extract average and maximum similarity from the first row of the summary CSV."""

	with csv_path.open(newline="", encoding="utf-8") as fh:
		reader = csv.DictReader(fh)
		row = next(reader, None)

	if not row:
		raise ValueError(f"No rows found in summary CSV: {csv_path}")

	try:
		average = Decimal(row["avg_similarity"])
		maximum = Decimal(row["max_similarity"])
	except KeyError as exc:
		raise KeyError(
			f"Summary CSV {csv_path} missing required column: {exc.args[0]}"
		) from exc
	except Exception as exc:  # noqa: BLE001
		raise ValueError(
			f"Invalid numeric value in summary CSV {csv_path}: {exc}"
		) from exc

	return _normalize_decimal(average), _normalize_decimal(maximum)


def _normalize_decimal(value: Decimal) -> str:
	"""Format Decimal values without scientific notation or trailing zeros."""

	normalized = value.normalize()
	return format(normalized, "f").rstrip("0").rstrip(".") or "0"


def _sum_metrics(metrics: Dict[str, str], columns: Iterable[str]) -> str:
	"""Sum the specified metric columns and return a normalized string."""

	total = Decimal("0")
	for column in columns:
		value = metrics.get(column, "").strip()
		if not value:
			continue
		try:
			total += Decimal(value)
		except Exception as exc:  # noqa: BLE001
			raise ValueError(
				f"Invalid numeric value for column '{column}': {metrics.get(column, '')}"
			) from exc

	return _normalize_decimal(total)


def build_output_row(
	repository: str,
	metrics: Dict[str, str],
	average_similarity: str,
	maximum_similarity: str,
	stars: str,
) -> Dict[str, str]:
	repository_link = f"https://github.com/{repository}".rstrip("/")

	row = {
		"Repository": repository,
		"Repository Link": repository_link,
		"Stars": stars,
		"Average Similarity": average_similarity,
		"Maximum Similarity": maximum_similarity,
	}

	for column in METRIC_COLUMNS:
		row[column] = metrics[column]

	row["Total Classes"] = _sum_metrics(
		metrics,
		("Public Classes", "Private Classes", "Protected Classes"),
	)
	row["Total Methods"] = _sum_metrics(
		metrics,
		("Public Methods", "Private Methods", "Protected Methods"),
	)

	# Ensure output order and fill missing keys with empty strings.
	return {field: row.get(field, "") for field in OUTPUT_FIELDS}


def write_output(path: Path, rows: Iterable[Dict[str, str]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as fh:
		writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def main() -> None:
	args = parse_args()

	generated_metrics = read_metrics(args.generated_metrics)
	baseline_metrics = read_metrics(args.baseline_metrics)

	generated_avg, generated_max = read_similarity_summary(args.generated_summary)
	baseline_avg, baseline_max = read_similarity_summary(args.baseline_summary)

	generated_row = build_output_row(
		repository=args.repository,
		metrics=generated_metrics,
		average_similarity=generated_avg,
		maximum_similarity=generated_max,
		stars=args.stars,
	)

	baseline_row = build_output_row(
		repository=args.repository,
		metrics=baseline_metrics,
		average_similarity=baseline_avg,
		maximum_similarity=baseline_max,
		stars=args.stars,
	)

	write_output(args.generated_output, [generated_row])
	write_output(args.baseline_output, [baseline_row])


if __name__ == "__main__":
	main()
