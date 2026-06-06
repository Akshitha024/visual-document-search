from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from tabulate import tabulate

from ..runner import bench
from ..viz.charts import (
    plot_first_relevant_rank,
    plot_metric_curves,
    plot_per_topic_accuracy,
    plot_score_margin,
    plot_topic_confusion,
)

app = typer.Typer(add_completion=False, help="cps: ColPali-style document search")


@app.command("bench")
def cmd_bench(
    out_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
    top_k: Annotated[int, typer.Option(help="primary top-k")] = 10,
) -> None:
    metrics = bench(out_dir, top_k=top_k)
    rows = [(k, f"{v:.4f}") for k, v in sorted(metrics.items())]
    print()
    print(tabulate(rows, headers=["metric", "value"], tablefmt="github"))


@app.command("plots")
def cmd_plots(
    results_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
    figures_dir: Annotated[Path, typer.Option(help="figures dir")] = Path("results/figures"),
) -> None:
    plot_metric_curves(results_dir / "metrics.json", figures_dir / "metric_curves.png")
    plot_first_relevant_rank(results_dir / "queries.jsonl", figures_dir / "first_relevant_rank.png")
    plot_per_topic_accuracy(
        results_dir / "queries.jsonl",
        results_dir / "layout.json",
        figures_dir / "per_topic_accuracy.png",
    )
    plot_score_margin(results_dir / "queries.jsonl", figures_dir / "score_margin.png")
    plot_topic_confusion(
        results_dir / "queries.jsonl",
        results_dir / "layout.json",
        figures_dir / "topic_confusion.png",
    )
    typer.echo(f"wrote 5 figures to {figures_dir}")


if __name__ == "__main__":
    app()
