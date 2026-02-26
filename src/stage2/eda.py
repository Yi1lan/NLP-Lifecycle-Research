"""EDA generation for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .preprocessing import PreprocessOutputs


@dataclass(frozen=True)
class EDAOutputs:
    class_distribution_table: Path
    length_summary_table: Path
    lexical_analysis_table: Path
    class_distribution_figure: Path
    length_distribution_figure: Path
    lexical_analysis_figure: Path
    key_metrics: dict[str, float | int | str]


LEXICAL_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def _load_split_frames(preprocessed: PreprocessOutputs) -> pd.DataFrame:
    full_df = pd.read_csv(preprocessed.full_dataset_path)
    full_df["split"] = "full"

    train_df = pd.read_csv(preprocessed.train_path)
    train_df["split"] = "train"

    dev_df = pd.read_csv(preprocessed.dev_path)
    dev_df["split"] = "dev"

    combined = pd.concat([full_df, train_df, dev_df], ignore_index=True)
    combined["label_name"] = combined["label_binary"].map({0: "No PCL", 1: "PCL"})
    return combined


def _tokenize_for_lexical_analysis(text: object) -> list[str]:
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", str(text).lower())
    return [
        token
        for token in tokens
        if len(token) >= 3 and token not in LEXICAL_STOPWORDS
    ]


def _class_distribution(combined: pd.DataFrame, table_path: Path, figure_path: Path) -> dict[str, float | int]:
    counts = (
        combined.groupby(["split", "label_binary", "label_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    split_totals = counts.groupby("split", as_index=False)["count"].sum().rename(
        columns={"count": "split_total"}
    )
    table = counts.merge(split_totals, on="split", how="left")
    table["percentage"] = (table["count"] / table["split_total"] * 100.0).round(3)
    table = table.sort_values(["split", "label_binary"]).reset_index(drop=True)
    table.to_csv(table_path, index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5))
    chart = sns.barplot(data=table, x="split", y="count", hue="label_name")
    
    # Add count labels on top of bars
    for container in chart.containers:
        chart.bar_label(container)

    chart.set_title("Class Distribution by Split")
    chart.set_xlabel("Split")
    chart.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close()

    metric_table = table[table["label_binary"] == 1].set_index("split")

    def _metric(split: str, column: str, default: float = 0.0) -> float:
        if split not in metric_table.index:
            return default
        return float(metric_table.loc[split, column])

    metrics = {
        "train_positive_count": int(_metric("train", "count")),
        "train_positive_pct": _metric("train", "percentage"),
        "dev_positive_count": int(_metric("dev", "count")),
        "dev_positive_pct": _metric("dev", "percentage"),
    }
    return metrics


def _length_analysis(
    combined: pd.DataFrame, table_path: Path, figure_path: Path
) -> dict[str, float | int]:
    stats = (
        combined.groupby(["split", "label_binary", "label_name"])["token_count"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            min="min",
            max="max",
            p90=lambda x: x.quantile(0.90),
            p95=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )
    stats = stats.round(3).sort_values(["split", "label_binary"]).reset_index(drop=True)
    stats.to_csv(table_path, index=False)

    plot_df = combined[combined["split"].isin(["train", "dev"])].copy()
    
    # Plot across the full observed token range for train/dev.
    x_limit = max(float(plot_df["token_count"].max()), 1.0)

    sns.set_theme(style="whitegrid")
    g = sns.displot(
        data=plot_df,
        x="token_count",
        hue="label_name",
        col="split",
        kind="hist",
        bins=30,
        element="bars",
        fill=True,
        alpha=0.5,
        stat="density",
        common_norm=False,
        height=4,
        aspect=1.2,
    )
    
    g.set_axis_labels("Token Count", "Density")
    g.set(xlim=(0, x_limit))
    
    g.figure.suptitle("Token-Length Distribution by Label and Split", y=1.05)
    g.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close("all")

    train_stats = stats[stats["split"] == "train"]
    train_pos = train_stats[train_stats["label_binary"] == 1]
    train_neg = train_stats[train_stats["label_binary"] == 0]

    train_p95 = float(train_stats["p95"].max()) if not train_stats.empty else 0.0
    metrics = {
        "train_p95_token_count": train_p95,
        "train_pos_mean_tokens": float(train_pos["mean"].iloc[0]) if not train_pos.empty else 0.0,
        "train_neg_mean_tokens": float(train_neg["mean"].iloc[0]) if not train_neg.empty else 0.0,
    }
    return metrics


def _lexical_analysis(
    combined: pd.DataFrame,
    table_path: Path,
    figure_path: Path,
    min_token_total_count: int = 6,
    top_k: int = 20,
) -> dict[str, float | int | str]:
    train_df = combined[combined["split"] == "train"].copy()
    train_df = train_df.dropna(subset=["text", "label_binary"])

    token_counts: dict[int, dict[str, int]] = {0: {}, 1: {}}
    doc_counts: dict[int, dict[str, int]] = {0: {}, 1: {}}

    for _, row in train_df.iterrows():
        label = int(row["label_binary"])
        if label not in {0, 1}:
            continue

        tokens = _tokenize_for_lexical_analysis(row["text"])
        if not tokens:
            continue

        label_token_counts = token_counts[label]
        for token in tokens:
            label_token_counts[token] = label_token_counts.get(token, 0) + 1

        label_doc_counts = doc_counts[label]
        for token in set(tokens):
            label_doc_counts[token] = label_doc_counts.get(token, 0) + 1

    vocab = set(token_counts[0]) | set(token_counts[1])
    vocab_size = len(vocab)
    total_tokens_0 = sum(token_counts[0].values())
    total_tokens_1 = sum(token_counts[1].values())

    lexical_rows: list[dict[str, float | int | str]] = []
    for token in sorted(vocab):
        count_0 = token_counts[0].get(token, 0)
        count_1 = token_counts[1].get(token, 0)
        total_count = count_0 + count_1
        if total_count < min_token_total_count:
            continue

        p_0 = (count_0 + 1.0) / (total_tokens_0 + vocab_size) if (total_tokens_0 + vocab_size) > 0 else 0.0
        p_1 = (count_1 + 1.0) / (total_tokens_1 + vocab_size) if (total_tokens_1 + vocab_size) > 0 else 0.0
        log_odds = math.log(p_1 / p_0) if p_0 > 0 and p_1 > 0 else 0.0

        lexical_rows.append(
            {
                "token": token,
                "count_no_pcl": count_0,
                "count_pcl": count_1,
                "doc_count_no_pcl": doc_counts[0].get(token, 0),
                "doc_count_pcl": doc_counts[1].get(token, 0),
                "log_odds_pcl_vs_no_pcl": log_odds,
            }
        )

    lexical_df = pd.DataFrame(lexical_rows)
    if lexical_df.empty:
        empty_table = pd.DataFrame(
            columns=[
                "label_name",
                "token",
                "count_in_label",
                "count_in_other_label",
                "doc_frequency_pct",
                "rate_per_10k_tokens",
                "log_odds_pcl_vs_no_pcl",
            ]
        )
        empty_table.to_csv(table_path, index=False)

        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, "No lexical signals found with current thresholds", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=220)
        plt.close()

        return {
            "lexical_vocab_size": int(vocab_size),
            "top_pcl_token": "n/a",
            "top_no_pcl_token": "n/a",
        }

    no_pcl_doc_total = max(int((train_df["label_binary"] == 0).sum()), 1)
    pcl_doc_total = max(int((train_df["label_binary"] == 1).sum()), 1)

    lexical_df["rate_no_pcl_per_10k"] = (
        lexical_df["count_no_pcl"] / max(total_tokens_0, 1) * 10000.0
    )
    lexical_df["rate_pcl_per_10k"] = (
        lexical_df["count_pcl"] / max(total_tokens_1, 1) * 10000.0
    )
    lexical_df["doc_pct_no_pcl"] = lexical_df["doc_count_no_pcl"] / no_pcl_doc_total * 100.0
    lexical_df["doc_pct_pcl"] = lexical_df["doc_count_pcl"] / pcl_doc_total * 100.0

    top_pcl = lexical_df.sort_values(
        ["log_odds_pcl_vs_no_pcl", "count_pcl"],
        ascending=[False, False],
    ).head(top_k).copy()
    top_pcl["label_name"] = "PCL"
    top_pcl["count_in_label"] = top_pcl["count_pcl"]
    top_pcl["count_in_other_label"] = top_pcl["count_no_pcl"]
    top_pcl["doc_frequency_pct"] = top_pcl["doc_pct_pcl"]
    top_pcl["rate_per_10k_tokens"] = top_pcl["rate_pcl_per_10k"]
    top_pcl["class_advantage"] = top_pcl["log_odds_pcl_vs_no_pcl"]

    top_no_pcl = lexical_df.sort_values(
        ["log_odds_pcl_vs_no_pcl", "count_no_pcl"],
        ascending=[True, False],
    ).head(top_k).copy()
    top_no_pcl["label_name"] = "No PCL"
    top_no_pcl["count_in_label"] = top_no_pcl["count_no_pcl"]
    top_no_pcl["count_in_other_label"] = top_no_pcl["count_pcl"]
    top_no_pcl["doc_frequency_pct"] = top_no_pcl["doc_pct_no_pcl"]
    top_no_pcl["rate_per_10k_tokens"] = top_no_pcl["rate_no_pcl_per_10k"]
    top_no_pcl["class_advantage"] = -top_no_pcl["log_odds_pcl_vs_no_pcl"]

    table = pd.concat([top_no_pcl, top_pcl], ignore_index=True)
    table = table[
        [
            "label_name",
            "token",
            "count_in_label",
            "count_in_other_label",
            "doc_frequency_pct",
            "rate_per_10k_tokens",
            "log_odds_pcl_vs_no_pcl",
        ]
    ]
    table = table.round(3)
    table.to_csv(table_path, index=False)

    plot_pcl = top_pcl.head(12).sort_values("class_advantage", ascending=True)
    plot_no_pcl = top_no_pcl.head(12).sort_values("class_advantage", ascending=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    sns.barplot(
        data=plot_no_pcl,
        x="class_advantage",
        y="token",
        color="#4C72B0",
        ax=axes[0],
    )
    axes[0].set_title("Top Distinctive Tokens: No PCL")
    axes[0].set_xlabel("Class Advantage Score")
    axes[0].set_ylabel("Token")

    sns.barplot(
        data=plot_pcl,
        x="class_advantage",
        y="token",
        color="#C44E52",
        ax=axes[1],
    )
    axes[1].set_title("Top Distinctive Tokens: PCL")
    axes[1].set_xlabel("Class Advantage Score")
    axes[1].set_ylabel("Token")

    fig.suptitle("Lexical Signal Analysis (Train Split, Unigrams)", y=1.02)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    top_pcl_token = str(top_pcl["token"].iloc[0]) if not top_pcl.empty else "n/a"
    top_no_pcl_token = str(top_no_pcl["token"].iloc[0]) if not top_no_pcl.empty else "n/a"

    return {
        "lexical_vocab_size": int(vocab_size),
        "top_pcl_token": top_pcl_token,
        "top_no_pcl_token": top_no_pcl_token,
    }


def run_eda(preprocessed: PreprocessOutputs, outputs_root: Path) -> EDAOutputs:
    """Generate Stage 2 EDA tables and figures."""
    figures_dir = outputs_root / "figures"
    tables_dir = outputs_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    combined = _load_split_frames(preprocessed)

    class_distribution_table = tables_dir / "class_distribution.csv"
    class_distribution_figure = figures_dir / "class_distribution.png"
    length_summary_table = tables_dir / "length_summary.csv"
    length_distribution_figure = figures_dir / "length_distribution.png"
    lexical_analysis_table = tables_dir / "lexical_analysis.csv"
    lexical_analysis_figure = figures_dir / "lexical_analysis.png"

    class_metrics = _class_distribution(
        combined, class_distribution_table, class_distribution_figure
    )
    length_metrics = _length_analysis(
        combined, length_summary_table, length_distribution_figure
    )
    lexical_metrics = _lexical_analysis(
        combined, lexical_analysis_table, lexical_analysis_figure
    )

    key_metrics = {**class_metrics, **length_metrics, **lexical_metrics}

    return EDAOutputs(
        class_distribution_table=class_distribution_table,
        length_summary_table=length_summary_table,
        lexical_analysis_table=lexical_analysis_table,
        class_distribution_figure=class_distribution_figure,
        length_distribution_figure=length_distribution_figure,
        lexical_analysis_figure=lexical_analysis_figure,
        key_metrics=key_metrics,
    )
