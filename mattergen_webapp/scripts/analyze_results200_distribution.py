#!/usr/bin/env python3
"""Analyze and visualize Li-Nb-O-Cl generation campaigns."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIRS = (
    PROJECT_ROOT / "runs_e" / "results200",
    PROJECT_ROOT / "results10",
)
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "results200_plus_results10"

HULL_CUTOFF_EV = 0.05

COLORS = {
    "blue": "#2f6fbb",
    "teal": "#26897a",
    "orange": "#c77c2b",
    "red": "#b64b4b",
    "purple": "#7a5aa6",
    "gray": "#6f7782",
    "light_gray": "#d9dee7",
    "dark": "#222831",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-ready distribution analyses for one or more result directories."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        nargs="+",
        default=list(DEFAULT_RESULTS_DIRS),
        help="Directory or directories containing stage2_candidates.csv and top300_run.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=f"Output directory. Defaults to {DEFAULT_ANALYSIS_DIR}.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("pdf", "svg", "png"),
        choices=("pdf", "svg", "png"),
        help="Figure formats to write.",
    )
    return parser.parse_args()


def configure_plots() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#2f343a",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.grid": True,
            "grid.color": "#e8ebf0",
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
        }
    )


def ensure_inputs(results_dir: Path) -> dict[str, Path]:
    paths = {
        "stage2": results_dir / "stage2_candidates.csv",
        "hull": results_dir / "top300_run" / "chgnet_hull_top300.csv",
        "hull_filtered": results_dir
        / "top300_run"
        / "chgnet_hull_top300_filtered.csv",
        "voltage": results_dir / "top300_run" / "chgnet_voltage_window_top300.csv",
        "export_index": results_dir
        / "top300_run"
        / "exported_300cifs"
        / "export_300index.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    return paths


def make_dataset_labels(results_dirs: Iterable[Path]) -> dict[Path, str]:
    labels: dict[Path, str] = {}
    used: set[str] = set()
    for idx, results_dir in enumerate(results_dirs, start=1):
        resolved = results_dir.resolve()
        base = results_dir.name or resolved.name or f"dataset{idx}"
        label = base
        if label in used:
            label = f"{base}_{idx}"
        used.add(label)
        labels[resolved] = label
    return labels


def extract_batch(text: object) -> str:
    match = re.search(r"batch\d+", str(text))
    return match.group(0) if match else "unknown"


def run_label(dataset: str, batch: str) -> str:
    short_dataset = dataset.replace("results", "R")
    if str(batch) == dataset:
        return short_dataset
    match = re.fullmatch(r"batch(\d+)", str(batch))
    if match:
        return f"{short_dataset}-B{match.group(1)}"
    if batch and batch != "unknown":
        return f"{short_dataset}-{batch}"
    return short_dataset


def extract_rank(file_name: object) -> float:
    match = re.search(r"cand300_(\d+)_", str(file_name))
    return float(match.group(1)) if match else math.nan


def safe_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def count_extxyz_frames(path: Path) -> int:
    count = 0
    with path.open("r") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                natoms = int(line)
            except ValueError as exc:
                raise ValueError(f"Invalid extxyz atom-count line in {path}: {line}") from exc
            handle.readline()
            for _ in range(natoms):
                handle.readline()
            count += 1
    return count


def count_campaign_frames(results_dir: Path, dataset: str) -> pd.DataFrame:
    rows = []
    paths = sorted((results_dir / "_segments").glob("batch*/Li-Nb-O-Cl/*.extxyz"))
    if not paths:
        paths = sorted(
            path
            for path in results_dir.glob("*/*.extxyz")
            if path.name in {"generated_crystals.extxyz", "relaxed.extxyz"}
        )
    for path in paths:
        batch = extract_batch(path)
        if batch == "unknown":
            batch = dataset
        rows.append(
            {
                "dataset": dataset,
                "batch": batch,
                "run_label": run_label(dataset, batch),
                "kind": path.stem,
                "path": str(path),
                "n_frames": count_extxyz_frames(path),
            }
        )
    return pd.DataFrame(rows)


def load_stage2(path: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    safe_numeric(
        df,
        [
            "frame",
            "n_atoms",
            "density",
            "f_li",
            "f_o",
            "hal_entropy",
            "li_conn",
            "min_li_li",
            "quick_score",
        ],
    )
    for col in ("passes_light_oxy", "charge_balance_ok", "smact_ok"):
        if col in df.columns and df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})
    df["dataset"] = dataset
    df["batch"] = df["path"].map(extract_batch)
    df.loc[df["batch"] == "unknown", "batch"] = dataset
    df["run_label"] = df["batch"].map(lambda value: run_label(dataset, value))
    df["ref"] = df["path"].astype(str) + "::" + df["frame"].astype("Int64").astype(str)
    df = df.sort_values("quick_score", ascending=False).reset_index(drop=True)
    df["local_quick_score_rank"] = np.arange(1, len(df) + 1)
    df["min_li_li_for_plot"] = df["min_li_li"].where(df["min_li_li"] < 100)
    return df


def load_batch_metrics(results_dir: Path, dataset: str) -> pd.DataFrame:
    rows = []
    paths = sorted((results_dir / "_segments").glob("batch*/Li-Nb-O-Cl/metrics.json"))
    if not paths:
        paths = sorted(results_dir.glob("*/metrics.json"))
    for path in paths:
        data = json.loads(path.read_text())
        batch = extract_batch(path)
        if batch == "unknown":
            batch = dataset
        row = {
            "dataset": dataset,
            "batch": batch,
            "run_label": run_label(dataset, batch),
            "path": str(path),
        }
        for key, value in data.items():
            if isinstance(value, dict) and "value" in value:
                row[key] = value["value"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["dataset", "batch"])
    metric_cols = [col for col in df.columns if col not in {"dataset", "batch", "run_label", "path"}]
    safe_numeric(df, metric_cols)
    return df


def rank_stage2_global(stage2: pd.DataFrame, topk: int = 300) -> pd.DataFrame:
    ranked = stage2.sort_values("quick_score", ascending=False).reset_index(drop=True)
    ranked["quick_score_rank"] = np.arange(1, len(ranked) + 1)
    ranked["is_top300_by_score"] = ranked["quick_score_rank"] <= topk
    return ranked


def load_top300(
    paths: dict[str, Path],
    dataset: str,
    stage2: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    export_index = pd.read_csv(paths["export_index"]).rename(
        columns={
            "formula": "formula_export",
            "spg": "spg_export",
            "n_atoms": "n_atoms_export",
            "density": "density_export",
        }
    )
    export_index["dataset"] = dataset
    export_index["file"] = export_index["cif"].map(lambda x: Path(str(x)).name)
    export_index["rank"] = export_index["file"].map(extract_rank)
    export_index["batch"] = export_index["ref"].map(extract_batch)
    export_index.loc[export_index["batch"] == "unknown", "batch"] = dataset
    export_index["run_label"] = export_index["batch"].map(lambda value: run_label(dataset, value))
    export_index["candidate_id"] = export_index["dataset"] + ":" + export_index["file"]

    hull = pd.read_csv(paths["hull"]).rename(
        columns={
            "path": "hull_path",
            "formula": "formula_hull",
            "chemsys": "chemsys_hull",
        }
    )
    hull["dataset"] = dataset
    safe_numeric(
        hull,
        [
            "natoms_cell",
            "energy_per_atom_eV",
            "energy_total_eV",
            "formation_energy_per_atom_eV",
            "energy_above_hull_eV",
            "is_stable",
        ],
    )

    hull_filtered = pd.read_csv(paths["hull_filtered"]).rename(
        columns={
            "path": "hull_path",
            "formula": "formula_hull",
            "chemsys": "chemsys_hull",
        }
    )
    hull_filtered["dataset"] = dataset
    safe_numeric(hull_filtered, ["energy_above_hull_eV", "is_stable"])

    voltage = pd.read_csv(paths["voltage"]).rename(
        columns={"formula": "formula_voltage", "chemsys": "chemsys_voltage"}
    )
    voltage["dataset"] = dataset
    safe_numeric(voltage, ["V_red", "V_ox", "window"])

    stage_cols = [
        "dataset",
        "ref",
        "quick_score",
        "quick_score_rank",
        "local_quick_score_rank",
        "is_top300_by_score",
        "spg",
        "formula",
        "n_atoms",
        "density",
        "f_li",
        "f_o",
        "hal_entropy",
        "li_conn",
        "min_li_li",
        "passes_light_oxy",
        "charge_balance_ok",
        "smact_ok",
    ]
    top = (
        export_index.merge(hull, on=["dataset", "file"], how="left")
        .merge(voltage, on=["dataset", "file"], how="left")
        .merge(stage2[stage_cols], on=["dataset", "ref"], how="left", suffixes=("", "_stage2"))
    )
    top["formula_plot"] = (
        top.get("formula")
        .fillna(top.get("formula_export"))
        .fillna(top.get("formula_hull"))
        .fillna(top.get("formula_voltage"))
    )
    top["spg_plot"] = top.get("spg").fillna(top.get("spg_export"))
    top["near_hull_0p05"] = top["energy_above_hull_eV"] <= HULL_CUTOFF_EV
    top["finite_voltage_window"] = top[["V_red", "V_ox", "window"]].notna().all(axis=1)
    return top.sort_values(["dataset", "rank"]), hull_filtered, voltage


def make_funnel(frame_counts: pd.DataFrame, stage2: pd.DataFrame, top300: pd.DataFrame) -> pd.DataFrame:
    generated = int(frame_counts.loc[frame_counts["kind"] == "generated_crystals", "n_frames"].sum())
    relaxed = int(frame_counts.loc[frame_counts["kind"] == "relaxed", "n_frames"].sum())
    parsed = int(len(stage2))
    top = int(len(top300))
    near_hull = int(top300["near_hull_0p05"].sum())
    voltage_rows = int(top300["formula_voltage"].notna().sum())
    finite_voltage = int(top300["finite_voltage_window"].sum())

    rows = [
        ("Generated structures", generated),
        ("Relaxed structures", relaxed),
        ("Parsed stage-2 rows", parsed),
        ("Top quick-score candidates", top),
        (f"Near-hull candidates <= {HULL_CUTOFF_EV:.2f} eV/atom", near_hull),
        ("Voltage rows", voltage_rows),
        ("Finite voltage windows", finite_voltage),
    ]
    out = pd.DataFrame(rows, columns=["step", "count"])
    out["pct_of_generated"] = out["count"] / generated if generated else np.nan
    out["pct_of_previous"] = out["count"] / out["count"].shift(1)
    out.loc[0, "pct_of_previous"] = 1.0
    return out


def write_tables(
    outdir: Path,
    frame_counts: pd.DataFrame,
    stage2: pd.DataFrame,
    batch_metrics: pd.DataFrame,
    evaluated_all: pd.DataFrame,
    top300: pd.DataFrame,
    funnel: pd.DataFrame,
) -> dict[str, object]:
    tables = outdir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    frame_counts.to_csv(tables / "frame_counts.csv", index=False)
    funnel.to_csv(tables / "screening_funnel.csv", index=False)
    batch_metrics.to_csv(tables / "batch_metrics.csv", index=False)
    evaluated_all.to_csv(tables / "all_evaluated_candidates.csv", index=False)
    top300.to_csv(tables / "top300_merged_metrics.csv", index=False)

    numeric_cols = [
        "n_atoms",
        "density",
        "f_li",
        "f_o",
        "hal_entropy",
        "li_conn",
        "min_li_li",
        "quick_score",
    ]
    stage2[numeric_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.to_csv(
        tables / "stage2_descriptive_stats.csv"
    )
    stage2["formula"].value_counts().rename_axis("formula").reset_index(name="count").to_csv(
        tables / "formula_counts_stage2.csv", index=False
    )
    stage2["spg"].value_counts().rename_axis("space_group").reset_index(name="count").to_csv(
        tables / "spacegroup_counts_stage2.csv", index=False
    )

    per_dataset = (
        frame_counts.pivot_table(
            index="dataset",
            columns="kind",
            values="n_frames",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    stage_counts = stage2.groupby("dataset").size().rename("stage2_rows").reset_index()
    eval_counts = evaluated_all.groupby("dataset").size().rename("evaluated_candidates").reset_index()
    top_counts = (
        top300.groupby("dataset")
        .agg(
            global_top300=("candidate_id", "size"),
            near_hull_0p05=("near_hull_0p05", "sum"),
            finite_voltage_windows=("finite_voltage_window", "sum"),
        )
        .reset_index()
    )
    per_dataset = (
        per_dataset.merge(stage_counts, on="dataset", how="left")
        .merge(eval_counts, on="dataset", how="left")
        .merge(top_counts, on="dataset", how="left")
    )
    per_dataset.to_csv(tables / "per_dataset_summary.csv", index=False)

    finite_voltage = top300.loc[top300["finite_voltage_window"]].copy()
    finite_voltage.sort_values("window", ascending=False).to_csv(
        tables / "finite_voltage_candidates.csv", index=False
    )
    finite_voltage.sort_values("window", ascending=False).head(20).to_csv(
        tables / "top_voltage_candidates.csv", index=False
    )

    corr_cols = ["density", "f_li", "f_o", "li_conn", "min_li_li_for_plot", "quick_score"]
    stage2[corr_cols].corr(method="spearman").to_csv(tables / "stage2_spearman_correlations.csv")

    quick_threshold = float(stage2.loc[stage2["is_top300_by_score"], "quick_score"].min())
    key_numbers = {
        "generated_structures": int(
            frame_counts.loc[frame_counts["kind"] == "generated_crystals", "n_frames"].sum()
        ),
        "relaxed_structures": int(frame_counts.loc[frame_counts["kind"] == "relaxed", "n_frames"].sum()),
        "stage2_rows": int(len(stage2)),
        "top300_rows": int(len(top300)),
        "all_evaluated_candidate_rows": int(len(evaluated_all)),
        "top300_quick_score_threshold": quick_threshold,
        "near_hull_0p05_count": int(top300["near_hull_0p05"].sum()),
        "stable_label_count": int((top300["is_stable"] == 1).sum()),
        "voltage_rows": int(top300["formula_voltage"].notna().sum()),
        "finite_voltage_window_count": int(top300["finite_voltage_window"].sum()),
        "max_voltage_window": float(finite_voltage["window"].max()) if len(finite_voltage) else None,
        "median_voltage_window": float(finite_voltage["window"].median()) if len(finite_voltage) else None,
        "mean_top300_energy_above_hull": float(top300["energy_above_hull_eV"].mean()),
        "median_top300_energy_above_hull": float(top300["energy_above_hull_eV"].median()),
        "mean_batch_frac_stable": float(batch_metrics["frac_stable_structures"].mean()),
        "mean_batch_frac_novel_unique_stable": float(
            batch_metrics["frac_novel_unique_stable_structures"].mean()
        ),
        "mean_batch_comp_validity": float(batch_metrics["avg_comp_validity"].mean()),
        "mean_batch_avg_ehull": float(batch_metrics["avg_energy_above_hull_per_atom"].mean()),
        "mean_batch_rmsd": float(batch_metrics["avg_rmsd_from_relaxation"].mean()),
        "light_oxygen_pass_count": int(stage2["passes_light_oxy"].sum()),
        "min_li_li_sentinel_count": int((stage2["min_li_li"] >= 100).sum()),
    }
    (tables / "key_numbers.json").write_text(json.dumps(key_numbers, indent=2))
    return key_numbers


def save_figure(fig: plt.Figure, outdir: Path, name: str, formats: Iterable[str]) -> None:
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(fig_dir / f"{name}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def annotate_panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def plot_overview(
    outdir: Path,
    formats: Iterable[str],
    stage2: pd.DataFrame,
    top300: pd.DataFrame,
    funnel: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2), constrained_layout=True)
    axes = axes.ravel()

    funnel_plot = funnel.loc[
        funnel["step"].isin(
            [
                "Generated structures",
                "Parsed stage-2 rows",
                "Top quick-score candidates",
                f"Near-hull candidates <= {HULL_CUTOFF_EV:.2f} eV/atom",
                "Finite voltage windows",
            ]
        )
    ].copy()
    short_steps = ["Generated", "Parsed", "Top 300", "Near-hull", "Finite window"]
    y = np.arange(len(funnel_plot))
    axes[0].barh(y, funnel_plot["count"], color=[COLORS["blue"], COLORS["teal"], COLORS["orange"], COLORS["red"], COLORS["purple"]])
    axes[0].set_yticks(y, short_steps)
    axes[0].invert_yaxis()
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Count (log scale)")
    axes[0].set_title("Screening funnel")
    for yi, (_, row) in enumerate(funnel_plot.iterrows()):
        axes[0].text(
            row["count"] * 1.08,
            yi,
            f"{int(row['count'])}\n({row['pct_of_generated']:.1%})",
            va="center",
            fontsize=6.5,
        )
    annotate_panel(axes[0], "A")

    threshold = stage2.loc[stage2["is_top300_by_score"], "quick_score"].min()
    sns.histplot(stage2["quick_score"], bins=48, ax=axes[1], color=COLORS["blue"], edgecolor="white")
    axes[1].axvline(threshold, color=COLORS["red"], lw=1.2, ls="--")
    axes[1].text(
        0.97,
        0.94,
        f"Top-300 cutoff = {threshold:.3f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )
    axes[1].set_xlabel("Quick score")
    axes[1].set_ylabel("Structures")
    axes[1].set_title("Ranking score distribution")
    annotate_panel(axes[1], "B")

    all_points = stage2.loc[~stage2["is_top300_by_score"]]
    top_points = stage2.loc[stage2["is_top300_by_score"]]
    axes[2].axhspan(0.05, 0.35, color=COLORS["teal"], alpha=0.12, lw=0, label="Light-oxygen target")
    axes[2].scatter(
        all_points["f_li"],
        all_points["f_o"],
        s=7,
        c=COLORS["light_gray"],
        alpha=0.55,
        linewidths=0,
        label="Other parsed",
    )
    scatter = axes[2].scatter(
        top_points["f_li"],
        top_points["f_o"],
        s=14,
        c=top_points["quick_score"],
        cmap="viridis",
        edgecolors="white",
        linewidths=0.2,
        label="Top 300",
    )
    axes[2].set_xlim(-0.02, max(0.7, stage2["f_li"].max() + 0.02))
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("Li atomic fraction")
    axes[2].set_ylabel("O / (O + halide)")
    axes[2].set_title("Composition space")
    axes[2].legend(frameon=False, loc="lower right")
    cbar = fig.colorbar(scatter, ax=axes[2], fraction=0.046, pad=0.02)
    cbar.set_label("Quick score")
    annotate_panel(axes[2], "C")

    formula_counts = stage2["formula"].value_counts().head(12).sort_values()
    axes[3].barh(formula_counts.index, formula_counts.values, color=COLORS["teal"])
    axes[3].set_xlabel("Structures")
    axes[3].set_title("Most frequent formulas")
    annotate_panel(axes[3], "D")

    spg_counts = stage2["spg"].value_counts().head(12).sort_values()
    axes[4].barh(spg_counts.index, spg_counts.values, color=COLORS["purple"])
    axes[4].set_xlabel("Structures")
    axes[4].set_title("Most frequent space groups")
    annotate_panel(axes[4], "E")

    bins = np.arange(stage2["n_atoms"].min() - 0.5, stage2["n_atoms"].max() + 1.5, 1)
    axes[5].hist(
        stage2["n_atoms"],
        bins=bins,
        color=COLORS["gray"],
        alpha=0.65,
        label="Parsed",
        edgecolor="white",
    )
    axes[5].hist(
        top300["n_atoms"].dropna(),
        bins=bins,
        color=COLORS["orange"],
        alpha=0.75,
        label="Top 300",
        edgecolor="white",
    )
    axes[5].set_xlabel("Atoms per cell")
    axes[5].set_ylabel("Structures")
    axes[5].set_title("Cell-size distribution")
    axes[5].legend(frameon=False)
    annotate_panel(axes[5], "F")

    save_figure(fig, outdir, "paper_overview_distributions", formats)


def plot_top300(outdir: Path, formats: Iterable[str], top300: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    axes = axes.ravel()

    ehull = top300["energy_above_hull_eV"].dropna()
    axes[0].hist(ehull, bins=30, color=COLORS["blue"], edgecolor="white", alpha=0.85)
    axes[0].axvline(HULL_CUTOFF_EV, color=COLORS["red"], ls="--", lw=1.3, label=f"{HULL_CUTOFF_EV:.2f} eV/atom cutoff")
    axes[0].set_xlabel("Energy above hull (eV/atom)")
    axes[0].set_ylabel("Top-300 structures")
    axes[0].set_title("CHGNet hull distribution")
    axes[0].legend(frameon=False)
    axes[0].text(
        0.98,
        0.92,
        f"{int((ehull <= HULL_CUTOFF_EV).sum())}/{len(top300)} <= cutoff",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=COLORS["dark"],
    )
    annotate_panel(axes[0], "A")

    near = top300["near_hull_0p05"].fillna(False)
    axes[1].scatter(
        top300.loc[~near, "quick_score"],
        top300.loc[~near, "energy_above_hull_eV"],
        s=18,
        color=COLORS["light_gray"],
        edgecolor="white",
        linewidth=0.3,
        label="> 0.05 eV/atom",
    )
    axes[1].scatter(
        top300.loc[near, "quick_score"],
        top300.loc[near, "energy_above_hull_eV"],
        s=22,
        color=COLORS["red"],
        edgecolor="white",
        linewidth=0.3,
        label="<= 0.05 eV/atom",
    )
    axes[1].axhline(HULL_CUTOFF_EV, color=COLORS["red"], ls="--", lw=1.0)
    axes[1].set_xlabel("Quick score")
    axes[1].set_ylabel("Energy above hull (eV/atom)")
    axes[1].set_title("Score vs. stability")
    axes[1].legend(frameon=False, loc="upper left")
    annotate_panel(axes[1], "B")

    finite = top300.loc[top300["finite_voltage_window"]].copy()
    if len(finite):
        axes[2].hist(finite["window"], bins=np.linspace(0, max(3.0, finite["window"].max() + 0.15), 18), color=COLORS["teal"], edgecolor="white")
        axes[2].axvline(finite["window"].median(), color=COLORS["dark"], lw=1.1, ls=":", label=f"Median = {finite['window'].median():.2f} V")
        axes[2].set_xlabel("Voltage window (V)")
        axes[2].set_ylabel("Near-hull structures")
        axes[2].legend(frameon=False)
    else:
        axes[2].text(0.5, 0.5, "No finite voltage windows", ha="center", va="center")
    axes[2].set_title("Finite voltage windows")
    annotate_panel(axes[2], "C")

    if len(finite):
        top_window = finite.sort_values("window", ascending=False).head(10).copy()
        top_window["label"] = top_window.apply(
            lambda row: f"{row['dataset'].replace('results', 'R')} #{int(row['rank']):03d} {row['formula_plot']}",
            axis=1,
        )
        top_window = top_window.sort_values("window")
        axes[3].barh(top_window["label"], top_window["window"], color=COLORS["orange"])
        axes[3].set_xlabel("Voltage window (V)")
        axes[3].set_title("Largest finite windows")
        for y, value in enumerate(top_window["window"]):
            axes[3].text(value + 0.03, y, f"{value:.2f}", va="center", fontsize=6.5)
        axes[3].set_xlim(0, max(3.1, top_window["window"].max() + 0.35))
    else:
        axes[3].text(0.5, 0.5, "No candidates", ha="center", va="center")
        axes[3].set_axis_off()
    annotate_panel(axes[3], "D")

    save_figure(fig, outdir, "paper_top300_chgnet_voltage", formats)


def plot_batch_metrics(outdir: Path, formats: Iterable[str], batch_metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    axes = axes.ravel()
    batches = batch_metrics["run_label"]

    axes[0].plot(batches, batch_metrics["avg_energy_above_hull_per_atom"], marker="o", color=COLORS["blue"], lw=1.4)
    axes[0].set_ylabel("Mean E above hull (eV/atom)")
    axes[0].set_title("Batch-level stability metric")
    axes[0].tick_params(axis="x", rotation=35)
    annotate_panel(axes[0], "A")

    axes[1].plot(batches, batch_metrics["avg_rmsd_from_relaxation"], marker="o", color=COLORS["orange"], lw=1.4)
    axes[1].set_ylabel("Mean RMSD from relaxation")
    axes[1].set_title("Relaxation displacement")
    axes[1].tick_params(axis="x", rotation=35)
    annotate_panel(axes[1], "B")

    axes[2].plot(batches, batch_metrics["frac_stable_structures"], marker="o", color=COLORS["teal"], lw=1.4, label="Stable")
    axes[2].plot(
        batches,
        batch_metrics["frac_novel_unique_stable_structures"],
        marker="s",
        color=COLORS["red"],
        lw=1.4,
        label="Novel, unique, stable",
    )
    axes[2].set_ylim(0, max(0.22, batch_metrics["frac_stable_structures"].max() + 0.03))
    axes[2].set_ylabel("Fraction")
    axes[2].set_title("Stable and high-value fractions")
    axes[2].legend(frameon=False)
    axes[2].tick_params(axis="x", rotation=35)
    annotate_panel(axes[2], "C")

    axes[3].plot(batches, batch_metrics["avg_comp_validity"], marker="o", color=COLORS["purple"], lw=1.4, label="Composition validity")
    axes[3].plot(batches, batch_metrics["frac_unique_structures"], marker="s", color=COLORS["gray"], lw=1.4, label="Unique structures")
    axes[3].plot(batches, batch_metrics["frac_novel_structures"], marker="^", color=COLORS["blue"], lw=1.4, label="Novel structures")
    axes[3].set_ylim(0.7, 1.02)
    axes[3].set_ylabel("Fraction")
    axes[3].set_title("Validity, uniqueness, novelty")
    axes[3].legend(frameon=False, ncol=1)
    axes[3].tick_params(axis="x", rotation=35)
    annotate_panel(axes[3], "D")

    save_figure(fig, outdir, "paper_batch_metrics", formats)


def plot_feature_correlations(outdir: Path, formats: Iterable[str], stage2: pd.DataFrame) -> None:
    cols = {
        "density": "Density",
        "f_li": "Li fraction",
        "f_o": "O/(O+halide)",
        "li_conn": "Li connectivity",
        "min_li_li_for_plot": "Min Li-Li",
        "quick_score": "Quick score",
    }
    corr = stage2[list(cols)].corr(method="spearman").rename(index=cols, columns=cols)
    fig, ax = plt.subplots(figsize=(4.8, 4.0), constrained_layout=True)
    sns.heatmap(
        corr,
        ax=ax,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        square=True,
        cbar_kws={"label": "Spearman r"},
    )
    ax.set_title("Feature correlations in parsed structures")
    save_figure(fig, outdir, "paper_feature_correlations", formats)


def make_report(
    outdir: Path,
    key_numbers: dict[str, object],
    stage2: pd.DataFrame,
    batch_metrics: pd.DataFrame,
    top300: pd.DataFrame,
) -> None:
    report_path = outdir / "combined_results_analysis_report.md"
    top_formula = stage2["formula"].value_counts().head(5)
    top_spg = stage2["spg"].value_counts().head(5)
    finite = top300.loc[top300["finite_voltage_window"]].sort_values("window", ascending=False)
    top_voltage_lines = []
    for _, row in finite.head(5).iterrows():
        top_voltage_lines.append(
            f"- `{row['dataset']} #{int(row['rank']):03d}` {row['formula_plot']} ({row['spg_plot']}): "
            f"{row['window']:.2f} V, V_red={row['V_red']:.2f} V, V_ox={row['V_ox']:.2f} V"
        )
    if not top_voltage_lines:
        top_voltage_lines = ["- No finite voltage windows were found in the voltage CSV."]

    text = f"""# Combined Li-Nb-O-Cl Distribution Analysis

## Key numbers

- Generated and relaxed structures: {key_numbers['generated_structures']} generated, {key_numbers['relaxed_structures']} relaxed.
- Parsed stage-2 rows: {key_numbers['stage2_rows']} ({key_numbers['stage2_rows'] / key_numbers['generated_structures']:.1%} of generated structures).
- Evaluated candidate rows available from existing top300 runs: {key_numbers['all_evaluated_candidate_rows']}.
- Top-300 quick-score cutoff: {key_numbers['top300_quick_score_threshold']:.4f}.
- CHGNet near-hull global top-300 candidates: {key_numbers['near_hull_0p05_count']} / {key_numbers['top300_rows']} at <= {HULL_CUTOFF_EV:.2f} eV/atom.
- CHGNet stable hull labels: {key_numbers['stable_label_count']} / {key_numbers['top300_rows']}.
- Voltage rows: {key_numbers['voltage_rows']}; finite voltage windows: {key_numbers['finite_voltage_window_count']}.
- Maximum finite voltage window: {key_numbers['max_voltage_window']:.2f} V.

## Batch-level metrics

- Mean batch energy above hull: {key_numbers['mean_batch_avg_ehull']:.3f} eV/atom.
- Mean batch RMSD from relaxation: {key_numbers['mean_batch_rmsd']:.3f}.
- Mean stable fraction: {key_numbers['mean_batch_frac_stable']:.1%}.
- Mean novel, unique, stable fraction: {key_numbers['mean_batch_frac_novel_unique_stable']:.1%}.
- Mean composition validity: {key_numbers['mean_batch_comp_validity']:.1%}.

## Distribution observations

- The most frequent formulas are: {', '.join(f'{k} ({v})' for k, v in top_formula.items())}.
- The most frequent space groups are: {', '.join(f'{k} ({v})' for k, v in top_spg.items())}.
- Median quick score is {stage2['quick_score'].median():.3f}; maximum quick score is {stage2['quick_score'].max():.3f}.
- Median Li fraction is {stage2['f_li'].median():.3f}; median O/(O+halide) is {stage2['f_o'].median():.3f}.
- Structures passing the light-oxygen threshold in `screen_all_extxyz.py`: {key_numbers['light_oxygen_pass_count']} / {key_numbers['stage2_rows']}.
- `min_li_li = 999` appears in {key_numbers['min_li_li_sentinel_count']} rows and indicates structures with fewer than two Li sites in the supercell-based connectivity calculation.

## Best finite voltage-window candidates

{chr(10).join(top_voltage_lines)}

## Figure files

- `figures/paper_overview_distributions.*`: screening funnel, quick-score distribution, composition space, formulas, space groups, cell-size distribution.
- `figures/paper_top300_chgnet_voltage.*`: top-300 hull distribution, score-stability relation, voltage-window distribution, largest finite windows.
- `figures/paper_batch_metrics.*`: batch-to-batch stability, relaxation, validity, novelty, uniqueness.
- `figures/paper_feature_correlations.*`: Spearman correlations among screening features.

## Suggested caption text

**Overview figure.** Distribution of {key_numbers['generated_structures']} generated Li-Nb-O-Cl structures and downstream screening results. The combined campaign produced {key_numbers['relaxed_structures']} relaxed structures, of which {key_numbers['stage2_rows']} were parsed into stage-2 descriptors. The global top 300 candidates were selected by the heuristic quick score, then matched to existing CHGNet hull-energy and electrochemical voltage-window scans.

**Top-300 figure.** CHGNet analysis of the global top-scoring {key_numbers['top300_rows']} candidates. {key_numbers['near_hull_0p05_count']} structures lie within {HULL_CUTOFF_EV:.2f} eV/atom of the convex hull, and {key_numbers['finite_voltage_window_count']} near-hull entries have finite voltage windows in the current voltage scan. The largest finite windows reach {key_numbers['max_voltage_window']:.2f} V.
"""
    report_path.write_text(text)


def main() -> None:
    args = parse_args()
    results_dirs = [path.resolve() for path in args.results_dir]
    outdir = (args.outdir or DEFAULT_ANALYSIS_DIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    configure_plots()
    dataset_labels = make_dataset_labels(args.results_dir)

    inputs: list[tuple[str, Path, dict[str, Path]]] = []
    frame_parts = []
    stage2_parts = []
    batch_parts = []
    for results_dir in results_dirs:
        dataset = dataset_labels[results_dir]
        paths = ensure_inputs(results_dir)
        inputs.append((dataset, results_dir, paths))
        frame_parts.append(count_campaign_frames(results_dir, dataset))
        stage2_parts.append(load_stage2(paths["stage2"], dataset))
        batch_parts.append(load_batch_metrics(results_dir, dataset))

    frame_counts = pd.concat(frame_parts, ignore_index=True)
    stage2 = rank_stage2_global(pd.concat(stage2_parts, ignore_index=True), topk=300)
    batch_metrics = pd.concat(batch_parts, ignore_index=True).sort_values(["dataset", "batch"])

    evaluated_parts = []
    for dataset, _results_dir, paths in inputs:
        top_part, _hull_filtered, _voltage = load_top300(paths, dataset, stage2)
        evaluated_parts.append(top_part)
    evaluated_all = pd.concat(evaluated_parts, ignore_index=True)
    top300 = (
        evaluated_all.loc[evaluated_all["is_top300_by_score"].fillna(False)]
        .copy()
        .sort_values("quick_score_rank")
    )

    expected_top = int(stage2["is_top300_by_score"].sum())
    if len(top300) != expected_top:
        covered = top300[["dataset", "ref"]].drop_duplicates()
        missing = (
            stage2.loc[stage2["is_top300_by_score"], ["dataset", "ref", "quick_score", "quick_score_rank"]]
            .merge(covered, on=["dataset", "ref"], how="left", indicator=True)
            .query("_merge == 'left_only'")
            .drop(columns="_merge")
        )
        missing.to_csv(outdir / "missing_global_top300_evaluations.csv", index=False)
        print(
            f"[WARN] Only {len(top300)} of {expected_top} global top-300 rows have existing evaluations. "
            f"Missing refs were written to {outdir / 'missing_global_top300_evaluations.csv'}"
        )

    funnel = make_funnel(frame_counts, stage2, top300)

    key_numbers = write_tables(outdir, frame_counts, stage2, batch_metrics, evaluated_all, top300, funnel)
    plot_overview(outdir, args.formats, stage2, top300, funnel)
    plot_top300(outdir, args.formats, top300)
    plot_batch_metrics(outdir, args.formats, batch_metrics)
    plot_feature_correlations(outdir, args.formats, stage2)
    make_report(outdir, key_numbers, stage2, batch_metrics, top300)

    print(f"Wrote analysis to {outdir}")
    print(json.dumps(key_numbers, indent=2))


if __name__ == "__main__":
    main()
