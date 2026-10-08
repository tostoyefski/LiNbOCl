#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

TRANSPORT_RESULTS = Path(__file__).resolve().parents[2] / "results" / "transport"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_list(raw: str) -> list[float]:
    if raw is None:
        return []
    items = [item.strip() for item in raw.split(";") if item.strip()]
    values = []
    for item in items:
        try:
            values.append(float(item))
        except ValueError:
            continue
    return values


def sanitize_name(name: str) -> str:
    safe = name.replace("/", "_").replace("\\", "_").strip()
    return safe if safe else "unknown"

def extract_label_name(file_name: str) -> str:
    stem = sanitize_name(Path(file_name).stem)
    tokens = [token for token in stem.split("_") if token]
    for token in tokens:
        if token.lower().startswith("cand"):
            continue
        if token.isdigit():
            continue
        if any(ch.isdigit() for ch in token):
            return token
    for token in tokens:
        if token.lower().startswith("cand"):
            continue
        if any(ch.isupper() for ch in token):
            return token
    return stem


def extract_xy(row: dict, x_scale: float) -> tuple[list[float], list[float]]:
    temps = parse_list(row.get("temps_K", ""))
    sigmas = parse_list(row.get("sigma_S_m_mean", ""))

    if not temps or not sigmas:
        return [], []

    if len(temps) != len(sigmas):
        n = min(len(temps), len(sigmas))
        temps = temps[:n]
        sigmas = sigmas[:n]

    x_vals = []
    y_vals = []
    for temp, sigma in zip(temps, sigmas):
        if temp <= 0 or sigma <= 0:
            continue
        inv_t = (x_scale / temp)
        x_vals.append(inv_t)
        y_vals.append(math.log(sigma * temp))

    return x_vals, y_vals


def format_value(raw: str, digits: int = 3) -> str:
    if raw is None or raw == "":
        return ""
    try:
        value = float(raw)
    except ValueError:
        return str(raw)
    return f"{value:.{digits}g}"


def plot_combined(rows: list[dict], output_dir: Path, x_scale: float, fmt: str, dpi: int) -> int:
    materials = []
    for row in rows:
        x_vals, y_vals = extract_xy(row, x_scale)
        if len(x_vals) < 2:
            continue
        file_name = row.get("file", "unknown")
        label_name = extract_label_name(file_name)
        materials.append(
            {
                "name": label_name,
                "x": x_vals,
                "y": y_vals,
                "slope": row.get("slope_ln_sigmaT_vs_invT", ""),
                "intercept": row.get("intercept", ""),
                "ea": row.get("Ea_eV", ""),
                "r2": row.get("r2", ""),
            }
        )

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("hsv")
    total = len(materials)

    label_specs = []
    for idx, mat in enumerate(materials):
        color = cmap(0.0 if total <= 1 else idx / (total - 1))
        ax.scatter(mat["x"], mat["y"], s=20, color=color, alpha=0.7, edgecolors="none")

        slope_raw = mat["slope"]
        intercept_raw = mat["intercept"]
        try:
            slope = float(slope_raw) if slope_raw != "" else None
            intercept = float(intercept_raw) if intercept_raw != "" else None
        except ValueError:
            slope = None
            intercept = None

        if slope is not None and intercept is not None:
            slope_plot = slope / x_scale
            x_min, x_max = min(mat["x"]), max(mat["x"])
            y_min = slope_plot * x_min + intercept
            y_max = slope_plot * x_max + intercept
            ax.plot([x_min, x_max], [y_min, y_max], color=color, linewidth=1.4, alpha=0.95)

            label_parts = [mat["name"]]
            ea = format_value(mat["ea"])
            r2 = format_value(mat["r2"])
            if ea:
                label_parts.append(f"Ea={ea}eV")
            if r2:
                label_parts.append(f"R2={r2}")
            label = " ".join(label_parts)
            label_specs.append(
                {
                    "label": label,
                    "color": color,
                    "x": x_max,
                    "y": y_max,
                }
            )

    x_label = "1/T (K$^{-1}$)" if x_scale == 1.0 else "1000/T (K$^{-1}$)"
    ax.set_xlabel(x_label)
    ax.set_ylabel("ln(sigma*T)")
    ax.set_title("Arrhenius plots (all materials)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.margins(x=0.25)

    if label_specs:
        y_values = [spec["y"] for spec in label_specs]
        y_min_data = min(y_values)
        y_max_data = max(y_values)
        y_span = y_max_data - y_min_data if y_max_data > y_min_data else 1.0
        min_sep = 0.04 * y_span
        label_specs.sort(key=lambda item: item["y"])
        for idx, spec in enumerate(label_specs):
            if idx == 0:
                continue
            prev = label_specs[idx - 1]
            if spec["y"] - prev["y"] < min_sep:
                spec["y"] = prev["y"] + min_sep

        x_vals_all = [spec["x"] for spec in label_specs]
        x_span = (max(x_vals_all) - min(x_vals_all)) if len(x_vals_all) > 1 else 1.0
        x_offset = 0.02 * x_span

        for spec in label_specs:
            ax.annotate(
                spec["label"],
                xy=(spec["x"], spec["y"]),
                xytext=(spec["x"] + x_offset, spec["y"]),
                textcoords="data",
                fontsize=8,
                color=spec["color"],
                ha="left",
                va="center",
                arrowprops=dict(arrowstyle="-", color=spec["color"], linewidth=0.8, alpha=0.9),
            )

        y_top = max(spec["y"] for spec in label_specs) + 0.05 * y_span
        current_ylim = ax.get_ylim()
        if y_top > current_ylim[1]:
            ax.set_ylim(current_ylim[0], y_top)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"arrhenius_all.{fmt}"
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a combined Arrhenius ln(sigma*T) vs 1/T for all materials."
    )
    parser.add_argument(
        "--input",
        default=TRANSPORT_RESULTS / "chgnet_ionic_conductivity_arrhenius.csv",
        type=Path,
        help="Path to chgnet_ionic_conductivity_arrhenius.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for plots (default: alongside input).",
    )
    parser.add_argument(
        "--x-scale",
        type=float,
        default=1.0,
        choices=[1.0, 1000.0],
        help="Use 1/T or 1000/T on the x-axis.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Image format for saved plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for raster formats.",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_dir = args.output if args.output else input_path.parent / "arrhenius_plots"

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    plotted = plot_combined(rows, output_dir, args.x_scale, args.format, args.dpi)
    print(f"Saved combined plot to {output_dir} (from {len(rows)} rows, {plotted} materials).")


if __name__ == "__main__":
    main()
