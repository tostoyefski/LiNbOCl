#!/usr/bin/env python3
import argparse
import csv
import math
import re

import matplotlib.pyplot as plt


def read_rows(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "formula": row["formula"],
                    "V_red": float(row["V_red"]),
                    "V_ox": float(row["V_ox"]),
                    "window": float(row["window"]),
                }
            )
    return rows


def sort_rows(rows, sort_key, descending):
    if not sort_key:
        return rows
    return sorted(rows, key=lambda r: r[sort_key], reverse=descending)


def format_formula(formula):
    return re.sub(r"(\d+)", r"$_{\1}$", formula)


def plot_voltage_window(rows, out_path, show):
    labels = [format_formula(r["formula"]) for r in rows]
    left = [r["V_red"] for r in rows]
    widths = [r["V_ox"] - r["V_red"] for r in rows]

    fig_h = max(3.0, 0.45 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(6.0, fig_h))

    ax.barh(
        range(len(rows)),
        widths,
        left=left,
        color="#5aa6d1",
        edgecolor="white",
        alpha=0.9,
    )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Stability Window (V vs. Li$^+$/Li)")
    max_v = max(r["V_ox"] for r in rows)
    ax.set_xlim(0, math.ceil(max_v + 0.2))
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot voltage window bars from a CSV table."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="chgnet_voltage_window_top300.csv",
        help="Input CSV path (positional)",
    )
    parser.add_argument(
        "out_path",
        nargs="?",
        default="chgnet_voltage_window_top300.png",
        help="Output image path (positional)",
    )
    parser.add_argument(
        "--csv",
        default="chgnet_voltage_window_top300.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--out",
        default="chgnet_voltage_window_top300.png",
        help="Output image path",
    )
    parser.add_argument(
        "--sort",
        choices=["V_red", "V_ox", "window"],
        default=None,
        help="Sort rows by a column",
    )
    parser.add_argument(
        "--desc",
        action="store_true",
        help="Sort in descending order",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window")
    args = parser.parse_args()

    csv_path = (
        args.csv
        if args.csv != parser.get_default("csv")
        else args.csv_path
    )
    out_path = (
        args.out
        if args.out != parser.get_default("out")
        else args.out_path
    )

    rows = read_rows(csv_path)
    rows = sort_rows(rows, args.sort, args.desc)
    plot_voltage_window(rows, out_path, args.show)


if __name__ == "__main__":
    main()
