#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "results/top300_run/chgnet_voltage_window_top300.csv"
DEFAULT_OUT = REPO_ROOT / "results/analysis/chgnet_voltage_window_top300.png"


def finite_number(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def censor_flag(value):
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"", "false", "0", "no"}:
        return False
    raise ValueError(f"invalid censor flag: {value}")


def parse_interval(interval):
    lower = finite_number(interval["V_red"], "V_red")
    upper = finite_number(interval["V_ox"], "V_ox")
    width = finite_number(interval.get("window", upper - lower), "window")
    if upper < lower or width < 0:
        raise ValueError("invalid stable interval bounds")
    if not math.isclose(width, upper - lower, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("window disagrees with voltage bounds")
    return {
        "V_red": lower,
        "V_ox": upper,
        "window": width,
        "lower_bound_censored": censor_flag(interval.get("lower_bound_censored")),
        "upper_bound_censored": censor_flag(interval.get("upper_bound_censored")),
    }


def read_rows(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader, 1):
            label = row.get("formula") or row.get("file") or f"row {index}"
            try:
                status = row.get("window_status")
                if "window_status" in row and status not in {"stable_window", "scan_censored"}:
                    raise ValueError(f"window_status={status}: {row.get('error') or 'no valid stable window'}")
                if row.get("error"):
                    raise ValueError(row["error"])
                if "passes_voltage_filter" in row and not censor_flag(row["passes_voltage_filter"]):
                    raise ValueError("passes_voltage_filter is not true")
                intervals_json = row.get("stable_intervals_json")
                if intervals_json:
                    decoded = json.loads(intervals_json)
                    if not isinstance(decoded, list) or not decoded:
                        raise ValueError("no stable intervals recorded")
                    intervals = [parse_interval(interval) for interval in decoded]
                else:
                    # Manual legacy tables contain only the three scalar columns.
                    intervals = [parse_interval(row)]
                if not any(interval["window"] > 0 for interval in intervals):
                    raise ValueError("stable samples have no positive verified window width")
                primary = min(intervals, key=lambda interval: (-interval["window"], interval["V_red"]))
                rows.append({"formula": label, **primary, "stable_intervals": intervals})
            except (KeyError, TypeError, ValueError) as exc:
                print(f"[SKIP] {label}: {exc}")
    return rows


def sort_rows(rows, sort_key, descending):
    if not sort_key:
        return rows
    return sorted(rows, key=lambda r: r[sort_key], reverse=descending)


def format_formula(formula):
    return re.sub(r"(\d+)", r"$_{\1}$", formula)


def plot_voltage_window(rows, out_path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    labels = [format_formula(r["formula"]) for r in rows]

    fig_h = max(3.0, 0.45 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(6.0, fig_h))

    intervals = []
    has_censored = False
    for index, row in enumerate(rows):
        for interval in row.get("stable_intervals", [row]):
            intervals.append(interval)
            if interval["window"] > 0:
                ax.barh(index, interval["window"], left=interval["V_red"], color="#5aa6d1", edgecolor="white", alpha=0.9)
            else:
                ax.scatter(interval["V_red"], index, marker="|", color="#23495d", zorder=3)
            for field, bound, marker in (
                ("lower_bound_censored", "V_red", "<"),
                ("upper_bound_censored", "V_ox", ">"),
            ):
                if interval.get(field):
                    has_censored = True
                    ax.scatter(interval[bound], index, marker=marker, color="#23495d", zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Stability Window (V vs. Li$^+$/Li)")
    if intervals:
        min_v = min(interval["V_red"] for interval in intervals)
        max_v = max(interval["V_ox"] for interval in intervals)
        ax.set_xlim(min(0, math.floor(min_v - 0.2)), math.ceil(max_v + 0.2))
    else:
        ax.text(0.5, 0.5, "No valid stability windows", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        print("[INFO] No valid stability windows to plot.")
    if has_censored:
        ax.legend(handles=[Line2D([], [], marker=">", linestyle="None", color="#23495d", label="Scan limit (boundary unobserved)")], loc="best", fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot voltage window bars from a CSV table."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="Input CSV path (positional)",
    )
    parser.add_argument(
        "out_path",
        nargs="?",
        default=None,
        help="Output image path (positional)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Input CSV path",
    )
    parser.add_argument(
        "--out",
        default=None,
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
    args = parser.parse_args(argv)
    args.csv = args.csv or args.csv_path or DEFAULT_CSV
    args.out = args.out or args.out_path or DEFAULT_OUT
    return args


def main(argv=None):
    args = parse_args(argv)

    rows = read_rows(args.csv)
    rows = sort_rows(rows, args.sort, args.desc)
    plot_voltage_window(rows, args.out, args.show)


if __name__ == "__main__":
    main()
