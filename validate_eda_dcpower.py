#!/usr/bin/env python3
"""
DCPower-ICS Dataset — Validation & EDA
=======================================
Checks data integrity, label distribution, per-scenario stats,
sensor statistics, and generates publication-ready plots.

Usage
-----
  python validate_eda_dcpower.py                          # default: dcpower_dataset/
  python validate_eda_dcpower.py --data-dir dcpower_dataset_full
  python validate_eda_dcpower.py --data-dir dcpower_dataset --no-plots
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ── colour palette (SWaT-paper style) ────────────────────────────────────────
C_NORMAL  = "#2196F3"
C_ATTACK  = "#F44336"
C_ACCENT  = "#FF9800"
C_GREY    = "#9E9E9E"

NUMERIC_COLS = [
    "utility_available", "utility_degraded",
    "PCC_breaker", "PCC_breaker_cmd",
    "GEN1_cmd", "GEN2_cmd",
    "GEN1_running", "GEN2_running",
    "GEN1_ready", "GEN2_ready",
    "GEN1_fault", "GEN2_fault",
    "GEN1_breaker", "GEN2_breaker",
    "GEN1_P", "GEN2_P", "GEN1_Q", "GEN2_Q",
    "GEN1_f", "GEN2_f", "GEN1_V", "GEN2_V",
    "UPS_bypass", "UPS_in_P", "UPS_out_P",
    "BESS_cmd", "BESS_P", "BESS_SOC", "BESS_current",
    "P_IT", "P_cooling", "P_noncritical",
    "load_shed_stage",
    "bus_voltage", "bus_frequency",
    "PCC_P", "PCC_Q", "PCC_I",
    "ambient_c", "maintenance_bypass",
]


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load(data_dir: str):
    train_path = os.path.join(data_dir, "dcpower_train.csv")
    test_path  = os.path.join(data_dir, "dcpower_test.csv")
    meta_path  = os.path.join(data_dir, "dcpower_meta.json")

    for p in [train_path, test_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Pass --data-dir pointing to your dataset folder.")
            sys.exit(1)

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    meta  = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    # normalise attack_scenario column
    train["attack_scenario"] = train["attack_scenario"].fillna("Normal")
    test["attack_scenario"]  = test["attack_scenario"].fillna("Normal")

    return train, test, meta


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Basic integrity checks
# ─────────────────────────────────────────────────────────────────────────────

def check_integrity(train: pd.DataFrame, test: pd.DataFrame) -> bool:
    print("\n" + "="*60)
    print("  SECTION 1 — DATA INTEGRITY")
    print("="*60)
    ok = True

    for name, df in [("train", train), ("test", test)]:
        print(f"\n  [{name.upper()}]  {len(df):,} rows × {len(df.columns)} cols")

        # missing values in numeric cols
        num_nulls = df[NUMERIC_COLS].isnull().sum().sum()
        print(f"    Numeric nulls         : {num_nulls}")
        if num_nulls > 0:
            print(f"    !! Null columns: {df[NUMERIC_COLS].isnull().sum()[df[NUMERIC_COLS].isnull().sum()>0].to_dict()}")
            ok = False

        # timestamp monotonic
        mono = df["timestamp"].is_monotonic_increasing
        print(f"    Timestamp monotonic   : {mono}")
        if not mono:
            ok = False

        # duplicate timestamps
        dupes = df["timestamp"].duplicated().sum()
        print(f"    Duplicate timestamps  : {dupes}")

        # label values
        label_vals = sorted(df["label"].unique())
        print(f"    Label values          : {label_vals}")
        if set(label_vals) - {0, 1}:
            print("    !! Unexpected label values")
            ok = False

        # timestamp gap uniformity
        diffs = df["timestamp"].diff().dropna()
        dt_unique = diffs.round(1).unique()
        print(f"    Timestep values (s)   : {dt_unique}")

    # train should be all-normal
    train_attacks = (train["label"] == 1).sum()
    print(f"\n  Train attack rows (should be 0) : {train_attacks}")
    if train_attacks > 0:
        ok = False

    print(f"\n  Integrity check: {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Label & scenario distribution
# ─────────────────────────────────────────────────────────────────────────────

def label_distribution(test: pd.DataFrame, meta: dict) -> pd.DataFrame:
    print("\n" + "="*60)
    print("  SECTION 2 — LABEL & SCENARIO DISTRIBUTION")
    print("="*60)

    n_total  = len(test)
    n_normal = (test["label"] == 0).sum()
    n_attack = (test["label"] == 1).sum()
    print(f"\n  Test split  |  total={n_total:,}  normal={n_normal:,} ({100*n_normal/n_total:.1f}%)  attack={n_attack:,} ({100*n_attack/n_total:.1f}%)")

    # per-scenario
    sc_rows = test[test["label"] == 1].groupby("attack_scenario").size().rename("rows")
    sc_pct  = (sc_rows / n_total * 100).round(2).rename("pct_of_test")

    # count attack windows (consecutive runs)
    test2 = test.copy()
    test2["window"] = (test2["attack_scenario"] != test2["attack_scenario"].shift()).cumsum()
    window_counts = (
        test2[test2["label"] == 1]
        .groupby("attack_scenario")["window"]
        .nunique()
        .rename("windows")
    )

    sc_df = pd.concat([sc_rows, sc_pct, window_counts], axis=1).sort_values("rows", ascending=False)
    print(f"\n  Per-scenario breakdown ({len(sc_df)} scenarios):")
    print(f"  {'Scenario':<30} {'Rows':>8} {'% Test':>8} {'Windows':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for sc, row in sc_df.iterrows():
        print(f"  {sc:<30} {int(row['rows']):>8,} {row['pct_of_test']:>8.2f} {int(row['windows']):>8}")

    all_scenarios = list(meta.get("scenario_catalogue", {}).keys())
    if all_scenarios:
        missing = set(all_scenarios) - set(sc_df.index)
        if missing:
            print(f"\n  !! Scenarios NOT in test: {missing}")
        else:
            print(f"\n  All {len(all_scenarios)} scenarios present in test ✓")

    return sc_df


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Sensor statistics
# ─────────────────────────────────────────────────────────────────────────────

def sensor_stats(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("  SECTION 3 — SENSOR STATISTICS (train vs test-normal)")
    print("="*60)

    test_normal = test[test["label"] == 0]

    stats = []
    for col in NUMERIC_COLS:
        tr = train[col]
        te = test_normal[col]
        stats.append({
            "sensor": col,
            "train_mean": tr.mean(),
            "train_std":  tr.std(),
            "train_min":  tr.min(),
            "train_max":  tr.max(),
            "test_mean":  te.mean(),
            "test_std":   te.std(),
            "mean_shift": abs(tr.mean() - te.mean()),
        })
    df = pd.DataFrame(stats).set_index("sensor")

    # flag sensors with large mean shift (possible concern)
    threshold = df["train_std"].clip(lower=0.01)
    df["norm_shift"] = df["mean_shift"] / threshold
    flagged = df[df["norm_shift"] > 2].sort_values("norm_shift", ascending=False)

    print(f"\n  Sensors with mean shift > 2σ between train and test-normal:")
    if flagged.empty:
        print("    None — distributions look consistent ✓")
    else:
        for s, row in flagged.iterrows():
            print(f"    {s:<30}  shift={row['mean_shift']:.3f}  ({row['norm_shift']:.1f}σ)")

    print(f"\n  Sensor value ranges (train):")
    print(f"  {'Sensor':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for s, row in df.iterrows():
        print(f"  {s:<25} {row['train_mean']:>10.3f} {row['train_std']:>10.3f} {row['train_min']:>10.3f} {row['train_max']:>10.3f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Plots
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(train: pd.DataFrame, test: pd.DataFrame, sc_df: pd.DataFrame, out_dir: str):
    print("\n" + "="*60)
    print("  SECTION 4 — GENERATING PLOTS")
    print("="*60)

    os.makedirs(out_dir, exist_ok=True)

    # ── Plot 1: Label timeline ───────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    fig.suptitle("DCPower-ICS Dataset Overview", fontsize=14, fontweight="bold")

    key_sensors = ["PCC_P", "bus_voltage", "BESS_SOC"]
    labels_arr  = test["label"].values
    t_arr       = test["timestamp"].values / 3600  # hours

    for ax, col in zip(axes, key_sensors):
        vals = test[col].values
        ax.plot(t_arr, vals, color=C_GREY, linewidth=0.4, alpha=0.8)
        ax.fill_between(t_arr, vals.min(), vals.max(),
                        where=(labels_arr == 1), alpha=0.25, color=C_ATTACK, label="Attack")
        ax.set_ylabel(col, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (hours)", fontsize=9)
    plt.tight_layout()
    p = os.path.join(out_dir, "fig1_timeline.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")

    # ── Plot 2: Per-scenario bar chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    sc_sorted = sc_df.sort_values("rows")
    colors = [C_ATTACK] * len(sc_sorted)
    bars = ax.barh(sc_sorted.index, sc_sorted["rows"], color=colors, alpha=0.85)
    ax.set_xlabel("Attack rows in test split", fontsize=10)
    ax.set_title("Per-Scenario Attack Row Counts", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)
    for bar, (_, row) in zip(bars, sc_sorted.iterrows()):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f"{int(row['rows']):,}  ({row['pct_of_test']:.1f}%)",
                va="center", fontsize=7)
    plt.tight_layout()
    p = os.path.join(out_dir, "fig2_scenario_distribution.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")

    # ── Plot 3: Sensor distributions normal vs attack ────────────────────────
    plot_sensors = ["PCC_P", "bus_voltage", "bus_frequency", "BESS_SOC",
                    "GEN1_P", "P_IT", "P_cooling", "UPS_in_P"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle("Sensor Distributions: Normal vs Attack", fontsize=13, fontweight="bold")

    test_norm = test[test["label"] == 0]
    test_atk  = test[test["label"] == 1]

    for ax, col in zip(axes.flat, plot_sensors):
        vmin = test[col].quantile(0.01)
        vmax = test[col].quantile(0.99)
        bins = np.linspace(vmin, vmax, 50)
        ax.hist(test_norm[col].clip(vmin, vmax), bins=bins, alpha=0.6,
                color=C_NORMAL, density=True, label="Normal")
        ax.hist(test_atk[col].clip(vmin, vmax),  bins=bins, alpha=0.6,
                color=C_ATTACK, density=True, label="Attack")
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    plt.tight_layout()
    p = os.path.join(out_dir, "fig3_sensor_distributions.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")

    # ── Plot 4: Correlation heatmap (train) ──────────────────────────────────
    cont_sensors = [c for c in NUMERIC_COLS
                    if train[c].nunique() > 10]
    corr = train[cont_sensors].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cont_sensors)))
    ax.set_yticks(range(len(cont_sensors)))
    ax.set_xticklabels(cont_sensors, rotation=90, fontsize=7)
    ax.set_yticklabels(cont_sensors, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title("Sensor Correlation Matrix (Train — Normal)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(out_dir, "fig4_correlation_heatmap.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", default="dcpower_dataset")
    p.add_argument("--plot-dir", default=None, help="Where to save plots (default: <data-dir>/eda_plots)")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    plot_dir = args.plot_dir or os.path.join(args.data_dir, "eda_plots")

    print(f"\n  DCPower-ICS Validation & EDA")
    print(f"  Data dir : {args.data_dir}")

    train, test, meta = load(args.data_dir)

    check_integrity(train, test)
    sc_df = label_distribution(test, meta)
    sensor_stats(train, test)

    if not args.no_plots:
        try:
            make_plots(train, test, sc_df, plot_dir)
            print(f"\n  All plots saved to: {plot_dir}/")
        except Exception as e:
            print(f"\n  Plot error (non-fatal): {e}")

    print("\n  EDA complete.\n")


if __name__ == "__main__":
    main()
