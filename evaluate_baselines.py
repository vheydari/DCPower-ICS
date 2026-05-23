#!/usr/bin/env python3
"""
DCPower-ICS — Baseline ML Anomaly Detection Evaluation
=======================================================
Trains and evaluates classical anomaly detection baselines
on the DCPower-ICS dataset, following the SWaT/WADI evaluation
protocol (point-adjust F1 and standard F1/precision/recall).

Baselines implemented
---------------------
  1. Isolation Forest         (unsupervised, tree-based)
  2. One-Class SVM            (unsupervised, kernel)
  3. Local Outlier Factor     (unsupervised, density)
  4. PCA Reconstruction       (unsupervised, linear)
  5. AutoEncoder              (unsupervised, neural — sklearn MLP)

Evaluation metrics
------------------
  - Precision, Recall, F1  (standard point-wise)
  - Point-Adjust F1        (SWaT protocol: if any point in an attack
                             window is detected, all points are credited)
  - Per-scenario F1 breakdown

Usage
-----
  pip install -r requirements.txt
  python evaluate_baselines.py
  python evaluate_baselines.py --data-dir dcpower_dataset
  python evaluate_baselines.py --data-dir dcpower_dataset --models iforest,pca,ae
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

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
# Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "dcpower_train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "dcpower_test.csv"))
    train["attack_scenario"] = train["attack_scenario"].fillna("Normal")
    test["attack_scenario"]  = test["attack_scenario"].fillna("Normal")

    X_train = train[NUMERIC_COLS].values.astype(np.float32)
    y_train = train["label"].values.astype(int)

    X_test  = test[NUMERIC_COLS].values.astype(np.float32)
    y_test  = test["label"].values.astype(int)
    sc_test = test["attack_scenario"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"  Train: {X_train_s.shape}  (all normal)")
    print(f"  Test : {X_test_s.shape}  | attack={y_test.sum():,} ({100*y_test.mean():.1f}%)")

    return X_train_s, y_train, X_test_s, y_test, sc_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    SWaT point-adjust protocol:
    If any point within a contiguous attack window is predicted as attack,
    all points in that window are credited as correctly detected.
    """
    y_adj = y_pred.copy()
    in_attack = False
    window_detected = False
    window_start = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_attack:
            in_attack = True
            window_start = i
            window_detected = False

        if in_attack:
            if y_pred[i] == 1:
                window_detected = True
            if y_true[i] == 0 or i == len(y_true) - 1:
                if window_detected:
                    end = i if y_true[i] == 0 else i + 1
                    y_adj[window_start:end] = 1
                in_attack = False

    return y_adj


def attack_segments(y_true: np.ndarray):
    """Return list of (start, end) for contiguous attack runs."""
    segs, in_seg, s = [], False, 0
    for i, v in enumerate(y_true):
        if v == 1 and not in_seg: in_seg = True; s = i
        elif v == 0 and in_seg:   in_seg = False; segs.append((s, i - 1))
    if in_seg: segs.append((s, len(y_true) - 1))
    return segs


def segment_recall_metric(y_true: np.ndarray, y_pred: np.ndarray):
    """Fraction of contiguous attack windows where detector fires ≥1 tick."""
    segs = attack_segments(y_true)
    if not segs: return 0.0, 0, 0
    det = sum(1 for s, e in segs if y_pred[s:e + 1].any())
    return det / len(segs), det, len(segs)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, sc_labels: np.ndarray,
             model_name: str) -> dict:
    y_adj = point_adjust(y_true, y_pred)
    seg_r, seg_det, seg_tot = segment_recall_metric(y_true, y_pred)

    results = {
        "model":        model_name,
        "precision":    precision_score(y_true, y_pred, zero_division=0),
        "recall":       recall_score(y_true, y_pred, zero_division=0),
        "f1":           f1_score(y_true, y_pred, zero_division=0),
        "f1_pa":        f1_score(y_true, y_adj, zero_division=0),
        "prec_pa":      precision_score(y_true, y_adj, zero_division=0),
        "rec_pa":       recall_score(y_true, y_adj, zero_division=0),
        "seg_recall":   round(seg_r, 4),
        "seg_detected": seg_det,
        "seg_total":    seg_tot,
    }

    # Per-scenario SEGMENT RECALL — correct metric for multi-window evaluation.
    # Per-scenario point F1 is structurally bounded by global precision because
    # all 53K normal rows enter every scenario mask (~1.5K attack rows).
    # Segment recall (did detector fire at least once per window?) is the
    # metric used in ICS anomaly detection papers (SWaT, HAI evaluations).
    sc_sr = {}
    for sc in np.unique(sc_labels[y_true == 1]):
        sc_y = (sc_labels == sc).astype(int)
        r, d, t = segment_recall_metric(sc_y, y_pred)
        sc_sr[sc] = {"seg_recall": round(r, 4), "detected": d, "total": t}
    results["per_scenario_seg_recall"] = sc_sr

    return results


def threshold_from_train(scores: np.ndarray, contamination: float = 0.05) -> float:
    """Set threshold at (1-contamination) quantile of anomaly scores on train set."""
    return np.quantile(scores, 1.0 - contamination)


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

def run_isolation_forest(X_train, X_test, y_test, sc_test, contamination=0.05):
    print("  Training Isolation Forest...")
    t0 = time.time()
    clf = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=42, n_jobs=-1)
    clf.fit(X_train)
    # score_samples: higher = more normal; flip for anomaly score
    train_scores = -clf.score_samples(X_train)
    test_scores  = -clf.score_samples(X_test)
    thr = threshold_from_train(train_scores, contamination)
    y_pred = (test_scores > thr).astype(int)
    print(f"    Done in {time.time()-t0:.1f}s  |  predicted attack={y_pred.sum():,}")
    return evaluate(y_test, y_pred, sc_test, "Isolation Forest"), test_scores


def run_ocsvm(X_train, X_test, y_test, sc_test, contamination=0.05):
    print("  Training One-Class SVM (subsample for speed)...")
    t0 = time.time()
    # OCSVM is O(n²) — subsample train
    max_n = min(10000, len(X_train))
    idx = np.random.RandomState(42).choice(len(X_train), max_n, replace=False)
    clf = OneClassSVM(kernel="rbf", nu=contamination, gamma="scale")
    clf.fit(X_train[idx])
    train_scores = -clf.score_samples(X_train[idx])
    test_scores  = -clf.score_samples(X_test)
    thr = threshold_from_train(train_scores, contamination)
    y_pred = (test_scores > thr).astype(int)
    print(f"    Done in {time.time()-t0:.1f}s  |  predicted attack={y_pred.sum():,}")
    return evaluate(y_test, y_pred, sc_test, "One-Class SVM"), test_scores


def run_lof(X_train, X_test, y_test, sc_test, contamination=0.05):
    print("  Training Local Outlier Factor (novelty mode)...")
    t0 = time.time()
    max_n = min(20000, len(X_train))
    idx = np.random.RandomState(42).choice(len(X_train), max_n, replace=False)
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True,
                             contamination=contamination, n_jobs=-1)
    clf.fit(X_train[idx])
    train_scores = -clf.score_samples(X_train[idx])
    test_scores  = -clf.score_samples(X_test)
    thr = threshold_from_train(train_scores, contamination)
    y_pred = (test_scores > thr).astype(int)
    print(f"    Done in {time.time()-t0:.1f}s  |  predicted attack={y_pred.sum():,}")
    return evaluate(y_test, y_pred, sc_test, "LOF"), test_scores


def run_pca(X_train, X_test, y_test, sc_test, n_components=10, contamination=0.05):
    print(f"  Training PCA Reconstruction (n_components={n_components})...")
    t0 = time.time()
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train)
    expl = pca.explained_variance_ratio_.sum()

    recon_train = pca.inverse_transform(pca.transform(X_train))
    recon_test  = pca.inverse_transform(pca.transform(X_test))
    train_scores = np.mean((X_train - recon_train) ** 2, axis=1)
    test_scores  = np.mean((X_test  - recon_test)  ** 2, axis=1)

    thr = threshold_from_train(train_scores, contamination)
    y_pred = (test_scores > thr).astype(int)
    print(f"    Done in {time.time()-t0:.1f}s  |  explained var={expl:.3f}  |  predicted attack={y_pred.sum():,}")
    return evaluate(y_test, y_pred, sc_test, f"PCA (k={n_components})"), test_scores


def run_autoencoder(X_train, X_test, y_test, sc_test, contamination=0.05):
    print("  Training AutoEncoder (MLP)...")
    t0 = time.time()
    n_feat = X_train.shape[1]
    ae = MLPRegressor(
        hidden_layer_sizes=(32, 16, 8, 16, 32),
        activation="relu",
        solver="adam",
        max_iter=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        verbose=False,
    )
    ae.fit(X_train, X_train)

    recon_train = ae.predict(X_train)
    recon_test  = ae.predict(X_test)
    train_scores = np.mean((X_train - recon_train) ** 2, axis=1)
    test_scores  = np.mean((X_test  - recon_test)  ** 2, axis=1)

    thr = threshold_from_train(train_scores, contamination)
    y_pred = (test_scores > thr).astype(int)
    print(f"    Done in {time.time()-t0:.1f}s  |  predicted attack={y_pred.sum():,}")
    return evaluate(y_test, y_pred, sc_test, "AutoEncoder (MLP)"), test_scores


# ─────────────────────────────────────────────────────────────────────────────
# Results table & plots
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: list):
    print("\n" + "="*80)
    print("  RESULTS SUMMARY")
    print("="*80)
    print(f"  {'Model':<28} {'Prec':>7} {'Rec':>7} {'F1':>7} {'F1-PA':>7} {'Prec-PA':>8} {'Rec-PA':>7}")
    print(f"  {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
    for r in results:
        print(f"  {r['model']:<28} "
              f"{r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} "
              f"{r['f1']:>7.4f} "
              f"{r['f1_pa']:>7.4f} "
              f"{r['prec_pa']:>8.4f} "
              f"{r['rec_pa']:>7.4f}")

    print("\n  Note: F1-PA = Point-Adjust F1 (SWaT protocol)\n")


def print_per_scenario_table(results: list):
    all_scenarios = sorted(set(
        sc for r in results
        for sc in r.get("per_scenario_seg_recall", {}).keys()
    ))
    if not all_scenarios:
        return

    print("\n" + "="*80)
    print("  PER-SCENARIO SEGMENT RECALL")
    print("  (fraction of attack windows where detector fires at least once)")
    print("="*80)

    header = f"  {'Scenario':<32}"
    for r in results:
        name = r['model'][:12]
        header += f" {name:>12}"
    print(header)
    print("  " + "-"*32 + ("-"*13)*len(results))

    for sc in all_scenarios:
        row = f"  {sc:<32}"
        for r in results:
            sr = r.get("per_scenario_seg_recall", {}).get(sc, None)
            val = sr["seg_recall"] if sr else None
            det = f"{sr['detected']}/{sr['total']}" if sr else ""
            row += f" {val:>8.4f} {det:>4}" if val is not None else f" {'N/A':>12}"
        print(row)


def save_results_csv(results: list, out_path: str):
    rows = []
    for r in results:
        base = {k: v for k, v in r.items() if k != "per_scenario_seg_recall"}
        rows.append(base)
        # also add per-scenario rows
        for sc, sr in r.get("per_scenario_seg_recall", {}).items():
            rows.append({"model": r["model"] + f"[{sc}]",
                         "precision": None, "recall": None,
                         "f1": sr["seg_recall"] if isinstance(sr, dict) else sr,
                         "f1_pa": None, "prec_pa": None, "rec_pa": None})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Results saved to: {out_path}")


def make_results_plot(results: list, out_path: str):
    models = [r["model"] for r in results]
    f1s    = [r["f1"]    for r in results]
    f1pas  = [r["f1_pa"] for r in results]
    precs  = [r["precision"] for r in results]
    recs   = [r["recall"] for r in results]

    x = np.arange(len(models))
    w = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("DCPower-ICS Baseline Results", fontsize=13, fontweight="bold")

    # left: grouped bar
    ax = axes[0]
    ax.bar(x - 1.5*w, precs, w, label="Precision", color="#2196F3", alpha=0.85)
    ax.bar(x - 0.5*w, recs,  w, label="Recall",    color="#4CAF50", alpha=0.85)
    ax.bar(x + 0.5*w, f1s,   w, label="F1",        color="#FF9800", alpha=0.85)
    ax.bar(x + 1.5*w, f1pas, w, label="F1-PA",     color="#F44336", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.set_title("Precision / Recall / F1 / F1-PA per Model")
    ax.grid(axis="y", alpha=0.3)

    # right: per-scenario segment recall heatmap
    all_scenarios = sorted(set(
        sc for r in results for sc in r.get("per_scenario_seg_recall", {}).keys()
    ))
    if all_scenarios:
        mat = np.full((len(all_scenarios), len(results)), np.nan)
        for j, r in enumerate(results):
            for i, sc in enumerate(all_scenarios):
                sr = r.get("per_scenario_seg_recall", {}).get(sc, None)
                mat[i, j] = sr["seg_recall"] if isinstance(sr, dict) else (float(sr) if sr is not None else np.nan)

        ax2 = axes[1]
        im = ax2.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax2.set_xticks(range(len(results)))
        ax2.set_xticklabels([r["model"] for r in results], rotation=25, ha="right", fontsize=7)
        ax2.set_yticks(range(len(all_scenarios)))
        ax2.set_yticklabels(all_scenarios, fontsize=7)
        plt.colorbar(im, ax=ax2, shrink=0.8, label="Segment Recall")
        ax2.set_title("Per-Scenario Segment Recall")

        for i in range(len(all_scenarios)):
            for j in range(len(results)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                             fontsize=5.5, color="black" if v > 0.4 else "white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", default="dcpower_dataset")
    p.add_argument("--out-dir",  default=None,
                   help="Output directory for results (default: <data-dir>/baseline_results)")
    p.add_argument("--models", default="iforest,ocsvm,lof,pca,ae",
                   help="Comma-separated list of models to run")
    p.add_argument("--contamination", type=float, default=0.05,
                   help="Training-score threshold tail probability; 0.05 uses the 95th percentile")
    p.add_argument("--pca-k", type=int, default=10,
                   help="Number of PCA components")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.data_dir, "baseline_results")
    os.makedirs(out_dir, exist_ok=True)

    models_to_run = set(m.strip().lower() for m in args.models.split(","))

    print(f"\n  DCPower-ICS Baseline ML Evaluation")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Out dir  : {out_dir}")
    print(f"  Models   : {models_to_run}")

    print("\n" + "="*60)
    print("  Loading data...")
    print("="*60)
    X_train, y_train, X_test, y_test, sc_test, scaler = load_data(args.data_dir)

    results = []

    print(f"  Training-score threshold tail probability: {args.contamination:.4f}")

    valid_models = {"iforest", "ocsvm", "lof", "pca", "ae"}
    unknown = models_to_run - valid_models
    if unknown:
        print(f"  Warning: ignoring unsupported model name(s): {sorted(unknown)}")

    if "iforest" in models_to_run:
        r, _ = run_isolation_forest(X_train, X_test, y_test, sc_test, args.contamination)
        results.append(r)

    if "ocsvm" in models_to_run:
        r, _ = run_ocsvm(X_train, X_test, y_test, sc_test, args.contamination)
        results.append(r)

    if "lof" in models_to_run:
        r, _ = run_lof(X_train, X_test, y_test, sc_test, args.contamination)
        results.append(r)

    if "pca" in models_to_run:
        r, _ = run_pca(X_train, X_test, y_test, sc_test, args.pca_k, args.contamination)
        results.append(r)

    if "ae" in models_to_run:
        r, _ = run_autoencoder(X_train, X_test, y_test, sc_test, args.contamination)
        results.append(r)

    print_results_table(results)
    print_per_scenario_table(results)
    save_results_csv(results, os.path.join(out_dir, "baseline_results.csv"))

    if not args.no_plots and results:
        make_results_plot(results, os.path.join(out_dir, "baseline_results.png"))

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
