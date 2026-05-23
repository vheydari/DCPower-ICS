#!/usr/bin/env python3
"""
DCPower-ICS Dataset Generator
==============================
Generates the DCPower-ICS benchmark dataset for data center power infrastructure.

DESIGN
------
The released benchmark follows the two-part design described in the paper:

1. Training split: 24 hours of normal operation with planned maintenance
   events. Generator load tests, UPS bypass windows, and load shed drills are
   all labeled normal (label=0).
2. Test split: pure grid-connected normal windows alternating with labeled
   fault/anomaly windows. Test normal windows do not include planned
   maintenance events, following the SWaT/WADI-style fault-detection protocol.
3. Maintenance robustness: evaluated separately using maintenance windows in
   the training split.

LABEL CONVENTION
----------------
  label = 0  -> Normal operation, including planned maintenance in train
  label = 1  -> Active fault/anomaly scenario in test
  attack_scenario = scenario code when label=1, "Normal" when label=0

Usage
-----
  python generate_dataset.py --train-hours 24 --test-hours 24 --seed 42 --out-dir dcpower_dataset
  python generate_dataset.py --train-hours 6 --test-hours 6 --out-dir dcpower_dataset_small
  python generate_dataset.py --help
"""


import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import plant_model as pm
    import scenario_library as sl
except ModuleNotFoundError:
    print("ERROR: plant_model.py / scenario_library.py not found.")
    sys.exit(1)

ALL_SCENARIOS: List[str] = list(sl.SCENARIOS.keys())

NUMERIC_FIELDS = [
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
ALL_COLUMNS = ["timestamp"] + NUMERIC_FIELDS + ["label", "attack_scenario"]

# ── Maintenance event definitions ─────────────────────────────────────────────
# SHORT events fit within a normal window (120-300s)
SHORT_EVENTS = [
    {"name": "ups_bypass",     "min_s": 60,  "max_s": 120, "settle_s": 0},
    {"name": "load_shed_test", "min_s": 60,  "max_s": 120, "settle_s": 30},
]
# LONG events get dedicated standalone windows (labeled normal, no attack follows)
LONG_EVENT = {"name": "generator_test", "min_s": 300, "max_s": 900, "settle_s": 60}
# How often a dedicated generator-test window is inserted (seconds)
GEN_TEST_INTERVAL_S = 45 * 60   # every ~45 minutes


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fresh_state(seed: int) -> pm.PowerPlantState:
    s = pm.PowerPlantState()
    s.rng = random.Random(seed)
    return s


def _warmup(state, seconds, dt):
    for _ in range(int(seconds / dt)):
        pm.step(state, dt)


def _row(obs, t, label, scenario):
    r = {"timestamp": round(t, 1)}
    for col in NUMERIC_FIELDS:
        v = obs.get(col, 0.0)
        r[col] = round(float(v), 4) if isinstance(v, (int, float)) else 0.0
    r["label"] = label
    r["attack_scenario"] = scenario if scenario else "Normal"
    return r


def _step_normal(state, dt, t, rows):
    obs = pm.step(state, dt)
    rows.append(_row(obs, t, 0, ""))
    return t + dt


# ─────────────────────────────────────────────────────────────────────────────
# Maintenance event runners (all produce label=0)
# ─────────────────────────────────────────────────────────────────────────────

def _run_generator_test(state, duration_s, dt, t, rows, log):
    state.utility_available = False
    state.utility_degraded  = True
    state.ats_timer         = 0.0
    t_start = t
    for _ in range(int(duration_s / dt)):
        obs = pm.step(state, dt)
        rows.append(_row(obs, t, 0, ""))
        t += dt
    state.utility_available = True
    state.utility_degraded  = False
    state.ats_timer         = 0.0
    for _ in range(int(LONG_EVENT["settle_s"] / dt)):
        obs = pm.step(state, dt)
        rows.append(_row(obs, t, 0, ""))
        t += dt
    log.append({"type": "generator_test", "start_s": round(t_start, 1),
                "duration_s": duration_s})
    return t


def _run_ups_bypass(state, duration_s, dt, t, rows, log):
    t_start = t
    state.health.maintenance_bypass = True
    for _ in range(int(duration_s / dt)):
        obs = pm.step(state, dt)
        rows.append(_row(obs, t, 0, ""))
        t += dt
    state.health.maintenance_bypass = False
    log.append({"type": "ups_bypass", "start_s": round(t_start, 1),
                "duration_s": duration_s})
    return t


def _run_load_shed_test(state, duration_s, dt, t, rows, log):
    t_start = t
    state.utility_degraded  = True
    state.utility_available = False
    state.ats_timer         = 0.0
    for _ in range(int(duration_s / dt)):
        obs = pm.step(state, dt)
        rows.append(_row(obs, t, 0, ""))
        t += dt
    state.utility_available = True
    state.utility_degraded  = False
    state.ats_timer         = 0.0
    for _ in range(int(30 / dt)):
        obs = pm.step(state, dt)
        rows.append(_row(obs, t, 0, ""))
        t += dt
    log.append({"type": "load_shed_test", "start_s": round(t_start, 1),
                "duration_s": duration_s})
    return t


RUNNERS = {
    "generator_test": _run_generator_test,
    "ups_bypass":     _run_ups_bypass,
    "load_shed_test": _run_load_shed_test,
}


# ─────────────────────────────────────────────────────────────────────────────
# Train split — long runs with maintenance events
# ─────────────────────────────────────────────────────────────────────────────

def generate_train(hours, dt, seed, warmup_s=300, verbose=True):
    if verbose:
        print(f"\n{'='*60}")
        print(f"  TRAIN SPLIT  |  {hours}h  |  dt={dt}s  |  seed={seed}")
        print(f"{'='*60}")

    state = _fresh_state(seed)
    rng   = random.Random(seed + 100)
    if verbose: print(f"  Warmup {warmup_s}s...", end="", flush=True)
    _warmup(state, warmup_s, dt)
    if verbose: print(" done.")

    total_s   = hours * 3600
    t         = 0.0
    rows      = []
    event_log = []
    next_ev   = rng.uniform(600, 1800)
    last_type = None
    t0 = time.time()

    while t < total_s:
        obs = pm.step(state, dt)
        rows.append(_row(obs, t, 0, ""))
        t += dt

        if t >= next_ev and t + 400 < total_s:
            # choose event (alternate between short and long)
            if last_type != "generator_test" and rng.random() < 0.4:
                ev = LONG_EVENT
            else:
                ev = rng.choice([e for e in SHORT_EVENTS if e["name"] != last_type]
                                 or SHORT_EVENTS)
            dur = rng.randint(ev["min_s"], ev["max_s"])
            t   = RUNNERS[ev["name"]](state, dur, dt, t, rows, event_log)
            last_type = ev["name"]
            next_ev   = t + rng.uniform(600, 1800)
            if verbose:
                print(f"    [{100*t/total_s:5.1f}%] Maintenance: {ev['name']} ({dur}s)")

        if verbose and len(rows) % 36000 == 0 and len(rows) > 0:
            print(f"  Train {100*t/total_s:5.1f}%  |  {len(rows):,} rows  |  {time.time()-t0:.0f}s")

    # trim to exact row count
    target = int(hours * 3600 / dt)
    rows = rows[:target]

    if verbose:
        import pandas as pd
        df = pd.DataFrame(rows)
        print(f"  Train done: {len(rows):,} rows  |  events: {len(event_log)}")
        print(f"  GEN1_cmd=1: {(df['GEN1_cmd']>0.5).mean():.3f}  "
              f"UPS_bypass=1: {(df['UPS_bypass']>0.5).mean():.3f}  "
              f"shed>0: {(df['load_shed_stage']>0).mean():.3f}")
    return rows, event_log


# ─────────────────────────────────────────────────────────────────────────────
# Test split — SWaT/WADI-style density with pure normal windows
# ─────────────────────────────────────────────────────────────────────────────

def generate_test(hours, dt, seed, normal_min=120, normal_max=300,
                  attack_min=90, attack_max=180,
                  warmup_s=300, scenarios=None, verbose=True):
    """
    Test split following the SWaT/WADI evaluation protocol.

    Normal windows between attacks are PURE GRID-CONNECTED (no maintenance
    events). This matches the SWaT methodology and produces per-scenario F1
    directly comparable to published ICS anomaly detection benchmarks.

    Detectors trained on the diverse training split (which includes generator
    tests, UPS bypass, load shed drills) must generalise correctly to pure
    grid-connected normal operation — this is the intended evaluation.

    Normal: 120-300s  |  Attack: 90-180s  |  Attack ratio: ~38%
    """
    if scenarios is None:
        scenarios = ALL_SCENARIOS

    state = _fresh_state(seed + 9999)
    rng   = random.Random(seed + 1)
    if verbose: print(f"  Warmup {warmup_s}s...", end="", flush=True)
    _warmup(state, warmup_s, dt)
    if verbose: print(" done.")

    total_s    = hours * 3600
    t          = 0.0
    rows       = []
    attack_log = []
    sc_pool    = list(scenarios); rng.shuffle(sc_pool); sc_idx = 0
    t0 = time.time()

    while t < total_s:
        # ── pure grid-connected normal window ─────────────────────────────────
        normal_dur = rng.randint(normal_min, normal_max)
        for _ in range(int(normal_dur / dt)):
            if t >= total_s: break
            t = _step_normal(state, dt, t, rows)

        if t >= total_s: break

        # ── attack window ─────────────────────────────────────────────────────
        code = sc_pool[sc_idx % len(sc_pool)]; sc_idx += 1
        if sc_idx % len(sc_pool) == 0: rng.shuffle(sc_pool)

        attack_dur = rng.randint(attack_min, attack_max)
        pm.set_scenario(state, code)
        t_atk = t
        for _ in range(int(attack_dur / dt)):
            if t >= total_s: break
            obs = pm.step(state, dt)
            rows.append(_row(obs, t, 1, code))
            t += dt
        pm.set_scenario(state, None)
        attack_log.append({"scenario": code,
                            "start_s":   round(t_atk, 1),
                            "duration_s": round(t - t_atk, 1)})

        if verbose and len(rows) % 36000 < (normal_dur + attack_dur + 10):
            pct = t / total_s * 100
            print(f"  Test  {pct:5.1f}%  |  {len(rows):,} rows  |  "
                  f"last: {code:<28}  |  {time.time()-t0:.0f}s")

    rows = rows[:int(hours * 3600 / dt)]

    if verbose:
        n_n = sum(1 for r in rows if r["label"]==0)
        n_a = sum(1 for r in rows if r["label"]==1)
        print(f"  Test done: {len(rows):,} rows  |  "
              f"normal={n_n:,} ({100*n_n/max(len(rows),1):.1f}%)  "
              f"attack={n_a:,} ({100*n_a/max(len(rows),1):.1f}%)")
        print(f"  Attack windows: {len(attack_log)}  "
              f"({len(set(e['scenario'] for e in attack_log))} unique scenarios)")
    return rows, attack_log, []   # no planned maintenance events in test



def write_csv(rows, path, verbose=True):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    mb = os.path.getsize(path) / 1e6
    if verbose: print(f"  Wrote {path}  ({len(rows):,} rows, {mb:.1f} MB)")


def write_meta(out_dir, train_rows, test_rows, train_events,
               attack_log, test_maint_log, args):
    sc_cat = {code: {
        "title": sc.title,
        "group": sc.group,
        "category": sc.category,
        "affected_assets": sc.affected_assets,
        "summary": sc.summary,
    } for code, sc in sl.SCENARIOS.items()}

    n_n = sum(1 for r in test_rows if r["label"]==0)
    n_a = sum(1 for r in test_rows if r["label"]==1)
    meta = {
        "dataset_name": "DCPower-ICS",
        "version": "1.0",
        "description": (
            "Labeled ICS anomaly detection benchmark for data center power "
            "infrastructure. The training split includes planned maintenance "
            "events labeled as normal. The test split alternates pure normal "
            "windows and labeled fault/anomaly windows."
        ),
        "generation": {k: v for k, v in vars(args).items()},
        "splits": {
            "train": {
                "file": "dcpower_train.csv",
                "rows": len(train_rows),
                "duration_hours": round(len(train_rows)*args.dt/3600, 2),
                "all_label_0": True,
                "maintenance_events": len(train_events),
            },
            "test": {
                "file": "dcpower_test.csv",
                "rows": len(test_rows),
                "duration_hours": round(len(test_rows)*args.dt/3600, 2),
                "normal_rows": n_n,
                "attack_rows": n_a,
                "attack_ratio": round(n_a/max(len(test_rows),1), 4),
                "attack_windows": len(attack_log),
                "unique_scenarios": len(set(e["scenario"] for e in attack_log)),
                "planned_maintenance_events": 0,
            },
        },
        "columns": {
            "timestamp": "Elapsed simulation time (seconds)",
            "sensors": NUMERIC_FIELDS,
            "label": "0=Normal (incl. planned maintenance), 1=Attack/Anomaly",
            "attack_scenario": "Scenario code when label=1, 'Normal' when label=0",
        },
        "scenario_catalogue": sc_cat,
        "train_maintenance_log": train_events,
        "test_attack_log": attack_log,
    }
    path = os.path.join(out_dir, "dcpower_meta.json")
    with open(path, "w") as f: json.dump(meta, f, indent=2)
    print(f"  Wrote {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate DCPower-ICS benchmark dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-hours",   type=float, default=24.0)
    p.add_argument("--test-hours",    type=float, default=24.0)
    p.add_argument("--dt",            type=float, default=1.0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--warmup",        type=int,   default=300)
    p.add_argument("--normal-min",    type=int,   default=120)
    p.add_argument("--normal-max",    type=int,   default=300)
    p.add_argument("--attack-min",    type=int,   default=90)
    p.add_argument("--attack-max",    type=int,   default=180)
    p.add_argument("--out-dir",       type=str,   default="dcpower_dataset")
    p.add_argument("--quiet",         action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    verbose = not args.quiet
    os.makedirs(args.out_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  DCPower-ICS Dataset Generator")
        print(f"{'='*60}")
        print(f"  Output dir : {args.out_dir}")
        print(f"  Train: {args.train_hours}h  |  Test: {args.test_hours}h")
        print(f"  dt={args.dt}s  seed={args.seed}")

    t0 = time.time()

    train_rows, train_ev = generate_train(
        args.train_hours, args.dt, args.seed, args.warmup, verbose)
    write_csv(train_rows, os.path.join(args.out_dir, "dcpower_train.csv"), verbose)

    test_rows, attack_log, test_maint = generate_test(
        args.test_hours, args.dt, args.seed,
        normal_min=args.normal_min, normal_max=args.normal_max,
        attack_min=args.attack_min, attack_max=args.attack_max,
        warmup_s=args.warmup, verbose=verbose)
    write_csv(test_rows, os.path.join(args.out_dir, "dcpower_test.csv"), verbose)
    write_meta(args.out_dir, train_rows, test_rows, train_ev,
               attack_log, test_maint, args)

    if verbose:
        total = len(train_rows) + len(test_rows)
        print(f"\n  DONE  |  {total:,} total rows  |  {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
