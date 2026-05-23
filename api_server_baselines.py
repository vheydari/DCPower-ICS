#!/usr/bin/env python3
"""
DCPower-ICS Baseline Demo Server  —  per-session simulation
=============================================================
Each visitor gets their own isolated PowerPlantState, trend buffer, and
simulation loop. Clicking Inject/Reset on one browser has no effect on any
other visitor's session.

Session management
------------------
- A UUID session token is stored in a browser cookie (dcpower_sid).
- Flask sets the cookie automatically on the first request to / or /stream.
- Sessions expire after SESSION_TTL_MINUTES of inactivity and are cleaned up
  by a background thread every CLEANUP_INTERVAL_MINUTES.
- Maximum MAX_SESSIONS concurrent sessions are allowed; oldest is evicted
  when the cap is reached.

Usage
-----
    python api_server_baselines.py
    python api_server_baselines.py --port 5003 --data-dir dcpower_dataset
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
import warnings
from collections import deque
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from flask import (Flask, Response, jsonify, make_response,
                   request, send_from_directory)
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from plant_model import PowerPlantState, set_scenario, step
from scenario_library import SCENARIOS, scenario_groups

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--host",       default="127.0.0.1")
ap.add_argument("--port",       type=int, default=5003)
ap.add_argument("--data-dir",   default="dcpower_dataset")
ap.add_argument("--train-ticks",type=int, default=5000)
args = ap.parse_args()

# ── Session constants ─────────────────────────────────────────────────────────
SESSION_COOKIE       = "dcpower_sid"
SESSION_TTL_MINUTES  = 30          # evict after this many minutes idle
CLEANUP_INTERVAL_S   = 120         # run cleanup every 2 minutes
MAX_SESSIONS         = 50          # cap concurrent sessions
TREND_LEN            = 300         # trend buffer length per session

app = Flask(__name__)

# ── Sensor columns ────────────────────────────────────────────────────────────
SENSOR_COLS = [
    "utility_available","utility_degraded",
    "PCC_breaker","PCC_breaker_cmd",
    "GEN1_cmd","GEN2_cmd","GEN1_running","GEN2_running",
    "GEN1_ready","GEN2_ready","GEN1_fault","GEN2_fault",
    "GEN1_breaker","GEN2_breaker",
    "GEN1_P","GEN2_P","GEN1_Q","GEN2_Q",
    "GEN1_f","GEN2_f","GEN1_V","GEN2_V",
    "UPS_bypass","UPS_in_P","UPS_out_P",
    "BESS_cmd","BESS_P","BESS_SOC","BESS_current",
    "P_IT","P_cooling","P_noncritical","load_shed_stage",
    "bus_voltage","bus_frequency",
    "PCC_P","PCC_Q","PCC_I",
    "ambient_c","maintenance_bypass",
]

# ── Load training data & fit detectors (once at startup, shared) ─────────────
def load_train_data() -> np.ndarray:
    csv_path = Path(args.data_dir) / "dcpower_train.csv"
    if csv_path.exists():
        print(f"  Loading training data from {csv_path} ...", end="", flush=True)
        import csv
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row.get("label", 0)) == 0:
                    rows.append([float(row.get(c, 0.0)) for c in SENSOR_COLS])
        X = np.array(rows, dtype=np.float32)
        if len(X) > 20000:
            idx = np.random.RandomState(42).choice(len(X), 20000, replace=False)
            X = X[idx]
        print(f" {len(X):,} rows loaded.")
        return X
    else:
        print(f"  CSV not found — generating {args.train_ticks} ticks online ...",
              end="", flush=True)
        plant = PowerPlantState()
        for _ in range(300): step(plant, 1.0)
        rows = []
        for _ in range(args.train_ticks):
            obs = step(plant, 1.0)
            rows.append([float(obs.get(c, 0.0)) for c in SENSOR_COLS])
        print(" done.")
        return np.array(rows, dtype=np.float32)


print("\n  DCPower-ICS Baseline Demo Server (per-session)")
print("  Fitting 5 detectors on normal training data ...")

X_train_raw = load_train_data()
SCALER = StandardScaler()
X_train = SCALER.fit_transform(X_train_raw)

rng_sub = np.random.RandomState(42)
idx10k  = rng_sub.choice(len(X_train), min(10000, len(X_train)), replace=False)
idx20k  = rng_sub.choice(len(X_train), min(20000, len(X_train)), replace=False)

print("  [1/5] Isolation Forest ...", end="", flush=True)
IFOREST = IsolationForest(n_estimators=200, contamination=0.05,
                           random_state=42, n_jobs=-1)
IFOREST.fit(X_train)
_sn_if = -IFOREST.score_samples(X_train)
THR_IF  = float(np.quantile(_sn_if, 0.95))
P1_IF, P99_IF = float(np.quantile(_sn_if, 0.01)), float(np.quantile(_sn_if, 0.99))
print(f" done  thr={THR_IF:.4f}")

print("  [2/5] One-Class SVM ...", end="", flush=True)
OCSVM = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
OCSVM.fit(X_train[idx10k])
_sn_oc = -OCSVM.score_samples(X_train[idx10k])
THR_OC  = float(np.quantile(_sn_oc, 0.95))
P1_OC, P99_OC = float(np.quantile(_sn_oc, 0.01)), float(np.quantile(_sn_oc, 0.99))
print(f" done  thr={THR_OC:.4f}")

print("  [3/5] LOF ...", end="", flush=True)
LOF = LocalOutlierFactor(n_neighbors=20, novelty=True,
                          contamination=0.05, n_jobs=-1)
LOF.fit(X_train[idx20k])
_sn_lof = -LOF.score_samples(X_train[idx20k])
THR_LOF  = float(np.quantile(_sn_lof, 0.95))
P1_LOF, P99_LOF = float(np.quantile(_sn_lof, 0.01)), float(np.quantile(_sn_lof, 0.99))
print(f" done  thr={THR_LOF:.4f}")

print("  [4/5] PCA reconstruction ...", end="", flush=True)
PCA_MODEL = PCA(n_components=10, random_state=42)
PCA_MODEL.fit(X_train)
_sn_pca = np.mean((X_train -
                   PCA_MODEL.inverse_transform(PCA_MODEL.transform(X_train)))**2,
                  axis=1)
THR_PCA  = float(np.quantile(_sn_pca, 0.95))
P1_PCA, P99_PCA = float(np.quantile(_sn_pca, 0.01)), float(np.quantile(_sn_pca, 0.99))
print(f" done  thr={THR_PCA:.6f}")

print("  [5/5] AutoEncoder MLP ...", end="", flush=True)
AE = MLPRegressor(hidden_layer_sizes=(32, 16, 8, 16, 32),
                  activation="relu", max_iter=80, random_state=42,
                  early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=5, verbose=False)
AE.fit(X_train, X_train)
_sn_ae = np.mean((X_train - AE.predict(X_train))**2, axis=1)
THR_AE  = float(np.quantile(_sn_ae, 0.95))
P1_AE, P99_AE = float(np.quantile(_sn_ae, 0.01)), float(np.quantile(_sn_ae, 0.99))
print(f" done  thr={THR_AE:.6f}")

print("  All detectors ready.\n")


# ── Score one observation ─────────────────────────────────────────────────────
def _norm(raw: float, p1: float, p99: float) -> float:
    return float(min(9.99, max(0.0, (raw - p1) / max(p99 - p1, 1e-9))))


def score_obs(obs: dict) -> list[dict]:
    x  = np.array([[float(obs.get(c, 0.0)) for c in SENSOR_COLS]],
                  dtype=np.float32)
    xs = SCALER.transform(x)
    raw = {
        "IForest": float(-IFOREST.score_samples(xs)[0]),
        "OCSVM":   float(-OCSVM.score_samples(xs)[0]),
        "LOF":     float(-LOF.score_samples(xs)[0]),
        "PCA":     float(np.mean((xs - PCA_MODEL.inverse_transform(
                                  PCA_MODEL.transform(xs)))**2)),
        "AE":      float(np.mean((xs - AE.predict(xs))**2)),
    }
    thresholds = dict(IForest=THR_IF, OCSVM=THR_OC, LOF=THR_LOF,
                      PCA=THR_PCA, AE=THR_AE)
    p1s  = dict(IForest=P1_IF,  OCSVM=P1_OC,  LOF=P1_LOF,
                PCA=P1_PCA,     AE=P1_AE)
    p99s = dict(IForest=P99_IF, OCSVM=P99_OC, LOF=P99_LOF,
                PCA=P99_PCA,    AE=P99_AE)
    return [
        {
            "name":       name,
            "raw_score":  round(raw[name], 5),
            "norm_score": round(_norm(raw[name], p1s[name], p99s[name]), 2),
            "alarming":   bool(raw[name] >= thresholds[name]),
        }
        for name in ["IForest", "OCSVM", "LOF", "PCA", "AE"]
    ]


# ── Per-session state ─────────────────────────────────────────────────────────
class Session:
    """One isolated simulation instance per visitor."""

    def __init__(self, sid: str):
        self.sid       = sid
        self.lock      = threading.Lock()
        self.trend     = deque(maxlen=TREND_LEN)
        self.tick      = 0
        self.latest    = None        # most recent obs dict
        self.last_seen = time.time() # for TTL eviction

        # warm up a fresh plant
        self.plant = PowerPlantState()
        for _ in range(300):
            step(self.plant, 1.0)
        self.latest = step(self.plant, 1.0)

        # start this session's simulation thread
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        while True:
            t0 = time.time()
            with self.lock:
                self.latest  = step(self.plant, 1.0)
                self.tick   += 1
            time.sleep(max(0.0, 1.0 - (time.time() - t0)))

    def touch(self):
        self.last_seen = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.last_seen) > SESSION_TTL_MINUTES * 60

    def build_payload(self) -> dict:
        self.touch()
        with self.lock:
            obs = dict(self.latest)

        detectors = score_obs(obs)
        sc_code   = obs.get("active_scenario") or None

        source = "Utility"
        if obs.get("PCC_P", 0) < 30 and obs.get("ATS_state") != "UTILITY":
            source = "Generators / islanded"
        if obs.get("UPS_bypass", 0) > 0.5:
            source = "UPS bypass / maintenance"

        alarm_score = max(d["norm_score"] for d in detectors)
        trend_point = {
            "tick":           self.tick,
            "bus_voltage":    obs.get("bus_voltage"),
            "bus_frequency":  obs.get("bus_frequency"),
            "gen_kw":         round((obs.get("GEN1_P", 0) + obs.get("GEN2_P", 0)), 1),
            "bess_soc":       obs.get("BESS_SOC"),
            "bess_kw":        obs.get("BESS_P"),
            "it_kw":          obs.get("P_IT"),
            "cooling_kw":     obs.get("P_cooling"),
            "noncritical_kw": obs.get("P_noncritical"),
            "alarm_score":    alarm_score,
            "scenario":       sc_code,
        }
        self.trend.append(trend_point)

        return {
            "status":          "ready",
            "tick":            self.tick,
            "mode_banner":     obs.get("controller_mode", "--"),
            "source_of_power": source,
            "scenario_code":   sc_code,
            "sensors":         obs,
            "detectors":       detectors,
            "trend":           list(self.trend),
            "trend_point":     trend_point,
            "loads": {
                "it":          round(obs.get("P_IT", 0), 1),
                "cooling":     round(obs.get("P_cooling", 0), 1),
                "noncritical": round(obs.get("P_noncritical", 0), 1),
            },
        }

    def inject(self, code: str):
        self.touch()
        with self.lock:
            set_scenario(self.plant, code)
            self.trend.clear()

    def reset(self):
        self.touch()
        with self.lock:
            self.plant = PowerPlantState()
            for _ in range(300):
                step(self.plant, 1.0)
            self.latest = step(self.plant, 1.0)
            self.tick   = 0
            self.trend.clear()


# ── Session registry ──────────────────────────────────────────────────────────
SESSIONS: dict[str, Session] = {}
SESSIONS_LOCK = threading.Lock()


def get_or_create_session(sid: str | None) -> tuple[str, Session]:
    """Return (sid, session), creating one if needed."""
    with SESSIONS_LOCK:
        # validate existing session
        if sid and sid in SESSIONS:
            return sid, SESSIONS[sid]

        # evict oldest if at cap
        if len(SESSIONS) >= MAX_SESSIONS:
            oldest_sid = min(SESSIONS, key=lambda s: SESSIONS[s].last_seen)
            del SESSIONS[oldest_sid]
            print(f"  [sessions] evicted {oldest_sid[:8]}... (cap reached)")

        new_sid     = str(uuid.uuid4())
        new_session = Session(new_sid)
        SESSIONS[new_sid] = new_session
        print(f"  [sessions] created {new_sid[:8]}...  total={len(SESSIONS)}")
        return new_sid, new_session


def cleanup_sessions():
    """Background thread: remove expired sessions."""
    while True:
        time.sleep(CLEANUP_INTERVAL_S)
        with SESSIONS_LOCK:
            expired = [s for s, sess in SESSIONS.items() if sess.is_expired()]
            for s in expired:
                del SESSIONS[s]
            if expired:
                print(f"  [sessions] cleaned up {len(expired)} expired sessions, "
                      f"{len(SESSIONS)} remaining")


threading.Thread(target=cleanup_sessions, daemon=True).start()


# ── Cookie helpers ────────────────────────────────────────────────────────────
def _sid_from_request() -> str | None:
    return request.cookies.get(SESSION_COOKIE)


def _set_cookie(response, sid: str):
    response.set_cookie(
        SESSION_COOKIE, sid,
        max_age=SESSION_TTL_MINUTES * 60,
        samesite="Lax",
        httponly=True,
    )
    return response


# ── Flask routes ──────────────────────────────────────────────────────────────
HERE = Path(__file__).parent


@app.route("/")
def index():
    sid, _  = get_or_create_session(_sid_from_request())
    resp    = make_response(send_from_directory(HERE, "dcpower_demo.html"))
    _set_cookie(resp, sid)
    return resp


@app.route("/dcpower_demo.html")
def html_file():
    sid, _  = get_or_create_session(_sid_from_request())
    resp    = make_response(send_from_directory(HERE, "dcpower_demo.html"))
    _set_cookie(resp, sid)
    return resp


@app.route("/scenarios")
def scenarios():
    return jsonify({
        "groups": scenario_groups(),
        "all":    {k: _sc_dict(v) for k, v in SCENARIOS.items()},
    })


def _sc_dict(s):
    if not s:
        return None
    return {
        "code": s.code,
        "title": s.title,
        "group": s.group,
        "category": s.category,
        "affected_assets": s.affected_assets,
        "summary": getattr(s, "summary", ""),
    }


@app.route("/stream_latest")
def stream_latest():
    sid, sess = get_or_create_session(_sid_from_request())
    resp = make_response(jsonify(sess.build_payload()))
    _set_cookie(resp, sid)
    return resp


@app.route("/stream")
def stream():
    sid, sess = get_or_create_session(_sid_from_request())

    def generate():
        last_tick = -1
        while True:
            time.sleep(0.5)
            payload = sess.build_payload()
            if payload["tick"] == last_tick:
                continue
            last_tick = payload["tick"]
            yield f"data: {json.dumps(payload)}\n\n"

    resp = Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    _set_cookie(resp, sid)
    return resp


@app.route("/inject", methods=["POST"])
def inject():
    sid, sess = get_or_create_session(_sid_from_request())
    code = request.get_json(force=True).get("scenario")
    if code not in SCENARIOS:
        return jsonify({"error": f"Unknown scenario: {code}"}), 400
    sess.inject(code)
    resp = make_response(jsonify({"ok": True}))
    _set_cookie(resp, sid)
    return resp


@app.route("/reset", methods=["POST"])
def reset():
    sid, sess = get_or_create_session(_sid_from_request())
    sess.reset()
    resp = make_response(jsonify({"ok": True}))
    _set_cookie(resp, sid)
    return resp


@app.route("/status")
def status():
    with SESSIONS_LOCK:
        n = len(SESSIONS)
    return jsonify({"active_sessions": n, "max_sessions": MAX_SESSIONS})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    url = f"http://{args.host}:{args.port}/"
    print(f"  DCPower-ICS Baseline Demo (per-session): {url}")
    print(f"  Session TTL: {SESSION_TTL_MINUTES} min  |  "
          f"Max sessions: {MAX_SESSIONS}\n")
    app.run(host=args.host, port=args.port, threaded=True)
