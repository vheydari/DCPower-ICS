# DCPower-ICS: A Labeled ICS Dataset for Data Center Power Infrastructure

[![Zenodo Dataset](https://img.shields.io/badge/Data-Zenodo-green)](https://doi.org/10.5281/zenodo.20358618)
[![Zenodo Code](https://img.shields.io/badge/Code%20DOI-Zenodo-blue)](https://doi.org/10.5281/zenodo.20358594)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-online-brightgreen)](http://157.151.204.244:5003/)

**DCPower-ICS** is a fully synthetic labeled ICS anomaly detection benchmark for data center power infrastructure, covering utility feed, PCC breaker, ATS, backup generators, BESS, UPS, IT load, cooling load, and sheddable load in a reduced-order physics-informed simulator.

This repository contains only the code needed to reproduce the paper dataset, validate the released files, run the baseline sanity checks, and launch the live baseline demo. The generated CSV dataset should be downloaded from or archived through Zenodo rather than committed to GitHub.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the paper dataset: 24h train + 24h test at 1 Hz
python generate_dataset.py --train-hours 24 --test-hours 24 --seed 42 --out-dir dcpower_dataset

# 3. Validate and explore the generated dataset
python validate_eda_dcpower.py --data-dir dcpower_dataset

# 4. Run the five unsupervised baseline sanity checks
python evaluate_baselines.py --data-dir dcpower_dataset

# 5. Launch the Flask baseline demo locally
python api_server_baselines.py --data-dir dcpower_dataset --host 127.0.0.1 --port 5003
```

Open `http://127.0.0.1:5003` after starting the demo server.

## Repository Contents

```text
dcpower-ics/
├── README.md
├── LICENSE
├── requirements.txt
├── Procfile
├── plant_model.py              # reduced-order simulator; standard library only
├── scenario_library.py         # 22 scenario metadata definitions
├── generate_dataset.py         # train/test generation and metadata writing
├── validate_eda_dcpower.py     # integrity checks and exploratory plots
├── evaluate_baselines.py       # five unsupervised baseline sanity checks
├── api_server_baselines.py     # Flask backend for live demo
└── dcpower_demo.html           # interactive schematic frontend
```

## Dataset Structure

```text
dcpower_dataset/
├── dcpower_train.csv      # 86,400 rows; all label=0; includes planned maintenance windows
├── dcpower_test.csv       # 86,400 rows; normal windows + labeled fault windows
└── dcpower_meta.json      # dataset card, scenario catalogue, event logs, generation parameters
```

Each CSV has 43 columns: `timestamp`, 40 numeric process variables, `label`, and `attack_scenario`.

## Benchmark Design

DCPower-ICS follows the two-part design described in the manuscript:

1. **Fault detection:** the test split alternates normal windows and labeled fault/anomaly windows. Test normal windows do not include scheduled maintenance, but may contain recovery from prior scenarios.
2. **Maintenance robustness:** planned generator load tests, UPS bypass windows, and load shed drills appear in the training split and are labeled normal. These windows can be used to evaluate maintenance-mode false-alarm behavior.

## Scenario Catalogue

The dataset includes 22 labeled fault/anomaly scenarios across four categories:

| Category | Scenarios |
|---|---|
| Maintenance (archived metadata) | `GEN_EFF_LOSS`, `GEN_START_DELAY`, `BREAKER_POSITION_MISMATCH`, `ATS_SLOW_TRANSFER`, `BATTERY_DEGRADATION`, `SOC_CAL_DRIFT`, `COOLING_EFF_LOSS`, `PCC_METER_BIAS`, `BUS_VOLTAGE_FREEZE`, `UPS_BYPASS_STUCK` |
| Configuration (archived metadata) | `WRONG_ATS_TIMING`, `WRONG_BATT_SETPOINT`, `LOAD_SHED_MISCONFIG`, `SWAPPED_SENSOR_MAPPING`, `BYPASS_LEFT_ENABLED` |
| Cyber (archived metadata) | `BREAKER_STATUS_SPOOF`, `POWER_METER_SPOOF`, `COORDINATED_MASKING`, `STEALTH_GEN_BIAS`, `FALSE_HEALTHY_SUBSYSTEM` |
| Operations (archived metadata) | `GRID_DISTURBANCE`, `UNSCHEDULED_BLACK_START` |

Planned maintenance events are not fault scenarios and are labeled normal.

## Baselines

`evaluate_baselines.py` runs the five unsupervised baseline sanity checks reported in the paper:

- Isolation Forest
- One-Class SVM
- Local Outlier Factor
- PCA reconstruction
- MLP AutoEncoder

The default threshold is the 95th percentile of fitting anomaly scores (`--contamination 0.05`): full training for Isolation Forest/PCA/MLP, and deterministic 10,000/20,000-row fitting subsets for SVM/LOF. The archived per-scenario CSV column named `f1` actually reports segment recall; the revision results use the explicit name `segment_recall`. These results are intended as technical validation/sanity checks, not optimized leaderboard claims.

## Revision analyses and scope

See [`revision/README.md`](revision/README.md) for the measurement-only maintenance ablation, physical diagnostics, and whole-event 1–5% evaluation subsets. The original simulator and dataset-generation files are unchanged. The new scripts read the deposited data; the three-seed analysis also generates supplementary training traces in memory without replacing the deposited files.

The simulator is fully synthetic and not calibrated to a facility. Frequency/voltage are algebraic heuristics, without swing/governor/AVR dynamics. Exact bus power balance is not enforced. The maintenance ATS timer resets each step; the nominal generator startup delay is zero. Labels mark scenario activation rather than a validated physical cause; `FALSE_HEALTHY_SUBSYSTEM` has no direct effect on exported numeric channels and the two breaker scenarios share an implementation. These limitations exclude physical transient, protection, power-quality, or standards-compliance validation. Maintenance FAR depends on feature selection and calibration and does not establish a novel unsolved detection problem.

Version-specific dataset: https://doi.org/10.5281/zenodo.20358618 (concept: 20358617). Original archived code: https://doi.org/10.5281/zenodo.20358594 (concept: 20358593). These code DOIs identify the original release, not the newly added revision analysis folder.

## Live Demo

An optional live demo is available at:

**http://157.151.204.244:5003/**

The live demo uses the same simulator and five baseline detectors. Each browser session receives an isolated simulation state.

> **Note:** The live demo is provided as a non-archival illustration only. It is not required to reproduce the dataset, validation checks, or baseline results. For reproducibility, use the Zenodo dataset DOI and the archived code release described in the paper.

To run the demo locally:

```bash
python api_server_baselines.py --data-dir dcpower_dataset --host 127.0.0.1 --port 5003
```

## License

Code is released under the MIT License. The dataset files should be cited from the archived Zenodo dataset record described in the paper.
