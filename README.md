# DCPower-ICS: A Labeled ICS Dataset for Data Center Power Infrastructure

[![Scientific Data](https://img.shields.io/badge/Scientific%20Data-under%20review-blue)]()
[![Zenodo Dataset](https://img.shields.io/badge/Data-Zenodo-green)]()
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)

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
├── dcpower_test.csv       # 86,400 rows; pure normal windows + labeled fault windows
└── dcpower_meta.json      # dataset card, scenario catalogue, event logs, generation parameters
```

Each CSV has 43 columns: `timestamp`, 40 numeric process variables, `label`, and `attack_scenario`.

## Benchmark Design

DCPower-ICS follows the two-part design described in the manuscript:

1. **Fault detection:** the test split alternates pure grid-connected normal windows and labeled fault/anomaly windows. Test normal windows do not include planned maintenance events.
2. **Maintenance robustness:** planned generator load tests, UPS bypass windows, and load shed drills appear in the training split and are labeled normal. These windows can be used to evaluate maintenance-mode false-alarm behavior.

## Scenario Catalogue

The dataset includes 22 labeled fault/anomaly scenarios across four categories:

| Category | Scenarios |
|---|---|
| Maintenance error | `GEN_EFF_LOSS`, `GEN_START_DELAY`, `BATTERY_DEGRADATION`, `SOC_CAL_DRIFT`, `COOLING_EFF_LOSS`, `UPS_BYPASS_STUCK` |
| Commissioning fault | `WRONG_ATS_TIMING`, `WRONG_BATT_SETPOINT`, `LOAD_SHED_MISCONFIG`, `SWAPPED_SENSOR_MAPPING`, `BYPASS_LEFT_ENABLED`, `BREAKER_POSITION_MISMATCH` |
| Cyber-physical | `BREAKER_STATUS_SPOOF`, `POWER_METER_SPOOF`, `PCC_METER_BIAS`, `BUS_VOLTAGE_FREEZE`, `COORDINATED_MASKING`, `STEALTH_GEN_BIAS` |
| Operational disturbance | `GRID_DISTURBANCE`, `UNSCHEDULED_BLACK_START`, `ATS_SLOW_TRANSFER`, `FALSE_HEALTHY_SUBSYSTEM` |

Planned maintenance events are not fault scenarios and are labeled normal.

## Baselines

`evaluate_baselines.py` runs the five unsupervised baseline sanity checks reported in the paper:

- Isolation Forest
- One-Class SVM
- Local Outlier Factor
- PCA reconstruction
- MLP AutoEncoder

The default threshold is the 95th percentile of training anomaly scores (`--contamination 0.05`). These results are intended as technical validation/sanity checks, not optimized leaderboard claims.

## Live Demo

The live demo uses the same simulator and five baseline detectors. Each browser session receives an isolated simulation state.

```bash
python api_server_baselines.py --data-dir dcpower_dataset --host 127.0.0.1 --port 5003
```

For Render/Heroku-like deployment, the included `Procfile` starts the same Flask server.

## License

Code is released under the MIT License. The dataset files should be cited from the archived Zenodo dataset record described in the paper.
