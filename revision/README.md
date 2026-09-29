# Manuscript revision analyses

These additions analyze dataset version https://doi.org/10.5281/zenodo.20358618. They do not repair or replace the original simulator or change the deposited data.

Run from the repository root with Python 3.12.14 and `pip install -r revision/requirements.txt`. Download all three deposited files into `dcpower_dataset/` and verify MD5 using `revision/results/download_verification.json`.

```bash
python revision/revision_analysis.py --data-dir dcpower_dataset --code-dir . --out-dir revision/rerun
python revision/low_prevalence.py --data-dir dcpower_dataset --target .01 --seed 42 --output revision/rerun/subset_1pct.csv
python revision/low_prevalence.py --data-dir dcpower_dataset --target .05 --seed 42 --output revision/rerun/subset_5pct.csv
```

`revision_analysis.py` writes physical diagnostics, environment/source hashes, and maintenance FAR. It independently refits Isolation Forest, PCA and MLP on all 40 channels and on 22 measurements after excluding 18 status/command inputs. Grid calibration uses the explicit mask in `masks()`; event intervals come from metadata. Training rows are used for fitting and scoring, so FAR is an in-sample diagnostic. Detector random state remains 42 while data seeds are 42, 123 and 777; extra seed traces are generated only in memory. No new dataset release is produced. The file also includes all-normal calibration diagnostics; the manuscript reports the `calibration=grid` rows. Thresholds are recalculated for each refitted model/feature set. Metadata define evaluation groups but labels and status/command channels do not enter the measurement-only detectors.

`low_prevalence.py` selects complete events plus all normal test rows and writes source-row manifests, not altered telemetry. Its subset-sum selection uses seed 42; separate `block_id` values mark gaps. Do not feed gaps to temporal detectors as continuous samples or claim all 22 scenarios are retained. Summary JSON files give actual prevalence and coverage.

`results/` contains the executed results used for the revision. The five original global baseline metrics were reproduced with unchanged `evaluate_baselines.py`. The companion `baseline_scenario_segment_recall.csv` explicitly labels the original script's per-scenario segment-recall output.

The inspected repository commit was `bc1ced5e89e9eb717835f400a21422ade7a7e4c2`; the original code archive corresponds to `3c879bb`, with identical core generator/model/scenario files. Data and core-file SHA-256 values are in `results/environment.json`. The new analysis files need their own release/commit citation after publication; the original code DOI does not archive this folder.
