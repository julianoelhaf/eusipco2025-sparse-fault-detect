# Sparse Fault Detection

Robust machine learning for fault detection under data constraints in power system protection.

[![Paper: EUSIPCO 2025](https://img.shields.io/badge/paper-EUSIPCO%202025-b31b1b.svg)](https://doi.org/10.23919/EUSIPCO63237.2025.11226584)
[![DOI](https://img.shields.io/badge/DOI-10.23919%2FEUSIPCO63237.2025.11226584-blue.svg)](https://doi.org/10.23919/EUSIPCO63237.2025.11226584)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)

Code for the paper:

> J. Oelhaf, G. Kordowich, C. Kim, P. A. Pérez-Toro, A. Maier, J. Jäger, S. Bayer,
> "Impact of Data Sparsity on Machine Learning for Fault Detection in Power System Protection,"
> in *2025 33rd European Signal Processing Conference (EUSIPCO)*, Palermo, Italy, 2025,
> pp. 1997–2001, doi: 10.23919/EUSIPCO63237.2025.11226584.

This repository studies the impact of data sparsity on scikit-learn models for fault
detection and fault line identification in power system protection. Training datasets
typically assume ideal data availability, whereas real deployments face communication
failures, sensor malfunctions, low-sampling-rate legacy equipment, and environmental
disturbances. The experiments quantify how much data is required to sustain a fault
detection accuracy of ≥ 99.9 %, and which strategies mitigate degradation when data is
limited, degraded, or missing.

> ### 📄 Official code for the EUSIPCO 2025 paper
>
> J. Oelhaf, G. Kordowich, C. Kim, P. A. Pérez-Toro, A. Maier, J. Jäger, and S. Bayer,
> **"Impact of Data Sparsity on Machine Learning for Fault Detection in Power System Protection,"**
> in *2025 33rd European Signal Processing Conference (EUSIPCO)*, Palermo, Italy, 2025, pp. 1997–2001.
>
> **[📄 IEEE Xplore](https://ieeexplore.ieee.org/document/11226584)** ·
> **[📚 arXiv](https://arxiv.org/abs/2505.15560)** ·
> **[🔗 DOI](https://doi.org/10.23919/EUSIPCO63237.2025.11226584)** ·
> BibTeX & `CITATION.cff` in the **Citation** section below

---

## Requirements

- Python 3.10

```bash
git clone https://github.com/julianoelhaf/eusipco2025-sparse-fault-detect
cd eusipco2025-sparse-fault-detect
pip install -e ".[dev]"
```

Runtime dependencies (`hydra-core`, `mlflow`, `numpy`, `pandas`, `scikit-learn`,
`matplotlib`, `seaborn`, `joblib`, `natsort`, `tqdm`, `pyyaml`, `omegaconf`) are declared in
`pyproject.toml`. Experiments are launched from the repository root so the Hydra app resolves
the `config/` group.

---

## Data

The experiments use a private FAU EMT simulation dataset (not publicly released). Key
properties, configured in `config/dataset/default.yaml`:

- 4000 simulation episodes, `double_line_90kV` topology
- 20 kHz sampling, 6400 timesteps per episode
- 8 protection relays, 3 voltage + 3 current channels each (48 channels total)

Point the config at a local copy via `dataset.raw_data_directory` /
`dataset.preprocessed_data_directory` (or edit `config/dataset/default.yaml`).

---

## Running experiments

The experiments below sweep window lengths and the fault target while varying one data
sparsity condition. The fault target is set with `training.fault_target` (`fault` for
detection, `fault_target` for line identification).

### 1. Individual relay failures

Drops all channels of one relay at a time:

```bash
python sparse_fault_detect/models/run_model.py --multirun \
    window_extraction.window_length=0.005,0.01,0.02,0.03,0.04,0.05 \
    training.fault_target=fault,fault_target \
    data_sparsity.relay_failure_ids=[0],[1],[2],[3],[4],[5],[6],[7],[8]
```

### 2. Downsampling (lower sampling frequencies)

Reduces the sampling frequency by the given factor (relative to 20 kHz):

| Factor | Resulting rate |
|--------|----------------|
| 2  | 10 kHz |
| 5  | 4 kHz  |
| 10 | 2 kHz  |
| 25 | 800 Hz |
| 50 | 400 Hz |

```bash
python sparse_fault_detect/models/run_model.py --multirun \
    window_extraction.window_length=0.005,0.01,0.02,0.03,0.04,0.05 \
    training.fault_target=fault,fault_target \
    data_sparsity.downsampling_factor=2,5,10,25,50
```

### 3. Communication loss at bus level

Drops all sensors at a bus (bus 1 → relays 1,2; bus 2 → relays 3,4,5,6; bus 3 → relays 7,8):

```bash
python sparse_fault_detect/models/run_model.py --multirun \
    window_extraction.window_length=0.005,0.01,0.02,0.03,0.04,0.05 \
    training.fault_target=fault,fault_target \
    data_sparsity.bus_failure_id=1,2,3
```

### 4. Missing voltage or current measurements

```bash
# Loss of voltage measurements
python sparse_fault_detect/models/run_model.py --multirun \
    window_extraction.window_length=0.005,0.01,0.02,0.03,0.04,0.05 \
    training.fault_target=fault,fault_target \
    data_sparsity.voltage_loss=True
```

```bash
# Loss of current measurements
python sparse_fault_detect/models/run_model.py --multirun \
    window_extraction.window_length=0.005,0.01,0.02,0.03,0.04,0.05 \
    training.fault_target=fault,fault_target \
    data_sparsity.current_loss=True
```

### 5. Intermittent communication loss (block-based zeroing)

Sets contiguous time intervals to zero to model transmission dropouts. `zeroing_duration_s`
is the dropout length in seconds:

```bash
python sparse_fault_detect/models/run_model.py --multirun \
    window_extraction.window_length=0.005,0.01,0.02,0.03,0.04,0.05 \
    training.fault_target=fault,fault_target \
    data_sparsity.zeroing_duration_s=0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045
```

Runs are tracked with MLflow; start a tracking server with `mlflow_server.sh` (see
`config/cluster/default.yaml` for the port).

---

## Configuration

All configuration is managed with [Hydra](https://hydra.cc). The root config is
`config/main-config.yaml`. Config groups:

| Group | File | What it controls |
|-------|------|------------------|
| `cluster` | `config/cluster/default.yaml` | MLflow tracking server port |
| `dataset` | `config/dataset/default.yaml` | topology, paths, sampling, episodes |
| `data_sparsity` | `config/data_sparsity/default.yaml` | relay/bus/phase failures, downsampling, zeroing, V/I loss |
| `model` | `config/model/default.yaml` | model selection (`model.model`) |
| `training` | `config/training/default.yaml` | fault target, CV splits, seed |
| `window_extraction` | `config/window_extraction/default.yaml` | window length, step, period of interest |

### Supported models

Selected via `model.model` (see `create_model_from_name` in
`sparse_fault_detect/models/run_model.py`):

| Name | Estimator |
|------|-----------|
| `logistic_regression` | `LogisticRegression` |
| `decision_tree_classifier` | `DecisionTreeClassifier` |
| `random_forest_classifier` | `RandomForestClassifier` (default) |
| `extra_tree_classifier` | `ExtraTreeClassifier` |

### Target tasks

Selected via `training.fault_target`:

| Value | Task |
|-------|------|
| `fault` | Fault detection (binary) |
| `fault_target` | Fault line identification |
| `sc_location` | Fault localization |

---

## Tests

A unit test suite covers the data-sparsity transforms (relay/bus/phase failures,
downsampling, block zeroing, voltage/current masks) plus an import smoke test. It runs in CI
(GitLab, see `.gitlab-ci.yml`) and locally:

```bash
pip install -e ".[dev]"
pytest          # run the suite
make lint       # flake8 + isort/black checks
```

---

## Citation

If you use this work, please cite (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@inproceedings{11226584,
  author    = {Oelhaf, Julian and Kordowich, Georg and Kim, Changhun and Pérez-Toro, Paula Andrea and Maier, Andreas and Jäger, Johann and Bayer, Siming},
  title     = {Impact of Data Sparsity on Machine Learning for Fault Detection in Power System Protection},
  booktitle = {2025 33rd European Signal Processing Conference (EUSIPCO)},
  year      = {2025},
  pages     = {1997-2001},
  doi       = {10.23919/EUSIPCO63237.2025.11226584}
}
```

Links: [IEEE Xplore](https://ieeexplore.ieee.org/document/11226584) ·
[arXiv:2505.15560](https://arxiv.org/abs/2505.15560)

---

## Acknowledgments

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) — 535389056.

---

## Contact

Julian Oelhaf — [julian.oelhaf@fau.de](mailto:julian.oelhaf@fau.de) ·
[Website](https://lme.tf.fau.de/persons/julian-oelhaf/) ·
[github.com/julianoelhaf](https://github.com/julianoelhaf)

---

## License

MIT. See [LICENSE](LICENSE).
