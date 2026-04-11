# EPACC Waveform Regression — Technical Reference

Predicting fluid responsiveness from high-frequency waveform data using deep learning.

---

## Clinical Context

The goal of this project is **fluid responsiveness prediction**: given a fluid bolus administered during resuscitation, predict whether the patient responds — specifically, whether Stroke Volume increases sufficiently to indicate successful resuscitation.

The pipeline predicts continuous Stroke Volume change from arterial pressure waveform recordings in animal studies. Bolus waveforms are segmented into fixed-length **wavelets** (224 samples). Models are trained at the wavelet level while evaluation is reported at both wavelet level and the clinically meaningful **bolus level**.

The pipeline supports:

- Deterministic 5-fold cross-validation
- Split-wise full model training
- Holdout evaluation from saved checkpoints
- Multi-split experimental execution with aggregated reporting

---

## Problem Structure

The dataset has a four-level hierarchy:

```
Dataset
└── Pig
    └── Bolus
        └── Wavelets  (224 samples each)
```

Each wavelet inherits the continuous regression label of its parent bolus. Training at the wavelet level maximises sample efficiency; predictions are aggregated back to bolus level for evaluation.

---

## Evaluation

Performance is reported at two levels:

### Wavelet-Level Metrics
Computed across all individual wavelets.
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Bolus-Level Metrics (Primary Endpoint)
Wavelet predictions are aggregated by `(dataset, pig, batch)` and then evaluated. This represents the clinically interpretable unit — one prediction per administered bolus.

---

## Identifier Semantics

Correct identifier usage is critical for leakage prevention.

| Column | Purpose |
|---|---|
| `pig_id` | Cross-validation fold grouping (subject identity) |
| `pig` | Bolus identifier component |
| `batch` | Bolus instance component |
| `dataset` | Dataset namespace |
| `label` | Continuous regression target (Stroke Volume change) |

CV fold assignment uses `pig_id`; bolus identity is defined as `(pig, batch)`. These are intentionally separate so no subject appears in both train and validation sets.

---

## Repository Structure

```
deep_learning/
├── configs/
│   └── default.yaml              # Experiment configuration
├── data/
│   └── EPACC/
│       ├── time_series_splits/
│       │   ├── train/            # SV_PS_train_*.csv
│       │   └── test/             # SV_PS_test_*.csv
│       └── fold_metadata/
│           └── fold_pigs_SV.csv  # CV fold assignments
├── src/epacc_mle/
│   ├── cli.py                    # Experiment entry point
│   ├── config.py                 # Config dataclass loader
│   ├── paths.py                  # Run directory management
│   ├── data/                     # IO, WaveletDataset, folds, collation
│   ├── models/                   # Architectures + registry
│   ├── train/                    # CV loop, full training
│   ├── eval/                     # Holdout evaluation, metrics
│   ├── experiments/              # Multi-split orchestration
│   └── viz/                      # Training curve plots
├── legacy/                       # Earlier implementation (reference only)
├── artifacts/                    # Experiment outputs (gitignored)
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

pip install -U pip
pip install -e .
```

---

## Required Data Layout

```
data/EPACC/
├── time_series_splits/
│   ├── train/
│   │   ├── SV_PS_train_1.csv
│   │   └── ...
│   └── test/
│       ├── SV_PS_test_1.csv
│       └── ...
└── fold_metadata/
    └── fold_pigs_SV.csv
```

Each CSV must contain columns: `label`, `dataset`, `pig`, `pig_id`, `batch`, `bit_1` ... `bit_224`.

---

## Configuration

All experiments are configuration-driven. Example `configs/default.yaml`:

```yaml
data:
  base_dir: data/EPACC
  num_splits: 26
  sequence_length: 224

model:
  name: cnn1d_sebn_reg

train:
  epochs: 150
  batch_size: 16
  learning_rate: 5e-4
```

Every run stores a resolved configuration snapshot at `artifacts/runs/<run_id>/config_resolved.yaml` for full reproducibility.

---

## Experiment Modes

Experiments are controlled via `cli.py`. Set exactly one mode flag to `True`:

### 1. Cross-Validation (Single Split)
Runs deterministic 5-fold CV on one training split. Use for architecture comparison and hyperparameter validation.

Output: `cv/split_01_cv_summary.csv`

### 2. Full Training (Single Split)
Trains on the entire training split and saves a checkpoint.

Output: `checkpoints/split_01_full_train_last.pth`

### 3. Holdout Evaluation from Checkpoint
Loads a saved checkpoint and evaluates on the corresponding test split. No retraining required.

### 4. End-to-End Split Experiment
Runs the full pipeline for a single split: train → checkpoint → holdout evaluation.

### 5. Multi-Split Experiment (Final Evaluation)
For each split: train, evaluate on holdout, aggregate metrics across all splits.

Outputs:
```
holdout_summary.csv
holdout_overall.json   # mean ± std across splits
```

---

## Experiment Outputs

All runs write to `artifacts/runs/<run_id>/`:

```
artifacts/runs/dev_20260301_123000/
├── config_resolved.yaml
├── checkpoints/
├── curves/
├── cv/
└── holdout/
```

---

## Recommended Workflow

| Phase | Setting |
|---|---|
| Development / debugging | `epochs = 2–5`, single split |
| Model selection | CV on split 1 |
| Final evaluation | CV → full train → holdout per split, then aggregate |

---

## Available Models

| Key | Architecture |
|---|---|
| `cnn1d_basic` | Baseline 1D CNN |
| `cnn1d_sebn_reg` | 1D CNN with Squeeze-and-Excitation blocks |
| `resnet1d` | ResNet-style 1D architecture |
| `wavenet1d` | WaveNet-inspired dilated convolutions |
| `transformer1d` | Transformer encoder |

---

## Reproducibility Guarantees

- Deterministic fold assignment (fixed by `pig_id`)
- Configuration snapshotting on every run
- Strict train/test/fold separation
- Checkpoint-based evaluation (no retraining required)
- Split isolation prevents cross-split leakage

---

## Hardware

CUDA is used automatically when available. For CPU debugging, use reduced epochs and a larger batch size.

---

## Author

**Chitrabhanu Gupta** — Clinical Machine Learning Researcher
