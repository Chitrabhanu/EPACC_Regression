# EPACC Waveform Regression

> Predicting fluid responsiveness from high-frequency waveform data using deep learning.

---

## Overview

This repository implements a production-structured ML pipeline developed for the **EPACC** project. The clinical goal is **fluid responsiveness prediction**: given a fluid bolus administered during resuscitation, can we predict whether the patient responds — specifically, whether Stroke Volume increases sufficiently to indicate successful resuscitation?

The pipeline predicts continuous Stroke Volume change from arterial pressure waveform recordings in animal studies, providing a data-driven approach to guide fluid bolus therapy decisions.

The pipeline is designed with research-to-production principles in mind: configuration-driven experiments, strict data leakage prevention, deterministic cross-validation, and checkpoint-based evaluation.

---

## Repository Structure

```
EPACC_Regression/
├── streamlined_scripts/
│   └── deep_learning/          # Production pipeline (main codebase)
│       ├── src/epacc_mle/      # Python package
│       │   ├── data/           # Data loading and fold construction
│       │   ├── models/         # Model architectures + registry
│       │   ├── train/          # CV and full-training loops
│       │   ├── eval/           # Holdout evaluation and aggregation
│       │   ├── experiments/    # Multi-split experiment runners
│       │   └── cli.py          # Experiment entry point
│       ├── configs/            # YAML experiment configs
│       ├── data/               # Dataset splits and fold metadata
│       ├── legacy/             # Earlier implementation (reference)
│       ├── pyproject.toml
│       └── README.md           # Detailed technical documentation
│
└── deep_learning_raw_scripts/  # Early prototyping scripts
```

---

## Quick Start

```bash
cd streamlined_scripts/deep_learning

# Create and activate environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# Install package
pip install -U pip
pip install -e .

# Run an experiment (edit cli.py to select mode)
python -m epacc_mle.cli
```

---

## Key Design Decisions

**Wavelet-level training, bolus-level evaluation.** Physiological bolus waveforms are segmented into fixed-length wavelets (224 samples). Models are trained at the wavelet level to maximise sample efficiency; predictions are aggregated back to the bolus level for clinically interpretable evaluation.

**Strict leakage prevention.** Cross-validation fold assignment uses `pig_id` (subject identity), while bolus identity uses `(pig, batch)`. These are intentionally separate to prevent any subject appearing in both train and validation folds.

**Deterministic reproducibility.** Every run snapshots its resolved configuration. Fold assignments are fixed. Checkpoints enable evaluation without retraining.

---

## Models Available

| Architecture | Description |
|---|---|
| `cnn1d_basic` | Baseline 1D CNN |
| `cnn1d_sebn_reg` | 1D CNN with Squeeze-and-Excitation blocks |
| `resnet1d` | ResNet-style 1D architecture |
| `wavenet1d` | WaveNet-inspired dilated convolutions |
| `transformer1d` | Transformer encoder over wavelet sequences |

---

## Experiment Modes

| Mode flag | Description |
|---|---|
| `RUN_CV_SINGLE_SPLIT` | 5-fold CV on one training split |
| `RUN_TRAIN_FULL_SINGLE_SPLIT` | Full training on one split |
| `RUN_EVAL_HOLDOUT_FROM_CKPT` | Holdout evaluation from saved checkpoint |
| `RUN_HOLDOUT_SINGLE_SPLIT` | End-to-end: train + holdout evaluation |
| `RUN_HOLDOUT_ALL_SPLITS` | Full multi-split experiment (final evaluation) |

---

## Author

**Chitrabhanu Gupta** — Clinical Machine Learning Researcher

For detailed technical documentation see [`streamlined_scripts/deep_learning/README.md`](streamlined_scripts/deep_learning/README.md).
