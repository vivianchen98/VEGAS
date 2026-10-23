# VEGAS: Human-Aligned Video Caption Evaluation via Gaze

This repository contains source code for **VEGAS: Human-Aligned Video Caption Evaluation via Gaze**.

VEGAS is accepted to **Transactions on Machine Learning Research (TMLR), 2026**.

## TL;DR

VEGAS is a training-free metric that uses gaze as a test-time signal to evaluate and select video captions that better align with what a viewer attended to. Given paired visual inputs (raw and gaze-conditioned) and candidate captions, VEGAS scores how much caption tokens depend on gaze-relevant visual evidence.

## Resources

- Project page: <https://vivianchen98.github.io/VEGAS_website/>
- Paper: <https://arxiv.org/abs/2607.08489>
- Dataset: <https://huggingface.co/datasets/vivianchen98/VEGAS>

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Running Experiments](#running-experiments)
- [Key Components](#key-components)
- [Citation](#citation)

## Project Structure

```text
.
├── config/                     # Hydra configs for AEA and SlideVQA experiments
│   ├── base.yaml
│   ├── prompt.yaml
│   ├── ariaeveryday.yaml
│   └── slidevqa.yaml
├── src/
│   ├── data/                   # Dataset download and media-subset preparation
│   │   ├── download_dataset.py
│   │   └── prepare_media_subset.py
│   ├── experiments/            # Runnable experiment scripts
│   │   ├── compute_vegas.py
│   │   └── evaluate_captions.py
│   └── utils/                  # Reusable implementation modules
│       ├── vegas.py
│       ├── model_inference.py
│       ├── keyword_masking_utils.py
│       └── metrics.py
├── pyproject.toml
└── README.md
```

Local downloaded data and prepared media are ignored by Git:

```text
data/
├── VEGAS/                      # Hugging Face metadata/caption/gaze dataset
└── VEGAS_MEDIA/                # Local raw and gaze-conditioned media subset
```

## Setup

Requires Python 3.10 or newer.

Using conda:

```bash
conda create -n vegas python=3.10 -y
conda activate vegas
pip install -e .
```

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy the environment template:

```bash
cp .env.example .env
```

Add model API keys as needed:

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

## Data Preparation

### 1. Download VEGAS metadata

Download the public VEGAS dataset from Hugging Face:

```bash
python src/data/download_dataset.py --output-dir data/VEGAS
```

This downloads captions, gaze/fixation data, heatmaps, and segment metadata from `vivianchen98/VEGAS`, then updates `.env` with dataset roots.

### 2. Prepare a runnable media subset

The VEGAS metadata dataset does not redistribute the original AEA videos or raw SlideVQA slide images. To run scoring, prepare paired raw and gaze-conditioned media locally.

For a small smoke test using local source datasets:

```bash
python src/data/prepare_media_subset.py \
  --dataset all \
  --limit 5 \
  --aea-source-root /path/to/original/AriaEverydayActivity \
  --slidevqa-source-root /path/to/original/SlideVQA/slides
```

This writes:

```text
data/VEGAS_MEDIA/manifest.csv
```

and prepares paths such as:

```text
data/VEGAS_MEDIA/AriaEverydayActivities/raw/*.mp4
data/VEGAS_MEDIA/AriaEverydayActivities/foveated/*.mp4
data/VEGAS_MEDIA/SlideVQA/raw/*.jpg
data/VEGAS_MEDIA/SlideVQA/heatmap/*.png
```

If source media roots are omitted, the script still writes a manifest showing which files are missing.

## Running Experiments

Validate metadata and media paths without model calls:

```bash
python src/experiments/compute_vegas.py --config-name=ariaeveryday dry_run=true limit=5
python src/experiments/compute_vegas.py --config-name=slidevqa dry_run=true limit=5
```

Compute VEGAS scores:

```bash
python src/experiments/compute_vegas.py --config-name=ariaeveryday limit=5
python src/experiments/compute_vegas.py --config-name=slidevqa limit=5
```

Compute caption comparison metrics:

```bash
python src/experiments/evaluate_captions.py --config-name=ariaeveryday
```

## Key Components

### VEGAS Score Calculation

- `src/utils/vegas.py`: core VEGAS scoring implementation.
- `src/experiments/compute_vegas.py`: dataset-agnostic runner for AEA and SlideVQA.

### Model Inference

- `src/utils/model_inference.py`: Gemini, OpenAI GPT, and OpenAI-compatible local model adapters.

### Keyword Masking

- `src/utils/keyword_masking_utils.py`: GPT-based and TF-IDF keyword extraction plus masking helpers.

### Evaluation Metrics

- `src/utils/metrics.py`: CIDEr, BLEU/mBLEU, token IoU, lexical alpha, SBERT, SPICE, and related helpers.
- `src/experiments/evaluate_captions.py`: applies evaluation metrics to experiment outputs.

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{chen2026vegas,
  title={VEGAS: Human-Aligned Video Caption Evaluation via Gaze},
  author={Chen, Shenghui and Li, Po-han and Sun, Ximeng and Yang, Shijia and Barsoum, Emad and Liu, Zicheng and Chinchali, Sandeep and Topcu, Ufuk},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://arxiv.org/abs/2607.08489}
}
```
