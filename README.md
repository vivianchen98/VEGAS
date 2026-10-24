<div align="center">

# VEGAS: Human-Aligned Video Caption Evaluation via Gaze

**Transactions on Machine Learning Research (TMLR), 2026**

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://vivianchen98.github.io/VEGAS_website/)
[![Paper](https://img.shields.io/badge/arXiv-2607.08489-b31b1b)](https://arxiv.org/abs/2607.08489)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/vivianchen98/VEGAS)

</div>

## TL;DR

VEGAS is a training-free metric for evaluating video captions with human gaze. It compares model behavior on raw and gaze-conditioned visual inputs to score whether caption tokens are grounded in visual evidence that people actually attended to.

## Resources

- **Project page:** <https://vivianchen98.github.io/VEGAS_website/>
- **Paper:** <https://arxiv.org/abs/2607.08489>
- **Dataset:** <https://huggingface.co/datasets/vivianchen98/VEGAS>

## Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Running VEGAS](#running-vegas)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Overview

VEGAS evaluates captions using paired visual inputs:

- a **raw** image/video input;
- a **gaze-conditioned** version that emphasizes attended regions;
- a candidate caption to be evaluated.

The metric estimates how much caption-token likelihood changes under gaze conditioning, producing a gaze-aligned caption quality score. This public repository contains the cleaned implementation used for the released VEGAS experiments on Aria Everyday Activities and SlideVQA-derived examples.

## Repository Structure

```text
VEGAS/
├── config/                     # Hydra configs for released experiments
│   ├── base.yaml
│   ├── prompt.yaml
│   ├── ariaeveryday.yaml
│   └── slidevqa.yaml
├── src/
│   ├── data/                   # Dataset download and media subset preparation
│   │   ├── download_dataset.py
│   │   └── prepare_media_subset.py
│   ├── experiments/            # Experiment entry points
│   │   ├── compute_vegas.py
│   │   └── evaluate_captions.py
│   └── utils/                  # Reusable VEGAS/model/metric utilities
│       ├── vegas.py
│       ├── model_inference.py
│       ├── keyword_masking_utils.py
│       └── metrics.py
├── .env.example
├── pyproject.toml
└── README.md
```

Local data, prepared media, and experiment outputs are ignored by Git:

```text
data/
├── VEGAS/                      # Hugging Face metadata, captions, gaze, heatmaps
└── VEGAS_MEDIA/                # Local raw and gaze-conditioned media subset
```

## Setup

Create a conda environment:

```bash
conda create -n vegas python=3.10 -y
conda activate vegas
pip install -e .
```

Alternatively, use `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create a local environment file:

```bash
cp .env.example .env
```

Add API keys if you use hosted model backends:

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

## Data Preparation

### 1. Download VEGAS metadata

Download the released VEGAS dataset from Hugging Face:

```bash
python src/data/download_dataset.py --output-dir data/VEGAS
```

This downloads the public captions, gaze/fixation metadata, heatmaps, and segment metadata from `vivianchen98/VEGAS`, then updates `.env` with dataset roots.

### 2. Prepare media for runnable examples

The Hugging Face release includes VEGAS metadata and derived gaze resources, but does not redistribute all original raw media from Aria Everyday Activities or SlideVQA. To run end-to-end scoring, prepare a small local media subset from your local copies of those datasets:

```bash
python src/data/prepare_media_subset.py \
  --dataset all \
  --limit 5 \
  --aea-source-root /path/to/original/AriaEverydayActivity \
  --slidevqa-source-root /path/to/original/SlideVQA/slides
```

This creates:

```text
data/VEGAS_MEDIA/manifest.csv
data/VEGAS_MEDIA/AriaEverydayActivities/raw/*.mp4
data/VEGAS_MEDIA/AriaEverydayActivities/foveated/*.mp4
data/VEGAS_MEDIA/SlideVQA/raw/*.jpg
data/VEGAS_MEDIA/SlideVQA/heatmap/*.png
```

If source media roots are omitted, the script still writes a manifest that reports missing files.

## Running VEGAS

Run a quick dry run to validate metadata and media paths without model calls:

```bash
python src/experiments/compute_vegas.py --config-name=ariaeveryday dry_run=true limit=5
python src/experiments/compute_vegas.py --config-name=slidevqa dry_run=true limit=5
```

Compute VEGAS scores:

```bash
python src/experiments/compute_vegas.py --config-name=ariaeveryday limit=5
python src/experiments/compute_vegas.py --config-name=slidevqa limit=5
```

Remove `limit=5` to run the full released split after your data and model backend are configured.

## Evaluation

Compute caption comparison metrics:

```bash
python src/experiments/evaluate_captions.py --config-name=ariaeveryday
```

Core implementation files:

- `src/utils/vegas.py`: VEGAS scoring implementation.
- `src/utils/model_inference.py`: Gemini, OpenAI GPT, and OpenAI-compatible local model adapters.
- `src/utils/keyword_masking_utils.py`: GPT-based and TF-IDF keyword extraction/masking utilities.
- `src/utils/metrics.py`: CIDEr, BLEU/mBLEU, token IoU, lexical alpha, SBERT, SPICE, and related caption metrics.

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
