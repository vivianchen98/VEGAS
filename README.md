<div align="center">

# VEGAS: Human-Aligned Video Caption Evaluation via Gaze

**Transactions on Machine Learning Research (TMLR), 2026**

<a href="https://vivianchen98.github.io/VEGAS_website/"><img src="https://img.shields.io/badge/Website-VEGAS-blue?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Website"></a>
<a href="https://arxiv.org/abs/2607.08489"><img src="https://img.shields.io/badge/arXiv-2607.08489-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
<a href="https://huggingface.co/datasets/vivianchen98/VEGAS"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow?style=for-the-badge&logo=huggingface&logoColor=black" alt="Dataset"></a>

</div>

## TL;DR

VEGAS is a training-free metric for evaluating video captions with human gaze. It compares model behavior on raw and gaze-conditioned visual inputs to score whether caption tokens are grounded in visual evidence that people actually attended to.

## Setup

```bash
conda create -n vegas python=3.10 -y
conda activate vegas
pip install -e .
cp .env.example .env
```

Add API keys to `.env` if you use hosted model backends:

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

## Data

Download the released VEGAS metadata, captions, gaze, heatmaps, and segment files from Hugging Face:

```bash
python src/data/download_dataset.py --output-dir data/VEGAS
```

The Hugging Face release does not redistribute all original raw media from the source datasets. To run end-to-end scoring, first obtain local copies of the original datasets:

- Aria Everyday Activities: <https://huggingface.co/papers/2402.13349>
- SlideVQA: <https://huggingface.co/papers/2301.04883>

Then prepare local media files from your copies of those datasets:

```bash
python src/data/prepare_media_subset.py \
  --dataset all \
  --aea-source-root /path/to/original/AriaEverydayActivity \
  --slidevqa-source-root /path/to/original/SlideVQA/slides
```

## Run VEGAS

Compute VEGAS scores:

```bash
python src/experiments/compute_vegas.py --config-name=ariaeveryday
python src/experiments/compute_vegas.py --config-name=slidevqa
```

## Evaluation

Compute caption comparison metrics:

```bash
python src/experiments/evaluate_captions.py --config-name=ariaeveryday
```

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
