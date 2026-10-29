
import logging

logger = logging.getLogger(__name__)
import csv
import os
import re
import sys
from pathlib import Path
from string import Formatter

from hydra import main
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm


sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.vegas import vegas_scoring_with_masking
from src.utils.runtime import load_environment, configure_logging

load_environment()

# Example:
# python src/experiments/compute_vegas.py --config-name=ariaeveryday video_type=gaze_foveated caption_col=gemini_groudtruth_edited
# python src/experiments/compute_vegas.py --config-name=slidevqa caption_col=caption

def build_inference_instance(cfg: DictConfig):
    from src.utils.model_inference import GeminiInference, GPTInference, LocalModelInference

    model_name = str(cfg.get("vegas_model", cfg.get("caption_model", "gemini"))).lower()
    if "gemini" in model_name:
        return model_name, GeminiInference(cfg)
    if "gpt" in model_name:
        return model_name, GPTInference(cfg)
    return "local", LocalModelInference(cfg)


def get_scoring_controls(cfg: DictConfig) -> dict:
    return {
        "gamma": float(cfg.get("vegas_gamma", 1.0)),
        "top_logprobs_k": int(cfg.get("top_logprobs_k", 5)),
        "default_logprob": float(cfg.get("default_logprob", -5.0)),
        "length_normalize": bool(cfg.get("length_normalize", False)),
        "video_fps": cfg.get("video_fps"),
        "video_frame_sample_rate": cfg.get("video_frame_sample_rate"),
        "video_max_frames": cfg.get("video_max_frames"),
    }


def format_path(template: str, row: pd.Series, ds_root: Path) -> Path:
    values = row.to_dict()
    values["root_path"] = str(ds_root)
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name and field_name not in values:
            raise KeyError(f"Missing column '{field_name}' required by path template: {template}")
    return Path(template.format(**values))


def resolve_existing_captions_csv(cfg: DictConfig, ds_root: Path) -> tuple[Path, Path]:
    captions_csv = Path(cfg.get("captions_csv", cfg.ds.get("captions_csv", ds_root / "aggregated_captions.csv")))
    if captions_csv.exists():
        return ds_root, captions_csv

    for child_name in ("AriaEverydayActivities", "AriaEveryday", "SlideVQA"):
        candidate_root = ds_root / child_name
        candidate_csv = candidate_root / "aggregated_captions.csv"
        if candidate_csv.exists():
            logger.info('%s', f"Using detected dataset root: {candidate_root}")
            return candidate_root, candidate_csv

    return ds_root, captions_csv


@main(version_base=None, config_path="../../config", config_name="ariaeveryday")
def compute_vegas(cfg: DictConfig) -> None:
    configure_logging()
    caption_col: str = cfg.get("caption_col", cfg.ds.get("caption_column", "caption"))
    logger.info('%s', f"Using caption column: {caption_col}")

    ds_root = Path(cfg.ds.root_path)
    if not str(ds_root):
        raise ValueError(
            "Dataset root is empty. Run `python src/data/download_dataset.py --output-dir data/VEGAS` "
            "or set ARIAEVERYDAY_ROOT / SLIDEVQA_ROOT in your shell environment."
        )
    masked_video_type = cfg.get("video_type", "gaze_foveated")  # gaze_foveated

    ds_root, captions_csv = resolve_existing_captions_csv(cfg, ds_root)
    caption_sep = str(cfg.get("caption_sep", cfg.ds.get("caption_sep", ",")))
    if not captions_csv.exists():
        raise FileNotFoundError(
            f"Captions CSV not found: {captions_csv}\n"
            "Check that the Hugging Face dataset downloaded successfully and that the dataset root in `.env` is correct."
        )
    caption_df = pd.read_csv(captions_csv, sep=caption_sep)
    limit = int(cfg.get("limit", 0) or 0)
    if limit > 0:
        caption_df = caption_df.head(limit)
    id_column = str(cfg.get("id_column", cfg.ds.get("id_column", "video_id")))
    action_id_column = cfg.get("action_id_column", cfg.ds.get("action_id_column", "action_id"))
    raw_video_template = cfg.get("raw_video_template", cfg.ds.get("raw_video_template"))
    masked_video_template = cfg.get("masked_video_template", cfg.ds.get("masked_video_template"))
    require_media = bool(cfg.get("require_media", True))

    # Prepare output directory
    output_dir = Path(cfg.get("output_dir", ds_root / "metrics" / "vegas"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dry_run = bool(cfg.get("dry_run", False))
    if dry_run:
        model_key, inference_instance = "dryrun", None
    else:
        model_key, inference_instance = build_inference_instance(cfg)
    output_file = output_dir / f"vegas_scores_{caption_col}_{model_key}_{masked_video_type}.csv"
    logger.info('%s %s', "output_file", output_file)
    scoring_controls = get_scoring_controls(cfg)

    # Load existing results if any to support resume
    processed_pairs = set()
    if output_file.exists() and output_file.stat().st_size > 0:
        try:
            results_df = pd.read_csv(output_file)
            for _, r in results_df.iterrows():
                processed_pairs.add(str(r["example_id"]))
        except Exception as e:
            logger.warning('%s', f"Warning: Could not load existing CSV {output_file}: {e}")
            processed_pairs = set()
        logger.debug('%s %s %s %s', "processed_pairs", processed_pairs, "len(processed_pairs)", len(processed_pairs))

    result_fields = [
        "video_id",
        "action_id",
        "example_id",
        "score",
        "raw_logprobs",
        "masked_logprobs",
        "raw_rankings",
        "masked_rankings",
        "caption",
        "original_guess",
        "masked_guess",
        "scorer_model",
    ]
    output_exists = output_file.exists() and output_file.stat().st_size > 0
    out_f = open(output_file, "a", newline="")
    csv_writer = csv.DictWriter(out_f, fieldnames=result_fields)
    if not output_exists:
        csv_writer.writeheader()
        out_f.flush()
        os.fsync(out_f.fileno())

    try:
        for _, row in tqdm(caption_df.iterrows(), total=len(caption_df), desc="Computing VEGAS scores"):
            video_id = str(row.get(id_column))
            action_id = row.get(action_id_column) if action_id_column else ""
            example_id = f"{video_id}_{action_id}" if action_id_column else video_id
            logger.debug('%s', f"Processing row with example_id={example_id}")

            # Use the user-selected caption column
            caption = row.get(caption_col)
            logger.debug('%s', f"Caption for {example_id}: {caption}")
            if pd.isna(caption) or caption is None:
                logger.warning('%s', f"Warning: no caption for {video_id} {action_id}, skipping.")
                continue

            video_path = format_path(str(raw_video_template), row, ds_root) if raw_video_template else None
            masked_video_path = format_path(str(masked_video_template), row, ds_root) if masked_video_template else None

            if example_id in processed_pairs:
                logger.info('%s', f"Skipping {example_id} as it has been processed.")
                continue

            media_available = bool(video_path and masked_video_path and video_path.exists() and masked_video_path.exists())
            if require_media and not media_available:
                if not dry_run:
                    if video_path and not video_path.exists():
                        logger.warning('%s', f"Warning: Raw video {video_path} not found. Skipping.")
                    if masked_video_path and not masked_video_path.exists():
                        logger.warning('%s', f"Warning: Masked video {masked_video_path} not found. Skipping.")
                    if not video_path or not masked_video_path:
                        logger.warning('%s', "Warning: raw_video_template/masked_video_template not configured. Skipping.")
                    continue

            if dry_run:
                media_msg = "media OK" if media_available else "metadata OK; media not configured/found"
                logger.info('%s', f"Dry run OK: {example_id} ({media_msg})")
                continue

            # Parse keywords and candidates
            keywords = [k.lower() for k in re.split(r"[-\s.,;!?]+", caption) if k and k != "I"]
            candidates = [caption]
            candidate_keywords = [keywords]

            logger.debug('%s %s', "Processing", example_id)

            try:
                scores = vegas_scoring_with_masking(
                    inference_instance=inference_instance,
                    video_raw_path=str(video_path),
                    video_masked_path=str(masked_video_path),
                    prompt=cfg.vegas_prompt,
                    candidates=candidates,
                    candidate_keywords=candidate_keywords,
                    multithread=False,
                    **scoring_controls,
                )

                # scores is a dict: {caption: {'score': ..., 'raw_logprobs': ..., 'raw_caption': ..., 'masked_caption': ...}}
                result_data = scores[caption]
                result_row = {
                    "video_id": video_id,
                    "action_id": action_id,
                    "example_id": example_id,
                    "score": result_data["score"],
                    "raw_logprobs": result_data["raw_logprobs"],
                    "masked_logprobs": result_data["masked_logprobs"],
                    "raw_rankings": result_data["raw_rankings"],
                    "masked_rankings": result_data["masked_rankings"],
                    "caption": caption,
                    "original_guess": result_data["original_guess"],
                    "masked_guess": result_data["masked_guess"],
                    "scorer_model": model_key,
                }

                # Persist each row immediately to be resilient to unexpected breaks.
                csv_writer.writerow(result_row)
                out_f.flush()
                os.fsync(out_f.fileno())

                processed_pairs.add(example_id)
            except Exception as e:
                logger.error('%s', f"Error processing {video_id} (action={action_id}): {e}")
                continue
    finally:
        out_f.close()


if __name__ == "__main__":
    compute_vegas()
