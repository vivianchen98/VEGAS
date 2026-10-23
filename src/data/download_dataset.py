from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DATASET_ID = "vivianchen98/VEGAS"


def download_dataset(output_dir: str | Path = "data/VEGAS", repo_id: str = DATASET_ID) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=output_path,
        local_dir_use_symlinks=False,
    )
    return output_path


def write_env_file(dataset_root: Path, env_path: str | Path = ".env") -> None:
    dataset_root = dataset_root.resolve()
    aria_root = dataset_root / "AriaEverydayActivities"
    slidevqa_root = dataset_root / "SlideVQA"

    if not aria_root.exists():
        aria_root = dataset_root / "AriaEveryday"
    if not aria_root.exists():
        aria_root = dataset_root / "ariaeveryday"
    if not slidevqa_root.exists():
        slidevqa_root = dataset_root / "slidevqa"

    lines = [
        f"ARIAEVERYDAY_ROOT={aria_root if aria_root.exists() else dataset_root}",
        f"SLIDEVQA_ROOT={slidevqa_root if slidevqa_root.exists() else dataset_root}",
    ]

    env_file = Path(env_path)
    existing = env_file.read_text().splitlines() if env_file.exists() else []
    kept = [line for line in existing if not line.startswith(("ARIAEVERYDAY_ROOT=", "SLIDEVQA_ROOT="))]
    env_file.write_text("\n".join([*kept, *lines, ""]) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public VEGAS dataset from Hugging Face.")
    parser.add_argument("--repo-id", default=DATASET_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--output-dir", default="data/VEGAS", help="Local dataset directory.")
    parser.add_argument("--env-path", default=".env", help="Path to update with dataset roots.")
    parser.add_argument("--skip-env", action="store_true", help="Do not write dataset roots to .env.")
    args = parser.parse_args()

    output_path = download_dataset(args.output_dir, args.repo_id)
    print(f"Downloaded {args.repo_id} to {output_path}")
    if not args.skip_env:
        write_env_file(output_path, args.env_path)
        print(f"Updated {args.env_path} with ARIAEVERYDAY_ROOT and SLIDEVQA_ROOT")


if __name__ == "__main__":
    main()
