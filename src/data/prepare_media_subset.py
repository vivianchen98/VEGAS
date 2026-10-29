from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_ffmpeg_clip(source_video: Path, dest_video: Path, start: float, end: float, overwrite: bool) -> bool:
    if dest_video.exists() and not overwrite:
        return True
    ensure_dir(dest_video.parent)
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-ss",
        str(start),
        "-to",
        str(end),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dest_video),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return dest_video.exists()


def find_aea_video(source_root: Path, vid: str, filename: str) -> Path:
    candidates = [
        source_root / f"{vid}.mp4",
        source_root / vid / filename,
        source_root / "data" / vid / filename,
    ]
    return next((path for path in candidates if path.exists()), candidates[-1])


def first_heatmap_for_slide(slide_root: Path, slide_num: int) -> Path | None:
    matches = sorted(slide_root.glob(f"*/{slide_num}_heatmap.png"))
    return matches[0] if matches else None


def copy_file(src: Path, dst: Path, overwrite: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists() and not overwrite:
        return True
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def prepare_aria(args: argparse.Namespace) -> list[dict[str, str]]:
    vegas_root = Path(args.vegas_root)
    metadata_root = vegas_root / "AriaEverydayActivities"
    captions = pd.read_csv(metadata_root / "aggregated_captions.csv")
    if args.limit > 0:
        captions = captions.head(args.limit)

    output_root = Path(args.output_root) / "AriaEverydayActivities"
    manifest = []
    source_root = Path(args.aea_source_root).expanduser() if args.aea_source_root else None

    for _, row in captions.iterrows():
        vid = str(row["vid"])
        action_id = str(row["action_id"])
        example_id = f"{vid}_{action_id}"
        start = float(row["start_time"])
        end = float(row["end_time"])
        raw_out = output_root / "raw" / f"{example_id}.mp4"
        masked_out = output_root / "foveated" / f"{example_id}.mp4"

        source_video = find_aea_video(source_root, vid, "video_rgb_raw.mp4") if source_root else None
        source_masked_video = find_aea_video(source_root, vid, "mps/eye_gaze/gaze_foveated_video_new.mp4") if source_root else None
        raw_ready = False
        masked_ready = False
        if source_video is not None:
            raw_ready = run_ffmpeg_clip(source_video, raw_out, start, end, args.overwrite)
        if source_masked_video is not None:
            masked_ready = run_ffmpeg_clip(source_masked_video, masked_out, start, end, args.overwrite)

        manifest.append(
            {
                "dataset": "ariaeveryday",
                "example_id": example_id,
                "source_id": vid,
                "action_id": action_id,
                "start_time": str(start),
                "end_time": str(end),
                "raw_path": str(raw_out),
                "masked_path": str(masked_out),
                "raw_ready": str(raw_ready),
                "masked_ready": str(masked_ready),
                "source_hint": str(source_video) if source_video is not None else "set --aea-source-root to original AEA videos",
                "note": "Uses mps/eye_gaze/gaze_foveated_video_new.mp4 when available under the AEA source root.",
            },
        )
    return manifest


def prepare_slidevqa(args: argparse.Namespace) -> list[dict[str, str]]:
    vegas_root = Path(args.vegas_root)
    metadata_root = vegas_root / "SlideVQA"
    captions = pd.read_csv(metadata_root / "aggregated_captions.csv")
    if args.limit > 0:
        captions = captions.head(args.limit)

    output_root = Path(args.output_root) / "SlideVQA"
    manifest = []
    slide_source_root = Path(args.slidevqa_source_root).expanduser() if args.slidevqa_source_root else None

    for _, row in captions.iterrows():
        slide_id = str(row["slide_id"])
        slide_num = int(row["slide_num"])
        example_id = f"{slide_id}_{slide_num}"
        raw_out = output_root / "raw" / f"{example_id}.jpg"
        masked_out = output_root / "heatmap" / f"{example_id}.png"

        raw_source = None
        if slide_source_root is not None:
            candidates = [
                slide_source_root / slide_id / f"{slide_num}.png",
                slide_source_root / slide_id / f"{slide_num}.jpg",
                slide_source_root / f"{slide_id}_{slide_num}.png",
                slide_source_root / f"{slide_id}_{slide_num}.jpg",
            ]
            raw_source = next((path for path in candidates if path.exists()), candidates[0])

        heatmap_source = first_heatmap_for_slide(metadata_root / "heatmap" / slide_id, slide_num)
        raw_ready = copy_file(raw_source, raw_out, args.overwrite) if raw_source is not None else False
        masked_ready = copy_file(heatmap_source, masked_out, args.overwrite) if heatmap_source is not None else False

        manifest.append(
            {
                "dataset": "slidevqa",
                "example_id": example_id,
                "source_id": slide_id,
                "action_id": str(slide_num),
                "start_time": "",
                "end_time": "",
                "raw_path": str(raw_out),
                "masked_path": str(masked_out),
                "raw_ready": str(raw_ready),
                "masked_ready": str(masked_ready),
                "source_hint": str(raw_source) if raw_source is not None else "set --slidevqa-source-root to original SlideVQA slide images",
                "note": "masked_path uses the first available participant heatmap from the VEGAS metadata dataset.",
            },
        )
    return manifest


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "dataset",
        "example_id",
        "source_id",
        "action_id",
        "start_time",
        "end_time",
        "raw_path",
        "masked_path",
        "raw_ready",
        "masked_ready",
        "source_hint",
        "note",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local media subset paths for VEGAS scoring.")
    parser.add_argument("--dataset", choices=["ariaeveryday", "slidevqa", "all"], default="all")
    parser.add_argument("--vegas-root", default="data/VEGAS", help="Downloaded vivianchen98/VEGAS dataset root.")
    parser.add_argument("--output-root", default="data/VEGAS_MEDIA", help="Output directory for prepared media subset.")
    parser.add_argument("--manifest", default="data/VEGAS_MEDIA/manifest.csv", help="Output manifest CSV path.")
    parser.add_argument("--limit", type=int, default=5, help="Number of rows to prepare per dataset; 0 means all.")
    parser.add_argument("--aea-source-root", default="", help="Directory containing original AEA videos named <vid>.mp4.")
    parser.add_argument("--slidevqa-source-root", default="", help="Directory containing original SlideVQA slide images.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    if args.dataset in {"ariaeveryday", "all"}:
        rows.extend(prepare_aria(args))
    if args.dataset in {"slidevqa", "all"}:
        rows.extend(prepare_slidevqa(args))

    manifest_path = Path(args.manifest)
    write_manifest(rows, manifest_path)
    ready = sum(row["raw_ready"] == "True" and row["masked_ready"] == "True" for row in rows)
    logger.info('%s', f"Wrote {len(rows)} rows to {manifest_path}")
    logger.info('%s', f"Fully ready examples: {ready}/{len(rows)}")
    logger.info('%s', "Use raw_path/masked_path from the manifest to configure raw_video_template and masked_video_template.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
