import os
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.metrics import (
    calculate_pair_cider,
    calculate_cider,
    calculate_lexical_alpha,
    calculate_mbleu,
    calculate_mbleu_pairwise,
    calculate_sbert_pairwise,
    calculate_self_sbert,
    calculate_spice_pairwise,
    calculate_token_IoU,
)

def process_metrics_human(
    cfg,
    df,
    metrics_map={
        # "token_IoU": calculate_token_IoU,
        # "mbleu": calculate_mbleu,
        # "alpha": calculate_lexical_alpha,
        # "sbert": calculate_self_sbert,
        # "sbert_gte": calculate_self_sbert,
        # "spice": calculate_self_spice,
        "cider": calculate_cider,
    },
    human_caption_column_names=["human_0", "human_1", "human_2", "human_3", "human_4"],
    vid_column_name="vid",
):
    # put all 5 human captions into one column into a list
    df["captions"] = df[human_caption_column_names].values.tolist()
    if vid_column_name not in df.columns:
        df[vid_column_name] = df["video_id"] + "_" + df["action_id"].astype(str)

    # Define and create save path
    save_dir = os.path.join(cfg.ds.root_path, "metrics", "human")
    os.makedirs(save_dir, exist_ok=True)

    for name, func in metrics_map.items():
        if name == "cider":
            continue

        # Calculate scores
        res = df["captions"].apply(func).reset_index(name="score")
        dest_path = os.path.join(save_dir, f"{name}.csv")
        res.to_csv(dest_path, index=False)
        print(f"Saved {name} to {dest_path}")
        
    # Handle CIDEr separately (special function signature)
    if "cider" in metrics_map:
        cider_vids, cider_scores = metrics_map["cider"](
            df[vid_column_name].tolist(),
            df["captions"].tolist(),
        )
        dest_path = os.path.join(cfg.ds.root_path, "metrics", "human", "cider.csv")
        pd.DataFrame({"vid": cider_vids, "score": cider_scores}).to_csv(
            dest_path, index=False
        )
        print(f"Saved cider to {dest_path}")

def process_metrics_human_vlm(
    cfg,
    vlm_df,
    save_subdir="human_vlm",
    metrics_map={
        "cider": calculate_pair_cider,
        "token_IoU": calculate_token_IoU,
        "mbleu": calculate_mbleu_pairwise,
        "alpha": calculate_lexical_alpha,
        "sbert": calculate_sbert_pairwise,
        # "spice": calculate_spice_pairwise,
        # "bertscore_recall": calculate_bertscore_recall,
    },
    vlm_caption_column_name="gemini_groudtruth",
    human_caption_column_names=["human_0", "human_1", "human_2", "human_3", "human_4"],
    vid_column_name="vid",
):
    # put all 5 human captions into one column into a list
    vlm_df["human_captions"] = vlm_df[human_caption_column_names].values.tolist()
    # Define and create save path
    save_dir = os.path.join(cfg.ds.root_path, "metrics", save_subdir)
    os.makedirs(save_dir, exist_ok=True)

    for name, func in metrics_map.items():
        if name == "cider":
            continue
        all_scores = np.zeros((len(vlm_df), len(human_caption_column_names)))
        for i, row in vlm_df.iterrows():
            vlm_caption = [row[vlm_caption_column_name]]
            for j, human_caption in enumerate(row["human_captions"]):
                # compute scores
                if name in ["token_IoU", "alpha"]:
                    score = func(vlm_caption + [human_caption])
                    all_scores[i, j] = score
                elif name in ["mbleu", "sbert", "spice", "sbert_gte", "bertscore_recall"]:
                    score = func(vlm_caption, [human_caption])
                    all_scores[i, j] = score
                else:
                    raise ValueError(f"Unknown metric: {name}")

        scores = all_scores.mean(axis=1)
        res = pd.DataFrame({"vid": vlm_df[vid_column_name], "score": scores})
       
        dest_path = os.path.join(save_dir, f"{name}.csv")
        res.to_csv(dest_path, index=False)
        print(f"Saved {name} to {dest_path}")

    # Handle CIDEr separately (special function signature)
    if "cider" in metrics_map:
        vids = vlm_df[vid_column_name].unique().tolist()
        human_captions_per_video = []
        vlm_captions = []
        for vid in vids:
            # print("vid", vid)
            vlm_caption = vlm_df[vlm_df[vid_column_name] == vid][vlm_caption_column_name].tolist()
            human_captions = vlm_df[vlm_df[vid_column_name] == vid][human_caption_column_names].values[0]
            # print(vlm_caption)
            # print(human_captions)
            # input()
            human_captions_per_video.append(human_captions)
            vlm_captions.append(vlm_caption)

        cider_vids, cider_scores = metrics_map["cider"](
            vids = vids,
            all_references = vlm_captions,
            all_captions = human_captions_per_video,
        )
        dest_path = os.path.join(cfg.ds.root_path, "metrics", save_subdir, "cider.csv")
        pd.DataFrame({"vid": cider_vids, "score": cider_scores}).to_csv(
            dest_path, index=False
        )
        print(f"Saved cider to {dest_path}")


def process_metrics_orthogonal(
    cfg,
    orthogonal_df,
    save_subdir="orig_gaze_masked",
    metrics_map={
        "cider": calculate_pair_cider,
        # "token_IoU": calculate_token_IoU,
        # "mbleu": calculate_mbleu_pairwise,
        # "alpha": calculate_lexical_alpha,
        # "sbert": calculate_sbert_pairwise,
        # "spice": calculate_spice_pairwise,
        # "bertscore_recall": calculate_bertscore_recall,
    },
    orig_caption_column_name="orig_caption",
    orth_caption_column_name="orth_caption",
    vid_column_name="vid",
):
    save_dir = os.path.join(cfg.ds.root_path, "metrics", save_subdir)
    os.makedirs(save_dir, exist_ok=True)

    # Clean data: drop rows with nan captions
    df = orthogonal_df.dropna(
        subset=[orig_caption_column_name, orth_caption_column_name],
    ).copy()

    vids = df[vid_column_name].unique().tolist()
    vid2caption_pairs = {}
    for vid in vids:
        vid2caption_pairs[vid] = (
            df[df[vid_column_name] == vid][orig_caption_column_name].tolist(),
            df[df[vid_column_name] == vid][orth_caption_column_name].tolist(),
        )

    # Save all pairwise results
    for name, func in metrics_map.items():
        # compute scores
        if name in ["token_IoU", "alpha"]:
            scores = []
            for vid in vids:
                score = func(vid2caption_pairs[vid][0] + vid2caption_pairs[vid][1])
                scores.append(score)
        elif name in ["mbleu", "sbert", "spice", "sbert_gte", "bertscore_recall"]:
            scores = []
            for vid in vids:
                scores_per_vid = []
                for i in range(len(vid2caption_pairs[vid][0])):
                    score = func(
                        [vid2caption_pairs[vid][0][i]],
                        [vid2caption_pairs[vid][1][i]],
                    )
                    scores_per_vid.append(score)
                scores.append(np.mean(scores_per_vid))
        elif name == "cider":
            all_references = []
            all_captions = []
            for vid in vids:
                all_references.extend([(vid2caption_pairs[vid][0])])
                all_captions.extend([(vid2caption_pairs[vid][1])])

            _, scores = func(
                vids,
                all_references,
                all_captions,
            )
        else:
            raise ValueError(f"Unknown metric: {name}")

        assert len(scores) == len(
            vids,
        ), f"Length mismatch for {name}, {len(scores)} != {len(vids)}"

        # save scores
        dest = os.path.join(save_dir, f"{name}.csv")
        pd.DataFrame({"vid": vids, "score": scores}).to_csv(dest, index=False)
        print(f"Saved {name} to {dest}")


@hydra.main(version_base=None, config_path="../../config", config_name="ariaeveryday")
def main(cfg):
    vegas_df = pd.read_csv(
        os.path.join(
            cfg.ds.root_path,
            "metrics/vegas/vegas_scores_full_gemma4_foveated.csv",
        )
    )
    vegas_df["orig_caption"] = "I " + vegas_df["original_guess"].astype(str) + "."
    vegas_df["orth_caption"] = "I " + vegas_df["masked_guess"].astype(str) + "."
    vegas_df["vid"] = vegas_df["video_id"] + "_" + vegas_df["action_id"].astype(str)

    # Compare Original Guess vs Masked Guess
    # process_metrics_orthogonal(
    #     cfg,
    #     vegas_df,
    #     "vegas_orig_gaze_foveated",
    #     orig_caption_column_name="orig_caption",
    #     orth_caption_column_name="orth_caption",
    #     vid_column_name="vid",
    # )

    human_caption = pd.read_csv(
        os.path.join(
            cfg.ds.root_path,
            "aggregated_captions.csv",
        )
    )
    human_caption["vid"] = human_caption["video_id"] + "_" + human_caption["action_id"].astype(str)
    # process_metrics_human(
    #     cfg,
    #     human_caption,
    # )

    process_metrics_human_vlm(
        cfg,
        human_caption,
    )

if __name__ == "__main__":
    main()
