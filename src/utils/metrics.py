import krippendorff
import numpy as np
import warnings
# from bert_score import score as bertscore
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a high-performance model
sbert_encoder_model_name = "all-mpnet-base-v2"
# sbert_encoder_model_name = "Alibaba-NLP/gte-large-en-v1.5"
sbert_model = SentenceTransformer(sbert_encoder_model_name, trust_remote_code=True)


def calculate_ttr(captions) -> float:
    """Calculates Type-Token Ratio for a set of captions."""
    all_tokens = " ".join(captions).lower().split()
    if not all_tokens:
        return 0
    return len(set(all_tokens)) / len(all_tokens)

def calculate_token_IoU(captions) -> float:
    """Calculates mean token-level IoU between all unique pairs of captions."""
    if len(captions) < 2:
        return 0.0
    scores = []
    for i in range(len(captions)):
        for j in range(i + 1, len(captions)):
            tokens1 = set(word_tokenize(captions[i].lower()))
            tokens2 = set(word_tokenize(captions[j].lower()))
            union_len = len(tokens1.union(tokens2))
            if union_len == 0:
                scores.append(0.0)
            else:
                scores.append(len(tokens1.intersection(tokens2)) / union_len)
    return np.mean(scores)


def calculate_mbleu(captions) -> float:
    """Calculates mean BLEU (mBLEU): strictly pairwise for all unique pairs."""
    cc = SmoothingFunction()
    scores = []
    tokenized = [c.lower().split() for c in captions]
    for i in range(len(tokenized)):
        for j in range(len(tokenized)):
            if i != j:
                hypothesis = tokenized[i]
                references = [tokenized[j]]
                # BLEU-4 score
                score = sentence_bleu(references, hypothesis, smoothing_function=cc.method1)
                scores.append(score)
    if not scores:
        return 0.0
    return np.mean(scores)


def calculate_mbleu_pairwise(captions1, captions2) -> float:
    """Calculates mean BLEU (mBLEU): captions1 vs captions2."""
    cc = SmoothingFunction()
    scores = []
    tokenized1 = [c.lower().split() for c in captions1]
    tokenized2 = [c.lower().split() for c in captions2]
    for i in range(len(tokenized1)):
        hypothesis = tokenized1[i]
        references = tokenized2
        # BLEU-4 score
        score = sentence_bleu(references, hypothesis, smoothing_function=cc.method1)
        scores.append(score)
    return np.mean(scores)


def calculate_lexical_alpha(captions) -> float:
    """Calculates Krippendorff's Alpha for a single video based on vocabulary.
    Higher disagreement (diversity) results in a lower Alpha.
    """
    # Tokenize captions into sets of unique words
    tokenized_caps = [set(word_tokenize(c.lower())) for c in captions]
    vocab = sorted(list(set().union(*tokenized_caps)))

    if not vocab:
        return 0.0

    # Create Reliability Matrix: (Coders/Participants x Units/Words)
    matrix = np.zeros((len(captions), len(vocab)))
    for p_idx, cap_tokens in enumerate(tokenized_caps):
        for w_idx, word in enumerate(vocab):
            if word in cap_tokens:
                matrix[p_idx, w_idx] = 1

    try:
        # Use nominal level because we are checking for the presence/absence of words
        return krippendorff.alpha(
            reliability_data=matrix,
            level_of_measurement="nominal",
        )
    except Exception as e:
        warnings.warn(f"All participants used exactly the same vocabulary (no variance)? Exception: {e}")
        return 1.0

def calculate_cider(vids: list[str], captions: list[list[str]]) -> tuple[list[str], list[float]]:
    """Calculates mean CIDEr using rotation from a list of captions and vids."""
    scorer = Cider()

    # 1. Group captions by video_id
    grouped = {vid: captions[i] for i, vid in enumerate(vids)}

    all_gts = {}
    all_res = {}
    rotations_per_video = []
    counter = 0

    # 2. Build the LOO structures
    for vid_id, captions in grouped.items():
        video_rotation_indices = []

        for i in range(len(captions)):
            # Reference: the single caption being "tested"
            all_res[counter] = [captions[i]]
            # Ground Truth: the other captions for that video
            all_gts[counter] = [captions[j] for j in range(len(captions)) if j != i]
            # print("-----------------------------")
            # print(all_res[counter])
            # print(all_gts[counter])
            # input()

            video_rotation_indices.append(counter)
            counter += 1

        rotations_per_video.append((vid_id, video_rotation_indices))

    # 3. Compute scores in batch
    _, individual_scores = scorer.compute_score(all_gts, all_res)

    # 4. Map scores back to the video IDs
    vids = []
    video_cider_means = []
    for vid_id, indices in rotations_per_video:
        video_mean = np.mean([individual_scores[idx] for idx in indices])
        video_cider_means.append(video_mean)
        vids.append(vid_id)

    return vids, video_cider_means

def calculate_pair_cider(vids: list[str], all_references: list[list[str]], all_captions: list[list[str]]) -> tuple[list[str], list[float]]:
    """Calculates mean CIDEr using rotation from a list of captions and vids."""
    assert len(vids) == len(all_references) == len(all_captions), "Length mismatch"
    scorer = Cider()

    # 1. Group captions by video_id
    grouped = {vid: list(all_captions[i]) for i, vid in enumerate(vids)}
    grouped_ref = {vid: list(all_references[i]) for i, vid in enumerate(vids)}

    all_gts = {}
    all_res = {}
    rotations_per_video = []
    counter = 0

    # 2. Build the LOO structures
    for vid_id, captions in grouped.items():
        video_rotation_indices = []

        # Reference: the single caption being "tested"
        assert len(grouped_ref[vid_id]) == 1, "Length mismatch"
        all_res[counter] = grouped_ref[vid_id]
        # Ground Truth: the other captions for that video
        all_gts[counter] = captions
        # print("-----------------------------")
        # print(all_res[counter])
        # print(all_gts[counter])
        # input()

        video_rotation_indices.append(counter)
        counter += 1

        rotations_per_video.append((vid_id, video_rotation_indices))

    # 3. Compute scores in batch
    _, individual_scores = scorer.compute_score(all_gts, all_res)

    # 4. Map scores back to the video IDs
    vids = []
    video_cider_means = []
    for vid_id, indices in rotations_per_video:
        video_mean = np.mean([individual_scores[idx] for idx in indices])
        video_cider_means.append(video_mean)
        vids.append(vid_id)

    return vids, video_cider_means

# def calculate_cider_loo_df(df) -> tuple[list[str], list[float]]:
#     """Calculates mean CIDEr using LOO rotation from a DataFrame."""
#     scorer = Cider()

#     # 1. Group captions by video_id
#     grouped = df.groupby("vid")["caption"].apply(list).to_dict()

#     all_gts = {}
#     all_res = {}
#     rotations_per_video = []
#     counter = 0

#     # 2. Build the LOO structures
#     for vid_id, captions in grouped.items():
#         video_rotation_indices = []

#         for i in range(len(captions)):
#             # Reference: the single caption being "tested"
#             all_res[counter] = [captions[i]]
#             # Ground Truth: the other captions for that video
#             all_gts[counter] = [captions[j] for j in range(len(captions)) if j != i]

#             video_rotation_indices.append(counter)
#             counter += 1

#         rotations_per_video.append((vid_id, video_rotation_indices))

#     # 3. Compute scores in batch
#     _, individual_scores = scorer.compute_score(all_gts, all_res)

#     # 4. Map scores back to the video IDs
#     vids = []
#     video_cider_means = []
#     for vid_id, indices in rotations_per_video:
#         video_mean = np.mean([individual_scores[idx] for idx in indices])
#         video_cider_means.append(video_mean)
#         vids.append(vid_id)

#     return vids, video_cider_means


# def calculate_cider_loo_df_pairwise(
#     df,
#     column_name1,
#     column_name2,
#     diagonal=False,
# ) -> tuple[list[str], list[float]]:
#     """Calculates mean CIDEr using LOO rotation from a DataFrame.

#     Diagonal = True for orthogonal evaluation
#     """
#     scorer = Cider()
#     print(df[column_name1].head().apply(list).to_dict())
#     print(df[column_name2].head().apply(list).to_dict())
#     # 1. Group captions by video_id
#     if diagonal:
#         grouped1 = df.groupby("vid")[column_name1].apply(list).to_dict()
#         grouped2 = df.groupby("vid")[column_name2].apply(list).to_dict()
#     else:
#         grouped1 = df[column_name1].apply(list).to_dict()
#         grouped2 = df[column_name2].apply(list).to_dict()
#     assert len(grouped1) == len(
#         grouped2,
#     ), "grouped1 and grouped2 have different lengths"
#     print(grouped1)
#     print(list(grouped1.values())[0])
#     assert len(list(grouped1.values())[0]) == 1, "grouped1 has more than 1 caption per video"
#     assert len(list(grouped2.values())[0]) == 1, "grouped2 has more than 1 caption per video"

#     all_gts = {}
#     all_res = {}
#     rotations_per_video = []
#     counter = 0

#     # 2. Build the LOO structures
#     for vid_id, captions in grouped1.items():
#         video_rotation_indices = []

#         for i in range(len(captions)):
#             # Reference: the single caption being "tested"
#             all_res[counter] = [captions[i]]
#             # Ground Truth: the other captions for that video
#             if diagonal:
#                 all_gts[counter] = [grouped2[vid_id][i]]
#             else:
#                 all_gts[counter] = grouped2[vid_id]
#                 assert len(grouped2[vid_id]) >= 1

#             video_rotation_indices.append(counter)
#             counter += 1

#         rotations_per_video.append((vid_id, video_rotation_indices))

#     assert len(all_gts) == len(all_res), "all_gts and all_res have different lengths"

#     # 3. Compute scores in batch
#     print(all_gts)
#     print(all_res)
#     input()
#     _, individual_scores = scorer.compute_score(all_gts, all_res)

#     # 4. Map scores back to the video IDs
#     vids = []
#     video_cider_means = []
#     for vid_id, indices in rotations_per_video:
#         video_mean = np.mean([individual_scores[idx] for idx in indices])
#         video_cider_means.append(video_mean)
#         vids.append(vid_id)

#     return vids, video_cider_means


def calculate_self_sbert(captions) -> float:
    """Input: A list of strings (captions generated for ONE image).
    Output: Similarity score (0 to 1). Higher is more similar.

    # Example Usage
    generated_captions = [
        "A dog running on the grass.",
        "A puppy playing in a field.",  # Semantically similar to #1
        "A car driving down the street." # Semantically very different
    ]

    score = calculate_self_sbert(generated_captions)
    print(f"Self-SBERT Similarity: {score:.4f}")
    """
    if len(captions) < 2:
        return 0.0

    # Encode all captions into vectors
    embeddings = sbert_model.encode(captions)

    # Calculate pairwise cosine similarity
    # This creates a similarity matrix (N x N)
    sim_matrix = cosine_similarity(embeddings)

    # Extract upper triangle values (excluding self-similarity of 1.0)
    # We only want unique pairs (A vs B), not (A vs A) or (B vs A) twice.
    n = sim_matrix.shape[0]

    # Get indices for the upper triangle, offset by 1 to exclude diagonal
    upper_tri_indices = np.triu_indices(n, k=1)
    unique_pairwise_scores = sim_matrix[upper_tri_indices]

    # Calculate mean similarity
    mean_similarity = np.mean(unique_pairwise_scores)
    return mean_similarity


def calculate_sbert_pairwise(captions1, captions2) -> float:
    """Input: Two lists of strings.
    Output: Similarity score (0 to 1). Higher is more similar.

    # Example Usage
    generated_captions = [
        "A dog running on the grass.",
        "A puppy playing in a field.",  # Semantically similar to #1
        "A car driving down the street." # Semantically very different
    ]

    score = calculate_self_sbert(generated_captions)
    print(f"Self-SBERT Similarity: {score:.4f}")
    """
    # Encode all captions into vectors
    embeddings1 = sbert_model.encode(captions1)
    embeddings2 = sbert_model.encode(captions2)
    m, n = embeddings1.shape[0], embeddings2.shape[0]
    embeddings = np.concatenate((embeddings1, embeddings2), axis=0)

    # Calculate pairwise cosine similarity
    # This creates a similarity matrix (m+n x m+n)
    sim_matrix_all = cosine_similarity(embeddings1, embeddings2)
    a = np.mean(sim_matrix_all)

    sim_matrix = cosine_similarity(embeddings)

    # Extract block upper triangle values (excluding self-similarity of 1.0)
    indices = []
    for i in range(m):
        for j in range(m, m + n, 1):
            indices.append((i, j))

    # Correctly index the specific (i, j) coordinates
    unique_pairwise_scores = sim_matrix[tuple(zip(*indices))] if indices else []

    # Calculate mean similarity
    mean_similarity = np.mean(unique_pairwise_scores) if len(unique_pairwise_scores) > 0 else 0.0
    assert np.isclose(a, mean_similarity), f"a: {a}, mean_similarity: {mean_similarity}"
    return float(mean_similarity)


def calculate_self_spice(captions) -> float:
    """Input: List of strings (captions for ONE image)
    Output: Self-SPICE Similarity score.

    # Example Usage
    # Note: This runs slower than SBERT because it calls a Java subprocess
    captions = [
        "A red car",
        "A scarlet automobile", # SPICE will see high overlap (synonyms handled)
        "A blue truck"          # SPICE will see low overlap (diff attribute/object)
    ]

    score = calculate_self_spice(captions)
    print(f"Self-SPICE Similarity: {score}")
    """
    if len(captions) < 2:
        return 0.0

    # Initialize the SPICE scorer
    scorer = Spice()

    # We must format data exactly how pycocoevalcap expects it:
    # It expects {image_id: [list_of_references]} and {image_id: [list_of_candidates]}
    # We will fake this by creating temporary IDs for every pair.

    gts = {}  # Ground Truths (References)
    res = {}  # Results (Candidates)

    pair_id = 0
    for i in range(len(captions)):
        for j in range(i + 1, len(captions)):
            # Create a "fake" image ID for this specific pair
            current_id = str(pair_id)

            # Caption i is the "Reference", Caption j is the "Candidate"
            gts[current_id] = [captions[i]]
            res[current_id] = [captions[j]]

            pair_id += 1

    # Run SPICE
    # This computes the score for every pair we set up above
    avg_score, individual_scores = scorer.compute_score(gts, res)

    # individual_scores contains the SPICE score for every pair
    # Note: SPICE returns a float for the average and a list for individuals
    return avg_score


def calculate_spice_pairwise(captions1, captions2) -> float:
    # Initialize the SPICE scorer
    scorer = Spice()

    # We must format data exactly how pycocoevalcap expects it:
    # It expects {image_id: [list_of_references]} and {image_id: [list_of_candidates]}
    # We will fake this by creating temporary IDs for every pair.

    gts = {}  # Ground Truths (References)
    res = {}  # Results (Candidates)

    pair_id = 0
    for i in range(len(captions1)):
        for j in range(len(captions2)):
            # Create a "fake" image ID for this specific pair
            current_id = str(pair_id)

            # Caption i is the "Reference", Caption j is the "Candidate"
            gts[current_id] = [captions1[i]]
            res[current_id] = [captions2[j]]

            pair_id += 1

    # Run SPICE
    # This computes the score for every pair we set up above
    avg_score, individual_scores = scorer.compute_score(gts, res)

    # individual_scores contains the SPICE score for every pair
    # Note: SPICE returns a float for the average and a list for individuals
    return avg_score


def calculate_bertscore_recall(
    captions_ref: list[str],
    captions_cand: list[str],
    device="cuda:0",
) -> float:
    """Compute mean BERTScore Recall (IDF-weighted) between two caption sets.

    captions_ref: list[str]  # human captions (REFERENCE)
    captions_cand: list[str] # GPT / orthogonal captions (CANDIDATE)

    Returns:
        mean recall score (0–1), higher = more reference content covered

    """
    # Create all reference–candidate pairs
    refs = []
    cands = []
    for ref in captions_ref:
        for cand in captions_cand:
            refs.append(ref)
            cands.append(cand)

    # Compute BERTScore
    # P, R, F are tensors with one value per (ref, cand) pair
    P, R, F = bertscore(
        cands,
        refs,
        # model_type=model_type,
        lang="en",
        idf=False,
        rescale_with_baseline=True,
        device=device,
        use_fast_tokenizer=True,
    )

    # We only care about Recall
    mean_recall = R.mean().item()
    return mean_recall

if __name__ == "__main__":
    captions1 = ["I am putting magazine on the table."]
    captions2 = ["I am putting magazine on the table."]
    print(calculate_token_IoU(captions1 + captions2))
    