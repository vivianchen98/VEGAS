
import logging

logger = logging.getLogger(__name__)
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from hydra import main
from omegaconf import DictConfig

sys.path.append(str(Path(__file__).resolve().parents[2]))
try:
    from src.utils.model_inference import GeminiInference
except ImportError:
    class GeminiInference:
        pass

try:
    from src.utils.model_inference import GPTInference
except ImportError:
    class GPTInference:
        pass

try:
    from src.utils.model_inference import LocalModelInference
except ImportError:
    class LocalModelInference:
        pass

def cal_token_prob(
    token_prob_object: list[dict],
    masked_word_gt: list[str],
    model: str,
    default_logprob: float = -5.0,
    verbose: bool = False,
) -> tuple[list[float], list[int]]:
    """Match the guessed words with the token probabilities from the model output. The match is case-insensitive and only the prefix match is considered.

    Args:
        token_prob (list[dict]): List of token probabilities from the model output.
        masked_word_gt (list[str]): List of ground truth words for the masked positions.
        model (str): The model name, either 'gemini' or 'gpt'.
        default_logprob (float, optional): Default log probability to assign if no match is found. Defaults to -20.0.

    Returns:
        list[float]: List of log probabilities for each guessed word.
        list[int]: List of guessed rank of vlm model of the matched words. Rank is 0-indexed and the last rank is num_candidates + 1 if no match is found.

    """
    if token_prob_object == []:
        return [default_logprob], [-1]
    token_guessed, top_logprobs = [], []  # size of num masked words x num candidates
    if "gemini" in model:
        num_candidates = len(token_prob_object[0].candidates)
        for token_dict in token_prob_object:
            candidates = token_dict.candidates
            for candidate in candidates:  # LogprobsResultCandidate object
                token_guessed.append(candidate.token.strip())
                top_logprobs.append(candidate.log_probability)
    elif "gpt" in model or model == "local":
        num_candidates = len(token_prob_object.content[0].top_logprobs)
        for token_info in token_prob_object.content:
            for top_lp in token_info.top_logprobs:
                token_guessed.append(top_lp.token.strip())
                top_logprobs.append(top_lp.logprob)
    else:
        raise ValueError(f"Model {model} not supported.")

    # only keep the logprobs for correct words
    matched_indices = []
    logprob_list = []
    rank_list = []

    for i, word_gt in enumerate(masked_word_gt):
        word_gt = word_gt.strip().lower()
        correct_guess = False
        for j in range(i * num_candidates, len(token_guessed)):
            token_guessed[j] = token_guessed[j].strip().lower()
            if (
                # token already matched, skip
                j in matched_indices
            ):
                continue
            if token_guessed[j] != "" and word_gt.startswith(token_guessed[j]):
                if verbose:
                    logger.debug('%s', f"Match found for '{word_gt}' with '{token_guessed[j]}' at index {j}")
                logprob_list.append(top_logprobs[j])
                correct_guess = True
                matched_indices.append(j)
                rank_list.append(j % num_candidates)
                break
        if not correct_guess:
            logprob_list.append(default_logprob)
            rank_list.append(num_candidates + 1)
            if verbose:
                logger.debug('%s', f"No match found for {word_gt}. Assigning default logprob {default_logprob}")

    assert len(logprob_list) == len(
        masked_word_gt,
    ), f"Length of logprob_list does not match masked_word_gt. {len(logprob_list)} vs {len(masked_word_gt)}"
    return logprob_list, rank_list


def _count_defaulted_rankings(rankings: list[int], top_logprobs_k: int) -> int:
    return sum(1 for rank in rankings if rank < 0 or rank >= top_logprobs_k)


def vegas_scoring_with_masking(
    inference_instance: GeminiInference | GPTInference | LocalModelInference,
    video_raw_path: str,
    video_masked_path: str,
    prompt: str,
    candidates: list[str] | None,  # list of candidate responses
    candidate_keywords: list[str] | None,  # list of candidate keywords
    gamma: float = 1.0,  # hyperparameter to balance the two terms: logP(seq|raw) - gamma * logP(seq|masked)
    keyword_token: str = "[MASK]",
    multithread: bool = True,
    top_logprobs_k: int = 5,
    default_logprob: float = -5.0,
    length_normalize: bool = False,
    video_fps: int | None = None,
    video_frame_sample_rate: int | None = None,
    video_max_frames: int | None = None,
) -> dict[str, float]:
    """Score candidates using contrastive decoding.

    Args:
        inference_instance (GeminiInference | GPTInference): Inference instance.
        video_raw_path (str): Path to raw video.
        video_masked_path (str): Path to masked video.
        prompt (str): Prompt for scoring.
        candidates (list[str], optional): List of candidate responses.
        candidate_keywords (list[str], optional): List of candidate keywords.
        gamma (float, optional): Contrastive penalty weight. Defaults to 1.0.
        keyword_token (str, optional): Token to replace the keywords with. Defaults to "[MASK]".
        multithread (bool, optional): Whether to use multithread for scoring. Defaults to True.
        top_logprobs_k (int, optional): Number of top token log probabilities to request. Defaults to 5.
        default_logprob (float, optional): Log probability for target tokens outside top-k. Defaults to -5.0.
        length_normalize (bool, optional): Divide contrastive score by number of scored keywords. Defaults to False.
        video_fps (int, optional): FPS used by video inference. If omitted, uses the inference client's default.
        video_frame_sample_rate (int, optional): Frame stride used when video_fps is 0. If omitted, uses the inference client's default.
        video_max_frames (int, optional): Maximum sampled frames to send per video. If omitted, uses the inference client's default.

    Returns:
        dict[str, float]: dictionary where:
            score (float): contrastive score
            raw_logprobs (list[float]): list of raw logprobs for each candidate.
            masked_logprobs (list[float]): list of masked logprobs for each candidate.
            raw_rankings (list[int]): list of raw rankings for each candidate.
            masked_rankings (list[int]): list of masked rankings for each candidate.

    """
    if candidate_keywords == []:
        return 0, [], [], [], []
    if top_logprobs_k < 1:
        raise ValueError(f"top_logprobs_k must be >= 1, got {top_logprobs_k}")
    candidate_scores = {}
    video_kwargs = {}
    if video_fps is not None:
        video_kwargs["fps"] = video_fps
    if video_frame_sample_rate is not None:
        video_kwargs["frame_sample_rate"] = video_frame_sample_rate
    if video_max_frames is not None:
        video_kwargs["max_frames"] = video_max_frames

    # mask the candidates by replacing the keywords with [MASK]
    masked_candidates = []
    for cand, keywords in zip(candidates, candidate_keywords):
        masked_cand = cand
        for keyword in keywords:
            masked_cand = masked_cand.replace(keyword, keyword_token)
        masked_candidates.append(masked_cand)

    # Determine model type
    if isinstance(inference_instance, GeminiInference):
        model_type = "gemini"
        kwargs = {
            "logprobs": top_logprobs_k,
            "top_p": 1,
        }
    elif isinstance(inference_instance, GPTInference):
        model_type = "gpt"
        kwargs = {
            "logprobs": 1,
            "top_logprobs": top_logprobs_k,
            "max_tokens": 40,
        }
    elif isinstance(inference_instance, LocalModelInference):
        model_type = "local"
        kwargs = {
            "logprobs": 1,
            "top_logprobs": top_logprobs_k,
            "max_tokens": 40,
            "temperature": 0.0,
            "top_p": 1.0,
        }
    else:
        raise ValueError(f"Unsupported model type: {type(inference_instance)}")
    
    
    def _score_single_candidate(cand, masked_cand, keywords):
        score = 0.0

        # Construct prompt for the prefix
        input_prompt = f"{prompt}\nCaption:\n{masked_cand}"

        # Get logprobs of the next token
        original_guess, raw_lp = inference_instance.video_inference(
            video_raw_path,
            input_prompt,
            **video_kwargs,
            **kwargs,
        )
        logger.debug('%s %s', "Original video guess:", original_guess)
        masked_guess, masked_lp = inference_instance.video_inference(
            video_masked_path,
            input_prompt,
            **video_kwargs,
            **kwargs,
        )
        logger.debug('%s %s', "Masked video guess:", masked_guess)

        raw_logprob, raw_ranking = cal_token_prob(
            raw_lp,
            keywords,
            model=model_type,
            default_logprob=default_logprob,
        )
        masked_logprob, masked_ranking = cal_token_prob(
            masked_lp,
            keywords,
            model=model_type,
            default_logprob=default_logprob,
        )
        raw_score = np.sum(raw_logprob)
        masked_score = np.sum(masked_logprob)

        contrastive_score = raw_score - (gamma * masked_score)
        n_scored_tokens = len(keywords)
        if length_normalize and n_scored_tokens > 0:
            contrastive_score = contrastive_score / n_scored_tokens
        score += contrastive_score

        logger.debug('%s', f"Candidate: {cand} | Keywords: {keywords} | Raw: {raw_score:.4f}, Masked: {masked_score:.4f}, Contrastive: {contrastive_score:.4f}")
        return (
            cand,
            score,
            raw_logprob,
            masked_logprob,
            raw_ranking,
            masked_ranking,
            original_guess,
            masked_guess,
            n_scored_tokens,
            _count_defaulted_rankings(raw_ranking, top_logprobs_k),
            _count_defaulted_rankings(masked_ranking, top_logprobs_k),
        )

    if multithread:
        with ThreadPoolExecutor(max_workers=len(masked_candidates)) as executor:
            futures = [
                executor.submit(_score_single_candidate, c, mc, keywords)
                for c, mc, keywords in zip(
                    candidates,
                    masked_candidates,
                    candidate_keywords,
                )
            ]
            results = (f.result() for f in as_completed(futures))
    else:
        results = (
            _score_single_candidate(c, mc, keywords)
            for c, mc, keywords in zip(
                candidates,
                masked_candidates,
                candidate_keywords,
            )
        )

    for (
        cand,
        score,
        raw_logprobs,
        masked_logprobs,
        raw_rankings,
        masked_rankings,
        original_guess,
        masked_guess,
        n_scored_tokens,
        n_missing_raw,
        n_missing_masked,
    ) in results:
        candidate_scores[cand] = {
            "score": score,
            "raw_logprobs": raw_logprobs,
            "masked_logprobs": masked_logprobs,
            "raw_rankings": raw_rankings,
            "masked_rankings": masked_rankings,
            "original_guess": original_guess,
            "masked_guess": masked_guess,
            "n_scored_tokens": n_scored_tokens,
            "n_missing_raw": n_missing_raw,
            "n_missing_masked": n_missing_masked,
            "top_logprobs_k": top_logprobs_k,
            "default_logprob": default_logprob,
            "length_normalize": length_normalize,
        }

    return candidate_scores


def vegas_scoring(
    inference_instance: GeminiInference | GPTInference,
    video_raw_path: str,
    video_masked_path: str,
    prompt: str,
    prompt_captioning: str = None,
    candidates: list[str] | None = None,  # list of candidate responses
    num_candidates: int = 5,
    gamma: float = 1.0,  # hyperparameter to balance the two terms: logP(seq|raw) - gamma * logP(seq|masked)
    multithread: bool = True,
) -> dict[str, float]:
    """Score candidates using contrastive decoding.

    Args:
        inference_instance (GeminiInference | GPTInference): Inference instance.
        video_raw_path (str): Path to raw video.
        video_masked_path (str): Path to masked video.
        prompt (str): Prompt for scoring.
        prompt_captioning (str, optional): Prompt for captioning. Defaults to None.
        candidates (list[str], optional): List of candidate responses. Defaults to None.
        num_candidates (int, optional): Number of candidates to generate if none provided. Defaults to 5.
        gamma (float, optional): Contrastive penalty weight. Defaults to 1.0.
        multithread (bool, optional): Whether to use multithread for scoring. Defaults to True.

    Returns:
        dict[str, float]: dictionary where:
            score (float): contrastive score
            raw_logprobs (list[float]): list of raw logprobs for each candidate.
            masked_logprobs (list[float]): list of masked logprobs for each candidate.
            raw_rankings (list[int]): list of raw rankings for each candidate.
            masked_rankings (list[int]): list of masked rankings for each candidate.

    """
    if candidates == [] or candidates is None:
        # Generate diverse candidates using temperature (parallel)
        def _generate_candidate():
            text, _ = inference_instance.video_inference(
                video_path=video_raw_path,
                prompt=prompt_captioning,
                temperature=0.9,
                max_output_tokens=50,
            )
            return text

        with ThreadPoolExecutor(max_workers=num_candidates) as executor:
            futures = [executor.submit(_generate_candidate) for _ in range(num_candidates)]
            candidates = []
            for future in as_completed(futures):
                text = future.result()
                if text and text not in candidates:
                    candidates.append(text)

    candidate_keywords = []
    for cand in candidates:
        candidate_keywords.append(cand.split())

    candidate_scores = {}

    # Determine model type
    model_type = "gemini" if isinstance(inference_instance, GeminiInference) else "gpt"
    kwargs = (
        {
            "logprobs": 5,
            "top_p": 1,
        }
        if isinstance(inference_instance, GeminiInference)
        else {
            "logprobs": 1,
            "top_logprobs": 5,
            "max_tokens": 1,
        }
    )

    def _score_single_candidate(cand):
        raw_logprobs = []
        masked_logprobs = []
        raw_rankings = []
        masked_rankings = []
        score = 0.0
        cand_parts = cand.strip().split()
        if not cand_parts:
            return cand, 0.0

        for i in range(len(cand_parts)):
            token_text = cand_parts[i]
            prefix_text = " ".join(cand_parts[:i])
            # Construct prompt for the prefix
            input_prompt = f"{prompt}\nCaption: {prefix_text}" if prefix_text else f"{prompt}\nCaption:"

            # Get logprobs of the next token
            _, raw_lp = inference_instance.video_inference(
                video_raw_path,
                input_prompt,
                logprobs=1,
                max_output_tokens=1,
            )
            _, masked_lp = inference_instance.video_inference(
                video_masked_path,
                input_prompt,
                logprobs=1,
                max_output_tokens=1,
            )

            raw_logprob, raw_ranking = cal_token_prob(
                raw_lp,
                [token_text],
                model=model_type,
            )
            masked_logprob, masked_ranking = cal_token_prob(
                masked_lp,
                [token_text],
                model=model_type,
            )
            raw_score = raw_logprob[0]
            masked_score = masked_logprob[0]
            raw_logprobs.append(raw_score)
            masked_logprobs.append(masked_score)
            raw_rankings.append(raw_ranking[0])
            masked_rankings.append(masked_ranking[0])

            contrastive_score = raw_score - (gamma * masked_score)
            score += contrastive_score

            logger.debug('%s', f"Candidate: {cand} | Token: {token_text} | raw: {raw_score:.4f}, masked: {masked_score:.4f}, contrastive: {contrastive_score:.4f}")
        return cand, score, raw_logprobs, masked_logprobs, raw_rankings, masked_rankings

    if multithread:
        with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
            futures = [executor.submit(_score_single_candidate, c) for c in candidates]
            results = (f.result() for f in as_completed(futures))
    else:
        results = (_score_single_candidate(c) for c in candidates)

    for (
        cand,
        score,
        raw_logprobs,
        masked_logprobs,
        raw_rankings,
        masked_rankings,
    ) in results:
        candidate_scores[cand] = {
            "score": score,
            "raw_logprobs": raw_logprobs,
            "masked_logprobs": masked_logprobs,
            "raw_rankings": raw_rankings,
            "masked_rankings": masked_rankings,
        }

    return candidate_scores


def contrastive_token_decoding(
    inference_instance: GeminiInference | GPTInference,
    video_raw_path: str,
    video_masked_path: str,
    prompt: str,
    max_steps: int = 20,
    gamma: float = 0.05,
    temperature: float | None = None,
):
    """Step-by-step token selection.
    Note: Extremely slow on API as it requires 2 calls per token.
    """
    # Determine model type
    model_type = "gemini" if isinstance(inference_instance, GeminiInference) else "gpt"

    current_text = ""

    for _ in range(max_steps):
        # Get top-5 candidates for the next token from both inputs
        # We append current_text to the prompt to 'force' the state
        current_prompt = f"{prompt}\nCaption: {current_text}"
        logger.debug('%s %s', "current_prompt:", current_prompt)

        _, raw_top = inference_instance.video_inference(
            video_raw_path,
            current_prompt,
            logprobs=5,
            max_output_tokens=50,
        )
        _, masked_top = inference_instance.video_inference(
            video_masked_path,
            current_prompt,
            logprobs=5,
            max_output_tokens=50,
        )

        best_token_score = -float("inf")
        selected_token = ""

        if model_type == "gemini":
            # Extract candidates from the first (and only) token position
            # raw_top is a list of LogprobsResult with .candidates list
            raw_candidates = raw_top[0].candidates
            masked_candidates = masked_top[0].candidates

            # Compute log-normalizers for softmax normalization: p~ = softmax(logits/T)
            # In log-space: log(p~) = (logits/T) - log(sum(exp(logits/T)))
            T = temperature if temperature is not None else 1.0

            raw_lps_scaled = [c.log_probability / T for c in raw_candidates]
            masked_lps_scaled = [c.log_probability / T for c in masked_candidates]
            if temperature is None:
                log_Z_raw = 0
                log_Z_masked = 0
            else:
                log_Z_raw = np.logaddexp.reduce(raw_lps_scaled)
                log_Z_masked = np.logaddexp.reduce(masked_lps_scaled)
        else:
            # Placeholder for GPT logic if needed in future
            raise NotImplementedError("GPT not fully supported in token decoding yet.")

        # Contrastive Logic: iterate through tokens proposed by the 'Expert' (raw video)
        for cand in raw_candidates:
            token = cand.token
            raw_lp = cand.log_probability

            # Get masked logprob using helper
            masked_lp_list, _ = cal_token_prob(
                masked_top,
                [token],
                model=model_type,
                default_logprob=-20.0,
                verbose=False,
            )
            masked_lp = masked_lp_list[0]

            # Normalize scaled logits before subtraction
            raw_lp_norm = (raw_lp / T) - log_Z_raw
            masked_lp_norm = (masked_lp / T) - log_Z_masked
            score = raw_lp_norm - (gamma * masked_lp_norm)
            logger.debug('%s %s %s %s %s %s %s %s', "token", token, "raw_lp", raw_lp, "masked_lp", masked_lp, "score", score)

            if score > best_token_score:
                best_token_score = score
                selected_token = token

        if (
            not selected_token
            or "<|EOS|>" in current_text  # GPT-2 EOS
            or "<EOS>" in current_text  # GPT-2 EOS
            or "<|endoftext|>" in current_text  # Gemini EOS
            or selected_token.startswith("<ctrl")
        ):
            break
        current_text += selected_token
        logger.debug('%s', f"Generating: {current_text}")  # Feedback since this is slow
    return current_text


@main(version_base=None, config_path="../../config", config_name="slidevqa")
def test(cfg: DictConfig) -> None:
    inference_instance = GeminiInference(cfg)
    orig_video = cfg.get("orig_video", "path/to/original_video.mp4")
    heatmap_video = cfg.get("heatmap_video", "path/to/gaze_masked_video.mp4")

    candidate2score = vegas_scoring_with_masking(
        inference_instance,
        orig_video,
        heatmap_video,
        "Given the video, predict the next word of the caption. Do not say anything else but the next word.",
        candidates=[
            "People are eating meat.",
            "The room is well-lit.",
            "The wall is gray.",
        ],
        candidate_keywords=[
            ["eating", "meat"],
            ["room", "well-lit"],
            ["wall", "gray"],
        ],
    )
    logger.debug('%s', candidate2score)



if __name__ == "__main__":
    test()
