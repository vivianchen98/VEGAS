
import logging

logger = logging.getLogger(__name__)
import ast
import os
import re
import sys

from hydra import main
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from src.utils.model_inference import GPTInference


def mask_keywords(paragraph: str, keywords: list[str], masked_str: str = "[MASK]") -> str:
    masked_paragraph = paragraph
    for keyword in keywords:
        num_words_in_phrase = len(keyword.split())
        replacement = " ".join([masked_str] * num_words_in_phrase) if num_words_in_phrase > 1 else masked_str
        masked_paragraph = masked_paragraph.replace(keyword, replacement)
    return masked_paragraph


def mask_keywords_regex(paragraph: str, keywords: list[str], masked_str: str = "[MASK]") -> str:
    masked_paragraph = paragraph
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        masked_paragraph = re.sub(pattern, masked_str, masked_paragraph)
    return masked_paragraph


def mask_keywords_separated(paragraph: str, keywords: list[str], masked_str: str = "[MASK]") -> str:
    masked_paragraph = paragraph
    for keyword_text in keywords:
        words = [keyword_text] if " " not in keyword_text else keyword_text.split()[0]
        for keyword in words:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            masked_paragraph = re.sub(pattern, masked_str, masked_paragraph)
    return masked_paragraph


def tfidf_filter_keywords(
    paragraph: str,
    corpus: list[str] | None = None,
    max_keywords: int = 20,
    ngram_range: tuple[int, int] = (1, 3),
    stop_words: str | list[str] | None = "english",
) -> list[str]:
    """Extract keywords using TF-IDF scores.

    If no corpus is provided, the paragraph is scored against itself, which still
    gives a deterministic frequency-based keyword baseline. Passing a larger
    domain corpus makes the scores more discriminative.
    """
    documents = [paragraph] if corpus is None else [*corpus, paragraph]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=stop_words,
        ngram_range=ngram_range,
    )
    tfidf = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf[-1].toarray().ravel()
    ranked_indices = scores.argsort()[::-1]

    keywords = []
    for index in ranked_indices:
        if scores[index] <= 0:
            break
        keywords.append(feature_names[index])
        if len(keywords) >= max_keywords:
            break
    return keywords


def gpt_filter_keywords(
    paragraph: str,
    prompt: str | None = None,
    gpt_inference: GPTInference | None = None,
    cfg: DictConfig | None = None,
    max_tokens: int = 4096,
) -> list[str]:
    """Filters keywords from a given paragraph using a specified prompt.

    Args:
        paragraph (str): The input paragraph from which to extract keywords.
        prompt (Optional[str]): The prompt to guide the keyword extraction process.
        gpt_inference (Optional[GPTInference]): An instance of GPTInference for making API calls.
        cfg (Optional[DictConfig]): Configuration object containing settings. Used for default prompt and GPTInference if not provided.

    Returns:
        str: A string of extracted keywords.

    """
    if prompt is None:
        prompt = cfg.keyword_masking_prompt
    if gpt_inference is None:
        gpt_inference = GPTInference(cfg)

    response_format = """Your mandatory JSON-only output is validated against this API schema:
                        "json_schema": {
                            "name": "keywords",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "keywords": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of the selected keywords extracted from the paragraph.",
                                    }
                                },
                                "required": ["keywords"],
                                "additionalProperties": False,
                            },
                        },
                        """.strip()
    prompt = f"{prompt}\n\nParagraph: {paragraph}"
    response = gpt_inference.text_inference(
        prompt=prompt,
        max_tokens=max_tokens,
        system_prompt=response_format,
    )
    response = response.replace("```json", "").replace("```", "").strip()
    # convert response to dict
    response_dict = ast.literal_eval(response)
    keywords = response_dict.get("keywords", [])
    return keywords


if __name__ == "__main__":

    @main(version_base=None, config_path="../../config", config_name="base")
    def run(cfg):
        gpt_inference = GPTInference(cfg)
        paragraph = "This thrilling travel adventure montage opens with a couple ziplining through a mountainous jungle and enjoying a romantic sunset over a sprawling city. The clips continue, showing them swimming in beautiful turquoise waterfalls, snorkeling over coral reefs, and getting up close with majestic elephants. As the scenes of their journey unfold, the title 'Christian's Travel Vlogs' appears, capturing the excitement of their experiences."
        keywords = gpt_filter_keywords(
            paragraph,
            gpt_inference=gpt_inference,
            cfg=cfg,
        )
        # keywords = [
        #     "opens with",
        #     "ziplining through jungle",
        #     "enjoying",
        #     "continue",
        #     "showing",
        #     "swimming in",
        #     "snorkeling over",
        #     "getting up close with",
        #     "unfold",
        #     "appears",
        #     "capturing",
        # ]
        logger.debug('%s', keywords)
        masked_paragraph = mask_keywords_separated(
            paragraph,
            keywords,
        )
        logger.debug('%s', masked_paragraph)

    run()
