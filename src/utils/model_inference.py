"""Model inference adapters used by VEGAS.

This module contains the shared base class plus Gemini, OpenAI GPT, and
OpenAI-compatible local-model clients.
"""

# ---- base class ----
import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv


class BaseAPIInference(ABC):
    def __init__(self):
        # Load environment variables from .env.local
        # load_dotenv(dotenv_path=".env.local")  # loads .env in current directory
        load_dotenv()
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    @abstractmethod
    def text_inference(self):
        pass

    @abstractmethod
    def video_inference(self):
        pass


# ---- Gemini adapter ----
import math
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types
from hydra import main
from omegaconf import DictConfig
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))


class GeminiInference(BaseAPIInference):
    def __init__(self, cfg: DictConfig = None):
        super().__init__()
        self.cfg = cfg
        self.client = genai.Client(api_key=self.GOOGLE_API_KEY)
        self.model = cfg.gemini_model if "gemini_model" in cfg else "gemini-3.1-pro-preview"
        self.google_drive_file_path2file_id = {}

    def count_tokens(
        self,
        text: str | None = None,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
    ) -> int:
        prompt = []
        if text is not None and len(text) > 0:
            prompt = [text]
        if image_paths is not None and len(image_paths) > 0:
            prompt += [Image.open(image_path) for image_path in image_paths]
        if video_paths is not None and len(video_paths) > 0:
            video_files = [self.upload_video(video_path) for video_path in video_paths]
            prompt += [video_files]

        # Count tokens using the new client method.
        sdk_http_response = self.client.models.count_tokens(
            model="gemini-2.5-pro",
            contents=prompt,
        )
        return sdk_http_response.total_tokens

    def upload_video(self, video_path: str) -> str:
        assert Path(video_path).exists(), f"Video path {video_path} does not exist."
        if self.google_drive_file_path2file_id.get(video_path, None) is None:
            # Upload the video file and wait until it's active
            video_file = self.client.files.upload(file=video_path)
            state = video_file.state.name
            while state != "ACTIVE":
                print(f"[{state} UPLOAD] {video_file.name}")
                time.sleep(10)  # wait for 10 seconds before checking again
                video_file = self.client.files.get(name=video_file.name)
                state = video_file.state.name
            self.google_drive_file_path2file_id[video_path] = video_file.name
        else:
            print("Reusing previously uploaded video file.")
            video_file = self.client.files.get(
                name=self.google_drive_file_path2file_id[video_path],
            )
        return video_file

    def video_inference(
        self,
        video_path: str,  # local path or youtube url link
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        logprobs: int | None = None,
        seed: int = 42,
        image_paths: list[str] | None = None,
        response_schema: any = None,
        **kwargs,
    ) -> tuple[str, list]:
        """Perform video inference using Gemini model. This function uploads the video to Google Drive.
        If the video has been uploaded before, it reuses the existing file. If image_paths is provided, it includes those images in the prompt as well.

        Args:
            video_path (str): Local path to the video file or YouTube URL.
            prompt (str): The prompt to guide the model's response.
            temperature (float): Sampling temperature for generation.
            max_output_tokens (int): Maximum number of tokens to generate.
            logprobs (int): Number of top log probabilities to return.
            seed (int): Random seed for generation.
            image_paths (list[str] | None): Optional list of image paths to include in the prompt.
            **kwargs: Additional keyword arguments for generation configuration.

        Returns:
            tuple[str, list]: Generated text and list of log probabilities.

        """
        response_logprobs = False if logprobs is None else True
        generation_config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            response_logprobs=response_logprobs,
            logprobs=logprobs,
            seed=seed,
            response_mime_type="application/json" if response_schema is not None else None,
            response_schema=response_schema,
            **kwargs,
        )
        video_file = self.upload_video(video_path)
        contents = (
            [video_file, prompt]
            if image_paths is None
            else [video_file] + [Image.open(p) for p in image_paths] + [prompt]
        )

        num_trial = 0
        response = None
        while num_trial < 3:
            num_trial += 1
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generation_config,
            )
            if response is not None and response.text is not None and response.candidates and response.candidates[0]:
                break

        if response is None or response.text is None:
            return "", []

        response_text = response.text
        response_logprobs = []
        if (
            response.candidates
            and response.candidates[0].logprobs_result
            and response.candidates[0].logprobs_result.top_candidates
        ):
            response_logprobs = response.candidates[0].logprobs_result.top_candidates

        return response_text, response_logprobs

    def multiple_image_inference(
        self,
        image_paths: list[str] | None,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        seed: int = 42,
        logprobs: int | None = None,
        response_schema: any = None,
        **kwargs,
    ) -> tuple[str, list]:
        response_logprobs = False if logprobs is None else True
        if image_paths is None or len(image_paths) == 0:
            contents = [prompt]
        else:
            contents = [Image.open(image_path) for image_path in image_paths] + [prompt]
        generation_config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            response_logprobs=response_logprobs,
            logprobs=logprobs,
            seed=seed,
            response_mime_type="application/json" if response_schema is not None else None,
            response_schema=response_schema,
            **kwargs,
        )
        num_trial = 0
        response = None
        while num_trial < 3:
            num_trial += 1
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generation_config,
            )
            if response is not None and response.text is not None:
                break

        if response is None or response.text is None:
            return "", []

        response_text = response.text
        response_logprobs = []
        if logprobs is not None and logprobs > 0:
            if (
                response.candidates
                and response.candidates[0].logprobs_result
                and response.candidates[0].logprobs_result.top_candidates
            ):
                response_logprobs = response.candidates[0].logprobs_result.top_candidates

        return response_text, response_logprobs

    def text_inference(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        seed: int = 42,
        logprobs: int | None = None,
        response_schema: any = None,
        **kwargs,
    ) -> tuple[str, list]:
        response_logprobs = False if logprobs is None else True
        generation_config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            response_logprobs=response_logprobs,
            logprobs=logprobs,
            seed=seed,
            response_mime_type="application/json" if response_schema is not None else None,
            response_schema=response_schema,
            **kwargs,
        )
        num_trial = 0
        response = None
        while num_trial < 3:
            num_trial += 1
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=generation_config,
            )
            if response is not None and response.text is not None:
                break

        if response is None or response.text is None:
            return "", []

        response_text = response.text
        response_logprobs = []
        if logprobs is not None and logprobs > 0:
            if (
                response.candidates
                and response.candidates[0].logprobs_result
                and response.candidates[0].logprobs_result.top_candidates
            ):
                response_logprobs = response.candidates[0].logprobs_result.top_candidates

        return response_text, response_logprobs

    def pretty_print_gemini_result(self, text, logprobs, max_items=120):
        # 1) Print the final text
        print("\n=== Model Response ===")
        print(text.strip())

        # 2) Tabular per-token view
        print("\n=== Token Logprobs ===")
        header = f"{'#':>4}  {'token':<20}  {'id':>8}  {'logp':>12}  {'prob':>12}  {'cum_logp':>12}"
        print(header)
        print("-" * len(header))

        cum = 0.0
        for i, c in enumerate(logprobs[:max_items]):
            token = getattr(c, "token", getattr(c, "text", ""))
            token_id = getattr(c, "token_id", None)
            logp = getattr(c, "log_probability", None)

            # Compute prob and running cumulative logp (if numeric)
            if isinstance(logp, (int, float)):
                prob = math.exp(logp)  # assumes natural log
                cum += logp
                logp_s = f"{logp:12.6f}"
                prob_s = f"{prob:12.6f}"
                cum_s = f"{cum:12.6f}"
            else:
                logp_s = prob_s = cum_s = " " * 12

            # Make spaces/newlines visible while staying readable
            token_disp = token.encode("unicode_escape").decode("ascii")
            if len(token_disp) > 20:
                token_disp = token_disp[:17] + "..."

            print(
                f"{i:>4}  {token_disp:<20}  {token_id!s:>8}  {logp_s}  {prob_s}  {cum_s}",
            )

        if len(logprobs) > max_items:
            print(f"... {len(logprobs) - max_items} more tokens omitted ...")


if __name__ == "__main__":
    ### test functions
    @main(version_base=None, config_path="../../config", config_name="base")
    def run(cfg):
        gemini_inference = GeminiInference(cfg)
        num_tokens = gemini_inference.count_tokens(
            text="This is a sample text to count tokens.",
            image_paths=[
                "path/to/example_image.png",
                "path/to/example_image_2.png",
            ],
        )
        print(f"Total tokens counted: {num_tokens}")

    run()


# ---- OpenAI GPT adapter ----
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
from hydra import main
from omegaconf import DictConfig
from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parents[2]))


class GPTInference(BaseAPIInference):
    def __init__(self, cfg: DictConfig = None):
        super().__init__()
        self.cfg = cfg
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.model = cfg.gpt_model if "gpt_model" in cfg else "gpt-5.2"

    def count_tokens(
        self,
        text: str | None = None,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
    ) -> int:
        raise NotImplementedError("Token counting not implemented for GPTInference.")

    def text_inference(
        self,
        prompt: str = "",
        max_tokens: int = 4096,
        seed: int = 42,
        system_prompt: str = "",
        **kwargs,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
            seed=seed,
            **kwargs,
        )
        return response.choices[0].message.content.strip()

    def video_inference(
        self,
        video_path: str = "",
        prompt: str = "",
        max_tokens: int = 4096,
        frame_sample_rate: int = 25,
        fps: int = 0,
        seed: int = 42,
        **kwargs,
    ) -> tuple[str, list] | tuple[str, None]:
        assert Path(video_path).exists(), f"Video path {video_path} does not exist."
        video = cv2.VideoCapture(video_path)
        base64Frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        if fps == 0 and frame_sample_rate > 0:
            base64Frames = [frame for frame in base64Frames[0::frame_sample_rate][:100]]
        else:
            # get duration of video from cv2
            duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
            # uniformly sample frames every fps seconds
            frame_indices = np.linspace(0, len(base64Frames) - 1, int(duration * fps))
            base64Frames = [base64Frames[int(i)] for i in frame_indices]

        video.release()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                        }
                        for frame in base64Frames
                    ],
                ],
            },
        ]
        if kwargs.get("logprobs"):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                seed=seed,
                **kwargs,
            )
            return (
                response.choices[0].message.content.strip(),
                response.choices[0].logprobs,
            )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            seed=seed,
            **kwargs,
        )
        return response.choices[0].message.content.strip(), None

    def multiple_image_inference(
        self,
        image_paths: list[str] = [],
        prompt: str = "",
        max_tokens: int = 4096,
        seed: int = 42,
        **kwargs,
    ) -> tuple[str, None]:
        base64Frames = []
        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                base64Frames.append(base64.b64encode(image_file.read()).decode("utf-8"))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                        }
                        for frame in base64Frames
                    ],
                ],
            },
        ]
        if "logprobs" in kwargs and kwargs["logprobs"] > 0:
            kwargs["logprobs"] = True
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                seed=seed,
                **kwargs,
            )
            return (
                response.choices[0].message.content.strip(),
                response.choices[0].logprobs,
            )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            seed=seed,
            **kwargs,
        )
        return response.choices[0].message.content.strip(), None


if __name__ == "__main__":

    @main(version_base=None, config_path="../../config", config_name="base")
    def run(cfg):
        gpt_inference = GPTInference(cfg)
        response_text, _ = gpt_inference.video_inference(
            "path/to/example_video.mp4",
            "Describe the video.",
            fps=1,
        )
        # response_text = gpt_inference.text_inference(
        #     "Tell me what is Turing Test.",
        # )
        print(response_text)

    run()


# ---- OpenAI-compatible local model adapter ----
import asyncio
import base64
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from openai import AsyncOpenAI, OpenAI

sys.path.append(str(Path(__file__).resolve().parents[2]))


class LocalModelInference(BaseAPIInference):
    """Inference client for locally served models with OpenAI-compatible API (e.g., vLLM).

    Usage:
        In config, add:
            local_model_name: "Qwen/Qwen3.5-35B-A3B"
            local_model_base_url: "http://qwen3-5-ql98m.x-flannel-pretraining.svc.cluster.local:8000/v1"
    """

    def __init__(self, cfg: DictConfig = None):
        # Don't call super().__init__() to avoid loading .env.local
        self.cfg = cfg

        # Get model configuration
        self.model = cfg.get("local_model_name", "Qwen/Qwen3.5-35B-A3B")
        self.base_url = cfg.get(
            "local_model_base_url",
            "http://qwen3-5-ql98m.x-flannel-pretraining.svc.cluster.local:8000/v1",
        )

        # Initialize OpenAI client with local endpoint
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="EMPTY",  # vLLM doesn't require real API key
        )

        # Initialize async client for batch processing
        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="EMPTY",
        )

        print(f"✓ LocalModelInference initialized: {self.model}")
        print(f"  Base URL: {self.base_url}")

    @staticmethod
    def strip_cot_thinking(text: str) -> str:
        """Strip chain-of-thought thinking process from model output.

        Removes:
        - <think> or <thinking> tags and their content
        - Step-by-step analysis patterns (numbered lists, bullet points)
        - Meta-commentary and reasoning phrases
        - Draft/revision markers

        Returns only the final descriptive paragraph.
        """
        if not text:
            return text

        # Remove <think> or <thinking> tags and everything inside them
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Split into lines for pattern-based filtering
        lines = text.split("\n")

        # Filter out lines that are part of CoT reasoning
        filtered_lines = []
        skip_section = False

        for line in lines:
            line_stripped = line.strip()

            # Skip numbered analysis patterns (e.g., "1. Analyze the frames:", "2. Draft:")
            if re.match(r"^\d+\.\s*\*\*.*?\*\*:", line_stripped) or re.match(
                r"^\d+\.\s+(Analyze|Draft|Synthesize|Refining|Final)",
                line_stripped,
                re.IGNORECASE,
            ):
                skip_section = True
                continue

            # Skip meta-commentary patterns
            if any(
                pattern in line_stripped.lower()
                for pattern in [
                    "let me analyze",
                    "the user wants",
                    "i will",
                    "draft 1:",
                    "revised draft:",
                    "final check:",
                    "self-correction:",
                    "*draft",
                    "*refining",
                    "frame 1:",
                    "frame 2:",
                    "final:",
                    "final polish:",
                ]
            ):
                skip_section = True
                continue

            # Resume if we hit a paragraph that looks like actual description
            if line_stripped and not skip_section:
                filtered_lines.append(line)
            elif line_stripped and skip_section:
                # Check if this line starts a proper description (not meta)
                if (
                    not re.match(r"^\d+\.", line_stripped)
                    and not line_stripped.startswith("*")
                    and not line_stripped.startswith("-")
                    and len(line_stripped) > 30
                ):  # Actual descriptions are typically longer
                    skip_section = False
                    filtered_lines.append(line)

        # Join filtered lines
        result = "\n".join(filtered_lines).strip()

        # If result is too short or empty, try to extract the last substantial paragraph
        if not result or len(result) < 50:
            # Look for the last paragraph that's substantial
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for para in reversed(paragraphs):
                # Skip if it looks like meta-commentary
                if (
                    not any(
                        pattern in para.lower()
                        for pattern in [
                            "analyze",
                            "draft",
                            "frame 1",
                            "frame 2",
                            "steps:",
                            "synthesize",
                        ]
                    )
                    and len(para) > 50
                ):
                    result = para
                    break

        return result

    def count_tokens(
        self,
        text: str | None = None,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
    ) -> int:
        raise NotImplementedError("Token counting not implemented for LocalModelInference.")

    def text_inference(
        self,
        prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int = 42,
        system_prompt: str = "",
        **kwargs,
    ) -> str:
        """Basic text inference."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            **kwargs,
        )
        return self.strip_cot_thinking(response.choices[0].message.content.strip())

    def video_inference(
        self,
        video_path: str = "",
        prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        frame_sample_rate: int = 25,
        fps: int = 1,
        max_frames: int | None = None,
        seed: int = 42,
        **kwargs,
    ) -> tuple[str, list] | tuple[str, None]:
        """Video inference by sampling frames and encoding them as base64.

        Args:
            video_path: Path to video file
            prompt: Text prompt for captioning
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            frame_sample_rate: Sample every Nth frame (when fps=0)
            fps: Frames per second to sample (when > 0, overrides frame_sample_rate)
            max_frames: Maximum number of sampled frames to send. If set, frames are uniformly subsampled after fps/stride sampling.
            seed: Random seed
            **kwargs: Additional arguments for completion

        Returns:
            Tuple of (generated_text, logprobs or None)

        """
        video = cv2.VideoCapture(video_path)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = video.get(cv2.CAP_PROP_FPS)
        base64Frames = []

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        # Sample frames based on fps or frame_sample_rate
        if fps == 0 and frame_sample_rate > 0:
            base64Frames = [frame for frame in base64Frames[0::frame_sample_rate][:100]]
        else:
            if frame_count <= 0 or video_fps <= 0 or len(base64Frames) == 0:
                video.release()
                raise ValueError(f"Unreadable or empty video for inference: {video_path}")
            duration = frame_count / video_fps
            frame_indices = np.linspace(0, len(base64Frames) - 1, int(duration * fps))
            base64Frames = [base64Frames[int(i)] for i in frame_indices]

        if max_frames is not None and max_frames > 0 and len(base64Frames) > max_frames:
            frame_indices = np.linspace(0, len(base64Frames) - 1, max_frames)
            base64Frames = [base64Frames[int(i)] for i in frame_indices]

        video.release()

        # Construct messages with text and images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                        }
                        for frame in base64Frames
                    ],
                ],
            },
        ]

        # Make API call with retry logic for transient connection errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if "logprobs" in kwargs and kwargs["logprobs"] > 0:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        seed=seed,
                        **kwargs,
                    )
                    return (
                        self.strip_cot_thinking(response.choices[0].message.content.strip()),
                        response.choices[0].logprobs,
                    )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    **kwargs,
                )
                return self.strip_cot_thinking(response.choices[0].message.content.strip()), None
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 5 * (2 ** attempt)  # 5s, 10s, 20s
                    print(f"  [Retry {attempt + 1}/{max_retries}] Connection error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

    def multiple_image_inference(
        self,
        image_paths: list[str] = [],
        prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int = 42,
        **kwargs,
    ) -> tuple[str, None]:
        """Inference on multiple images (for frame-based captioning).

        Args:
            image_paths: List of image file paths
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            seed: Random seed
            **kwargs: Additional arguments

        Returns:
            Tuple of (generated_text, None)

        """
        base64Frames = []
        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                base64Frames.append(base64.b64encode(image_file.read()).decode("utf-8"))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                        }
                        for frame in base64Frames
                    ],
                ],
            },
        ]

        if "logprobs" in kwargs and kwargs["logprobs"] > 0:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                **kwargs,
            )
            return (
                self.strip_cot_thinking(response.choices[0].message.content.strip()),
                response.choices[0].logprobs,
            )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            **kwargs,
        )
        return self.strip_cot_thinking(response.choices[0].message.content.strip()), None

    async def video_inference_async(
        self,
        video_path: str = "",
        prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        frame_sample_rate: int = 25,
        fps: int = 1,
        seed: int = 42,
        **kwargs,
    ) -> tuple[str, None]:
        """Async video inference for concurrent batch processing.

        Args:
            video_path: Path to video file
            prompt: Text prompt for captioning
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            frame_sample_rate: Sample every Nth frame (when fps=0)
            fps: Frames per second to sample (when > 0, overrides frame_sample_rate)
            seed: Random seed
            **kwargs: Additional arguments for completion

        Returns:
            Tuple of (generated_text, None)

        """
        video = cv2.VideoCapture(video_path)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = video.get(cv2.CAP_PROP_FPS)
        base64Frames = []

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        # Sample frames based on fps or frame_sample_rate
        if fps == 0 and frame_sample_rate > 0:
            base64Frames = [frame for frame in base64Frames[0::frame_sample_rate][:100]]
        else:
            if frame_count <= 0 or video_fps <= 0 or len(base64Frames) == 0:
                video.release()
                raise ValueError(f"Unreadable or empty video for inference: {video_path}")
            duration = frame_count / video_fps
            frame_indices = np.linspace(0, len(base64Frames) - 1, int(duration * fps))
            base64Frames = [base64Frames[int(i)] for i in frame_indices]

        video.release()

        # Construct messages with text and images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                        }
                        for frame in base64Frames
                    ],
                ],
            },
        ]

        # Make async API call
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            **kwargs,
        )
        return self.strip_cot_thinking(response.choices[0].message.content.strip()), None

    async def batch_video_inference(
        self,
        video_data: list[tuple],
        max_concurrent: int = 8,
    ) -> list[tuple[str, str, Exception | None]]:
        """Process multiple videos concurrently with rate limiting.

        Args:
            video_data: List of tuples (video_path, video_id, prompt, kwargs_dict)
            max_concurrent: Maximum number of concurrent requests

        Returns:
            List of tuples (video_id, caption, error)

        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(video_path, video_id, prompt, kwargs):
            async with semaphore:
                try:
                    caption, _ = await self.video_inference_async(
                        video_path=video_path,
                        prompt=prompt,
                        **kwargs,
                    )
                    return (video_id, caption, None)
                except Exception as e:
                    return (video_id, None, e)

        tasks = [process_one(vpath, vid, prompt, kwargs) for vpath, vid, prompt, kwargs in video_data]

        return await asyncio.gather(*tasks)

