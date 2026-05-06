"""Shared artifact-path helpers for multi-model experiment runs."""

from __future__ import annotations

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
DEFAULT_ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"

MODEL_ARTIFACT_DIRS = {
    #"Qwen/Qwen3.5-2B": "qwen3.5-2b",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5-1.5b-instruct",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5-3b-instruct",

    "microsoft/Phi-4-mini-reasoning": "phi-4-mini-reasoning",
    #"microsoft/Phi-3-mini-4k-instruct": "phi-3-mini-instruct",

    #"google/gemma-4-E2B-it": "gemma-4-e2b-it",
    "google/gemma-3-1b-it": "gemma-3-1b-it",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
    "google/gemma-3-12b-it": "gemma-3-12b-it",
    "google/gemma-2-2b-it": "gemma-2-2b-it",
    "google/gemma-2-9b-it": "gemma-2-9b-it",
    
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1": "Llama-Nemotron-8B",
    "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1": "Llama-Nemotron-4B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf": "Llama-2-7b-chat-hf",

    #"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "Deepseek-R1-1.5B",

    #"allenai/OLMo-2-0425-1B-Instruct": "olmo-2-1b"
}


def artifact_subdir(model_id: str | None) -> str:
    """Readable folder name for a model's experiment artifacts."""
    if not model_id:
        raise ValueError("Pass --model-id explicitly; artifacts are keyed by model.")
    if model_id not in MODEL_ARTIFACT_DIRS:
        known = ", ".join(sorted(MODEL_ARTIFACT_DIRS))
        raise ValueError(
            f"No artifact folder registered for {model_id!r}. "
            f"Add it to MODEL_ARTIFACT_DIRS in experiment_paths.py. Known: {known}"
        )
    return MODEL_ARTIFACT_DIRS[model_id]


def artifact_dir_for(model_id: str | None) -> Path:
    """Resolve the per-model artifact directory."""
    return DEFAULT_ARTIFACT_DIR / artifact_subdir(model_id)


def refusal_path_for(model_id: str | None) -> Path:
    """Canonical cached refusal-direction path for a model."""
    return artifact_dir_for(model_id) / "refusal.pt"


def add_model_arg(parser, *, default: str | None = None):
    parser.add_argument(
        "--model-id",
        type=str,
        default=default,
        required=default is None,
        help="Hugging Face model id. Artifacts live under artifacts/<short-model-name>/.",
    )


def add_artifact_args(parser, *, include_model_id: bool = True):
    """Compatibility shim for older scripts; prefer add_model_arg."""
    if include_model_id:
        add_model_arg(parser)
