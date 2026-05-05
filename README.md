source .venv/bin/activate

model_library = {
    "gemma 4": "google/gemma-4-E2B-it",
    "gemma 2": "google/gemma-2-2b-it",
    "qwen3.5": "Qwen/Qwen3.5-2B",
    "qwen2.5": "Qwen/Qwen2.5-1.5B-Instruct",
    "phi": "microsoft/Phi-4-mini-reasoning",
    "llama": "meta-llama/Llama-3.2-3B-Instruct"
    "nemotron": "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF"
}


python3 experiment1/compute_refusal_direction.py --model-id "MODEL"
python3 experiment1/calibrated_projection.py --model-id "MODEL" --intervention shift --token-scope all