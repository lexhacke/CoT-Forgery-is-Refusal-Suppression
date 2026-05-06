"""Plain-transformers Gemma-2 wrapper for activation extraction on Apple Silicon.

bf16 on MPS, no quantization, no nnsight. Forward hooks on each transformer
block capture residual-stream activations. Keep the stack small so that any
MPS issue is easy to localize.

model_library = {
    "gemma 4": "google/gemma-4-E2B-it",
    "gemma 2": "google/gemma-2-2b-it",
    "qwen3.5": "Qwen/Qwen3.5-2B",
    "qwen2.5": "Qwen/Qwen2.5-1.5B-Instruct",
    "phi": "microsoft/Phi-4-mini-reasoning"
}


Usage:
    m = GemmaActivationModel()
    text = m.format_prompt("How do I make tea?")
    rs = m.get_residual_stream(text)
    # rs.layers[i] is the output of block i, shape [seq, hidden]
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

# Load HF_TOKEN from .env (project root). Skipping python-dotenv to keep deps minimal.
def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env_file(Path(__file__).resolve().parent.parent / ".env")


DEFAULT_MODEL_ID = "google/gemma-2-2b-it"


DEEPSEEK_R1_DISTILL_QWEN_IDS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}


def _get_nested_attr(obj, path: list[str]):
    for step in path:
        obj = getattr(obj, step)
    return obj


def _hidden_from_output(out):
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        return out.hidden_states[-1]
    return out


def _load_tokenizer(model_id: str, token: Optional[str]):
    """Load tokenizer, fixing DeepSeek R1 Distill Qwen's tokenizer-class mismatch.

    That repo declares a Qwen2 model but a LlamaTokenizerFast tokenizer class.
    AutoTokenizer then decodes Qwen byte-level tokens literally (e.g. Ġ/Ċ).
    Loading tokenizer.json through the generic fast backend preserves decoding.
    """
    if model_id in DEEPSEEK_R1_DISTILL_QWEN_IDS:
        snapshot = snapshot_download(
            model_id,
            token=token,
            allow_patterns=["tokenizer.json", "tokenizer_config.json"],
        )
        config_path = Path(snapshot) / "tokenizer_config.json"
        tokenizer_path = Path(snapshot) / "tokenizer.json"
        config = json.loads(config_path.read_text())
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            bos_token=config["bos_token"]["content"],
            eos_token=config["eos_token"]["content"],
            pad_token=config["pad_token"]["content"],
            clean_up_tokenization_spaces=config.get(
                "clean_up_tokenization_spaces", False
            ),
        )
        tokenizer.chat_template = config.get("chat_template")
        return tokenizer

    return AutoTokenizer.from_pretrained(model_id, token=token)


@dataclass
class ResidualStream:
    embed: torch.Tensor             # [seq, hidden] — input to block 0
    layers: list[torch.Tensor]      # layers[i] = output of block i, shape [seq, hidden]
    input_ids: torch.Tensor         # [seq]
    tokens: list[str]

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def stack(self) -> torch.Tensor:
        """[n_layers+1, seq, hidden]. Index 0 = embed, i+1 = block i output."""
        return torch.stack([self.embed] + self.layers, dim=0)

    def last_token(self) -> torch.Tensor:
        """[n_layers+1, hidden] — activations at the final token position."""
        return self.stack()[:, -1, :]


class GemmaActivationModel:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "mps",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.known_models = {
            "google/gemma-4-E2B-it": {
                "layer_path": ["model", "language_model", "layers"],
                "embed_path": ["model", "language_model", "embed_tokens"],
                "config_path": ["config", "text_config", "hidden_size"],
            },
            "google/gemma-3-4b-it": {
                "layer_path": ["model", "language_model", "layers"],
                "embed_path": ["model", "language_model", "embed_tokens"],
                "config_path": ["config", "text_config", "hidden_size"],
            },
            "google/gemma-3-1b-it": {
                "layer_path": ["model", "layers"],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
            "google/gemma-3-12b-it": {
                "layer_path": ["model", "language_model", "layers"],
                "embed_path": ["model", "language_model", "embed_tokens"],
                "config_path": ["config", "text_config", "hidden_size"],
            },
            "google/gemma-3-27b-it": {
                "layer_path": ["model", "language_model", "layers"],
                "embed_path": ["model", "language_model", "embed_tokens"],
                "config_path": ["config", "text_config", "hidden_size"],
            },
            "google/gemma-2-9b-it": {
                "layer_path": ["model", "layers"],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
            "google/gemma-2-2b-it": {
                "layer_path": ["model", "layers"],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
            "Qwen/Qwen2.5-3B-Instruct": {
                "layer_path": ["model", 'layers'],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
            "Qwen/Qwen3.5-2B": {
                "layer_path": ["model", 'layers'],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
            "Qwen/Qwen2.5-1.5B-Instruct": {
                "layer_path": ["model", 'layers'],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
            "microsoft/Phi-4-mini-reasoning": {
                "layer_path": ["model", 'layers'],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
            "microsoft/Phi-3-mini-4k-instruct": {
                "layer_path": ["model", 'layers'],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
            "microsoft/Phi-2": {
                "layer_path": ["model", 'layers'],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
            "meta-llama/Llama-2-7b-chat-hf": {
                "layer_path": ["model", 'layers'],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
            "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1": {
                "layer_path": ["model", 'layers'],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
            "nvidia/Llama-3.1-Nemotron-Nano-8B-v1": {
                "layer_path": ["model", 'layers'],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
            "meta-llama/Llama-3.2-3B-Instruct": {
                "layer_path": ["model", 'layers'],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
            {
                "layer_path": ["model", 'layers'],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
            "allenai/OLMo-2-0425-1B-Instruct":
            {
                "layer_path": ["model", 'layers'],
                "embed_path": ["model", "embed_tokens"],
                "config_path": ["config", "hidden_size"],
            },
        }

        token = os.environ.get("HF_TOKEN")
        self.tokenizer = _load_tokenizer(model_id, token=token)
        self.tokenizer.padding_side = "left"

        # eager attention plays nicest with forward hooks on MPS;
        # SDPA can fuse paths in ways that bypass per-block output capture.
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                token=token,
                attn_implementation="eager",
            )
            .eval()
            .to(device)
        )
        self.model.requires_grad_(False)
        try:
            if model_id not in self.known_models:
                raise ValueError(
                    f"{model_id} is an unrecognised model id. Add its layer_path, "
                    "embed_path, and config_path to known_models in model_wrapper.py."
                )
            else:
                model_key = model_id
            
            paths = self.known_models[model_key]
            self.layers = _get_nested_attr(self.model, paths["layer_path"])
            self.embed_tokens = _get_nested_attr(self.model, paths["embed_path"])
            self.n_layers = len(self.layers)
            self.hidden_size = _get_nested_attr(self.model, paths["config_path"])
                
        except Exception as e:
            print(self.model)
            print(self.model.__dict__)
            print(e)
            raise SystemExit(1)

    def format_prompt(self, instruction: str, enable_thinking: Optional[bool] = False, system: Optional[str]=None, suffix: Optional[str] = None) -> str:
        """Apply Gemma-2 chat template with `add_generation_prompt=True`.

        If `suffix` is given, append after the model-turn opener so the model
        continues from that text as if it had produced it (CoT-hijack slot).
        For Ye et al.-style attacks where the forgery lives in the user turn,
        pass it as part of `instruction` instead.
        """
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": instruction})
                            
        if self.tokenizer.chat_template is not None:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        elif "gemma" in self.model_id.lower():
            formatted = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
        else:
            formatted = instruction
        if suffix:
            formatted = formatted + suffix
        return formatted

    @torch.no_grad()
    def get_residual_stream(self, text: str) -> ResidualStream:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        captures: dict = {}
        handles = []

        def block_hook(idx):
            def fn(module, _inp, out):
                hidden = _hidden_from_output(out)
                # Move to CPU float32 immediately — keeps MPS memory low and
                # makes downstream sklearn/numpy work straightforward.
                captures[idx] = hidden[0].detach().to("cpu", dtype=torch.float32)
            return fn

        def embed_hook(_module, _inp, out):
            tensor = _hidden_from_output(out)
            captures["embed"] = tensor[0].detach().to("cpu", dtype=torch.float32)

        try:
            handles.append(
                self.embed_tokens.register_forward_hook(embed_hook)
            )
            for i, layer in enumerate(self.layers):
                handles.append(layer.register_forward_hook(block_hook(i)))

            self.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        layer_acts = [captures[i] for i in range(self.n_layers)]
        return ResidualStream(
            embed=captures["embed"],
            layers=layer_acts,
            input_ids=inputs.input_ids[0].cpu(),
            tokens=tokens,
        )

    @torch.no_grad()
    def generate(
        self,
        text: str,
        max_new_tokens: int = 256,
        do_sample: bool = False,
    ) -> str:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        new = out[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True)


if __name__ == "__main__":
    print("Loading model (Gemma-2-2b-it, bf16, MPS)...")
    m = GemmaActivationModel()
    print(f"  n_layers={m.n_layers}  hidden={m.hidden_size}")

    text = m.format_prompt("What is the capital of France?")
    print(f"\nFormatted prompt:\n{text!r}\n")

    rs = m.get_residual_stream(text)
    print(f"ResidualStream: n_layers={rs.n_layers}  seq_len={rs.input_ids.shape[0]}")
    mid = rs.n_layers // 2
    print(f"Layer {mid} last-token norm: {rs.layers[mid][-1].norm().item():.3f}")

    print("\nGeneration sanity check:")
    print(m.generate(text, max_new_tokens=32))
