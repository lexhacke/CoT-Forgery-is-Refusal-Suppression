import torch
from model_wrapper import GemmaActivationModel


MODEL_ARTIFACT_DIRS = {
    "Qwen/Qwen3.5-2B": "qwen3.5-2b",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5-1.5b-instruct",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5-3b-instruct",

    "microsoft/Phi-4-mini-reasoning": "phi-4-mini-reasoning",
    "microsoft/Phi-3-mini-4k-instruct": "phi-3-mini-instruct",

    "google/gemma-4-E2B-it": "gemma-4-e2b-it",
    "google/gemma-3-1b-it": "gemma-3-1b-it",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
    "google/gemma-3-12b-it": "gemma-3-12b-it",
    "google/gemma-2-2b-it": "gemma-2-2b-it",
    "google/gemma-2-9b-it": "gemma-2-9b-it",
    
    "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1": "Llama-Nemotron-4B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct",
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1": "Llama-Nemotron-8B",
    "meta-llama/Llama-2-7b-chat-hf": "Llama-2-7b-chat-hf",

    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "Deepseek-R1-1.5B"
}

model_choice = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"

def inject_thinking_trace(model: GemmaActivationModel, prompt: str, trace: str) -> str:
    """Format prompt, then place trace inside a Qwen-style <think> block."""
    text = model.format_prompt(prompt, enable_thinking=True)
    empty_think = "<think>\n\n</think>\n\n"
    filled_think = f"<think>\n{trace.strip()}\n</think>\n\n"
    if empty_think in text:
        return text.replace(empty_think, filled_think, 1)
    return text + filled_think


if __name__ == "__main__":
    print("\n\n==", model_choice, "Playground ==\n")

    model = GemmaActivationModel(model_id=model_choice)
    
    n = 1
    max_new_tokens = 1500

    while True:
        output = {}
        prompt = input("Enter Harmful Prompt: ")
        CoT = input("Enter CoT Trace: ")
        injected = prompt + '\n'*n + CoT
        text = model.format_prompt(prompt, system="detailed thinking on", enable_thinking=True)
        output["baseline"] = model.generate(text, max_new_tokens=max_new_tokens).strip()
        if CoT != "":
            # Injected in <user>
            text = model.format_prompt(injected)
            output["injected_user"] = model.generate(text, max_new_tokens=max_new_tokens).strip()

            # Injected in <think>
            text = inject_thinking_trace(model, prompt, CoT)
            output["injected_think"] = model.generate(text, max_new_tokens=max_new_tokens).strip()

        print("\n == Baseline Response ==\n"+output['baseline'])
        if CoT != "":
            print("\n\n == Injected Response (User) == \n"+output['injected_user'])
            print("\n\n == Injected Response (Think) == \n"+output['injected_think'])
        if input("Press Enter to Continue") != "":
            break
