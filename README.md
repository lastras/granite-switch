# Granite Switch — Build AI models like you build software

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![corelib](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fibm-granite%2Fgranitelib-core-r1.0&query=%24.downloads&label=corelib&logo=huggingface&color=yellow)](https://huggingface.co/ibm-granite/granitelib-core-r1.0)
[![raglib](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fibm-granite%2Fgranitelib-rag-r1.0&query=%24.downloads&label=raglib&logo=huggingface&color=yellow)](https://huggingface.co/ibm-granite/granitelib-rag-r1.0)
[![guardianlib](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fibm-granite%2Fgranitelib-guardian-r1.0&query=%24.downloads&label=guardianlib&logo=huggingface&color=yellow)](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0)

| [**Browse Adapters**](https://generative-computing.github.io/granite-switch/adapter_catalog.html) | [Pre-composed Models on HF](https://huggingface.co/ibm-granite/granite-switch-4.1-8b-preview) | [Tutorials](tutorials/README.md) |

Pick adapters for RAG, safety, factuality, and more — compose them into a single Granite model — deploy with one command. No training, no glue code, upgrade any adapter independently.

Small models with the right adapters consistently outperform much larger generalist models on targeted tasks. **Activated LoRA (aLoRA)** makes this practical at scale: all adapters share one KV cache, activating on demand — so one deployment serves many capabilities with no memory or latency overhead.

<p align="center">
  <img src="docs/benchmark_animation.svg" alt="Granite Switch: adapters stack, accuracy improves" width="820">
</p>

## Key Features

- **Composable** — Combine independently developed adapters into one checkpoint, whether IBM's or yours. Swap, upgrade, or customize without retraining.
- **Fast** — Built on IBM's Activated LoRA technology for efficient KV cache reuse, low latency, and [high inference throughput](https://generative-computing.github.io/granite-switch/race_live.html).
- **Accurate** — Task-specific adapters can match and even surpass the accuracy of significantly larger generalist models, while requiring only a fraction of the serving cost. See the [adapter catalog](https://generative-computing.github.io/granite-switch/adapter_catalog.html#hallucination-detection) for benchmark comparisons across all 12 adapters.
- **Inference-ready** — Deploy with vLLM for production or HuggingFace for prototyping. Same checkpoint, no conversion step.

## Quick Start

### Install

```bash
python -m venv venv && source venv/bin/activate

# Granite-Switch installation is based on your usecase:
pip install "granite-switch[compose]"   # Compose modular models
pip install "granite-switch[hf]"        # HuggingFace inference
pip install "granite-switch[vllm]"      # vLLM production inference (0.19.x)
pip install "granite-switch[vllm20]"    # vLLM 0.20+ (requires CUDA 13+)
pip install "granite-switch[dev]"       # Everything (uses vLLM 0.19.x by default)
pip install "granite-switch[dev-vllm20]" # Dev environment with vLLM 0.20+
```

Requires Python 3.9+ and PyTorch 2.0+.

> **Two vLLM backends available:** `.[vllm]` ships with vLLM 0.19.x for broad CUDA 12.x compatibility. `.[vllm20]` gives you vLLM 0.20+ with the latest performance improvements (requires CUDA 13+).

### Compose a Model

Compose a base Granite model with adapter libraries into a single deployable checkpoint:

```bash
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ibm-granite/granitelib-core-r1.0 ibm-granite/granitelib-rag-r1.0  ibm-granite/granitelib-guardian-r1.0 \
  --output ./my-model
```

Use the **[Adapter Composer](https://generative-computing.github.io/granite-switch/adapter_catalog.html)** to browse available adapters, compare benchmarks, and generate a ready-to-run compose command.

This downloads the base model, embeds compatible LoRA adapters (with a preference towards activated LoRA), adds control tokens and a chat template, and produces a model directory that works with both HuggingFace and vLLM.

For convenience, you can find already composed Granite Switch models for the Granite 4.1 model family here:

- [ibm-granite/granite-switch-4.1-3b-preview](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview)
- [ibm-granite/granite-switch-4.1-8b-preview](https://huggingface.co/ibm-granite/granite-switch-4.1-8b-preview)
- [ibm-granite/granite-switch-4.1-30b-preview](https://huggingface.co/ibm-granite/granite-switch-4.1-30b-preview)


### Run Inference

**vLLM + Mellea (recommended):**

```bash
pip install mellea
# Example with the 3B model 
python -m vllm.entrypoints.openai.api_server --model ibm-granite/granite-switch-4.1-3b-preview --port 8000
```

```python
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = OpenAIBackend(
    model_id="ibm-granite/granite-switch-4.1-3b-preview",
    base_url="http://localhost:8000/v1",
    api_key="unused",
)
backend.register_embedded_adapter_model("ibm-granite/granite-switch-4.1-3b-preview")

query = "I want to ask you something. what is...mmmm the the main city(capital you call it,right?) of France?"
ctx = ChatContext()

rewritten = rag.rewrite_question(query, ctx, backend)
print(f"original:  {query}")
print(f"rewritten: {rewritten}")
# => "What is the capital of France?"
```

**HuggingFace:**

```python
import granite_switch.hf  # Register HF backend

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-switch-4.1-3b-preview", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-switch-4.1-3b-preview")

messages = [{"role": "user", "content": "What is the capital of France?"}]
documents = [{"doc_id": "1", "text": "Paris is the capital of France."}]

prompt = tokenizer.apply_chat_template(
    messages,
    documents=documents,
    adapter_name="answerability",  # activates the answerability adapter
    add_generation_prompt=True,
    tokenize=False,
)
outputs = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device))
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# => "answerable"
```

## How It Works

Granite Switch uses a **switch layer**—a small attention-based mechanism that reads control tokens from the input and determines which adapter's LoRA weights to apply at each position.

**What makes composition work:**

- **KV cache normalization** — each adapter sees only the base model's KV cache, never another adapter's internal state
- **No joint training required** — adapters are developed, tested, and published independently
- **Standard inference** — The entire model loads in vLLM with zero code changes

## Tutorials

New here? Start with a 5-minute notebook and work your way up:

| | What you'll build | Time |
|---|---|---|
| [Hello Adapter](tutorials/notebooks/00_hello_adapter.ipynb) | Invoke your first adapter with HuggingFace | 5 min |
| [Hello Mellea](tutorials/notebooks/01_hello_mellea.ipynb) | Call adapters through a clean Python API | 5 min |
| [RAG Pipeline](tutorials/notebooks/03_01_govt_rag_pipeline_simple.ipynb) | Query rewrite + answerability + citations in one model | 30 min |
| [Compose Your Own](tutorials/notebooks/04_compose_granite_switch.ipynb) | Build a custom checkpoint from adapter libraries | 15 min |
| [aLoRA vs LoRA Race](tutorials/notebooks/05_alora_vs_lora_race.ipynb) | Measure throughput gains from activated LoRA | 20 min |

All notebooks run on Colab. See [tutorials/README.md](tutorials/README.md) for the full list and guided learning paths.

## IBM and Open Source AI

Granite Switch was started by IBM Research.

## License

Granite Switch has an Apache-2.0 license, as found in the [LICENSE](LICENSE) file.
