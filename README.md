# LoRA Fine-Tuning on BLOOM 1B

A Jupyter Notebook implementing **LoRA (Low-Rank Adaptation)** fine-tuning on the BLOOM 1B language model using the HuggingFace PEFT library — demonstrating how a 1-billion parameter model can be efficiently adapted to a downstream task by training only a small fraction of its parameters.

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [LoRA Architecture](#lora-architecture)
- [Notebook Contents](#notebook-contents)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [References](#references)
- [Contact](#contact)

---

## Overview

Full fine-tuning of a 1B+ parameter LLM requires updating every weight in the model, which is computationally expensive and memory-intensive. LoRA (Hu et al., 2022) addresses this by injecting trainable low-rank decomposition matrices into the attention layers of the frozen base model, reducing the number of trainable parameters by several orders of magnitude while retaining competitive task performance. This notebook demonstrates LoRA fine-tuning applied to BLOOM 1B.

---

## Background

### BLOOM

BLOOM (BigScience Large Open-science Open-access Multilingual Language Model) is a 176B parameter autoregressive language model trained on 46 natural languages and 13 programming languages. The 1B variant used here provides a computationally tractable base model for demonstrating fine-tuning techniques while retaining the core architecture.

### Why LoRA?

| Approach | Trainable Parameters | GPU Memory |
|---|---|---|
| Full fine-tuning | 100% (~1B params) | Very high |
| LoRA (r=8) | ~0.1–0.5% | Significantly reduced |
| LoRA (r=16) | ~0.2–1.0% | Moderately reduced |

LoRA achieves this by approximating the weight update matrix ΔW as a product of two low-rank matrices A and B, where rank r << d:

```
W_updated = W_frozen + ΔW = W_frozen + AB
where A ∈ R^(d×r), B ∈ R^(r×k), r << min(d, k)
```

Only A and B are trained; W_frozen remains unchanged throughout.

---

## LoRA Architecture

```
Input
  │
  ▼
┌──────────────────────────────┐
│  Frozen Base Model Layer     │  W_frozen (not updated)
│  (BLOOM attention weights)   │
└────────────┬─────────────────┘
             │         +
┌────────────▼─────────────────┐
│  LoRA Adapter                │  Only these weights train
│  ΔW = A × B                  │
│  A: (d × r), B: (r × k)     │
│  r = rank (e.g. 8, 16)       │
└────────────┬─────────────────┘
             │
             ▼
          Output
```

---

## Notebook Contents

| Section | Description |
|---|---|
| Setup & Model Loading | Loading BLOOM 1B tokenizer and model from HuggingFace |
| LoRA Configuration | Setting rank `r`, `lora_alpha`, target modules, and dropout |
| PEFT Model Wrapping | Applying `LoraConfig` to create the parameter-efficient model |
| Parameter Count Comparison | Demonstrating the reduction in trainable parameters vs. full fine-tuning |
| Dataset Preparation | Loading and formatting the downstream task dataset |
| Training | Fine-tuning the LoRA-wrapped model using HuggingFace `Trainer` |
| Inference | Generating text with the fine-tuned model |
| Results | Qualitative comparison of base model vs. LoRA fine-tuned outputs |

---

## Technologies Used

| Library | Purpose |
|---|---|
| `transformers` (HuggingFace) | BLOOM model and tokenizer loading |
| `peft` (HuggingFace) | LoRA implementation via `LoraConfig` and `get_peft_model` |
| `datasets` (HuggingFace) | Dataset loading and preprocessing |
| `torch` (PyTorch) | Training backend |
| `bitsandbytes` | Optional 8-bit quantisation for reduced memory footprint |

---

## Setup and Installation

```bash
git clone https://github.com/chetnapriyadarshini/LoRA_fine_tuning_on_Bloom.git
cd LoRA_fine_tuning_on_Bloom
pip install transformers peft datasets torch bitsandbytes accelerate
```

Launch the notebook:

```bash
jupyter notebook LoRA_Bloom_Fine_tuning_Implementation.ipynb
```

> **Note:** A GPU with at least 8GB VRAM is recommended. BLOOM 1B with LoRA can be fine-tuned on a Google Colab T4 GPU.

---

## References

- Hu, E.J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. arXiv:2106.09685.
- Le Scao, T. et al. (2022). *BLOOM: A 176B-Parameter Open-Access Multilingual Language Model*. arXiv:2211.05100.
- HuggingFace PEFT Documentation: https://huggingface.co/docs/peft

---

## Contact

Created by [@chetnapriyadarshini](https://github.com/chetnapriyadarshini) — feel free to reach out with questions or suggestions.
