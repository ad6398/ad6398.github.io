---
title: "Fine-tuning Phi-2 with DPO on the Anthropic HH Dataset"
pubDatetime: 2024-03-01T00:00:00Z
modDatetime: 2024-03-01T00:00:00Z
slug: rlhf-phi2-dpo
featured: false
draft: false
tags:
  - llm
  - rlhf
  - fine-tuning
  - research
description: "Fine-tuning Microsoft's Phi-2 using Direct Preference Optimization (DPO) on the Anthropic Helpful and Harmless dataset with LoRA and 8-bit quantization."
---

RLHF (Reinforcement Learning from Human Feedback) is the standard recipe for aligning language models to be helpful and harmless. But classical RLHF requires training a separate reward model — which is expensive and finicky. **Direct Preference Optimization (DPO)** sidesteps this by framing alignment directly as a classification problem over preference pairs.

I explored this with Microsoft's **Phi-2** — a 2.7B parameter model that punches well above its weight — on the [Anthropic Helpful and Harmless (HH) dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf).

## Setup

**Model:** Microsoft Phi-2 (2.7B params)
**Dataset:** Anthropic HH — pairs of (chosen, rejected) dialogue responses
**Method:** DPO via HuggingFace TRL
**Efficiency:** LoRA adapters + 8-bit quantization (bitsandbytes)

LoRA and quantization made this trainable on a single GPU without sacrificing much quality — a practical requirement for research-scale experiments.

## What DPO Does

Given a pair of responses (chosen, rejected) to the same prompt, DPO optimizes the policy to increase the relative likelihood of the chosen response while implicitly using the base model as a reference. No reward model, no PPO loop.

The loss is:

```
L_DPO = -E[log σ(β * (log π(y_w|x) - log π_ref(y_w|x)) - β * (log π(y_l|x) - log π_ref(y_l|x)))]
```

where `y_w` is the preferred response, `y_l` is the rejected one, and `β` controls how far the policy can deviate from the reference.

## Results

The fine-tuned model showed **improved reward margins** — when scored with a reward model, it consistently preferred the helpful/harmless responses over the rejected ones. You can see the full training run on [W&B](https://api.wandb.ai/links/ak11089/zcwc0nnz).

## Takeaways

- DPO is remarkably clean to implement with TRL — the training loop is almost identical to standard SFT
- Phi-2 responds well to preference tuning despite being small
- LoRA + 8-bit makes alignment experiments accessible without large GPU clusters

This is a good starting point if you want to experiment with alignment at a smaller scale before moving to larger models.
