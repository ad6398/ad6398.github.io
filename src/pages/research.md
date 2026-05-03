---
layout: ../layouts/AboutLayout.astro
title: "Research"
---

Full list on [Google Scholar](https://scholar.google.com/citations?user=qjFFoE0AAAAJ&hl=en).

---

## Publications

### GupShup: Summarizing Open-Domain Code-Switched Conversations
**EMNLP 2021** · [Paper](https://aclanthology.org/2021.emnlp-main.499.pdf)

First author. We introduce the task of abstractive summarization of Hindi-English code-switched conversations, along with **GupShup** — the first dataset of its kind, containing 6,800+ Hi-En conversations with human-annotated summaries. Multilingual mBART and multi-view seq2seq models achieve the best results.

---

### LDKP: A Dataset for Identifying Keyphrases from Long Scientific Documents
**CIKM 2022 (DL4SR Workshop)** · [Paper](https://ceur-ws.org/Vol-3317/Paper9.pdf)

First author. We release two large corpora mapping keyphrases of ~1.3M and ~100K scientific articles with fully extracted text and metadata. Transformer models capable of processing long sequences (Longformer) outperform traditional approaches on keyphrase extraction from long documents.

---

### Transformers on Sarcasm Detection with Context
**ACL 2020 (FigLang Workshop)** · [Paper](https://aclanthology.org/2020.figlang-1.13.pdf)

First author. We study the effect of conversational context on sarcasm detection, extending BERT, RoBERTa, and SpanBERT with single sentence, sentence-pair, and LSTM-Transformer hybrid architectures on Twitter and Reddit threads.

---

## Projects

### [Distributed LLM Inference Service](https://github.com/ad6398/llm-inference-service)
An efficient cloud-based inference service for LLMs on Kubernetes, using vLLM and paged attention to reduce memory consumption and latency.

### [RLHF on Phi-2 for Dialogues](https://api.wandb.ai/links/ak11089/zcwc0nnz)
Fine-tuning Microsoft Phi-2 using Direct Preference Optimization (DPO) on the Anthropic HH dataset with LoRA adapters and 8-bit quantization — showing improved reward margins.

### [Conditional Diffusion Model for Next Frame Generation](https://github.com/ad6398/Conditional-Diffusion-Model-for-Next-Frame-Generation)
A diffusion model conditioned on past 11 frames for auto-regressive generation of the next 11 frames. MSE of 0.004 for next frame prediction; Jaccard score of 30.89 for semantic segmentation.

### [transformerkp](https://github.com/Deep-Learning-for-Keyphrase/transformerkp)
Open source transformer-based library for keyphrase extraction and generation from text documents. Supports benchmark datasets and evaluation metrics for both tasks.

### [t-CRF](https://github.com/ad6398/t-crf)
CRF head on top of any transformer-based sequence tagger (NER, POS tagging, etc.) for improved structured prediction.

### [SpanElectra](https://github.com/ad6398/SpanElectra)
A language model combining SpanBERT's span boundary objective with ELECTRA's discriminator-generator training — SpanBERT-level accuracy at ELECTRA-level training efficiency.

### [Question Generation](https://github.com/ad6398/Question-Generation)
Given a paragraph, generate all possible questions. If a context is provided, generate questions whose answers correspond to that context.
