---
title: "GupShup: Summarizing Code-Switched Conversations"
pubDatetime: 2021-11-07T00:00:00Z
modDatetime: 2021-11-07T00:00:00Z
slug: gupshup-code-switching-summarization
featured: true
draft: false
tags:
  - nlp
  - research
  - llm
  - multilingual
description: "Our EMNLP 2021 paper on abstractive summarization of Hindi-English code-switched conversations — introducing the GupShup dataset."
---

Code-switching — mixing two languages within a single conversation — is ubiquitous in multilingual communities worldwide. In India, Hindi-English (Hi-En) mixing is especially common in everyday chat, social media, and customer support conversations. Yet, despite being so prevalent, code-switched text remains largely underserved in NLP.

Our paper at **EMNLP 2021**, *GupShup: Summarizing Open-Domain Code-Switched Conversations*, addresses this gap.

## The Problem

Automatic summarization of conversations has made great strides for English. But when conversations mix Hindi and English — sometimes mid-sentence — standard summarization models struggle. The vocabulary is mixed, grammar is hybrid, and training data is essentially nonexistent.

## What We Built

We introduce **GupShup**, the first dataset for abstractive summarization of Hi-En code-switched conversations:

- **6,800+ conversations** sourced from social media and chat platforms
- Human-annotated summaries in both English and Hi-En
- Detailed annotation guidelines we developed from scratch
- Code-switching statistics and linguistic analysis

## Key Findings

We benchmarked multiple summarization approaches. Multilingual **mBART** and a **multi-view seq2seq** model achieved the best performance. The results also surface interesting failure modes — models sometimes default to a single language even when the input and expected summary are mixed.

## Why It Matters

As LLMs become more capable, handling multilingual and code-switched text is increasingly important for real-world deployments. This dataset provides a foundation for that research.

**[Read the paper](https://aclanthology.org/2021.emnlp-main.499.pdf)**
