---
title: "How We Cut ML Inference Latency by 40% on Kubernetes"
pubDatetime: 2023-06-01T00:00:00Z
modDatetime: 2023-06-01T00:00:00Z
slug: async-model-serving-40-percent
featured: true
draft: false
tags:
  - ml-engineering
  - llm-ops
  - kubernetes
  - infrastructure
description: "The architecture behind our async model serving platform at Instabase — async workers, RabbitMQ, multi-level caching, and sticky routing to cut inference time by 40%."
---

When I joined Instabase's Model Service team, our ML inference pipeline was synchronous, single-level, and running on compute-limited Kubernetes nodes. Under load, latency was rough. Here's what we built to fix it.

## The Problem

Document AI inference is expensive. Each request runs a large model (sometimes LLM-scale) against a document that might be dozens of pages. In a Kubernetes environment with limited GPU resources, synchronous request handling meant:

- Long queues under burst traffic
- Wasted compute on duplicate documents
- No intelligence about where a model's warm state lived

## The Architecture

We built a fully async serving platform with four key components:

### 1. Async Workers

Instead of blocking on each inference request, we moved to an async worker pool. Requests are dispatched to available workers without holding a connection. This alone reduced p99 latency significantly under load.

### 2. RabbitMQ for Request Queuing

We introduced RabbitMQ as the message broker between the API layer and inference workers. This decouples request ingestion from processing, enabling backpressure handling and reliable retries without dropping requests.

### 3. Two-Level Caching

We implemented a two-level cache:
- **L1 (in-memory):** Hot results for recently processed documents
- **L2 (Redis):** Persistent cache keyed on document hash + model version

For document-heavy workflows, cache hit rates above 40% were common — especially for repeat documents in audit/compliance pipelines.

### 4. Sticky Routing

Model loading is expensive. We implemented sticky routing so that requests for the same model are preferentially routed to workers that already have it loaded in memory. This dramatically reduced model cold-start overhead.

### Hardware Acceleration

On top of the architecture changes, we applied ONNX export and model pruning to reduce per-inference compute. This compounded the latency gains.

## Results

**40% reduction in inference time** on a compute-limited Kubernetes environment. During peak load periods the improvement was even more pronounced due to the queueing and caching effects.

## Lessons

- Async is almost always worth the complexity in ML serving
- Caching at the document level (not just the model level) pays off in document AI
- Sticky routing is underrated — model loading overhead is real

If you're building ML inference infrastructure and want to talk through the details, reach out.
