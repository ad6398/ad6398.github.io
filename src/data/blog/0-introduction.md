---
title: "GenZ to AI Enz: A Roadmap for CS Grads Breaking into AI"
pubDatetime: 2026-05-01T00:00:00Z
slug: genz-to-ai-enz-intro
featured: true
draft: false
tags:
  - beginners
  - ml-engineering
  - llm
  - agents
  - genz-to-ai-enz
description: "A complete series taking CS students and early-career engineers from zero ML knowledge to building real AI systems with LLMs and agents."
---

Hi, I'm Amar - a Machine Learning Engineer at DoorDash, working on the GenAI Platform Foundation team. I spend my days building the infrastructure that powers LLM-based features at scale, and this series is my attempt to share what I've learned along the way.

Software engineering has changed. LLMs and agents aren't just a niche specialisation anymore - they're components of everyday software, the same way databases, APIs, and caches are. A backend engineer is expected to know how a database works, how to write a query, when to index. That same expectation is moving toward LLMs.

Knowing how to call an LLM API is table stakes now. What sets engineers apart is understanding what's happening underneath - how models generate text, why they hallucinate, how to make retrieval work reliably, how to build agents that don't break in production.

Most learning resources don't meet engineers where they are. Classical ML courses assume a math background. Research papers assume you already know the field. And the guides aimed at beginners usually stop at "call the API and print the output."

This series fills that gap.

## Why I wrote this

There are plenty of guides for becoming a software engineer. There are also courses on classical ML - regression, decision trees, CNNs. But I couldn't find anything that bridges the gap for the era we're actually in: LLMs, RAG, agents, and the infrastructure that runs them in production.

The tools freshers need to learn today are different from what was relevant five years ago. This is the guide I wish existed when I was starting out.

There's also a second reason. I'm planning a follow-up series on LLM system design and LLMOps - how to architect, scale, and operate AI systems at production load. Good system design resources for ML are already rare. For LLMs specifically, they barely exist. This series lays the foundation for that one. You need to understand how these systems work before you can design them well.

## Who this is for

You'll get the most out of this if you:

- Are a CS student or have 0-1 years of engineering experience
- Know data structures, algorithms, and basic CS from coursework or work
- Have written code in Python (or can pick it up quickly)
- Have zero machine learning background

You don't need linear algebra expertise, a research background, or prior ML experience. Those things help, but they're not required to start.

## What you'll learn

- **How LLMs work** - from neural networks and transformers all the way to how models are pre-trained and what happens during that process
- **How to fine-tune** - adapt an open-source model to your own data and use case
- **How to build reliable agents** - systems that use tools, plan across steps, and don't fall apart in production
- **How to deploy LLMs and agents** - serving, batching, latency, cost, and the infrastructure that keeps things running
- **How to evaluate** - measure what your system actually does, catch hallucinations, and know when it's good enough to ship

## How the series is structured

Each post covers one concept. There's no skipping ahead - each post builds on the last. 

The series starts from the ground up - neural networks, how transformers work, how LLMs are trained and fine-tuned - and builds toward the things you'll actually use at work: RAG pipelines, agents, evaluation, and running models in production. There are also hands-on walkthroughs throughout where you build real things end to end.

The full list of posts and walkthroughs is in the [series index](/posts/genz-to-ai-enz-index).

## Where to start

Start with post 1.1. If you already know deep learning basics, check the series index and jump to wherever makes sense.

→ **[Post 1.1 - What is a Neural Network?](/posts/what-is-a-neural-network)**
