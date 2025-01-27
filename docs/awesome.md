# Awesome OpenXLA

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="https://raw.githubusercontent.com/openxla/xla/refs/heads/main/docs/images/openxla_dark.svg"
    class="devsite-dark-theme">
  <img
    alt="OpenXLA Ecosystem"
    src="https://raw.githubusercontent.com/openxla/xla/refs/heads/main/docs/images/openxla.svg">
</picture>

[OpenXLA](https://openxla.org) is open ecosystem of performant, portable, and
extensible machine learning (ML) infrastructure components that simplify ML
development by defragmenting the tools between frontend frameworks and hardware
backends. Built by industry leaders in AI modeling, software, and hardware.

**How is the community using OpenXLA?** This page consolidates links to
repositories and projects using OpenXLA to provide inspiration and code pointers!

**Have a project that uses OpenXLA?** Send us a
[pull request](https://github.com/openxla/stablehlo/blob/main/docs/awesome.md)
and add it to this page!

## Frameworks

- [JAX](https://github.com/jax-ml/jax) is a ML framework with a
NumPy-like API for writing high-performance ML models <img align="center" src="https://img.shields.io/github/stars/jax-ml/jax?style=social">
- [PyTorch/XLA](https://github.com/pytorch/xla/) provides a bridge from PyTorch
to OpenXLA and StableHLO <img  align="center" src="https://img.shields.io/github/stars/pytorch/xla?style=social">
- [TensorFlow](https://github.com/tensorflow/tensorflow) is a long-standing ML
framework with a large ecosystem <img align="center" src="https://img.shields.io/github/stars/tensorflow/tensorflow?style=social">

## PJRT Plugins

- [libTPU](https://cloud.google.com/tpu/docs/runtimes) allows models to execute
on Google's Cloud TPUs

## Edge Compilation

- [Google AI Edge](https://ai.google.dev/edge) uses StableHLO as an input format
to deploy to mobile devices using [LiteRT](https://ai.google.dev/edge/litert)
  - [AI Edge Torch](https://github.com/google-ai-edge/ai-edge-torch) exports
  PyTorch models for mobile deployment via StableHLO <img align="center" src="https://img.shields.io/github/stars/google-ai-edge/ai-edge-torch?style=social">

## Tooling and Visualization

- [Model Explorer](https://github.com/google-ai-edge/model-explorer) offers
heirarchical graph visualization with support for StableHLO models
<img align="center" src="https://img.shields.io/github/stars/google-ai-edge/model-explorer?style=social">
