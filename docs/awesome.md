# Awesome OpenXLA

**How is the community using OpenXLA?** This page consolidates links to
repositories and projects using OpenXLA to provide inspiration and code pointers!

**Have a project that uses OpenXLA?** Send us a pull request and add it to this page!

## Frameworks

### JAX

<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo" width="80" height="40">

[JAX](https://github.com/jax-ml/jax) is a machine-learning framework with a
NumPy-like API for writing high-performance ML models. JAX lowers to StableHLO,
PJRT, and XLA for high-performance compilation and execution on CPUs, GPUs,
TPUs, and xPUs.

### PyTorch

<img src="https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png" alt="logo" width="200" height="40">

[PyTorch/XLA](https://github.com/pytorch/xla/) is a Python package that uses
OpenXLA to connect the PyTorch deep learning framework to TPUs, GPUs, and CPUs.

### TensorFlow

<img src="https://www.tensorflow.org/images/tf_logo_horizontal.png" alt="logo" width="200" height="60">

[TensorFlow](https://github.com/tensorflow/tensorflow) is an end-to-end
open-source platform for machine learning. It has a comprehensive, flexible
ecosystem of tools, libraries, and community resources for ML research and
application development. TensorFlow can use
[OpenXLA as an alternative backend](https://openxla.org/xla/tf2xla) for
compilation and execution.

## PJRT Plugins

### libTPU

The libTPU PJRT plugin enables frameworks to compile and run models on Cloud TPUs.

## Edge Compilation

### Google AI Edge

[Google AI Edge](https://ai.google.dev/edge) uses frameworks to generate
StableHLO, which is then converted into a mix of TFLite and StableHLO ops.
This is serialized in a flatbuffer and sent to resource-constrained edge devices.

## Tooling and Visualization

### Model Explorer

Model Explorer can visualize StableHLO representations of models, providing
insights into the compilation process within OpenXLA.
