# How awsome ML compilation community is using awesome OpenXLA?

## ML Frameworks

**JAX:** [uses OpenXLA]((https://jax.readthedocs.io/en/latest/quickstart.html)) as
its backend for compilation and execution on CPUs, GPUs, TPUs, xPUs.

**PyTorch:** [PyTorch/XLA](https://github.com/pytorch/xla/) is a Python package that uses the
XLA to connect the PyTorch framework and Cloud TPUs.

**TF:** TensorFlow can use
[OpenXLA as an alternative backend](https://openxla.org/xla/tf2xla) for
compilation and execution.

## PJRT Plugins

### JAX-Metal

[JAX-Metal](https://developer.apple.com/metal/jax/) uses OpenXLA to compile JAX
computations into Metal Shading Language(MSL) code, which can then be executed
on Apple GPUs.

### libTPU

libTPU is used by OpenXLA as a backend to target TPUs. OpenXLA compiles
computations into XLA, which is then further compiled and executed on TPUs using
libTPU.

## MLIR Bridges

## StableHLO Transformations

 StableHLO Transformations are a set of MLIR passes designed to optimize and
 transform programs in the StableHLO dialect.

## Tooling and Vizualization

### Model Explorer

Model Explorer can visualize StableHLO representations of models, providing
insights into the compilation process within OpenXLA.

### SHerLOC

SHerLOC (StableHLO Rule Optimization and Composition) is a framework for
defining and applying rewrite rules to StableHLO programs.

## Edge Compilation

### Google AI Edge

Google AI Edge leverages OpenXLA for compiling and optimizing models for
efficient execution on resource-constrained edge devices.

### StableHLO and Apple CoreML

StableHLO can act as an intermediate representation for converting models to
the Apple CoreML format, enabling deployment on Apple devices.

## Uncategorized

### ByteIR

### BladeDISC
