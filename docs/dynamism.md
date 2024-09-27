# Dynamism in StableHLO

The current state of dynamism is more formally spelled out in the
[Dynamism RFC][dynamism-rfc], this page will provide a high level overview of
the RFC and discuss important APIs and tooling for interacting with dynamic
programs.

[dynamism-rfc]:https://github.com/openxla/stablehlo/blob/main/rfcs/20230704-dynamism-101.md

## Dynamism Terminology & Support Overview

First, to cover a few terms that will appear in this doc, as well as a brief
intro to their support in StableHLO:

### Dynamic dimensions

Dynamic dimensions refers to any dimension whose dimension size is unknown.
In StableHLO we represent dynamic dimensions using `?`, i.e. `tensor<16x?xf32>`.

### Bounded dynamism

Bounded dynamism refers to a dynamic dimension whose value has a known upper
bound. Generally this is useful for padding the tensor during execution.
In StableHLO we represent bounded dynamism using `#stablehlo.bounds` as a
tensor encoding, i.e. a rank-2 tensor with one dynamic dimension bounded at 16
and the other without a bound can be represented as
`tensor<?x?xf32, #stablehlo.bounds<16, ?>>`.

StableHLO is able to represent bounded dynamism, but there is limited framework
support, originating in TensorFlow, and with some support in PyTorch/XLA.

### Unbounded dynamism

Unbounded dynamism as the name implies refers to a dynamic dimension with
no known bound on the size. This type of dynamism is very common in StableHLO,
with JAX, PyTorch/XLA, and TF support, often used for exporting models with
dynamic batch size or sequence length.

In StableHLO we simply elide the bounds encoding for this form of dynamism, i.e.
`tensor<?x?xf32>`.

### Shape polymorphism

Shape polymorphism is a [term we've inherited from JAX][shape-poly].

There are two key implications to shape polymorphism:

1. All dynamism in the program traces back to its input arguments.
2. All dynamism pertains to tensor _shapes_ only, i.e. not data-dependent.

With these two rules, once the static shapes of a program are known, we are able
to take a dynamic program and fully refine it into a static program for
compilation (see ["Compiler passes for refining dynamic programs"](#compiler-passes-for-refining-dynamic-programs)).

Generally shape polymorphism uses unbounded dynamism, if known argument shapes
can lead to a fully static program, there isn't a need to guess on how to bound
the values.

### Data-dependent dynamism

Data-dependent dynamism refers to dynamic dimensions sizes that pertain to
the _data_ inside a tensor. The canonical example is a `nonzeros` function which
returns the indices of all elements that are `0` in a tensor value. The shape
cannot be known without evaluating the data, but it can often be compiled using
bounded dynamism, spending extra memory on the potential output tensor size.

Many data-dependent dynamic ops can be modeled using bounded dynamism, where an
upper bound on a tensor size is specified, and hardware generally will implement
this via tensor padding. Today there is some support for data-dependent dynamism
in PyTorch/XLA and TensorFlow, but JAX does not currently trace operations which
lead to data dependent dynamism.

[shape-poly]:https://jax.readthedocs.io/en/latest/export/shape_poly.html

## Exporting programs with dynamic dimensions

See our StableHLO tutorials for information on how to export programs with
dynamic batch sizes or sequence lengths:

- [JAX Tutorial > Export with Dynamic Batch Size][jax-export-dynamic]
- [PyTorch/XLA Tutorial > Export with Dynamic Batch Size][pytorch-export-dynamic]

[jax-export-dynamic]:https://openxla.org/stablehlo/tutorials/jax-export#export_with_dynamic_batch_size
[pytorch-export-dynamic]:https://openxla.org/stablehlo/tutorials/pytorch-export#export_with_dynamic_batch_dimension

## Compiler passes for refining dynamic programs

### Remove dynamism pass pipeline

There are a few useful passes for refining shapes, conveniently they are all
bundled in a pass pipeline [`createStablehloRemoveDynamismPipeline`][remove-dynamism]:

```c++
void createStablehloRemoveDynamismPipeline(OpPassManager &pm,
                                           TypeRange refinedTypes);
```

### Individual passes for refining dynamism

Individually, the passes that tend to be useful for shape refinement are:

- [`stablehlo-refine-arguments`][refine-arguments] to replace input arguments
  with concrete tensor types.
- [`stablehlo-refine-shapes`][refine-shapes] to propagate the new input argument
  shape information throughout the entire program.
- [`stablehlo-canonicalize-dynamism`][canonicalize-dynamism] to replace dynamic
  ops with their static variants.

See linked documentation for up-to-date information and examples.

[remove-dynamism]:https://github.com/openxla/stablehlo/blob/ff13c96e56b73c62dcbb5b34b69f5ece9e71322f/stablehlo/transforms/Passes.h#L134
[canonicalize-dynamism]:https://openxla.org/stablehlo/generated/stablehlo_passes#-stablehlo-canonicalize-dynamism
[refine-arguments]:https://openxla.org/stablehlo/generated/stablehlo_passes#-stablehlo-refine-arguments
[refine-shapes]:https://openxla.org/stablehlo/generated/stablehlo_passes#-stablehlo-refine-shapes

## Example: How is dynamism useful, and how can I use it?

Dynamism has lots of uses, here we'll mainly focus on the common use case for
Shape Polymorphism - creating a flexible exported model representation,
generally used to represent dynamic batch size or sequence length.

### Static add_one model

We'll use the following simple `add_one` model to demonstrate this:

```py
def add_one(x):
  return x + 1
```

When traced using a `tensor<4xf32>` we'll get the following StableHLO program:

```mlir
// File: add_one.mlir
func.func @add_one(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
  %0 = stablehlo.add %arg0, %cst : tensor<4xf32>
  return %0 : tensor<4xf32>
}
```

This model will work _only_ for input arguments that have a `tensor<4xf32>`
shape. If we ever changed our batch size or sequence length, we would need to
re-trace the source code and re-lower to StableHLO, and there's no guarantee
that we even have access to the source code still!

### Dynamic add_one model

This is where shape polymorphic dynamism comes into play. Instead JAX and
PyTorch/XLA can emit the `add_one` model with dynamically valid IR which
will broadcast the constant to match the dynamic input shape as follows:

```mlir
// File: add_one_dynamic.mlir
func.func public @main(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %cst = stablehlo.constant dense<1.0> : tensor<f32>
  %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
  %2 = stablehlo.dynamic_broadcast_in_dim %cst, %1, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  %3 = stablehlo.add %arg0, %2 : tensor<?xf32>
  return %3 : tensor<?xf32>
}
```

This model representation is much more flexible, and allows deferred
specification of values like batch size or sequence length. This model can be
deployed on platforms with dynamic shape support (like [AI Edge][ai-edge]), or
it can be refined using the dynamism passes mentioned in this documentation.

[ai-edge]:https://github.com/google-ai-edge/ai-edge-torch

### Refining the dynamic model

For example the following pass ordering can fully refine this program:

```sh
stablehlo-opt add_one_dynamic.mlir \
  --stablehlo-refine-arguments='types=tensor<16xf32>' \
  --stablehlo-refine-shapes \
  --stablehlo-canonicalize-dynamism
```

Incrementally, this is how the program gets transformed:

```mlir
// After stablehlo-refine-arguments: Inputs updated, shapes not propagated
func.func public @main(%arg0: tensor<16xf32>) -> tensor<?xf32> {
  %c = stablehlo.constant dense<16> : tensor<1xi64>
  %0 = stablehlo.custom_call @stablehlo.shape_refinement_operand_wrapper(%arg0, %c) {indices_of_shape_operands = dense<1> : tensor<1xi64>} : (tensor<16xf32>, tensor<1xi64>) -> tensor<?xf32>
  ...
  %3 = stablehlo.dynamic_broadcast_in_dim %cst, %2, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  %4 = stablehlo.add %0, %3 : tensor<?xf32>
  return %4 : tensor<?xf32>
}

// After stablehlo-refine-shapes: Shapes propagated, dynamic ops still exist
func.func public @main(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %c = stablehlo.constant dense<16> : tensor<1xi32>
  %0 = stablehlo.dynamic_broadcast_in_dim %cst, %c, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<16xf32>
  %1 = stablehlo.add %arg0, %0 : tensor<16xf32>
  return %1 : tensor<16xf32>
}

// After stablehlo-canonicalize-dynamism: Dynamic ops replaced with static ops
func.func public @main(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
  %1 = stablehlo.add %arg0, %0 : tensor<16xf32>
  return %1 : tensor<16xf32>
}

// (Bonus) Use ` --stablehlo-aggressive-simplification` pass to canonicalize the
// constant broadcast, leaving us with the original static program in this case.
func.func public @main(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<16xf32>
  %0 = stablehlo.add %arg0, %cst : tensor<16xf32>
  return %0 : tensor<16xf32>
}
```
