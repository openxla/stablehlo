# RFC: Remove window_reversal from ConvolutionOp

## Summary

A proposal to remove window_reversal feature from StableHLO.

## Background of window_reversal

In the StableHLO specification, window_reversal is an input used in
ConvolutionOp. The semantics of the input can be referred to from
[here](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convolution).

In the StableHLO dialect: 1) window_reversal is implemented using an optional
MLIR attribute with default value as false, 2) in addition to ConvolutionOp,
it is also used in DynamicConvOp which is not yet specced.

## Details

The proposal is based on the observation that either the attribute is not used
in MHLO/StableHLO producers at all or is always used with the default value.
To confirm that, we looked into the code of the following projects
(in alphabetical order): JAX, ONNX-MLIR, PyTorch/XLA, TF/XLA and Torch-MLIR.
By bringing up this RFC, we are aiming to make sure that all usages are
accounted for.
