## About

When bootstrapping StableHLO from MHLO, we have inherited MHLO's implementation
of many things, including prettyprinting, verification and shape inference.
Thanks to that, we already have significant coverage of the opset, but there's
still plenty to do to review the existing implementations for completeness and
provide new implementations where none exist.

This live document is for the developers and the users to track the progress on
various aspects of the opset - pretty printing, verification, type inference,
specification, interpreter etc.

### How to use it

The progress of a StableHLO op, as mentioned in the corresponding row, on a
particular aspect, as mentioned in the corresponding column, is tracked using
one of the following tracking labels.

 - Generic labels
    - **yes**: complete
    - **wip**: semi-complete: Work in progress or under review.
    - **no**: not complete yet, but part of [the roadmap](https://github.com/openxla/stablehlo#roadmap).
 - Customized labels
    - Verifier
       - **match-xla**:  verifier in sync with  [XLA semantics](https://www.tensorflow.org/xla/operation_semantics).
       - **match-spec**: verifier in sync with [StableHLO semantics](https://github.com/openxla/stablehlo/blob/main/docs/spec_draft.md).

## Status

| StableHLO Op (114) | Specification (21) | Verification | Type Inference | Prettyprinting | Interpreter (2) |
|:--:|:--:|:--:|:--:|:--:|:--:|
| AbsOP |yes||||no|
| AddOP |yes|||| yes|
| AfterAllOP |no||||no |
| AllGatherOP |no||||no|
| AllReduceOP |no||||no|
| AllToAllOP |no||||no|
| AndOP |yes|||| no|
| Atan2Op |no||||no|
| BatchNormGradOp |no||||no|
| BatchNormInferenceOp |no||||no|
| BatchNormTrainingOp |no||||no|
| BitcastConvertOp |no||||no|
| BroadcastInDimOp |no||||no|
| BroadcastOp |no||||no|
| CaseOp |no||||no|
| CbrtOp |no||||no|
| CeilOp |yes||||no|
| CholeskyOp |no||||no|
| ClampOp |no||||no|
| ClzOp |no||||no|
| CollectivePermuteOp |no||||no|
| CompareOp |no||||no|
| ComplexOp |no||||no|
| ComputeReshapeShapeOp |no||||no|
| ConcatenateOp |no||||no|
| ConstantOp |yes|||| yes|
| ConvertOp |no||||no|
| ConvolutionOp |no||||no|
| CosineOp |yes||||no|
| CreateTokenOp |no||||no|
| CrossReplicaSumOp |no||||no|
| CstrReshapableOp |no||||no|
| CustomCallOp |no||||no|
| DivOp |yes||||no|
| DotGeneralOp |no||||no|
| DotOp |no||||no|
| DynamicBroadcastInDimOp |no||||no|
| DynamicConvOp |no||||no|
| DynamicGatherOp |no||||no|
| DynamicIotaOp |no||||no|
| DynamicPadOp |no||||no|
| DynamicReshapeOp |no||||no|
| DynamicSliceOp |no||||no|
| DynamicUpdateSliceOp |no||||no|
| EinsumOp |no||||no|
| Expm1Op |no||||no|
| ExpOp |no||||no|
| FftOp |no||||no|
| FloorOp |yes||||no|
| GatherOp |no||||no|
| GetDimensionSizeOp |no||||no|
| GetTupleElementOp |no||||no|
| IfOp |no||||no|
| ImagOp |no||||no|
| InfeedOp |no||||no|
| IotaOp |no||||no|
| IsFiniteOp |no||||no|
| Log1pOp |no||||no|
| LogisticOp |yes||||no|
| LogOp |yes||||no|
| MapOp |no||||no|
| MaxOp |yes||||no|
| MinOp |yes||||no|
| MulOp |no||||no|
| NegOp |yes||||no|
| NotOp |yes||||no|
| OptimizationBarrierOp |no||||no|
| OrOp |yes||||no|
| OutfeedOp |no||||no|
| PadOp |no||||no|
| PopulationCountOp |no||||no|
| PowOp |no||||no|
| RealDynamicSliceOp |no||||no|
| RealOp |no||||no|
| RecvOp |no||||no|
| ReduceOp |no||||no|
| ReducePrecisionOp |no||||no|
| ReduceScatterOp |no||||no|
| ReduceWindowOp |no||||no|
| RemOp |yes||||no|
| ReplicaIdOp |no||||no|
| ReshapeOp |no||||no|
| ReturnOp |no||||no|
| ReverseOp |no||||no|
| RngBitGeneratorOp |no||||no|
| RngOp |no||||no|
| RoundNearestEvenOp |no||||no|
| RoundOp |no||||no|
| RsqrtOp |yes||||no|
| ScatterOp |no||||no|
| SelectAndScatterOp |no||||no|
| SelectOp |no||||no|
| SendOp |no||||no|
| SetDimensionSizeOp |no||||no|
| ShiftLeftOp |no||||no|
| ShiftRightArithmeticOp |no||||no|
| ShiftRightLogicalOp |no||||no|
| SignOp |no||||no|
| SineOp |yes||||no|
| SliceOp |no||||no|
| SortOp |no||||no|
| SqrtOp |yes||||no|
| SubtractOp |no||||no|
| TanhOp |yes||||no|
| TorchIndexSelectOp |no||||no|
| TraceOp |no||||no|
| TransposeOp |no||||no|
| TriangularSolveOp |no||||no|
| TupleOp |no||||no|
| UnaryEinsumOp |no||||no|
| UniformDequantizeOp |no||||no|
| UniformQuantizeOp |no||||no|
| WhileOp |no||||no|
| XorOp |yes||||no|
