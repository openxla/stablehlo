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
    - **no**: not complete yet, but part of [the roadmap](https://github.com/openxla/stablehlo#roadmap).
 - Customized labels for Verifier and Type Inference
    - **yes\***: in sync with  [XLA semantics](https://www.tensorflow.org/xla/operation_semantics).
    - **yes**: in sync with [StableHLO semantics](https://github.com/openxla/stablehlo/blob/main/docs/spec_draft.md).
    - **yes(need-revisit)**: implemented but need revisit for the sync with XLA or spec
    - **infeasible**: infeasible to implement by design

## Status

| StableHLO Op (114) | Specification (22) | Verification (104) | Type Inference (82) | Prettyprinting (76) | Interpreter (7) |
|:--|:--:|:--:|:--:|:--:|:--:|
| AbsOp |yes|yes*|yes*|yes|no|
| AddOp |yes|yes*|yes*|yes| yes|
| AfterAllOp |no|no|no|yes|no |
| AllGatherOp |no|yes*|no|no|no|
| AllReduceOp |no|no|no|no|no|
| AllToAllOp |no|yes*|yes*|no|no|
| AndOp |yes|yes*|yes*|yes| no|
| Atan2Op |no|yes*|yes*|yes|no|
| BatchNormGradOp |no|yes*|yes*|no|no|
| BatchNormInferenceOp |no|yes*|yes*|no|no|
| BatchNormTrainingOp |no|yes*|yes*|no|no|
| BitcastConvertOp |no|yes*|infeasible|yes|no|
| BroadcastInDimOp |no|yes*|infeasible|no|no|
| BroadcastOp |no|yes*|yes*|no|no|
| CaseOp |no|yes*|yes*|no|no|
| CbrtOp |no|yes*|yes*|yes|no|
| CeilOp |yes|yes*|yes*|yes|yes|
| CholeskyOp |no|yes*|yes*|yes|no|
| ClampOp |no|yes*|yes*|yes|no|
| ClzOp |no|yes*|yes*|yes|no|
| CollectivePermuteOp |no|yes*|yes*|no|no|
| CompareOp |no|yes*|yes*|yes|no|
| ComplexOp |no|yes*|yes*|yes|no|
| ComputeReshapeShapeOp |no|no|no|yes|no|
| ConcatenateOp |no|yes*|yes*|yes|no|
| ConstantOp |yes|yes*|yes*|yes|yes|
| ConvertOp |no|yes*|infeasible|yes|no|
| ConvolutionOp |no|yes*|no|yes(need-revisit)|no|
| CosineOp |yes|yes*|yes*|yes|yes|
| CreateTokenOp |no|yes*|no|yes|no|
| CrossReplicaSumOp |no|no|yes*|no|no|
| CstrReshapableOp |no|yes*|no|yes|no|
| CustomCallOp |no|yes*|infeasible|yes|no|
| DivOp |yes|yes*|yes*|yes|no|
| DotGeneralOp |no|yes*|yes*|no|no|
| DotOp |no|yes*|yes(need-revisit)|yes|no|
| DynamicBroadcastInDimOp |no|yes*|no|no|no|
| DynamicConvOp |no|no|no|no|no|
| DynamicGatherOp |no|no|yes(need-revisit)|no|no|
| DynamicIotaOp |no|no|no|yes|no|
| DynamicPadOp |no|yes*|no|yes|no|
| DynamicReshapeOp |no|yes*|no|yes|no|
| DynamicSliceOp |no|yes*|yes*|no|no|
| DynamicUpdateSliceOp |no|yes*|no|yes|no|
| EinsumOp |no|no|no|no|no|
| Expm1Op |no|yes*|yes*|yes|no|
| ExpOp |no|yes*|yes*|yes|no|
| FftOp |no|yes*|yes*|no|no|
| FloorOp |yes|yes*|yes*|yes|yes|
| GatherOp |no|yes*|yes*|no|no|
| GetDimensionSizeOp |no|yes*|no|yes|no|
| GetTupleElementOp |no|yes*|yes(need-revisit)|yes|no|
| IfOp |no|yes*|yes*|no|no|
| ImagOp |no|yes*|yes*|yes|no|
| InfeedOp |no|yes*|no|no|no|
| IotaOp |no|yes*|infeasible|yes|no|
| IsFiniteOp |no|yes*|yes*|yes|no|
| Log1pOp |no|yes*|yes*|yes|no|
| LogisticOp |yes|yes*|yes*|yes|no|
| LogOp |yes|yes*|yes*|yes|no|
| MapOp |no|yes*|no|no|no|
| MaxOp |yes|yes*|yes*|yes|no|
| MinOp |yes|yes*|yes*|yes|no|
| MulOp |no|yes*|yes*|yes|no|
| NegOp |yes|yes*|yes*|yes|no|
| NotOp |yes|yes*|yes*|yes|no|
| OptimizationBarrierOp |no|yes*|no|yes|no|
| OrOp |yes|yes*|yes*|yes|no|
| OutfeedOp |no|yes*|no|no|no|
| PadOp |no|yes*|yes*|no|no|
| PopulationCountOp |no|yes*|yes*|yes|no|
| PowOp |no|yes*|yes*|yes|no|
| RealDynamicSliceOp |no|yes*|no|yes|no|
| RealOp |no|yes*|yes*|yes|no|
| RecvOp |no|yes*|no|no|no|
| ReduceOp |no|yes*|yes*|yes(need-revisit)|no|
| ReducePrecisionOp |no|yes*|yes*|yes|no|
| ReduceScatterOp |no|yes*|no|no|no|
| ReduceWindowOp |no|yes*|yes*|no|no|
| RemOp |yes|yes*|yes*|yes|no|
| ReplicaIdOp |no|yes*|yes(need-revisit)|yes|no|
| ReshapeOp |yes|yes|infeasible|yes|yes|
| ReturnOp |no|yes*|no|yes|no|
| ReverseOp |no|yes*|yes*|no|no|
| RngBitGeneratorOp |no|yes*|infeasible|yes|no|
| RngOp |no|yes*|yes*|yes|no|
| RoundNearestEvenOp |no|yes*|yes*|yes|no|
| RoundOp |no|yes*|yes*|yes|no|
| RsqrtOp |yes|yes*|yes*|yes|no|
| ScatterOp |no|yes*|no|no|no|
| SelectAndScatterOp |no|yes*|no|no|no|
| SelectOp |no|yes*|yes*|yes|no|
| SendOp |no|yes*|no|no|no|
| SetDimensionSizeOp |no|yes*|yes(need-revisit)|yes|no|
| ShiftLeftOp |no|yes*|yes*|yes|no|
| ShiftRightArithmeticOp |no|yes*|yes*|yes|no|
| ShiftRightLogicalOp |no|yes*|yes*|yes|no|
| SignOp |no|yes*|yes*|yes|no|
| SineOp |yes|yes*|yes*|yes|no|
| SliceOp |no|yes*|yes*|no|no|
| SortOp |no|yes*|no|no|no|
| SqrtOp |yes|yes*|yes*|yes|no|
| SubtractOp |no|yes*|yes*|yes|no|
| TanhOp |yes|yes*|yes*|yes|yes|
| TorchIndexSelectOp |no|no|no|no|no|
| TraceOp |no|yes*|no|yes|no|
| TransposeOp |no|yes*|yes*|no|no|
| TriangularSolveOp |no|yes*|no|no|no|
| TupleOp |no|yes*|yes(need-revisit)|yes|no|
| UnaryEinsumOp |no|no|no|no|no|
| UniformDequantizeOp |no|yes*|yes*|yes|no|
| UniformQuantizeOp |no|yes*|infeasible|yes|no|
| WhileOp |no|yes*|no|yes(need-revisit)|no|
| XorOp |yes|yes*|yes*|yes|no|
