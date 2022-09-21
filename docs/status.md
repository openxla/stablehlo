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
 - Customized labels for Verifier and Type Inference 
    - **yes(match-xla)**:  in sync with  [XLA semantics](https://www.tensorflow.org/xla/operation_semantics).
    - **yes(match-spec)**: in sync with [StableHLO semantics](https://github.com/openxla/stablehlo/blob/main/docs/spec_draft.md).
    - **yes(need-revisit)**: implemented but need revisit for the sync with XLA or spec
    - **no-plan**: infeasible or no plan to implement by design
    
## Status

| StableHLO Op (114) | Specification (21) | Verifier(104) | Type Inference(79) | Prettyprinting | Interpreter (2) |
|:--|:--:|:--:|:--:|:--:|:--:|
| AbsOp |yes|yes(match-xla)|yes(match-xla)||no|
| AddOp |yes|yes(match-xla)|yes(match-xla)|| yes|
| AfterAllOp |no|no|no||no |
| AllGatherOp |no|yes(match-xla)|no||no|
| AllReduceOp |no|no|no||no|
| AllToAllOp |no|yes(match-xla)|yes(match-xla)||no|
| AndOp |yes|yes(match-xla)|yes(match-xla)|| no|
| Atan2Op |no|yes(match-xla)|yes(match-xla)||no|
| BatchNormGradOp |no|yes(match-xla)|yes(match-xla)||no|
| BatchNormInferenceOp |no|yes(match-xla)|yes(match-xla)||no|
| BatchNormTrainingOp |no|yes(match-xla)|yes(match-xla)||no|
| BitcastConvertOp |no|yes(match-xla)|no-plan||no|
| BroadcastInDimOp |no|yes(match-xla)|no-plan||no|
| BroadcastOp |no|yes(match-xla)|yes(match-xla)||no|
| CaseOp |no|yes(match-xla)|wip||no|
| CbrtOp |no|yes(match-xla)|yes(match-xla)||no|
| CeilOp |yes|yes(match-xla)|yes(match-xla)||no|
| CholeskyOp |no|yes(match-xla)|yes(match-xla)||no|
| ClampOp |no|yes(match-xla)|yes(match-xla)||no|
| ClzOp |no|yes(match-xla)|yes(match-xla)||no|
| CollectivePermuteOp |no|yes(match-xla)|yes(match-xla)||no|
| CompareOp |no|yes(match-xla)|yes(match-xla)||no|
| ComplexOp |no|yes(match-xla)|yes(match-xla)||no|
| ComputeReshapeShapeOp |no|no|no||no|
| ConcatenateOp |no|yes(match-xla)|yes(match-xla)||no|
| ConstantOp |yes|yes(match-xla)|yes(match-xla)|| yes|
| ConvertOp |no|yes(match-xla)|no-plan||no|
| ConvolutionOp |no|yes(match-xla)|no||no|
| CosineOp |yes|yes(match-xla)|yes(match-xla)||no|
| CreateTokenOp |no|yes(match-xla)|no||no|
| CrossReplicaSumOp |no|no|yes(match-xla)||no|
| CstrReshapableOp |no|yes(match-xla)|no||no|
| CustomCallOp |no|yes(match-xla)|no-plan||no|
| DivOp |yes|yes(match-xla)|yes(match-xla)||no|
| DotGeneralOp |no|yes(match-xla)|wip||no|
| DotOp |no|yes(match-xla)|yes(need-revisit)||no|
| DynamicBroadcastInDimOp |no|yes(match-xla)|no||no|
| DynamicConvOp |no|no|no||no|
| DynamicGatherOp |no|no|yes(need-revisit)||no|
| DynamicIotaOp |no|no|no||no|
| DynamicPadOp |no|yes(match-xla)|no||no|
| DynamicReshapeOp |no|yes(match-xla)|no||no|
| DynamicSliceOp |no|yes(match-xla)|yes(match-xla)||no|
| DynamicUpdateSliceOp |no|yes(match-xla)|no||no|
| EinsumOp |no|no|no||no|
| Expm1Op |no|yes(match-xla)|yes(match-xla)||no|
| ExpOp |no|yes(match-xla)|yes(match-xla)||no|
| FftOp |no|yes(match-xla)|yes(match-xla)||no|
| FloorOp |yes|yes(match-xla)|yes(match-xla)||no|
| GatherOp |no|yes(match-xla)|yes(match-xla)||no|
| GetDimensionSizeOp |no|yes(match-xla)|no||no|
| GetTupleElementOp |no|yes(match-xla)|yes(need-revisit)||no|
| IfOp |no|yes(match-xla)|wip||no|
| ImagOp |no|yes(match-xla)|yes(match-xla)||no|
| InfeedOp |no|yes(match-xla)|no||no|
| IotaOp |no|yes(match-xla)|no-plan||no|
| IsFiniteOp |no|yes(match-xla)|yes(match-xla)||no|
| Log1pOp |no|yes(match-xla)|yes(match-xla)||no|
| LogisticOp |yes|yes(match-xla)|yes(match-xla)||no|
| LogOp |yes|yes(match-xla)|yes(match-xla)||no|
| MapOp |no|yes(match-xla)|no||no|
| MaxOp |yes|yes(match-xla)|yes(match-xla)||no|
| MinOp |yes|yes(match-xla)|yes(match-xla)||no|
| MulOp |no|yes(match-xla)|yes(match-xla)||no|
| NegOp |yes|yes(match-xla)|yes(match-xla)||no|
| NotOp |yes|yes(match-xla)|yes(match-xla)||no|
| OptimizationBarrierOp |no|yes(match-xla)|no||no|
| OrOp |yes|yes(match-xla)|yes(match-xla)||no|
| OutfeedOp |no|yes(match-xla)|no||no|
| PadOp |no|yes(match-xla)|yes(match-xla)||no|
| PopulationCountOp |no|yes(match-xla)|yes(match-xla)||no|
| PowOp |no|yes(match-xla)|yes(match-xla)||no|
| RealDynamicSliceOp |no|yes(match-xla)|no||no|
| RealOp |no|yes(match-xla)|yes(match-xla)||no|
| RecvOp |no|yes(match-xla)|no||no|
| ReduceOp |no|yes(match-xla)|yes(match-xla)||no|
| ReducePrecisionOp |no|yes(match-xla)|yes(match-xla)||no|
| ReduceScatterOp |no|yes(match-xla)|no||no|
| ReduceWindowOp |no|yes(match-xla)|yes(match-xla)||no|
| RemOp |yes|yes(match-xla)|yes(match-xla)||no|
| ReplicaIdOp |no|yes(match-xla)|yes(need-revisit)||no|
| ReshapeOp |no|yes(match-xla)|no-plan||no|
| ReturnOp |no|yes(match-xla)|no||no|
| ReverseOp |no|yes(match-xla)|yes(match-xla)||no|
| RngBitGeneratorOp |no|yes(match-xla)|no-plan||no|
| RngOp |no|yes(match-xla)|yes(match-xla)||no|
| RoundNearestEvenOp |no|yes(match-xla)|yes(match-xla)||no|
| RoundOp |no|yes(match-xla)|yes(match-xla)||no|
| RsqrtOp |yes|yes(match-xla)|yes(match-xla)||no|
| ScatterOp |no|yes(match-xla)|no||no|
| SelectAndScatterOp |no|yes(match-xla)|no||no|
| SelectOp |no|yes(match-xla)|yes(match-xla)||no|
| SendOp |no|yes(match-xla)|no||no|
| SetDimensionSizeOp |no|yes(match-xla)|yes(need-revisit)||no|
| ShiftLeftOp |no|yes(match-xla)|yes(match-xla)||no|
| ShiftRightArithmeticOp |no|yes(match-xla)|yes(match-xla)||no|
| ShiftRightLogicalOp |no|yes(match-xla)|yes(match-xla)||no|
| SignOp |no|yes(match-xla)|yes(match-xla)||no|
| SineOp |yes|yes(match-xla)|yes(match-xla)||no|
| SliceOp |no|yes(match-xla)|yes(match-xla)||no|
| SortOp |no|yes(match-xla)|no||no|
| SqrtOp |yes|yes(match-xla)|yes(match-xla)||no|
| SubtractOp |no|yes(match-xla)|yes(match-xla)||no|
| TanhOp |yes|yes(match-xla)|yes(match-xla)||no|
| TorchIndexSelectOp |no|no|no||no|
| TraceOp |no|yes(match-xla)|no||no|
| TransposeOp |no|yes(match-xla)|yes(match-xla)||no|
| TriangularSolveOp |no|yes(match-xla)|no||no|
| TupleOp |no|yes(match-xla)|yes(need-revisit)||no|
| UnaryEinsumOp |no|no|no||no|
| UniformDequantizeOp |no|yes(match-xla)|yes(match-xla)||no|
| UniformQuantizeOp |no|yes(match-xla)|no-plan||no|
| WhileOp |no|yes(match-xla)|no||no|
| XorOp |yes|yes(match-xla)|yes(match-xla)||no|
