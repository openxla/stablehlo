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
    - **yes\***:  in sync with  [XLA semantics](https://www.tensorflow.org/xla/operation_semantics).
    - **yes**: in sync with [StableHLO semantics](https://github.com/openxla/stablehlo/blob/main/docs/spec_draft.md).
    - **yes(need-revisit)**: implemented but need revisit for the sync with XLA or spec
    - **infeasible**: infeasible to implement by design
    
## Status

| StableHLO Op (114) | Specification (21) | Verifier(104) | Type Inference(82) | Prettyprinting | Interpreter (2) |
|:--|:--:|:--:|:--:|:--:|:--:|
| AbsOp |yes|yes*|yes*||no|
| AddOp |yes|yes*|yes*|| yes|
| AfterAllOp |no|no|no||no |
| AllGatherOp |no|yes*|no||no|
| AllReduceOp |no|no|no||no|
| AllToAllOp |no|yes*|yes*||no|
| AndOp |yes|yes*|yes*|| no|
| Atan2Op |no|yes*|yes*||no|
| BatchNormGradOp |no|yes*|yes*||no|
| BatchNormInferenceOp |no|yes*|yes*||no|
| BatchNormTrainingOp |no|yes*|yes*||no|
| BitcastConvertOp |no|yes*|infeasible||no|
| BroadcastInDimOp |no|yes*|infeasible||no|
| BroadcastOp |no|yes*|yes*||no|
| CaseOp |no|yes*|yes*||no|
| CbrtOp |no|yes*|yes*||no|
| CeilOp |yes|yes*|yes*||no|
| CholeskyOp |no|yes*|yes*||no|
| ClampOp |no|yes*|yes*||no|
| ClzOp |no|yes*|yes*||no|
| CollectivePermuteOp |no|yes*|yes*||no|
| CompareOp |no|yes*|yes*||no|
| ComplexOp |no|yes*|yes*||no|
| ComputeReshapeShapeOp |no|no|no||no|
| ConcatenateOp |no|yes*|yes*||no|
| ConstantOp |yes|yes*|yes*|| yes|
| ConvertOp |no|yes*|infeasible||no|
| ConvolutionOp |no|yes*|no||no|
| CosineOp |yes|yes*|yes*||no|
| CreateTokenOp |no|yes*|no||no|
| CrossReplicaSumOp |no|no|yes*||no|
| CstrReshapableOp |no|yes*|no||no|
| CustomCallOp |no|yes*|infeasible||no|
| DivOp |yes|yes*|yes*||no|
| DotGeneralOp |no|yes*|yes*||no|
| DotOp |no|yes*|yes(need-revisit)||no|
| DynamicBroadcastInDimOp |no|yes*|no||no|
| DynamicConvOp |no|no|no||no|
| DynamicGatherOp |no|no|yes(need-revisit)||no|
| DynamicIotaOp |no|no|no||no|
| DynamicPadOp |no|yes*|no||no|
| DynamicReshapeOp |no|yes*|no||no|
| DynamicSliceOp |no|yes*|yes*||no|
| DynamicUpdateSliceOp |no|yes*|no||no|
| EinsumOp |no|no|no||no|
| Expm1Op |no|yes*|yes*||no|
| ExpOp |no|yes*|yes*||no|
| FftOp |no|yes*|yes*||no|
| FloorOp |yes|yes*|yes*||no|
| GatherOp |no|yes*|yes*||no|
| GetDimensionSizeOp |no|yes*|no||no|
| GetTupleElementOp |no|yes*|yes(need-revisit)||no|
| IfOp |no|yes*|yes*||no|
| ImagOp |no|yes*|yes*||no|
| InfeedOp |no|yes*|no||no|
| IotaOp |no|yes*|infeasible||no|
| IsFiniteOp |no|yes*|yes*||no|
| Log1pOp |no|yes*|yes*||no|
| LogisticOp |yes|yes*|yes*||no|
| LogOp |yes|yes*|yes*||no|
| MapOp |no|yes*|no||no|
| MaxOp |yes|yes*|yes*||no|
| MinOp |yes|yes*|yes*||no|
| MulOp |no|yes*|yes*||no|
| NegOp |yes|yes*|yes*||no|
| NotOp |yes|yes*|yes*||no|
| OptimizationBarrierOp |no|yes*|no||no|
| OrOp |yes|yes*|yes*||no|
| OutfeedOp |no|yes*|no||no|
| PadOp |no|yes*|yes*||no|
| PopulationCountOp |no|yes*|yes*||no|
| PowOp |no|yes*|yes*||no|
| RealDynamicSliceOp |no|yes*|no||no|
| RealOp |no|yes*|yes*||no|
| RecvOp |no|yes*|no||no|
| ReduceOp |no|yes*|yes*||no|
| ReducePrecisionOp |no|yes*|yes*||no|
| ReduceScatterOp |no|yes*|no||no|
| ReduceWindowOp |no|yes*|yes*||no|
| RemOp |yes|yes*|yes*||no|
| ReplicaIdOp |no|yes*|yes(need-revisit)||no|
| ReshapeOp |no|yes*|infeasible||no|
| ReturnOp |no|yes*|no||no|
| ReverseOp |no|yes*|yes*||no|
| RngBitGeneratorOp |no|yes*|infeasible||no|
| RngOp |no|yes*|yes*||no|
| RoundNearestEvenOp |no|yes*|yes*||no|
| RoundOp |no|yes*|yes*||no|
| RsqrtOp |yes|yes*|yes*||no|
| ScatterOp |no|yes*|no||no|
| SelectAndScatterOp |no|yes*|no||no|
| SelectOp |no|yes*|yes*||no|
| SendOp |no|yes*|no||no|
| SetDimensionSizeOp |no|yes*|yes(need-revisit)||no|
| ShiftLeftOp |no|yes*|yes*||no|
| ShiftRightArithmeticOp |no|yes*|yes*||no|
| ShiftRightLogicalOp |no|yes*|yes*||no|
| SignOp |no|yes*|yes*||no|
| SineOp |yes|yes*|yes*||no|
| SliceOp |no|yes*|yes*||no|
| SortOp |no|yes*|no||no|
| SqrtOp |yes|yes*|yes*||no|
| SubtractOp |no|yes*|yes*||no|
| TanhOp |yes|yes*|yes*||no|
| TorchIndexSelectOp |no|no|no||no|
| TraceOp |no|yes*|no||no|
| TransposeOp |no|yes*|yes*||no|
| TriangularSolveOp |no|yes*|no||no|
| TupleOp |no|yes*|yes(need-revisit)||no|
| UnaryEinsumOp |no|no|no||no|
| UniformDequantizeOp |no|yes*|yes*||no|
| UniformQuantizeOp |no|yes*|infeasible||no|
| WhileOp |no|yes*|no||no|
| XorOp |yes|yes*|yes*||no|