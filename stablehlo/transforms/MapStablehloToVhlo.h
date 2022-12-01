/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_TRANSFORMS_MAPSTABLEHLOTOVHLO_H
#define STABLEHLO_TRANSFORMS_MAPSTABLEHLOTOVHLO_H

#include <type_traits>

#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace vhlo {

template <typename VhloOpTy>
struct VhloToStablehloOpImpl {
  using Type = std::false_type;
};
template <typename VhloOpTy>
using VhloToStablehloOp = typename VhloToStablehloOpImpl<VhloOpTy>::Type;

template <typename StablehloOpTy>
struct StablehloToVhloOpImpl {
  using Type = std::false_type;
};
template <typename StablehloOpTy>
using StablehloToVhloOp = typename StablehloToVhloOpImpl<StablehloOpTy>::Type;

#define MAP_STABLEHLO_TO_VERSION(OpName, OpVer)       \
  template <>                                         \
  struct StablehloToVhloOpImpl<stablehlo::OpName> {   \
    using Type = vhlo::OpName##OpVer;                 \
  };                                                  \
  template <>                                         \
  struct VhloToStablehloOpImpl<vhlo::OpName##OpVer> { \
    using Type = stablehlo::OpName;                   \
  };

#define MAP_STABLEHLO_TO_VERSION_V0(OpName) MAP_STABLEHLO_TO_VERSION(OpName, )

MAP_STABLEHLO_TO_VERSION_V0(AbsOp)
MAP_STABLEHLO_TO_VERSION_V0(AddOp)
MAP_STABLEHLO_TO_VERSION_V0(AfterAllOp)
MAP_STABLEHLO_TO_VERSION_V0(AllGatherOp)
MAP_STABLEHLO_TO_VERSION_V0(AllReduceOp)
MAP_STABLEHLO_TO_VERSION_V0(AllToAllOp)
MAP_STABLEHLO_TO_VERSION_V0(AndOp)
MAP_STABLEHLO_TO_VERSION_V0(Atan2Op)
MAP_STABLEHLO_TO_VERSION_V0(BatchNormGradOp)
MAP_STABLEHLO_TO_VERSION_V0(BatchNormInferenceOp)
MAP_STABLEHLO_TO_VERSION_V0(BatchNormTrainingOp)
MAP_STABLEHLO_TO_VERSION_V0(BitcastConvertOp)
MAP_STABLEHLO_TO_VERSION_V0(BroadcastInDimOp)
MAP_STABLEHLO_TO_VERSION_V0(BroadcastOp)
MAP_STABLEHLO_TO_VERSION_V0(CaseOp)
MAP_STABLEHLO_TO_VERSION_V0(CbrtOp)
MAP_STABLEHLO_TO_VERSION_V0(CeilOp)
MAP_STABLEHLO_TO_VERSION_V0(CholeskyOp)
MAP_STABLEHLO_TO_VERSION_V0(ClampOp)
MAP_STABLEHLO_TO_VERSION_V0(ClzOp)
MAP_STABLEHLO_TO_VERSION_V0(CollectivePermuteOp)
MAP_STABLEHLO_TO_VERSION_V0(CompareOp)
MAP_STABLEHLO_TO_VERSION_V0(ComplexOp)
MAP_STABLEHLO_TO_VERSION_V0(ComputeReshapeShapeOp)
MAP_STABLEHLO_TO_VERSION_V0(ConcatenateOp)
MAP_STABLEHLO_TO_VERSION_V0(ConstantOp)
MAP_STABLEHLO_TO_VERSION_V0(ConvertOp)
MAP_STABLEHLO_TO_VERSION_V0(ConvolutionOp)
MAP_STABLEHLO_TO_VERSION_V0(CosineOp)
MAP_STABLEHLO_TO_VERSION_V0(CreateTokenOp)
MAP_STABLEHLO_TO_VERSION_V0(CrossReplicaSumOp)
MAP_STABLEHLO_TO_VERSION_V0(CstrReshapableOp)
MAP_STABLEHLO_TO_VERSION(CustomCallOp, V2)
MAP_STABLEHLO_TO_VERSION_V0(DivOp)
MAP_STABLEHLO_TO_VERSION_V0(DotGeneralOp)
MAP_STABLEHLO_TO_VERSION_V0(DotOp)
MAP_STABLEHLO_TO_VERSION_V0(DynamicBroadcastInDimOp)
MAP_STABLEHLO_TO_VERSION_V0(DynamicConvOp)
MAP_STABLEHLO_TO_VERSION_V0(DynamicGatherOp)
MAP_STABLEHLO_TO_VERSION_V0(DynamicIotaOp)
MAP_STABLEHLO_TO_VERSION_V0(DynamicPadOp)
MAP_STABLEHLO_TO_VERSION_V0(DynamicReshapeOp)
MAP_STABLEHLO_TO_VERSION_V0(DynamicSliceOp)
MAP_STABLEHLO_TO_VERSION_V0(DynamicUpdateSliceOp)
MAP_STABLEHLO_TO_VERSION_V0(EinsumOp)
MAP_STABLEHLO_TO_VERSION_V0(Expm1Op)
MAP_STABLEHLO_TO_VERSION_V0(ExpOp)
MAP_STABLEHLO_TO_VERSION_V0(FftOp)
MAP_STABLEHLO_TO_VERSION_V0(FloorOp)
MAP_STABLEHLO_TO_VERSION_V0(GatherOp)
MAP_STABLEHLO_TO_VERSION_V0(GetDimensionSizeOp)
MAP_STABLEHLO_TO_VERSION_V0(GetTupleElementOp)
MAP_STABLEHLO_TO_VERSION_V0(IfOp)
MAP_STABLEHLO_TO_VERSION_V0(ImagOp)
MAP_STABLEHLO_TO_VERSION_V0(InfeedOp)
MAP_STABLEHLO_TO_VERSION_V0(IotaOp)
MAP_STABLEHLO_TO_VERSION_V0(IsFiniteOp)
MAP_STABLEHLO_TO_VERSION_V0(Log1pOp)
MAP_STABLEHLO_TO_VERSION_V0(LogisticOp)
MAP_STABLEHLO_TO_VERSION_V0(LogOp)
MAP_STABLEHLO_TO_VERSION_V0(MapOp)
MAP_STABLEHLO_TO_VERSION_V0(MaxOp)
MAP_STABLEHLO_TO_VERSION_V0(MinOp)
MAP_STABLEHLO_TO_VERSION_V0(MulOp)
MAP_STABLEHLO_TO_VERSION_V0(NegOp)
MAP_STABLEHLO_TO_VERSION_V0(NotOp)
MAP_STABLEHLO_TO_VERSION_V0(OptimizationBarrierOp)
MAP_STABLEHLO_TO_VERSION_V0(OrOp)
MAP_STABLEHLO_TO_VERSION_V0(OutfeedOp)
MAP_STABLEHLO_TO_VERSION_V0(PadOp)
MAP_STABLEHLO_TO_VERSION_V0(PopulationCountOp)
MAP_STABLEHLO_TO_VERSION_V0(PowOp)
MAP_STABLEHLO_TO_VERSION_V0(RealDynamicSliceOp)
MAP_STABLEHLO_TO_VERSION_V0(RealOp)
MAP_STABLEHLO_TO_VERSION_V0(RecvOp)
MAP_STABLEHLO_TO_VERSION_V0(ReduceOp)
MAP_STABLEHLO_TO_VERSION_V0(ReducePrecisionOp)
MAP_STABLEHLO_TO_VERSION_V0(ReduceScatterOp)
MAP_STABLEHLO_TO_VERSION_V0(ReduceWindowOp)
MAP_STABLEHLO_TO_VERSION_V0(RemOp)
MAP_STABLEHLO_TO_VERSION_V0(ReplicaIdOp)
MAP_STABLEHLO_TO_VERSION_V0(ReshapeOp)
MAP_STABLEHLO_TO_VERSION_V0(ReturnOp)
MAP_STABLEHLO_TO_VERSION_V0(ReverseOp)
MAP_STABLEHLO_TO_VERSION_V0(RngBitGeneratorOp)
MAP_STABLEHLO_TO_VERSION_V0(RngOp)
MAP_STABLEHLO_TO_VERSION_V0(RoundOp)
MAP_STABLEHLO_TO_VERSION_V0(RoundNearestEvenOp)
MAP_STABLEHLO_TO_VERSION_V0(RsqrtOp)
MAP_STABLEHLO_TO_VERSION_V0(ScatterOp)
MAP_STABLEHLO_TO_VERSION_V0(SelectAndScatterOp)
MAP_STABLEHLO_TO_VERSION_V0(SelectOp)
MAP_STABLEHLO_TO_VERSION_V0(SendOp)
MAP_STABLEHLO_TO_VERSION_V0(SetDimensionSizeOp)
MAP_STABLEHLO_TO_VERSION_V0(ShiftLeftOp)
MAP_STABLEHLO_TO_VERSION_V0(ShiftRightArithmeticOp)
MAP_STABLEHLO_TO_VERSION_V0(ShiftRightLogicalOp)
MAP_STABLEHLO_TO_VERSION_V0(SignOp)
MAP_STABLEHLO_TO_VERSION_V0(SineOp)
MAP_STABLEHLO_TO_VERSION_V0(SliceOp)
MAP_STABLEHLO_TO_VERSION_V0(SortOp)
MAP_STABLEHLO_TO_VERSION_V0(SqrtOp)
MAP_STABLEHLO_TO_VERSION_V0(SubtractOp)
MAP_STABLEHLO_TO_VERSION_V0(TanhOp)
MAP_STABLEHLO_TO_VERSION_V0(TorchIndexSelectOp)
MAP_STABLEHLO_TO_VERSION_V0(TraceOp)
MAP_STABLEHLO_TO_VERSION_V0(TransposeOp)
MAP_STABLEHLO_TO_VERSION_V0(TriangularSolveOp)
MAP_STABLEHLO_TO_VERSION_V0(TupleOp)
MAP_STABLEHLO_TO_VERSION_V0(UnaryEinsumOp)
MAP_STABLEHLO_TO_VERSION_V0(UniformDequantizeOp)
MAP_STABLEHLO_TO_VERSION_V0(UniformQuantizeOp)
MAP_STABLEHLO_TO_VERSION_V0(WhileOp)
MAP_STABLEHLO_TO_VERSION_V0(XorOp)

#undef MAP_STABLEHLO_TO_VERSION
#undef MAP_STABLEHLO_TO_VERSION_V0

}  // namespace vhlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_MAPSTABLEHLOTOVHLO_H