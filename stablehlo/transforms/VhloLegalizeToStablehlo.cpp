/* Copyright 2022 The StableHLO Authors.
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

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/AllocatorBase.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/dialect/VhloTypes.h"
#include "stablehlo/transforms/MapStablehloToVhlo.h"
#include "stablehlo/transforms/Passes.h"

#define DEBUG_TYPE "compat-passes"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_VHLOLEGALIZETOSTABLEHLOPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// VHLO --> StableHLO types
//===----------------------------------------------------------------------===//

class VhloToStablehloTypeConverter : public vhlo::VhloTypeConverter {
 public:
  VhloToStablehloTypeConverter() : vhlo::VhloTypeConverter() {
    addConversion([](Type type) -> Type { return type; });
    addConversion([](vhlo::TokenV1Type token) -> Type {
      LLVM_DEBUG(llvm::dbgs() << "Converting TokenType\n");
      return stablehlo::TokenType::get(token.getContext());
    });
    addVhloToBuiltinConversions();
  }

  Attribute convertEncoding(Attribute attr) const final {
    if (auto vhloAttr = dyn_cast_or_null<vhlo::TypeExtensionsV1Attr>(attr)) {
      return stablehlo::TypeExtensionsAttr::get(vhloAttr.getContext(),
                                                vhloAttr.getBounds());
    }
    // All encodings supported in StableHLO.
    return attr;
  }
};

//===----------------------------------------------------------------------===//
// VHLO --> StableHLO attributes: 1) Generic case.
// Applicable in areas where there is 1:1 mapping from VHLO to StableHLO.
// This is the predominant case.
//===----------------------------------------------------------------------===//

#define RETURN_CONVERTED_ENUM_ATTR(Name, Version)                   \
  auto vhloValue = vhlo::stringify##Name##Version(attr.getValue()); \
  auto stablehloValue = stablehlo::symbolize##Name(vhloValue);      \
  if (!stablehloValue.has_value()) return {};                       \
  return stablehlo::Name##Attr::get(attr.getContext(), stablehloValue.value())

Attribute convertGeneric(Attribute vhloAttr,
                         const TypeConverter* typeConverter) {
  LLVM_DEBUG(llvm::dbgs() << "Converting attr " << vhloAttr);
  if (auto vhloAttrs = dyn_cast<vhlo::ArrayV1Attr>(vhloAttr)) {
    SmallVector<Attribute> stablehloAttrs;
    for (auto vhloAttr : vhloAttrs.getValue()) {
      auto stablehloAttr = convertGeneric(vhloAttr, typeConverter);
      if (!stablehloAttr) return {};
      stablehloAttrs.push_back(stablehloAttr);
    }
    return ArrayAttr::get(vhloAttrs.getContext(), stablehloAttrs);
  }
  if (auto attr = dyn_cast<vhlo::BooleanV1Attr>(vhloAttr)) {
    return BoolAttr::get(attr.getContext(), attr.getValue());
  }
  if (auto attr = dyn_cast<vhlo::ComparisonDirectionV1Attr>(vhloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonDirection, V1);
  }
  if (auto attr = dyn_cast<vhlo::ComparisonTypeV1Attr>(vhloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonType, V1);
  }
  if (auto attr = dyn_cast<vhlo::CustomCallApiVersionV1Attr>(vhloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(CustomCallApiVersion, V1);
  }
  if (auto attr = dyn_cast<vhlo::DictionaryV1Attr>(vhloAttr)) {
    SmallVector<NamedAttribute> vhloAttrs;
    for (auto namedAttr : attr.getValue()) {
      auto builtinName = dyn_cast_or_null<StringAttr>(
          convertGeneric(namedAttr.first, typeConverter));
      auto builtinValue = convertGeneric(namedAttr.second, typeConverter);
      if (!builtinName || !builtinValue) return {};
      vhloAttrs.push_back({builtinName, builtinValue});
    }
    return DictionaryAttr::get(attr.getContext(), vhloAttrs);
  }
  if (auto attr = dyn_cast<vhlo::FftTypeV1Attr>(vhloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(FftType, V1);
  }
  if (auto attr = dyn_cast<vhlo::FloatV1Attr>(vhloAttr)) {
    auto builtinFloatType = typeConverter->convertType(attr.getType());
    if (!builtinFloatType) return {};
    // FIXME: What is the proper way to reconstruct a attr?
    return FloatAttr::get(builtinFloatType, attr.getValue().convertToDouble());
  }
  if (auto attr = dyn_cast<vhlo::IntegerV1Attr>(vhloAttr)) {
    auto builtinIntegerType = typeConverter->convertType(attr.getType());
    if (!builtinIntegerType) return {};
    return IntegerAttr::get(builtinIntegerType, attr.getValue());
  }
  if (auto attr = dyn_cast<vhlo::OutputOperandAliasV1Attr>(vhloAttr)) {
    return stablehlo::OutputOperandAliasAttr::get(
        attr.getContext(), attr.getOutputTupleIndices(), attr.getOperandIndex(),
        attr.getOperandTupleIndices());
  }
  if (auto attr = dyn_cast<vhlo::PrecisionV1Attr>(vhloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(Precision, V1);
  }
  if (auto attr = dyn_cast<vhlo::RngAlgorithmV1Attr>(vhloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(RngAlgorithm, V1);
  }
  if (auto attr = dyn_cast<vhlo::RngDistributionV1Attr>(vhloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(RngDistribution, V1);
  }
  if (auto attr = dyn_cast<vhlo::StringV1Attr>(vhloAttr)) {
    return StringAttr::get(attr.getContext(), attr.getValue());
  }
  if (auto attr = dyn_cast<vhlo::TensorV1Attr>(vhloAttr)) {
    auto builtinType =
        cast<ShapedType>(typeConverter->convertType(attr.getType()));
    if (!builtinType) return {};
    return DenseIntOrFPElementsAttr::getFromRawBuffer(builtinType,
                                                      attr.getData());
  }
  if (auto attr = dyn_cast<vhlo::TransposeV1Attr>(vhloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(Transpose, V1);
  }
  // NOTE: TypeExtensionsV1Attr is only used as a RankedTensorType's encoding,
  // so it's handled during type conversion (see convertEncoding above).
  if (auto attr = dyn_cast<vhlo::TypeV1Attr>(vhloAttr)) {
    auto builtinType = typeConverter->convertType(attr.getValue());
    if (!builtinType) return {};
    return TypeAttr::get(builtinType);
  }

  // All VHLO Attributes must be converted by now.
  if (vhloAttr.getDialect().getNamespace() ==
      vhlo::VhloDialect::getDialectNamespace()) {
    // All VHLO attributes must have counterparts in StableHLO.
    return {};
  }

  // This should be unreachable unless program is a mix of VHLO and other
  // dialects, e.g. due to user edits to textual assembly format.
  return {};
}

//===----------------------------------------------------------------------===//
// VHLO --> StableHLO attributes: 2) Special cases.
// Applicable in areas where there is no 1:1 mapping from VHLO to StableHLO.
// This is pretty infrequent.
//===----------------------------------------------------------------------===//

// Indicates the outcome of converting a potentially special case.
// NOT_SPECIAL means that this wasn't actually a special case.
// SPECIAL_FAILURE means that this was a special case and conversion failed.
// SPECIAL_SUCCESS means that this was a special case and conversion succeeded.
enum class SpecialResult {
  SPECIAL_SUCCESS = 0,
  SPECIAL_FAILURE = 1,
  NOT_SPECIAL = 2,
};

SpecialResult specialSuccess() { return SpecialResult::SPECIAL_SUCCESS; }
SpecialResult specialFailure() { return SpecialResult::SPECIAL_FAILURE; }
SpecialResult notSpecial() { return SpecialResult::NOT_SPECIAL; }

LogicalResult convertBool(Attribute vhloAttr, bool& result) {
  auto vhloBooleanAttr = dyn_cast<vhlo::BooleanV1Attr>(vhloAttr);
  if (!vhloBooleanAttr) return failure();
  result = vhloBooleanAttr.getValue();
  return success();
}

bool isNoneType(Attribute vhloAttr) {
  auto typeAttr = llvm::dyn_cast<vhlo::TypeV1Attr>(vhloAttr);
  if (!typeAttr) return false;
  return llvm::isa<vhlo::NoneV1Type>(typeAttr.getValue());
}

LogicalResult convertTypeAttr(Attribute vhloAttr, Type& result,
                              const TypeConverter* typeConverter) {
  auto stablehloAttr = convertGeneric(vhloAttr, typeConverter);
  if (!stablehloAttr || !isa<TypeAttr>(stablehloAttr)) return failure();
  result = cast<TypeAttr>(stablehloAttr).getValue();
  return success();
}

LogicalResult convertInt(Attribute vhloAttr, int64_t& result) {
  auto vhloIntegerAttr = dyn_cast<vhlo::IntegerV1Attr>(vhloAttr);
  if (!vhloIntegerAttr) return failure();
  result = vhloIntegerAttr.getValue().getSExtValue();
  return success();
}

LogicalResult convertInts(Attribute vhloAttr,
                          const TypeConverter* typeConverter,
                          SmallVector<int64_t>& result) {
  auto vhloTensorAttr = dyn_cast<vhlo::TensorV1Attr>(vhloAttr);
  if (!vhloTensorAttr) return failure();
  auto stablehloAttr = dyn_cast_or_null<DenseIntElementsAttr>(
      convertGeneric(vhloAttr, typeConverter));
  if (!stablehloAttr) return failure();
  llvm::append_range(result, stablehloAttr.getValues<int64_t>());
  return success();
}

Attribute convertSymbol(Attribute vhloAttr,
                        const TypeConverter* typeConverter) {
  auto vhloStringAttr = dyn_cast<vhlo::StringV1Attr>(vhloAttr);
  if (!vhloStringAttr) return {};
  auto stablehloStringAttr = dyn_cast_or_null<StringAttr>(
      convertGeneric(vhloStringAttr, typeConverter));
  if (!stablehloStringAttr) return {};
  return FlatSymbolRefAttr::get(stablehloStringAttr);
}

template <typename OpType>
Attribute convertChannelHandle(OpType vhloOp,
                               const TypeConverter* typeConverter) {
  int64_t channelId, channelType;
  if (failed(convertInt(vhloOp.getChannelId(), channelId)) ||
      failed(convertInt(vhloOp.getChannelType(), channelType)))
    return {};
  return stablehlo::ChannelHandleAttr::get(vhloOp.getContext(), channelId,
                                           channelType);
}

Attribute convertChannelId(Attribute vhloAttr,
                           const TypeConverter* typeConverter) {
  int64_t channelId;
  if (failed(convertInt(vhloAttr, channelId))) return {};
  return stablehlo::ChannelHandleAttr::get(vhloAttr.getContext(), channelId,
                                           /*channelType=*/0);
}

template <typename OpType>
Attribute convertConvDimensionNumbers(OpType vhloOp,
                                      const TypeConverter* typeConverter) {
  int64_t stablehloInputBatchDimension, stablehloInputFeatureDimension;
  SmallVector<int64_t> stablehloInputSpatialDimensions;
  int64_t stablehloKernelInputFeatureDimension,
      stablehloKernelOutputFeatureDimension;
  SmallVector<int64_t> stablehloKernelSpatialDimensions;
  int64_t stablehloOutputBatchDimension, stablehloOutputFeatureDimension;
  SmallVector<int64_t> stablehloOutputSpatialDimensions;
  if (failed(convertInt(vhloOp.getInputBatchDimension(),
                        stablehloInputBatchDimension)) ||
      failed(convertInt(vhloOp.getInputFeatureDimension(),
                        stablehloInputFeatureDimension)) ||
      failed(convertInts(vhloOp.getInputSpatialDimensions(), typeConverter,
                         stablehloInputSpatialDimensions)) ||
      failed(convertInt(vhloOp.getKernelInputFeatureDimension(),
                        stablehloKernelInputFeatureDimension)) ||
      failed(convertInt(vhloOp.getKernelOutputFeatureDimension(),
                        stablehloKernelOutputFeatureDimension)) ||
      failed(convertInts(vhloOp.getKernelSpatialDimensions(), typeConverter,
                         stablehloKernelSpatialDimensions)) ||
      failed(convertInt(vhloOp.getOutputBatchDimension(),
                        stablehloOutputBatchDimension)) ||
      failed(convertInt(vhloOp.getOutputFeatureDimension(),
                        stablehloOutputFeatureDimension)) ||
      failed(convertInts(vhloOp.getOutputSpatialDimensions(), typeConverter,
                         stablehloOutputSpatialDimensions)))
    return {};
  return stablehlo::ConvDimensionNumbersAttr::get(
      vhloOp.getContext(), stablehloInputBatchDimension,
      stablehloInputFeatureDimension, stablehloInputSpatialDimensions,
      stablehloKernelInputFeatureDimension,
      stablehloKernelOutputFeatureDimension, stablehloKernelSpatialDimensions,
      stablehloOutputBatchDimension, stablehloOutputFeatureDimension,
      stablehloOutputSpatialDimensions);
}

Attribute convertCustomCallCalledComputations(
    Attribute vhloAttr, const TypeConverter* typeConverter) {
  if (auto vhloArrayAttr = dyn_cast<vhlo::ArrayV1Attr>(vhloAttr)) {
    SmallVector<Attribute> stablehloAttrs;
    for (auto vhloAttr : vhloArrayAttr.getValue()) {
      auto stablehloAttr = convertSymbol(vhloAttr, typeConverter);
      if (!stablehloAttr) return {};
      stablehloAttrs.push_back(stablehloAttr);
    }
    return ArrayAttr::get(vhloAttr.getContext(), stablehloAttrs);
  }
  return {};
}

FailureOr<Attribute> convertDotAlgorithm(vhlo::DotGeneralOpV2 vhloOp,
                                         const TypeConverter* typeConverter) {
  Type lhsPrecisionType, rhsPrecisionType, accumulationType;
  if (isNoneType(vhloOp.getLhsComponentCount())) {
    // All must be nonetype
    if (!isNoneType(vhloOp.getRhsComponentCount()) ||
        !isNoneType(vhloOp.getAccumulationType()) ||
        !isNoneType(vhloOp.getLhsComponentCount()) ||
        !isNoneType(vhloOp.getRhsComponentCount()) ||
        !isNoneType(vhloOp.getNumPrimitiveOperations()) ||
        !isNoneType(vhloOp.getAllowImpreciseAccumulation()))
      return failure();

    // Otherwise, valid NoneTyped Algorithm
    return Attribute{};
  }
  int64_t lhsComponentCount, rhsComponentCount, numPrimitiveOperations;
  bool allowImpreciseAccumulation;
  if (failed(convertTypeAttr(vhloOp.getLhsPrecisionType(), lhsPrecisionType,
                             typeConverter)) ||
      failed(convertTypeAttr(vhloOp.getRhsPrecisionType(), rhsPrecisionType,
                             typeConverter)) ||
      failed(convertTypeAttr(vhloOp.getAccumulationType(), accumulationType,
                             typeConverter)) ||
      failed(convertInt(vhloOp.getLhsComponentCount(), lhsComponentCount)) ||
      failed(convertInt(vhloOp.getRhsComponentCount(), rhsComponentCount)) ||
      failed(convertInt(vhloOp.getNumPrimitiveOperations(),
                        numPrimitiveOperations)) ||
      failed(convertBool(vhloOp.getAllowImpreciseAccumulation(),
                         allowImpreciseAccumulation)))
    return failure();
  return stablehlo::DotAlgorithmAttr::get(
      vhloOp->getContext(), lhsPrecisionType, rhsPrecisionType,
      accumulationType, lhsComponentCount, rhsComponentCount,
      numPrimitiveOperations, allowImpreciseAccumulation);
}

Attribute convertDotDimensionNumbers(vhlo::DotGeneralOpV2 vhloOp,
                                     const TypeConverter* typeConverter) {
  SmallVector<int64_t> stablehloLhsBatchingDimensions,
      stablehloRhsBatchingDimensions, stablehloLhsContractingDimensions,
      stablehloRhsContractingDimensions;
  if (failed(convertInts(vhloOp.getLhsBatchingDimensions(), typeConverter,
                         stablehloLhsBatchingDimensions)) ||
      failed(convertInts(vhloOp.getRhsBatchingDimensions(), typeConverter,
                         stablehloRhsBatchingDimensions)) ||
      failed(convertInts(vhloOp.getLhsContractingDimensions(), typeConverter,
                         stablehloLhsContractingDimensions)) ||
      failed(convertInts(vhloOp.getRhsContractingDimensions(), typeConverter,
                         stablehloRhsContractingDimensions)))
    return {};
  return stablehlo::DotDimensionNumbersAttr::get(
      vhloOp.getContext(), stablehloLhsBatchingDimensions,
      stablehloRhsBatchingDimensions, stablehloLhsContractingDimensions,
      stablehloRhsContractingDimensions);
}

Attribute convertFuncCallee(Attribute vhloAttr,
                            const TypeConverter* typeConverter) {
  return convertSymbol(vhloAttr, typeConverter);
}

template <typename OpType>
Attribute convertGatherDimensionNumbers(OpType vhloOp,
                                        const TypeConverter* typeConverter) {
  SmallVector<int64_t> stablehloOffsetDims, stablehloCollapsedSliceDims,
      stablehloOperandBatchingDims, stablehloStartIndicesBatchingDims,
      stablehloStartIndexMap;
  int64_t stablehloIndexVectorDim;
  if (failed(convertInts(vhloOp.getOffsetDims(), typeConverter,
                         stablehloOffsetDims)) ||
      failed(convertInts(vhloOp.getCollapsedSliceDims(), typeConverter,
                         stablehloCollapsedSliceDims)) ||
      failed(convertInts(vhloOp.getOperandBatchingDims(), typeConverter,
                         stablehloOperandBatchingDims)) ||
      failed(convertInts(vhloOp.getStartIndicesBatchingDims(), typeConverter,
                         stablehloStartIndicesBatchingDims)) ||
      failed(convertInts(vhloOp.getStartIndexMap(), typeConverter,
                         stablehloStartIndexMap)) ||
      failed(convertInt(vhloOp.getIndexVectorDim(), stablehloIndexVectorDim)))
    return {};
  return stablehlo::GatherDimensionNumbersAttr::get(
      vhloOp.getContext(), stablehloOffsetDims, stablehloCollapsedSliceDims,
      stablehloOperandBatchingDims, stablehloStartIndicesBatchingDims,
      stablehloStartIndexMap, stablehloIndexVectorDim);
}

Attribute convertScatterDimensionNumbers(vhlo::ScatterOpV2 vhloOp,
                                         const TypeConverter* typeConverter) {
  SmallVector<int64_t> stablehloUpdateWindowDims, stablehloInsertedWindowDims,
      stablehloInputBatchingDims, stablehloScatterIndicesBatchingDims,
      stablehloScatterDimsToOperandDims;
  int64_t stablehloIndexVectorDim;
  if (failed(convertInts(vhloOp.getUpdateWindowDims(), typeConverter,
                         stablehloUpdateWindowDims)) ||
      failed(convertInts(vhloOp.getInsertedWindowDims(), typeConverter,
                         stablehloInsertedWindowDims)) ||
      failed(convertInts(vhloOp.getInputBatchingDims(), typeConverter,
                         stablehloInputBatchingDims)) ||
      failed(convertInts(vhloOp.getScatterIndicesBatchingDims(), typeConverter,
                         stablehloScatterIndicesBatchingDims)) ||
      failed(convertInts(vhloOp.getScatterDimsToOperandDims(), typeConverter,
                         stablehloScatterDimsToOperandDims)) ||
      failed(convertInt(vhloOp.getIndexVectorDim(), stablehloIndexVectorDim)))
    return {};
  return stablehlo::ScatterDimensionNumbersAttr::get(
      vhloOp.getContext(), stablehloUpdateWindowDims,
      stablehloInsertedWindowDims, stablehloInputBatchingDims,
      stablehloScatterIndicesBatchingDims, stablehloScatterDimsToOperandDims,
      stablehloIndexVectorDim);
}

#undef RETURN_CONVERTED_ENUM_ATTR

template <typename... StringTy>
void eraseAttrs(SmallVector<NamedAttribute>& attrs, StringTy... names) {
  StringSet<llvm::MallocAllocator> nameSet({names...});
  llvm::erase_if(attrs, [&](NamedAttribute attr) {
    return nameSet.contains(attr.getName());
  });
}

template <typename VhloOpTy>
LogicalResult implodeSpecial(const OpConversionPattern<VhloOpTy>& pattern,
                             VhloOpTy vhloOp,
                             SmallVector<NamedAttribute>& vhloAttrs,
                             SmallVector<NamedAttribute>& stablehloAttrs) {
  if constexpr (std::is_same<VhloOpTy, vhlo::ConvolutionOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::DynamicConvOpV2>::value) {
    auto stablehloAttr =
        convertConvDimensionNumbers(vhloOp, pattern.getTypeConverter());
    if (!stablehloAttr) return failure();
    stablehloAttrs.emplace_back(
        StringAttr::get(pattern.getContext(), "dimension_numbers"),
        stablehloAttr);
    eraseAttrs(vhloAttrs, "input_batch_dimension", "input_feature_dimension",
               "input_spatial_dimensions", "kernel_input_feature_dimension",
               "kernel_output_feature_dimension", "kernel_spatial_dimensions",
               "output_batch_dimension", "output_feature_dimension",
               "output_spatial_dimensions");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::DotGeneralOpV2>::value) {
    // Dot Dimension Numbers
    auto stablehloDotDimAttr =
        convertDotDimensionNumbers(vhloOp, pattern.getTypeConverter());
    if (!stablehloDotDimAttr) return failure();
    stablehloAttrs.emplace_back(
        StringAttr::get(pattern.getContext(), "dot_dimension_numbers"),
        stablehloDotDimAttr);
    eraseAttrs(vhloAttrs, "lhs_batching_dimensions", "rhs_batching_dimensions",
               "lhs_contracting_dimensions", "rhs_contracting_dimensions");

    // Dot Algorithm
    auto stablehloDotAlgorithmAttr =
        convertDotAlgorithm(vhloOp, pattern.getTypeConverter());
    if (failed(stablehloDotAlgorithmAttr)) return failure();
    if (stablehloDotAlgorithmAttr.value())
      stablehloAttrs.emplace_back(
          StringAttr::get(pattern.getContext(), "algorithm"),
          stablehloDotAlgorithmAttr.value());
    eraseAttrs(vhloAttrs, "lhs_precision_type", "rhs_precision_type",
               "accumulation_type", "lhs_component_count",
               "rhs_component_count", "num_primitive_operations",
               "allow_imprecise_accumulation");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::DynamicGatherOpV2>::value ||
                std::is_same<VhloOpTy, vhlo::GatherOpV2>::value) {
    auto stablehloAttr =
        convertGatherDimensionNumbers(vhloOp, pattern.getTypeConverter());
    if (!stablehloAttr) return failure();
    stablehloAttrs.emplace_back(
        StringAttr::get(pattern.getContext(), "dimension_numbers"),
        stablehloAttr);
    eraseAttrs(vhloAttrs, "offset_dims", "collapsed_slice_dims",
               "operand_batching_dims", "start_indices_batching_dims",
               "start_index_map", "index_vector_dim");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::ScatterOpV2>::value) {
    auto stablehloAttr =
        convertScatterDimensionNumbers(vhloOp, pattern.getTypeConverter());
    if (!stablehloAttr) return failure();
    stablehloAttrs.emplace_back(
        StringAttr::get(pattern.getContext(), "scatter_dimension_numbers"),
        stablehloAttr);
    eraseAttrs(vhloAttrs, "update_window_dims", "inserted_window_dims",
               "input_batching_dims", "scatter_indices_batching_dims",
               "scatter_dims_to_operand_dims", "index_vector_dim");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::RecvOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::SendOpV1>::value) {
    auto stablehloAttr =
        convertChannelHandle(vhloOp, pattern.getTypeConverter());
    if (!stablehloAttr) return failure();
    stablehloAttrs.emplace_back(
        StringAttr::get(pattern.getContext(), "channel_handle"), stablehloAttr);
    eraseAttrs(vhloAttrs, "channel_id", "channel_type");
  }
  return success();
}

template <typename T, typename DenseArrayAttr>
SpecialResult convertDenseArray(const TypeConverter* typeConverter,
                                StringAttr vhloName, Attribute vhloAttr,
                                SmallVector<NamedAttribute>& stablehloAttrs) {
  auto tensorAttr = dyn_cast<vhlo::TensorV1Attr>(vhloAttr);
  if (!tensorAttr) return specialFailure();

  auto type = dyn_cast<RankedTensorType>(
      typeConverter->convertType(tensorAttr.getType()));
  if (!type) return specialFailure();

  auto elems = DenseElementsAttr::getFromRawBuffer(type, tensorAttr.getData());

  stablehloAttrs.emplace_back(
      vhloName, DenseArrayAttr::get(vhloAttr.getContext(),
                                    llvm::to_vector(elems.getValues<T>())));
  return specialSuccess();
}

SpecialResult convertDenseBoolArray(
    const TypeConverter* typeConverter, StringAttr vhloName, Attribute vhloAttr,
    SmallVector<NamedAttribute>& stablehloAttrs) {
  return convertDenseArray<bool, DenseBoolArrayAttr>(typeConverter, vhloName,
                                                     vhloAttr, stablehloAttrs);
}

SpecialResult convertDenseI64Array(
    const TypeConverter* typeConverter, StringAttr vhloName, Attribute vhloAttr,
    SmallVector<NamedAttribute>& stablehloAttrs) {
  return convertDenseArray<int64_t, DenseI64ArrayAttr>(
      typeConverter, vhloName, vhloAttr, stablehloAttrs);
}

template <typename VhloOpTy>
SpecialResult convertSpecial(const OpConversionPattern<VhloOpTy>& pattern,
                             StringAttr vhloName, Attribute vhloAttr,
                             SmallVector<NamedAttribute>& stablehloAttrs) {
  StringAttr stablehloName = vhloName;
  Attribute stablehloAttr;
  auto typeConverter = pattern.getTypeConverter();

  if constexpr (std::is_same<VhloOpTy, vhlo::AllGatherOpV2>::value ||
                std::is_same<VhloOpTy, vhlo::AllReduceOpV2>::value ||
                std::is_same<VhloOpTy, vhlo::AllToAllOpV2>::value ||
                std::is_same<VhloOpTy, vhlo::CollectivePermuteOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::ReduceScatterOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::CollectiveBroadcastOpV1>::value) {
    if (vhloName == "channel_id") {
      stablehloName = StringAttr::get(pattern.getContext(), "channel_handle");
      stablehloAttr = convertChannelId(vhloAttr, pattern.getTypeConverter());
      if (!stablehloAttr) return specialFailure();
    }
    if (vhloName == "use_global_device_ids") {
      auto vhloBooleanAttr = dyn_cast<vhlo::BooleanV1Attr>(vhloAttr);
      if (!vhloBooleanAttr) return specialFailure();
      if (!vhloBooleanAttr.getValue()) return specialSuccess();
      stablehloAttr = UnitAttr::get(pattern.getContext());
    }
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::CustomCallOpV1>::value) {
    if (vhloName == "called_computations") {
      stablehloAttr = convertCustomCallCalledComputations(
          vhloAttr, pattern.getTypeConverter());
      if (!stablehloAttr) return specialFailure();
    }
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::CompositeOpV1>::value) {
    if (vhloName == "decomposition") {
      stablehloAttr = convertSymbol(vhloAttr, pattern.getTypeConverter());
      if (!stablehloAttr) return specialFailure();
    }
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::CallOpV1>::value) {
    if (vhloName == "callee") {
      stablehloAttr = convertFuncCallee(vhloAttr, pattern.getTypeConverter());
      if (!stablehloAttr) return specialFailure();
    }
  }
  if (stablehloAttr) {
    stablehloAttrs.emplace_back(stablehloName, stablehloAttr);
    return specialSuccess();
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::FftOpV1>::value) {
    if (vhloName == "fft_length")
      return convertDenseI64Array(typeConverter, vhloName, vhloAttr,
                                  stablehloAttrs);
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::BroadcastOpV1>::value) {
    if (vhloName == "broadcast_sizes")
      return convertDenseI64Array(typeConverter, vhloName, vhloAttr,
                                  stablehloAttrs);
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::DynamicSliceOpV1>::value) {
    if (vhloName == "slice_sizes")
      return convertDenseI64Array(typeConverter, vhloName, vhloAttr,
                                  stablehloAttrs);
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::ReverseOpV1>::value) {
    if (vhloName == "dimensions")
      return convertDenseI64Array(typeConverter, vhloName, vhloAttr,
                                  stablehloAttrs);
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::TransposeOpV1>::value) {
    if (vhloName == "permutation")
      return convertDenseI64Array(typeConverter, vhloName, vhloAttr,
                                  stablehloAttrs);
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::PadOpV1>::value) {
    if (vhloName == "edge_padding_low" || vhloName == "edge_padding_high" ||
        vhloName == "interior_padding")
      return convertDenseI64Array(typeConverter, vhloName, vhloAttr,
                                  stablehloAttrs);
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::SliceOpV1>::value) {
    if (vhloName == "start_indices" || vhloName == "limit_indices" ||
        vhloName == "strides")
      return convertDenseI64Array(typeConverter, vhloName, vhloAttr,
                                  stablehloAttrs);
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::BroadcastInDimOpV1>::value) {
    if (vhloName == "broadcast_dimensions")
      return convertDenseI64Array(typeConverter, vhloName, vhloAttr,
                                  stablehloAttrs);
  }
  if constexpr (std::is_same<VhloOpTy,
                             vhlo::DynamicBroadcastInDimOpV1>::value) {
    if (vhloName == "broadcast_dimensions" ||
        vhloName == "known_expanding_dimensions" ||
        vhloName == "known_nonexpanding_dimensions")
      return convertDenseI64Array(typeConverter, vhloName, vhloAttr,
                                  stablehloAttrs);
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::SelectAndScatterOpV1>::value) {
    if (vhloName == "window_dimensions" || vhloName == "window_strides")
      return convertDenseI64Array(typeConverter, vhloName, vhloAttr,
                                  stablehloAttrs);
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::ReduceWindowOpV1>::value) {
    if (vhloName == "window_dimensions" || vhloName == "window_strides" ||
        vhloName == "base_dilations" || vhloName == "window_dilations")
      return convertDenseI64Array(typeConverter, vhloName, vhloAttr,
                                  stablehloAttrs);
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::MapOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::ReduceOpV1>::value) {
    if (vhloName == "dimensions")
      return convertDenseI64Array(typeConverter, vhloName, vhloAttr,
                                  stablehloAttrs);
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::GatherOpV2>::value) {
    if (vhloName == "slice_sizes")
      return convertDenseI64Array(typeConverter, vhloName, vhloAttr,
                                  stablehloAttrs);
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::ConvolutionOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::DynamicConvOpV2>::value) {
    if (vhloName == "lhs_dilation" || vhloName == "rhs_dilation" ||
        vhloName == "window_strides")
      return convertDenseI64Array(typeConverter, vhloName, vhloAttr,
                                  stablehloAttrs);
    if (vhloName == "window_reversal")
      return convertDenseBoolArray(typeConverter, vhloName, vhloAttr,
                                   stablehloAttrs);
  }
  return notSpecial();
}

//===----------------------------------------------------------------------===//
// VHLO --> StableHLO attributes: 3) Default attributes.
// Unlike StableHLO, VHLO doesn't have default attributes, so the corresponding
// attributes are added explicitly during StableHLO --> VHLO conversion.
// These attributes are removed here. (Strictly speaking, don't have to do this
// but this makes eyeballing easier).
//===----------------------------------------------------------------------===//

bool isBoolean(Attribute vhloAttr, bool value) {
  auto attr = dyn_cast_or_null<vhlo::BooleanV1Attr>(vhloAttr);
  return attr && attr.getValue() == value;
}

bool isEmptyArray(Attribute vhloAttr) {
  auto attr = dyn_cast_or_null<vhlo::ArrayV1Attr>(vhloAttr);
  return attr && attr.getValue().empty();
}

bool isEmptyDictionary(Attribute vhloAttr) {
  auto attr = dyn_cast_or_null<vhlo::DictionaryV1Attr>(vhloAttr);
  return attr && attr.getValue().empty();
}

bool isEmptyString(Attribute vhloAttr) {
  auto attr = dyn_cast_or_null<vhlo::StringV1Attr>(vhloAttr);
  return attr && attr.getValue().empty();
}

bool isEmptyTensor(Attribute vhloAttr) {
  auto attr = dyn_cast_or_null<vhlo::TensorV1Attr>(vhloAttr);
  return attr && attr.getData().empty();
}

bool isEnum(Attribute vhloAttr, Attribute value) { return vhloAttr == value; }

bool isInteger(Attribute vhloAttr, int64_t value) {
  auto attr = dyn_cast_or_null<vhlo::IntegerV1Attr>(vhloAttr);
  return attr && attr.getValue().getSExtValue() == value;
}

bool isSplatArray(Attribute vhloAttr, Attribute splatValue) {
  auto attr = dyn_cast_or_null<vhlo::ArrayV1Attr>(vhloAttr);
  return attr && llvm::all_of(attr.getValue(), [&](Attribute attr) {
           return attr == splatValue;
         });
}

template <typename T>
bool isSplatTensor(const ConversionPattern& pattern, Attribute vhloAttr,
                   T splatValue) {
  auto attr = dyn_cast_or_null<DenseElementsAttr>(
      convertGeneric(vhloAttr, pattern.getTypeConverter()));
  return attr && attr.isSplat() &&
         attr.template getSplatValue<T>() == splatValue;
}

bool isString(Attribute vhloAttr, StringRef value) {
  auto attr = dyn_cast_or_null<vhlo::StringV1Attr>(vhloAttr);
  return attr && attr.getValue() == value;
}

// TODO(#1232): Also validate attributes before removing them.
// Current logic assumes that these attributes are valid, but that might not
// necessarily be the case because VHLO doesn't have verifiers.
template <typename VhloOpTy>
LogicalResult removeDefaults(const OpConversionPattern<VhloOpTy>& pattern,
                             VhloOpTy vhloOp,
                             SmallVector<NamedAttribute>& vhloAttrs) {
  if constexpr (std::is_same<VhloOpTy, vhlo::FuncOpV1>::value) {
    if (isString(vhloOp.getSymVisibilityAttr(), ""))
      eraseAttrs(vhloAttrs, "sym_visibility");
    if (isEmptyArray(vhloOp.getArgAttrsAttr()))
      eraseAttrs(vhloAttrs, "arg_attrs");
    if (isEmptyArray(vhloOp.getResAttrsAttr()))
      eraseAttrs(vhloAttrs, "res_attrs");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::AllGatherOpV2>::value ||
                std::is_same<VhloOpTy, vhlo::AllReduceOpV2>::value ||
                std::is_same<VhloOpTy, vhlo::ReduceScatterOpV1>::value) {
    if (isInteger(vhloOp.getChannelIdAttr(), 0))
      eraseAttrs(vhloAttrs, "channel_id");
    if (isBoolean(vhloOp.getUseGlobalDeviceIdsAttr(), false))
      eraseAttrs(vhloAttrs, "use_global_device_ids");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::AllToAllOpV2>::value ||
                std::is_same<VhloOpTy, vhlo::CollectivePermuteOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::CollectiveBroadcastOpV1>::value) {
    if (isInteger(vhloOp.getChannelIdAttr(), 0))
      eraseAttrs(vhloAttrs, "channel_id");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::CholeskyOpV1>::value) {
    if (isBoolean(vhloOp.getLowerAttr(), false)) eraseAttrs(vhloAttrs, "lower");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::CompareOpV1>::value) {
    if (isEnum(vhloOp.getCompareTypeAttr(),
               vhlo::ComparisonTypeV1Attr::get(pattern.getContext(),
                                               vhlo::ComparisonTypeV1::NOTYPE)))
      eraseAttrs(vhloAttrs, "compare_type");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::CompositeOpV1>::value) {
    if (isInteger(vhloOp.getVersionAttr(), 0)) {
      eraseAttrs(vhloAttrs, "version");
    }
    if (isEmptyDictionary(vhloOp.getCompositeAttributesAttr())) {
      eraseAttrs(vhloAttrs, "composite_attributes");
    }
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::ConvolutionOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::DynamicConvOpV2>::value) {
    if (isSplatTensor(pattern, vhloOp.getWindowStridesAttr(), 1ll))
      eraseAttrs(vhloAttrs, "window_strides");
    if constexpr (std::is_same<VhloOpTy, vhlo::ConvolutionOpV1>::value) {
      if (isSplatTensor(pattern, vhloOp.getPaddingAttr(), 0ll))
        eraseAttrs(vhloAttrs, "padding");
    }
    if (isSplatTensor(pattern, vhloOp.getLhsDilationAttr(), 1ll))
      eraseAttrs(vhloAttrs, "lhs_dilation");
    if (isSplatTensor(pattern, vhloOp.getRhsDilationAttr(), 1ll))
      eraseAttrs(vhloAttrs, "rhs_dilation");
    if (isSplatTensor(pattern, vhloOp.getWindowReversalAttr(), false))
      eraseAttrs(vhloAttrs, "window_reversal");
    if (isSplatArray(vhloOp.getPrecisionConfigAttr(),
                     vhlo::PrecisionV1Attr::get(pattern.getContext(),
                                                vhlo::PrecisionV1::DEFAULT)))
      eraseAttrs(vhloAttrs, "precision_config");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::CustomCallOpV1>::value) {
    if (isBoolean(vhloOp.getHasSideEffectAttr(), false))
      eraseAttrs(vhloAttrs, "has_side_effect");
    if (isEmptyString(vhloOp.getBackendConfigAttr()) ||
        isEmptyDictionary(vhloOp.getBackendConfigAttr()))
      eraseAttrs(vhloAttrs, "backend_config");
    if (isEnum(vhloOp.getApiVersionAttr(),
               vhlo::CustomCallApiVersionV1Attr::get(
                   pattern.getContext(),
                   vhlo::CustomCallApiVersionV1::API_VERSION_ORIGINAL)))
      eraseAttrs(vhloAttrs, "api_version");
    if (isEmptyArray(vhloOp.getCalledComputations()))
      eraseAttrs(vhloAttrs, "called_computations");
    if (isEmptyArray(vhloOp.getOperandLayouts()) &&
        isEmptyArray(vhloOp.getResultLayouts())) {
      eraseAttrs(vhloAttrs, "operand_layouts");
      eraseAttrs(vhloAttrs, "result_layouts");
    }
    if (isEmptyArray(vhloOp.getOutputOperandAliases()))
      eraseAttrs(vhloAttrs, "output_operand_aliases");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::DotGeneralOpV2>::value ||
                std::is_same<VhloOpTy, vhlo::DotOpV1>::value) {
    if (isSplatArray(vhloOp.getPrecisionConfigAttr(),
                     vhlo::PrecisionV1Attr::get(pattern.getContext(),
                                                vhlo::PrecisionV1::DEFAULT)))
      eraseAttrs(vhloAttrs, "precision_config");
  }
  if constexpr (std::is_same<VhloOpTy,
                             vhlo::DynamicBroadcastInDimOpV1>::value) {
    if (isEmptyTensor(vhloOp.getKnownExpandingDimensionsAttr()))
      eraseAttrs(vhloAttrs, "known_expanding_dimensions");
    if (isEmptyTensor(vhloOp.getKnownNonexpandingDimensionsAttr()))
      eraseAttrs(vhloAttrs, "known_nonexpanding_dimensions");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::DynamicGatherOpV2>::value ||
                std::is_same<VhloOpTy, vhlo::GatherOpV2>::value) {
    if (isBoolean(vhloOp.getIndicesAreSortedAttr(), false))
      eraseAttrs(vhloAttrs, "indices_are_sorted");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::InfeedOpV1>::value) {
    if (isEmptyString(vhloOp.getInfeedConfig()))
      eraseAttrs(vhloAttrs, "infeed_config");
    if (isEmptyArray(vhloOp.getLayout())) eraseAttrs(vhloAttrs, "layout");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::OutfeedOpV1>::value) {
    if (isEmptyString(vhloOp.getOutfeedConfig()))
      eraseAttrs(vhloAttrs, "outfeed_config");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::RecvOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::SendOpV1>::value) {
    if (isBoolean(vhloOp.getIsHostTransferAttr(), false))
      eraseAttrs(vhloAttrs, "is_host_transfer");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::ReduceWindowOpV1>::value) {
    if (isSplatTensor(pattern, vhloOp.getWindowStridesAttr(), 1ll))
      eraseAttrs(vhloAttrs, "window_strides");
    if (isSplatTensor(pattern, vhloOp.getBaseDilationsAttr(), 1ll))
      eraseAttrs(vhloAttrs, "base_dilations");
    if (isSplatTensor(pattern, vhloOp.getWindowDilationsAttr(), 1ll))
      eraseAttrs(vhloAttrs, "window_dilations");
    if (isSplatTensor(pattern, vhloOp.getPaddingAttr(), 0ll))
      eraseAttrs(vhloAttrs, "padding");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::ScatterOpV2>::value) {
    if (isBoolean(vhloOp.getIndicesAreSortedAttr(), false))
      eraseAttrs(vhloAttrs, "indices_are_sorted");
    if (isBoolean(vhloOp.getUniqueIndicesAttr(), false))
      eraseAttrs(vhloAttrs, "unique_indices");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::SelectAndScatterOpV1>::value) {
    if (isSplatTensor(pattern, vhloOp.getWindowStridesAttr(), 1ll))
      eraseAttrs(vhloAttrs, "window_strides");
    if (isSplatTensor(pattern, vhloOp.getPaddingAttr(), 0ll))
      eraseAttrs(vhloAttrs, "padding");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::SortOpV1>::value) {
    if (isInteger(vhloOp.getDimensionAttr(), -1))
      eraseAttrs(vhloAttrs, "dimension");
    if (isBoolean(vhloOp.getIsStableAttr(), false))
      eraseAttrs(vhloAttrs, "is_stable");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// VHLO --> StableHLO operations
//===----------------------------------------------------------------------===//

template <typename VhloOpTy>
class VhloToStablehloOpConverter : public OpConversionPattern<VhloOpTy> {
 public:
  using OpConversionPattern<VhloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      VhloOpTy vhloOp, typename VhloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    SmallVector<Type> stablehloTypes;
    if (failed(this->getTypeConverter()->convertTypes(vhloOp->getResultTypes(),
                                                      stablehloTypes)))
      return failure();

    // These operands have already been converted to StableHLO by
    // the dialect conversion infrastructure.
    ValueRange stablehloOperands = adaptor.getOperands();

    // Convert VHLO attributes to StableHLO equivalents.
    // There are two paths:
    //   1) Generic path where default logic applies, and there is a 1:1
    //      mapping from VHLO to StableHLO.
    //   2) Special cases (currently, about a dozen) where there is not 1:1
    //      mapping from VHLO to StableHLO.
    SmallVector<NamedAttribute> vhloAttrs = to_vector(vhloOp->getAttrs());
    SmallVector<NamedAttribute> stablehloAttrs;
    if (failed(removeDefaults(*this, vhloOp, vhloAttrs))) return failure();
    if (failed(implodeSpecial(*this, vhloOp, vhloAttrs, stablehloAttrs)))
      return failure();
    for (NamedAttribute vhloAttr : vhloAttrs) {
      auto result = convertSpecial(*this, vhloAttr.getName(),
                                   vhloAttr.getValue(), stablehloAttrs);
      switch (result) {
        case SpecialResult::SPECIAL_SUCCESS:
          break;
        case SpecialResult::SPECIAL_FAILURE:
          return failure();
        case SpecialResult::NOT_SPECIAL:
          auto stablehloAttr =
              convertGeneric(vhloAttr.getValue(), this->getTypeConverter());
          if (!stablehloAttr) return failure();
          stablehloAttrs.push_back({vhloAttr.getName(), stablehloAttr});
          break;
      }
    }

    // Replace vhlo.return --> func.return if direct parent is a func op.
    if constexpr (std::is_same<VhloOpTy, vhlo::ReturnOpV1>::value) {
      if (llvm::isa<vhlo::FuncOpV1, func::FuncOp>(vhloOp->getParentOp())) {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(
            vhloOp, stablehloTypes, stablehloOperands, stablehloAttrs);
        return success();
      }
    }

    // Convert the VHLO operation to a StableHLO equivalent.
    // This can almost be done in a generic fashion, except for
    // vhlo.case that uses a variadic number of regions which means an
    // additional argument for the generic builder.
    VhloToStablehloOp<VhloOpTy> stablehloOp;
    if constexpr (std::is_same<VhloOpTy, vhlo::CaseOpV1>::value) {
      stablehloOp = rewriter.create<stablehlo::CaseOp>(
          vhloOp.getLoc(), stablehloTypes, stablehloOperands, stablehloAttrs,
          vhloOp.getBranches().size());
    } else {
      stablehloOp = rewriter.create<VhloToStablehloOp<VhloOpTy>>(
          vhloOp.getLoc(), stablehloTypes, stablehloOperands, stablehloAttrs);
    }

    for (auto [vhloRegion, stablehloRegion] :
         llvm::zip(vhloOp->getRegions(), stablehloOp->getRegions())) {
      rewriter.inlineRegionBefore(vhloRegion, stablehloRegion,
                                  stablehloRegion.end());
      if (failed(rewriter.convertRegionTypes(&stablehloRegion,
                                             *this->getTypeConverter(),
                                             /*entryConversion=*/nullptr)))
        return failure();
    }

    rewriter.replaceOp(vhloOp, stablehloOp);
    return success();
  }
};

template <typename... StablehloOpTypes>
void populateVhloToStablehloPatterns(RewritePatternSet* patterns,
                                     TypeConverter* converter,
                                     MLIRContext* context) {
  patterns
      ->add<VhloToStablehloOpConverter<StablehloToVhloOp<StablehloOpTypes>>...>(
          *converter, context);
}

}  // namespace

struct VhloLegalizeToStablehloPass
    : public impl::VhloLegalizeToStablehloPassBase<
          VhloLegalizeToStablehloPass> {
  LogicalResult initialize(MLIRContext* context) override {
    target = std::make_shared<ConversionTarget>(*context);
    target->addIllegalDialect<vhlo::VhloDialect>();
    target->addLegalDialect<stablehlo::StablehloDialect>();
    target->addLegalDialect<func::FuncDialect>();

    RewritePatternSet patterns_(context);
    stablehlo::populateVhloToStablehloPatterns(&patterns_, &converter, context);
    patterns = std::move(patterns_);

    return success();
  }

  void runOnOperation() override {
    // Upgraded VHLO should always be convertible to StableHLO.
    // Arbitrary VHLO might not be convertible if it uses deprecated features
    // which are no longer available in StableHLO.
    if (failed(applyPartialConversion(getOperation(), *target, patterns))) {
      return signalPassFailure();
    }
  }

 private:
  VhloToStablehloTypeConverter converter;
  FrozenRewritePatternSet patterns;
  std::shared_ptr<ConversionTarget> target;
};

void populateVhloToStablehloPatterns(RewritePatternSet* patterns,
                                     TypeConverter* converter,
                                     MLIRContext* context) {
  populateVhloToStablehloPatterns<
#define GET_OP_LIST
#include "stablehlo/dialect/StablehloOps.cpp.inc"
      , func::CallOp, func::FuncOp>(patterns, converter, context);
  // Omit ReturnOp since it is handled during conversion of vhlo::ReturnOp
}

}  // namespace stablehlo
}  // namespace mlir
