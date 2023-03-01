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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
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

  Attribute convertEncoding(Attribute attr) final {
    if (auto vhloAttr = attr.dyn_cast_or_null<vhlo::TypeExtensionsV1Attr>()) {
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

Attribute convertGeneric(Attribute vhloAttr, TypeConverter* typeConverter) {
  LLVM_DEBUG(llvm::dbgs() << "Converting attr " << vhloAttr);
  // TODO: ArgResultAliasV1Attr isn't handled yet.
  if (auto vhloAttrs = vhloAttr.dyn_cast<vhlo::ArrayV1Attr>()) {
    SmallVector<Attribute> stablehloAttrs;
    for (auto vhloAttr : vhloAttrs.getValue()) {
      auto stablehloAttr = convertGeneric(vhloAttr, typeConverter);
      if (!stablehloAttr) return {};
      stablehloAttrs.push_back(stablehloAttr);
    }
    return ArrayAttr::get(vhloAttrs.getContext(), stablehloAttrs);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::BooleanV1Attr>()) {
    return BoolAttr::get(attr.getContext(), attr.getValue());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::ComparisonDirectionV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonDirection, V1);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::ComparisonTypeV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonType, V1);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::CustomCallApiVersionV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(CustomCallApiVersion, V1);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::DictionaryV1Attr>()) {
    SmallVector<NamedAttribute> vhloAttrs;
    for (auto namedAttr : attr.getValue()) {
      auto builtinName = convertGeneric(namedAttr.first, typeConverter)
                             .dyn_cast_or_null<StringAttr>();
      auto builtinValue = convertGeneric(namedAttr.second, typeConverter);
      if (!builtinName || !builtinValue) return {};
      vhloAttrs.push_back({builtinName, builtinValue});
    }
    return DictionaryAttr::get(attr.getContext(), vhloAttrs);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::FftTypeV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(FftType, V1);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::FloatV1Attr>()) {
    auto builtinFloatType = typeConverter->convertType(attr.getType());
    if (!builtinFloatType) return {};
    // FIXME: What is the proper way to reconstruct a attr?
    return FloatAttr::get(builtinFloatType, attr.getValue().convertToDouble());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::IntegerV1Attr>()) {
    auto builtinIntegerType = typeConverter->convertType(attr.getType());
    if (!builtinIntegerType) return {};
    return IntegerAttr::get(builtinIntegerType, attr.getValue());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::OutputOperandAliasV1Attr>()) {
    return stablehlo::OutputOperandAliasAttr::get(
        attr.getContext(), attr.getOutputTupleIndices(), attr.getOperandIndex(),
        attr.getOperandTupleIndices());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::PrecisionV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Precision, V1);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::RngAlgorithmV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngAlgorithm, V1);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::RngDistributionV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngDistribution, V1);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::StringV1Attr>()) {
    return StringAttr::get(attr.getContext(), attr.getValue());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::TensorV1Attr>()) {
    auto builtinType = typeConverter->convertType(attr.getType());
    if (!builtinType) return {};
    return DenseIntOrFPElementsAttr::getFromRawBuffer(builtinType,
                                                      attr.getData());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::TransposeV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Transpose, V1);
  }
  // NOTE: TypeExtensionsV1Attr is only used as a RankedTensorType's encoding,
  // so it's handled during type conversion (see convertEncoding above).
  if (auto attr = vhloAttr.dyn_cast<vhlo::TypeV1Attr>()) {
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

LogicalResult convertInt(Attribute vhloAttr, int64_t& result) {
  auto vhloIntegerAttr = vhloAttr.dyn_cast<vhlo::IntegerV1Attr>();
  if (!vhloIntegerAttr) return failure();
  result = vhloIntegerAttr.getValue().getSExtValue();
  return success();
}

LogicalResult convertInts(Attribute vhloAttr, TypeConverter* typeConverter,
                          SmallVector<int64_t>& result) {
  auto vhloTensorAttr = vhloAttr.dyn_cast<vhlo::TensorV1Attr>();
  if (!vhloTensorAttr) return failure();
  auto stablehloAttr = convertGeneric(vhloAttr, typeConverter)
                           .dyn_cast_or_null<DenseIntElementsAttr>();
  if (!stablehloAttr) return failure();
  llvm::append_range(result, stablehloAttr.getValues<int64_t>());
  return success();
}

Attribute convertSymbol(Attribute vhloAttr, TypeConverter* typeConverter) {
  auto vhloStringAttr = vhloAttr.dyn_cast<vhlo::StringV1Attr>();
  if (!vhloStringAttr) return {};
  auto stablehloStringAttr = convertGeneric(vhloStringAttr, typeConverter)
                                 .dyn_cast_or_null<StringAttr>();
  if (!stablehloStringAttr) return {};
  return FlatSymbolRefAttr::get(stablehloStringAttr);
}

template <typename OpType>
Attribute convertChannelHandle(OpType vhloOp, TypeConverter* typeConverter) {
  int64_t channelId, channelType;
  if (failed(convertInt(vhloOp.getChannelId(), channelId)) ||
      failed(convertInt(vhloOp.getChannelType(), channelType)))
    return {};
  return stablehlo::ChannelHandleAttr::get(vhloOp.getContext(), channelId,
                                           channelType);
}

Attribute convertChannelId(Attribute vhloAttr, TypeConverter* typeConverter) {
  int64_t channelId;
  if (failed(convertInt(vhloAttr, channelId))) return {};
  return stablehlo::ChannelHandleAttr::get(vhloAttr.getContext(), channelId,
                                           /*channelType=*/0);
}

template <typename OpType>
Attribute convertConvDimensionNumbers(OpType vhloOp,
                                      TypeConverter* typeConverter) {
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

Attribute convertCustomCallCalledComputations(Attribute vhloAttr,
                                              TypeConverter* typeConverter) {
  if (auto vhloArrayAttr = vhloAttr.dyn_cast<vhlo::ArrayV1Attr>()) {
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

Attribute convertDotDimensionNumbers(vhlo::DotGeneralOpV1 vhloOp,
                                     TypeConverter* typeConverter) {
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

Attribute convertFuncCallee(Attribute vhloAttr, TypeConverter* typeConverter) {
  return convertSymbol(vhloAttr, typeConverter);
}

template <typename OpType>
Attribute convertGatherDimensionNumbers(OpType vhloOp,
                                        TypeConverter* typeConverter) {
  SmallVector<int64_t> stablehloOffsetDims, stablehloCollapsedSliceDims,
      stablehloStartIndexMap;
  int64_t stablehloIndexVectorDim;
  if (failed(convertInts(vhloOp.getOffsetDims(), typeConverter,
                         stablehloOffsetDims)) ||
      failed(convertInts(vhloOp.getCollapsedSliceDims(), typeConverter,
                         stablehloCollapsedSliceDims)) ||
      failed(convertInts(vhloOp.getStartIndexMap(), typeConverter,
                         stablehloStartIndexMap)) ||
      failed(convertInt(vhloOp.getIndexVectorDim(), stablehloIndexVectorDim)))
    return {};
  return stablehlo::GatherDimensionNumbersAttr::get(
      vhloOp.getContext(), stablehloOffsetDims, stablehloCollapsedSliceDims,
      stablehloStartIndexMap, stablehloIndexVectorDim);
}

Attribute convertScatterDimensionNumbers(vhlo::ScatterOpV1 vhloOp,
                                         TypeConverter* typeConverter) {
  SmallVector<int64_t> stablehloUpdateWindowDims, stablehloInsertedWindowDims,
      stablehloScatterDimsToOperandDims;
  int64_t stablehloIndexVectorDim;
  if (failed(convertInts(vhloOp.getUpdateWindowDims(), typeConverter,
                         stablehloUpdateWindowDims)) ||
      failed(convertInts(vhloOp.getInsertedWindowDims(), typeConverter,
                         stablehloInsertedWindowDims)) ||
      failed(convertInts(vhloOp.getScatterDimsToOperandDims(), typeConverter,
                         stablehloScatterDimsToOperandDims)) ||
      failed(convertInt(vhloOp.getIndexVectorDim(), stablehloIndexVectorDim)))
    return {};
  return stablehlo::ScatterDimensionNumbersAttr::get(
      vhloOp.getContext(), stablehloUpdateWindowDims,
      stablehloInsertedWindowDims, stablehloScatterDimsToOperandDims,
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
                std::is_same<VhloOpTy, vhlo::DynamicConvOpV1>::value) {
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
  if constexpr (std::is_same<VhloOpTy, vhlo::DotGeneralOpV1>::value) {
    auto stablehloAttr =
        convertDotDimensionNumbers(vhloOp, pattern.getTypeConverter());
    if (!stablehloAttr) return failure();
    stablehloAttrs.emplace_back(
        StringAttr::get(pattern.getContext(), "dot_dimension_numbers"),
        stablehloAttr);
    eraseAttrs(vhloAttrs, "lhs_batching_dimensions", "rhs_batching_dimensions",
               "lhs_contracting_dimensions", "rhs_contracting_dimensions");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::DynamicGatherOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::GatherOpV1>::value) {
    auto stablehloAttr =
        convertGatherDimensionNumbers(vhloOp, pattern.getTypeConverter());
    if (!stablehloAttr) return failure();
    stablehloAttrs.emplace_back(
        StringAttr::get(pattern.getContext(), "dimension_numbers"),
        stablehloAttr);
    eraseAttrs(vhloAttrs, "offset_dims", "collapsed_slice_dims",
               "start_index_map", "index_vector_dim");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::ScatterOpV1>::value) {
    auto stablehloAttr =
        convertScatterDimensionNumbers(vhloOp, pattern.getTypeConverter());
    if (!stablehloAttr) return failure();
    stablehloAttrs.emplace_back(
        StringAttr::get(pattern.getContext(), "scatter_dimension_numbers"),
        stablehloAttr);
    eraseAttrs(vhloAttrs, "update_window_dims", "inserted_window_dims",
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

template <typename VhloOpTy>
SpecialResult convertSpecial(const OpConversionPattern<VhloOpTy>& pattern,
                             StringRef vhloName, Attribute vhloAttr,
                             SmallVector<NamedAttribute>& stablehloAttrs) {
  StringRef stablehloName = vhloName;
  Attribute stablehloAttr;
  if constexpr (std::is_same<VhloOpTy, vhlo::AllGatherOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::AllReduceOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::AllToAllOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::CollectivePermuteOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::ReduceScatterOpV1>::value) {
    if (vhloName == "channel_id") {
      stablehloName = StringAttr::get(pattern.getContext(), "channel_handle");
      stablehloAttr = convertChannelId(vhloAttr, pattern.getTypeConverter());
      if (!stablehloAttr) return specialFailure();
    }
    if (vhloName == "use_global_device_ids") {
      auto vhloBooleanAttr = vhloAttr.dyn_cast<vhlo::BooleanV1Attr>();
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
  if constexpr (std::is_same<VhloOpTy, vhlo::CallOpV1>::value) {
    if (vhloName == "callee") {
      stablehloAttr = convertFuncCallee(vhloAttr, pattern.getTypeConverter());
      if (!stablehloAttr) return specialFailure();
    }
  }
  if (stablehloAttr) {
    stablehloAttrs.emplace_back(
        StringAttr::get(pattern.getContext(), stablehloName), stablehloAttr);
    return specialSuccess();
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
  auto attr = vhloAttr.template dyn_cast_or_null<vhlo::BooleanV1Attr>();
  return attr && attr.getValue() == value;
}

bool isEmptyArray(Attribute vhloAttr) {
  auto attr = vhloAttr.dyn_cast_or_null<vhlo::ArrayV1Attr>();
  return attr && attr.getValue().empty();
}

bool isEmptyString(Attribute vhloAttr) {
  auto attr = vhloAttr.dyn_cast_or_null<vhlo::StringV1Attr>();
  return attr && attr.getValue().empty();
}

bool isEmptyTensor(Attribute vhloAttr) {
  auto attr = vhloAttr.dyn_cast_or_null<vhlo::TensorV1Attr>();
  return attr && attr.getData().empty();
}

bool isEnum(Attribute vhloAttr, Attribute value) { return vhloAttr == value; }

bool isInteger(Attribute vhloAttr, int64_t value) {
  auto attr = vhloAttr.template dyn_cast_or_null<vhlo::IntegerV1Attr>();
  return attr && attr.getValue().getSExtValue() == value;
}

bool isSplatArray(Attribute vhloAttr, Attribute splatValue) {
  auto attr = vhloAttr.dyn_cast_or_null<vhlo::ArrayV1Attr>();
  return attr && llvm::all_of(attr.getValue(), [&](Attribute attr) {
           return attr == splatValue;
         });
}

template <typename T>
bool isSplatTensor(const ConversionPattern& pattern, Attribute vhloAttr,
                   T splatValue) {
  auto attr = convertGeneric(vhloAttr, pattern.getTypeConverter())
                  .template dyn_cast_or_null<DenseElementsAttr>();
  return attr && attr.isSplat() &&
         attr.template getSplatValue<T>() == splatValue;
}

bool isString(Attribute vhloAttr, StringRef value) {
  auto attr = vhloAttr.dyn_cast_or_null<vhlo::StringV1Attr>();
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
  if constexpr (std::is_same<VhloOpTy, vhlo::AllGatherOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::AllReduceOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::ReduceScatterOpV1>::value) {
    if (isInteger(vhloOp.getChannelIdAttr(), 0))
      eraseAttrs(vhloAttrs, "channel_id");
    if (isBoolean(vhloOp.getUseGlobalDeviceIdsAttr(), false))
      eraseAttrs(vhloAttrs, "use_global_device_ids");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::AllToAllOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::CollectivePermuteOpV1>::value) {
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
  if constexpr (std::is_same<VhloOpTy, vhlo::ConvolutionOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::DynamicConvOpV1>::value) {
    if (isSplatTensor(pattern, vhloOp.getWindowStridesAttr(), 1l))
      eraseAttrs(vhloAttrs, "window_strides");
    if (isSplatTensor(pattern, vhloOp.getPaddingAttr(), 0l))
      eraseAttrs(vhloAttrs, "padding");
    if (isSplatTensor(pattern, vhloOp.getLhsDilationAttr(), 1l))
      eraseAttrs(vhloAttrs, "lhs_dilation");
    if (isSplatTensor(pattern, vhloOp.getRhsDilationAttr(), 1l))
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
    if (isEmptyString(vhloOp.getBackendConfigAttr()))
      eraseAttrs(vhloAttrs, "backend_config");
    if (isEnum(vhloOp.getApiVersionAttr(),
               vhlo::CustomCallApiVersionV1Attr::get(
                   pattern.getContext(),
                   vhlo::CustomCallApiVersionV1::API_VERSION_ORIGINAL)))
      eraseAttrs(vhloAttrs, "api_version");
    if (isEmptyArray(vhloOp.getCalledComputations()))
      eraseAttrs(vhloAttrs, "called_computations");
    if (isEmptyArray(vhloOp.getOperandLayouts()))
      eraseAttrs(vhloAttrs, "operand_layouts");
    if (isEmptyArray(vhloOp.getResultLayouts()))
      eraseAttrs(vhloAttrs, "result_layouts");
    if (isEmptyArray(vhloOp.getOutputOperandAliases()))
      eraseAttrs(vhloAttrs, "output_operand_aliases");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::DotGeneralOpV1>::value ||
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
  if constexpr (std::is_same<VhloOpTy, vhlo::DynamicGatherOpV1>::value ||
                std::is_same<VhloOpTy, vhlo::GatherOpV1>::value) {
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
    if (isSplatTensor(pattern, vhloOp.getWindowStridesAttr(), 1l))
      eraseAttrs(vhloAttrs, "window_strides");
    if (isSplatTensor(pattern, vhloOp.getBaseDilationsAttr(), 1l))
      eraseAttrs(vhloAttrs, "base_dilations");
    if (isSplatTensor(pattern, vhloOp.getWindowDilationsAttr(), 1l))
      eraseAttrs(vhloAttrs, "window_dilations");
    if (isSplatTensor(pattern, vhloOp.getPaddingAttr(), 0l))
      eraseAttrs(vhloAttrs, "padding");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::ScatterOpV1>::value) {
    if (isBoolean(vhloOp.getIndicesAreSortedAttr(), false))
      eraseAttrs(vhloAttrs, "indices_are_sorted");
    if (isBoolean(vhloOp.getUniqueIndicesAttr(), false))
      eraseAttrs(vhloAttrs, "unique_indices");
  }
  if constexpr (std::is_same<VhloOpTy, vhlo::SelectAndScatterOpV1>::value) {
    if (isSplatTensor(pattern, vhloOp.getWindowStridesAttr(), 1l))
      eraseAttrs(vhloAttrs, "window_strides");
    if (isSplatTensor(pattern, vhloOp.getPaddingAttr(), 0l))
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
      stablehloOp = rewriter.replaceOpWithNewOp<stablehlo::CaseOp>(
          vhloOp, stablehloTypes, stablehloOperands, stablehloAttrs,
          vhloOp.getBranches().size());
    } else {
      stablehloOp = rewriter.replaceOpWithNewOp<VhloToStablehloOp<VhloOpTy>>(
          vhloOp, stablehloTypes, stablehloOperands, stablehloAttrs);
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
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<vhlo::VhloDialect>();
    target.addLegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<func::FuncDialect>();

    VhloToStablehloTypeConverter converter;
    RewritePatternSet patterns(&getContext());
    stablehlo::populateVhloToStablehloPatterns(&patterns, &converter,
                                               &getContext());

    // Upgraded VHLO should always be convertible to StableHLO.
    // Arbitrary VHLO might not be convertible if it uses deprecated features
    // which are no longer available in StableHLO.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
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
