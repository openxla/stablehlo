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

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/dialect/VhloTypes.h"
#include "stablehlo/transforms/MapStablehloToVhlo.h"
#include "stablehlo/transforms/Passes.h"

#define DEBUG_TYPE "compat-passes"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOLEGALIZETOVHLOPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// StableHLO --> VHLO types
//===----------------------------------------------------------------------===//

class StablehloToVhloTypeConverter : public vhlo::VhloTypeConverter {
 public:
  StablehloToVhloTypeConverter() : vhlo::VhloTypeConverter() {
    addConversion([](Type type) -> Type {
      if (type.getDialect().getNamespace() ==
          vhlo::VhloDialect::getDialectNamespace()) {
        return type;
      }
      LLVM_DEBUG(llvm::dbgs() << "Invalid type: " << type << '\n');
      return {};
    });
    addConversion([](TokenType token) -> Type {
      return vhlo::TokenV1Type::get(token.getContext());
    });
    addBuiltinToVhloConversions();
  }

  Attribute convertEncoding(Attribute attr) final {
    LLVM_DEBUG(llvm::dbgs() << "Converting encoding.\n" << attr << '\n');
    // Must be VHLO encoding, or convertible to VHLO encoding.
    if (attr.getDialect().getNamespace() ==
        vhlo::VhloDialect::getDialectNamespace())
      return attr;

    if (auto stablehloAttr =
            attr.dyn_cast_or_null<stablehlo::TypeExtensionsAttr>()) {
      return vhlo::TypeExtensionsV1Attr::get(stablehloAttr.getContext(),
                                             stablehloAttr.getBounds());
    }

    // Was not VHLO encoding, or convertible.
    return {};
  }
};

//===----------------------------------------------------------------------===//
// StableHLO --> VHLO attributes: 1) Generic case.
// Applicable in areas where there is 1:1 mapping from StableHLO to VHLO.
// This is the predominant case.
//===----------------------------------------------------------------------===//

#define RETURN_CONVERTED_ENUM_ATTR(Name, Version)                    \
  auto stablehloValue = stablehlo::stringify##Name(attr.getValue()); \
  auto vhloValue = vhlo::symbolize##Name##Version(stablehloValue);   \
  if (!vhloValue.has_value()) return {};                             \
  return vhlo::Name##Version##Attr::get(attr.getContext(), vhloValue.value())

Attribute convertGeneric(Attribute stablehloAttr,
                         TypeConverter* typeConverter) {
  // Handle StableHLO attributes.
  // The logic that handles attributes from other dialects (e.g. builtin
  // attributes) lives below.
  if (auto attr =
          stablehloAttr.dyn_cast<stablehlo::ComparisonDirectionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonDirection, V1);
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::ComparisonTypeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonType, V1);
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::FftTypeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(FftType, V1);
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::OutputOperandAliasAttr>()) {
    return vhlo::OutputOperandAliasV1Attr::get(
        attr.getContext(), attr.getOutputTupleIndices(), attr.getOperandIndex(),
        attr.getOperandTupleIndices());
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::PrecisionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Precision, V1);
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::RngAlgorithmAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngAlgorithm, V1);
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::RngDistributionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngDistribution, V1);
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::TransposeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Transpose, V1);
  }
  if (stablehloAttr.getDialect().getNamespace() ==
      stablehlo::StablehloDialect::getDialectNamespace()) {
    // All StableHLO attributes must have counterparts in VHLO.
    return {};
  }

  // Handle supported non-StableHLO attributes.
  // Each of these attributes has a counterpart in the VHLO dialect -
  // VHLO programs never include attributes from other dialects.
  if (auto stablehloAttrs = stablehloAttr.dyn_cast<ArrayAttr>()) {
    SmallVector<Attribute> vhloAttrs;
    for (auto stablehloAttr : stablehloAttrs) {
      auto vhloAttr = convertGeneric(stablehloAttr, typeConverter);
      if (!vhloAttr) return {};
      vhloAttrs.push_back(vhloAttr);
    }
    return vhlo::ArrayV1Attr::get(stablehloAttrs.getContext(), vhloAttrs);
  }
  if (auto attr = stablehloAttr.dyn_cast<DenseIntOrFPElementsAttr>()) {
    auto vhloType = typeConverter->convertType(attr.getType());
    LLVM_DEBUG(llvm::dbgs() << "Converted " << vhloType << '\n');
    if (!vhloType) return {};
    return vhlo::TensorV1Attr::get(attr.getContext(), vhloType,
                                   attr.getRawData());
  }
  if (auto attr = stablehloAttr.dyn_cast<DictionaryAttr>()) {
    SmallVector<std::pair<Attribute, Attribute>> vhloAttrs;
    for (auto namedAttr : attr.getValue()) {
      auto vhloName = convertGeneric(namedAttr.getName(), typeConverter);
      auto vhloValue = convertGeneric(namedAttr.getValue(), typeConverter);
      if (!vhloName || !vhloValue) return {};
      vhloAttrs.push_back({vhloName, vhloValue});
    }
    return vhlo::DictionaryV1Attr::get(attr.getContext(), vhloAttrs);
  }
  if (auto attr = stablehloAttr.dyn_cast<FloatAttr>()) {
    auto vhloFloatType = typeConverter->convertType(attr.getType());
    if (!vhloFloatType) return {};
    return vhlo::FloatV1Attr::get(attr.getContext(), vhloFloatType,
                                  attr.getValue());
  }
  if (auto integerAttr = stablehloAttr.dyn_cast<IntegerAttr>()) {
    if (auto boolAttr = stablehloAttr.dyn_cast<BoolAttr>()) {
      return vhlo::BooleanV1Attr::get(boolAttr.getContext(),
                                      boolAttr.getValue());
    } else {
      auto vhloIntegerType = typeConverter->convertType(integerAttr.getType());
      if (!vhloIntegerType) return {};
      return vhlo::IntegerV1Attr::get(integerAttr.getContext(), vhloIntegerType,
                                      integerAttr.getValue());
    }
  }
  if (auto attr = stablehloAttr.dyn_cast<StringAttr>()) {
    if (!attr.getType().isa<NoneType>()) {
      // Don't support custom string types
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to convert string with type: " << attr << '\n');
      return {};
    }
    return vhlo::StringV1Attr::get(attr.getContext(), attr.getValue());
  }
  if (auto attr = stablehloAttr.dyn_cast<TypeAttr>()) {
    auto vhloType = typeConverter->convertType(attr.getValue());
    if (!vhloType) return {};
    return vhlo::TypeV1Attr::get(attr.getContext(), vhloType);
  }

  LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << stablehloAttr << '\n');
  return {};  // Failed to convert attribute.
}

//===----------------------------------------------------------------------===//
// StableHLO --> VHLO attributes: 2) Special cases.
// Applicable in areas where there is no 1:1 mapping from StableHLO to VHLO.
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

Attribute convertInt(const ConversionPattern& pattern, int64_t stablehloDim) {
  auto stablehloType = IntegerType::get(pattern.getContext(), 64);
  auto stablehloAttr = IntegerAttr::get(stablehloType, stablehloDim);
  return convertGeneric(stablehloAttr, pattern.getTypeConverter());
}

Attribute convertInts(const ConversionPattern& pattern,
                      ArrayRef<int64_t> stablehloDims) {
  auto stablehloType = RankedTensorType::get(
      stablehloDims.size(), IntegerType::get(pattern.getContext(), 64));
  auto stablehloAttr = DenseIntElementsAttr::get(stablehloType, stablehloDims);
  return convertGeneric(stablehloAttr, pattern.getTypeConverter());
}

Attribute convertSymbol(const ConversionPattern& pattern,
                        Attribute stablehloAttr) {
  auto stablehloSymbolAttr = stablehloAttr.dyn_cast<FlatSymbolRefAttr>();
  if (!stablehloSymbolAttr) return {};
  return convertGeneric(stablehloSymbolAttr.getAttr(),
                        pattern.getTypeConverter());
}

SpecialResult convertChannelHandle(const ConversionPattern& pattern,
                                   Attribute stablehloAttr,
                                   SmallVector<NamedAttribute>& vhloAttrs) {
  auto attr = stablehloAttr.dyn_cast<stablehlo::ChannelHandleAttr>();
  if (!attr) return specialFailure();

  auto vhloChannelId = convertInt(pattern, attr.getHandle());
  if (!vhloChannelId) return specialFailure();
  vhloAttrs.emplace_back(StringAttr::get(pattern.getContext(), "channel_id"),
                         vhloChannelId);

  auto vhloChannelType = convertInt(pattern, attr.getType());
  if (!vhloChannelType) return specialFailure();
  vhloAttrs.emplace_back(StringAttr::get(pattern.getContext(), "channel_type"),
                         vhloChannelType);
  return specialSuccess();
}

SpecialResult convertChannelId(const ConversionPattern& pattern,
                               Attribute stablehloAttr,
                               SmallVector<NamedAttribute>& vhloAttrs) {
  auto attr = stablehloAttr.dyn_cast<stablehlo::ChannelHandleAttr>();
  if (!attr) return specialFailure();

  auto vhloChannelId = convertInt(pattern, attr.getHandle());
  if (!vhloChannelId) return specialFailure();
  vhloAttrs.emplace_back(StringAttr::get(pattern.getContext(), "channel_id"),
                         vhloChannelId);
  return specialSuccess();
}

SpecialResult convertConvDimensionNumbers(
    const ConversionPattern& pattern, Attribute stablehloAttr,
    SmallVector<NamedAttribute>& vhloAttrs) {
  auto attr = stablehloAttr.dyn_cast<stablehlo::ConvDimensionNumbersAttr>();
  if (!attr) return specialFailure();

  auto vhloInputBatchDimension =
      convertInt(pattern, attr.getInputBatchDimension());
  if (!vhloInputBatchDimension) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "input_batch_dimension"),
      vhloInputBatchDimension);

  auto vhloInputFeatureDimension =
      convertInt(pattern, attr.getInputFeatureDimension());
  if (!vhloInputFeatureDimension) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "input_feature_dimension"),
      vhloInputFeatureDimension);

  auto vhloInputSpatialDimensions =
      convertInts(pattern, attr.getInputSpatialDimensions());
  if (!vhloInputSpatialDimensions) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "input_spatial_dimensions"),
      vhloInputSpatialDimensions);

  auto vhloKernelInputFeatureDimension =
      convertInt(pattern, attr.getKernelInputFeatureDimension());
  if (!vhloKernelInputFeatureDimension) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "kernel_input_feature_dimension"),
      vhloKernelInputFeatureDimension);

  auto vhloKernelOutputFeatureDimension =
      convertInt(pattern, attr.getKernelOutputFeatureDimension());
  if (!vhloKernelOutputFeatureDimension) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "kernel_output_feature_dimension"),
      vhloKernelOutputFeatureDimension);

  auto vhloKernelSpatialDimensions =
      convertInts(pattern, attr.getKernelSpatialDimensions());
  if (!vhloKernelSpatialDimensions) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "kernel_spatial_dimensions"),
      vhloKernelSpatialDimensions);

  auto vhloOutputBatchDimension =
      convertInt(pattern, attr.getOutputBatchDimension());
  if (!vhloOutputBatchDimension) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "output_batch_dimension"),
      vhloOutputBatchDimension);

  auto vhloOutputFeatureDimension =
      convertInt(pattern, attr.getOutputFeatureDimension());
  if (!vhloOutputFeatureDimension) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "output_feature_dimension"),
      vhloOutputFeatureDimension);

  auto vhloOutputSpatialDimensions =
      convertInts(pattern, attr.getOutputSpatialDimensions());
  if (!vhloOutputSpatialDimensions) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "output_spatial_dimensions"),
      vhloOutputSpatialDimensions);
  return specialSuccess();
}

SpecialResult convertCustomCallApiVersion(
    const ConversionPattern& pattern, Attribute stablehloAttr,
    SmallVector<NamedAttribute>& vhloAttrs) {
  auto attr = stablehloAttr.dyn_cast<CustomCallApiVersionAttr>();
  if (!attr) return specialFailure();

  auto convert = [&]() -> Attribute {
    RETURN_CONVERTED_ENUM_ATTR(CustomCallApiVersion, V1);
  };
  auto vhloAttr = convert();
  if (!vhloAttr) return specialFailure();
  vhloAttrs.emplace_back(StringAttr::get(pattern.getContext(), "api_version"),
                         vhloAttr);
  return specialSuccess();
}

SpecialResult convertCustomCallCalledComputations(
    const ConversionPattern& pattern, Attribute stablehloAttr,
    SmallVector<NamedAttribute>& vhloAttrs) {
  auto attr = stablehloAttr.dyn_cast<ArrayAttr>();
  if (!attr) return specialFailure();

  SmallVector<Attribute> vhloElementAttrs;
  for (auto stablehloElementAttr : attr) {
    auto vhloElementAttr = convertSymbol(pattern, stablehloElementAttr);
    if (!vhloElementAttr) return specialFailure();
    vhloElementAttrs.push_back(vhloElementAttr);
  }

  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "called_computations"),
      vhlo::ArrayV1Attr::get(pattern.getContext(), vhloElementAttrs));
  return specialSuccess();
}

SpecialResult convertDotDimensionNumbers(
    const ConversionPattern& pattern, Attribute stablehloAttr,
    SmallVector<NamedAttribute>& vhloAttrs) {
  auto attr = stablehloAttr.dyn_cast<stablehlo::DotDimensionNumbersAttr>();
  if (!attr) return specialFailure();

  auto vhloLhsBatchingDimensions =
      convertInts(pattern, attr.getLhsBatchingDimensions());
  if (!vhloLhsBatchingDimensions) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "lhs_batching_dimensions"),
      vhloLhsBatchingDimensions);

  auto vhloRhsBatchingDimensions =
      convertInts(pattern, attr.getRhsBatchingDimensions());
  if (!vhloRhsBatchingDimensions) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "rhs_batching_dimensions"),
      vhloRhsBatchingDimensions);

  auto vhloLhsContractingDimensions =
      convertInts(pattern, attr.getLhsContractingDimensions());
  if (!vhloLhsContractingDimensions) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "lhs_contracting_dimensions"),
      vhloLhsContractingDimensions);

  auto vhloRhsContractingDimensions =
      convertInts(pattern, attr.getRhsContractingDimensions());
  if (!vhloRhsContractingDimensions) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "rhs_contracting_dimensions"),
      vhloRhsContractingDimensions);
  return specialSuccess();
}

SpecialResult convertFuncCallee(const ConversionPattern& pattern,
                                Attribute stablehloAttr,
                                SmallVector<NamedAttribute>& vhloAttrs) {
  auto vhloAttr = convertSymbol(pattern, stablehloAttr);
  if (!vhloAttr) return specialFailure();
  vhloAttrs.emplace_back(StringAttr::get(pattern.getContext(), "callee"),
                         vhloAttr);
  return specialSuccess();
}

SpecialResult convertGatherDimensionNumbers(
    const ConversionPattern& pattern, Attribute stablehloAttr,
    SmallVector<NamedAttribute>& vhloAttrs) {
  auto attr = stablehloAttr.dyn_cast<stablehlo::GatherDimensionNumbersAttr>();
  if (!attr) return specialFailure();

  auto vhloOffsetDims = convertInts(pattern, attr.getOffsetDims());
  if (!vhloOffsetDims) return specialFailure();
  vhloAttrs.emplace_back(StringAttr::get(pattern.getContext(), "offset_dims"),
                         vhloOffsetDims);

  auto vhloCollapsedSliceDims =
      convertInts(pattern, attr.getCollapsedSliceDims());
  if (!vhloCollapsedSliceDims) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "collapsed_slice_dims"),
      vhloCollapsedSliceDims);

  auto vhloStartIndexMap = convertInts(pattern, attr.getStartIndexMap());
  if (!vhloStartIndexMap) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "start_index_map"),
      vhloStartIndexMap);

  auto vhloIndexVectorDim = convertInt(pattern, attr.getIndexVectorDim());
  if (!vhloIndexVectorDim) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "index_vector_dim"),
      vhloIndexVectorDim);
  return specialSuccess();
}

SpecialResult convertScatterDimensionNumbers(
    const ConversionPattern& pattern, Attribute stablehloAttr,
    SmallVector<NamedAttribute>& vhloAttrs) {
  auto attr = stablehloAttr.dyn_cast<stablehlo::ScatterDimensionNumbersAttr>();
  if (!attr) return specialFailure();

  auto vhloUpdateWindowDims = convertInts(pattern, attr.getUpdateWindowDims());
  if (!vhloUpdateWindowDims) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "update_window_dims"),
      vhloUpdateWindowDims);

  auto vhloInsertedWindowDims =
      convertInts(pattern, attr.getInsertedWindowDims());
  if (!vhloInsertedWindowDims) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "inserted_window_dims"),
      vhloInsertedWindowDims);

  auto vhloScatterDimsToOperandDims =
      convertInts(pattern, attr.getScatterDimsToOperandDims());
  if (!vhloScatterDimsToOperandDims) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "scatter_dims_to_operand_dims"),
      vhloScatterDimsToOperandDims);

  auto vhloIndexVectorDim = convertInt(pattern, attr.getIndexVectorDim());
  if (!vhloIndexVectorDim) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "index_vector_dim"),
      vhloIndexVectorDim);
  return specialSuccess();
}

SpecialResult convertUseGlobalDeviceIds(
    const ConversionPattern& pattern, Attribute stablehloAttr,
    SmallVector<NamedAttribute>& vhloAttrs) {
  if (!stablehloAttr.isa<UnitAttr>()) return specialFailure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "use_global_device_ids"),
      vhlo::BooleanV1Attr::get(pattern.getContext(), true));
  return specialSuccess();
}

template <typename StablehloOpTy>
SpecialResult convertSpecial(const OpConversionPattern<StablehloOpTy>& pattern,
                             StringRef stablehloName, Attribute stablehloAttr,
                             SmallVector<NamedAttribute>& vhloAttrs) {
  if constexpr (std::is_same<StablehloOpTy, stablehlo::AllGatherOp>::value ||
                std::is_same<StablehloOpTy, stablehlo::AllReduceOp>::value ||
                std::is_same<StablehloOpTy, stablehlo::AllToAllOp>::value ||
                std::is_same<StablehloOpTy,
                             stablehlo::CollectivePermuteOp>::value ||
                std::is_same<StablehloOpTy,
                             stablehlo::ReduceScatterOp>::value) {
    if (stablehloName == "channel_handle")
      return convertChannelId(pattern, stablehloAttr, vhloAttrs);
    if (stablehloName == "use_global_device_ids")
      return convertUseGlobalDeviceIds(pattern, stablehloAttr, vhloAttrs);
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::ConvolutionOp>::value ||
                std::is_same<StablehloOpTy, stablehlo::DynamicConvOp>::value) {
    if (stablehloName == "dimension_numbers")
      return convertConvDimensionNumbers(pattern, stablehloAttr, vhloAttrs);
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::CustomCallOp>::value) {
    if (stablehloName == "api_version")
      return convertCustomCallApiVersion(pattern, stablehloAttr, vhloAttrs);
    if (stablehloName == "called_computations")
      return convertCustomCallCalledComputations(pattern, stablehloAttr,
                                                 vhloAttrs);
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::DotGeneralOp>::value) {
    if (stablehloName == "dot_dimension_numbers")
      return convertDotDimensionNumbers(pattern, stablehloAttr, vhloAttrs);
  }
  if constexpr (std::is_same<StablehloOpTy,
                             stablehlo::DynamicGatherOp>::value ||
                std::is_same<StablehloOpTy, stablehlo::GatherOp>::value) {
    if (stablehloName == "dimension_numbers")
      return convertGatherDimensionNumbers(pattern, stablehloAttr, vhloAttrs);
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::RecvOp>::value ||
                std::is_same<StablehloOpTy, stablehlo::SendOp>::value) {
    if (stablehloName == "channel_handle")
      return convertChannelHandle(pattern, stablehloAttr, vhloAttrs);
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::ScatterOp>::value) {
    if (stablehloName == "scatter_dimension_numbers")
      return convertScatterDimensionNumbers(pattern, stablehloAttr, vhloAttrs);
  }
  if constexpr (std::is_same<StablehloOpTy, func::CallOp>::value) {
    if (stablehloName == "callee")
      return convertFuncCallee(pattern, stablehloAttr, vhloAttrs);
  }
  return notSpecial();
}

#undef RETURN_CONVERTED_ENUM_ATTR

//===----------------------------------------------------------------------===//
// StableHLO --> VHLO attributes: 3) Default attributes.
// Unlike StableHLO, VHLO doesn't have default attributes, so they need to be
// added here.
//===----------------------------------------------------------------------===//

template <typename StablehloOpTy>
LogicalResult addDefaults(const OpConversionPattern<StablehloOpTy>& pattern,
                          StablehloOpTy stablehloOp,
                          SmallVector<NamedAttribute>& vhloAttrs) {
  Builder builder(pattern.getContext());
  auto addDefaultAttr = [&](StringRef vhloName, Attribute stablehloAttr) {
    vhloAttrs.emplace_back(
        StringAttr::get(pattern.getContext(), vhloName),
        convertGeneric(stablehloAttr, pattern.getTypeConverter()));
  };
  if constexpr (std::is_same<StablehloOpTy, func::FuncOp>::value) {
    if (!stablehloOp.getSymVisibilityAttr())
      addDefaultAttr("sym_visibility",
                     StringAttr::get(pattern.getContext(), ""));
    if (!stablehloOp.getArgAttrsAttr())
      addDefaultAttr("arg_attrs", ArrayAttr::get(pattern.getContext(), {}));
    if (!stablehloOp.getResAttrsAttr())
      addDefaultAttr("res_attrs", ArrayAttr::get(pattern.getContext(), {}));
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::AllGatherOp>::value ||
                std::is_same<StablehloOpTy, stablehlo::AllReduceOp>::value ||
                std::is_same<StablehloOpTy,
                             stablehlo::ReduceScatterOp>::value) {
    if (!stablehloOp.getChannelHandleAttr())
      addDefaultAttr("channel_id", builder.getI64IntegerAttr(0));
    if (!stablehloOp.getUseGlobalDeviceIdsAttr())
      addDefaultAttr("use_global_device_ids", builder.getBoolAttr(false));
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::AllToAllOp>::value ||
                std::is_same<StablehloOpTy,
                             stablehlo::CollectivePermuteOp>::value) {
    if (!stablehloOp.getChannelHandleAttr())
      addDefaultAttr("channel_id", builder.getI64IntegerAttr(0));
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::CholeskyOp>::value) {
    if (!stablehloOp.getLowerAttr())
      addDefaultAttr("lower", builder.getBoolAttr(false));
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::CompareOp>::value) {
    if (!stablehloOp.getCompareTypeAttr())
      addDefaultAttr("compare_type", stablehlo::ComparisonTypeAttr::get(
                                         pattern.getContext(),
                                         stablehlo::ComparisonType::NOTYPE));
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::ConvolutionOp>::value ||
                std::is_same<StablehloOpTy, stablehlo::DynamicConvOp>::value) {
    auto numSpatialDimensions = static_cast<int64_t>(
        stablehloOp.getDimensionNumbers().getInputSpatialDimensions().size());
    if (!stablehloOp.getWindowStridesAttr())
      addDefaultAttr("window_strides",
                     builder.getI64TensorAttr(
                         SmallVector<int64_t>(numSpatialDimensions, 1l)));
    if (!stablehloOp.getPaddingAttr())
      addDefaultAttr("padding",
                     DenseIntElementsAttr::get(
                         RankedTensorType::get({numSpatialDimensions, 2},
                                               builder.getI64Type()),
                         SmallVector<int64_t>(numSpatialDimensions * 2, 0l)));
    if (!stablehloOp.getLhsDilationAttr())
      addDefaultAttr("lhs_dilation",
                     builder.getI64TensorAttr(
                         SmallVector<int64_t>(numSpatialDimensions, 1l)));
    if (!stablehloOp.getRhsDilationAttr())
      addDefaultAttr("rhs_dilation",
                     builder.getI64TensorAttr(
                         SmallVector<int64_t>(numSpatialDimensions, 1l)));
    if (!stablehloOp.getWindowReversalAttr())
      addDefaultAttr("window_reversal",
                     DenseIntElementsAttr::get(
                         RankedTensorType::get({numSpatialDimensions},
                                               builder.getI1Type()),
                         SmallVector<bool>(numSpatialDimensions, false)));
    if (!stablehloOp.getPrecisionConfigAttr())
      addDefaultAttr(
          "precision_config",
          builder.getArrayAttr(SmallVector<Attribute>(
              2, stablehlo::PrecisionAttr::get(
                     pattern.getContext(), stablehlo::Precision::DEFAULT))));
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::CustomCallOp>::value) {
    if (!stablehloOp.getHasSideEffectAttr())
      addDefaultAttr("has_side_effect", builder.getBoolAttr(false));
    if (!stablehloOp.getBackendConfigAttr())
      addDefaultAttr("backend_config", builder.getStringAttr(""));
    if (!stablehloOp.getApiVersionAttr())
      vhloAttrs.emplace_back(
          StringAttr::get(pattern.getContext(), "api_version"),
          vhlo::CustomCallApiVersionV1Attr::get(
              pattern.getContext(),
              vhlo::CustomCallApiVersionV1::API_VERSION_ORIGINAL));
    if (!stablehloOp.getCalledComputationsAttr())
      addDefaultAttr("called_computations",
                     ArrayAttr::get(pattern.getContext(), {}));
    if (!stablehloOp.getOperandLayoutsAttr())
      addDefaultAttr("operand_layouts",
                     ArrayAttr::get(pattern.getContext(), {}));
    if (!stablehloOp.getResultLayoutsAttr())
      addDefaultAttr("result_layouts",
                     ArrayAttr::get(pattern.getContext(), {}));
    if (!stablehloOp.getOutputOperandAliasesAttr())
      addDefaultAttr("output_operand_aliases",
                     ArrayAttr::get(pattern.getContext(), {}));
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::DotGeneralOp>::value ||
                std::is_same<StablehloOpTy, stablehlo::DotOp>::value) {
    if (!stablehloOp.getPrecisionConfigAttr())
      addDefaultAttr(
          "precision_config",
          builder.getArrayAttr(SmallVector<Attribute>(
              2, stablehlo::PrecisionAttr::get(
                     pattern.getContext(), stablehlo::Precision::DEFAULT))));
  }
  if constexpr (std::is_same<StablehloOpTy,
                             stablehlo::DynamicBroadcastInDimOp>::value) {
    if (!stablehloOp.getKnownExpandingDimensionsAttr())
      addDefaultAttr("known_expanding_dimensions",
                     builder.getI64TensorAttr({}));
    if (!stablehloOp.getKnownNonexpandingDimensionsAttr())
      addDefaultAttr("known_nonexpanding_dimensions",
                     builder.getI64TensorAttr({}));
  }
  if constexpr (std::is_same<StablehloOpTy,
                             stablehlo::DynamicGatherOp>::value ||
                std::is_same<StablehloOpTy, stablehlo::GatherOp>::value) {
    if (!stablehloOp.getIndicesAreSortedAttr())
      addDefaultAttr("indices_are_sorted", builder.getBoolAttr(false));
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::InfeedOp>::value) {
    if (!stablehloOp.getInfeedConfigAttr())
      addDefaultAttr("infeed_config", builder.getStringAttr(""));
    if (!stablehloOp.getLayoutAttr())
      addDefaultAttr("layout", ArrayAttr::get(pattern.getContext(), {}));
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::OutfeedOp>::value) {
    if (!stablehloOp.getOutfeedConfigAttr())
      addDefaultAttr("outfeed_config", builder.getStringAttr(""));
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::RecvOp>::value ||
                std::is_same<StablehloOpTy, stablehlo::SendOp>::value) {
    if (!stablehloOp.getIsHostTransferAttr())
      addDefaultAttr("is_host_transfer", builder.getBoolAttr(false));
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::ReduceWindowOp>::value) {
    auto numWindowDimensions =
        static_cast<int64_t>(stablehloOp.getWindowDimensions().size());
    if (!stablehloOp.getWindowStridesAttr())
      addDefaultAttr("window_strides",
                     builder.getI64TensorAttr(
                         SmallVector<int64_t>(numWindowDimensions, 1l)));
    if (!stablehloOp.getBaseDilationsAttr())
      addDefaultAttr("base_dilations",
                     builder.getI64TensorAttr(
                         SmallVector<int64_t>(numWindowDimensions, 1l)));
    if (!stablehloOp.getWindowDilationsAttr())
      addDefaultAttr("window_dilations",
                     builder.getI64TensorAttr(
                         SmallVector<int64_t>(numWindowDimensions, 1l)));
    if (!stablehloOp.getPaddingAttr())
      addDefaultAttr("padding",
                     DenseIntElementsAttr::get(
                         RankedTensorType::get({numWindowDimensions, 2},
                                               builder.getI64Type()),
                         SmallVector<int64_t>(numWindowDimensions * 2, 0l)));
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::ScatterOp>::value) {
    if (!stablehloOp.getIndicesAreSortedAttr())
      addDefaultAttr("indices_are_sorted", builder.getBoolAttr(false));
    if (!stablehloOp.getUniqueIndicesAttr())
      addDefaultAttr("unique_indices", builder.getBoolAttr(false));
  }
  if constexpr (std::is_same<StablehloOpTy,
                             stablehlo::SelectAndScatterOp>::value) {
    // TODO(#1055): Change window_dimensions in SelectAndScatterOp
    // from optional to non-optional.
    if (!stablehloOp.getWindowDimensionsAttr()) return failure();
    auto numWindowDimensions =
        static_cast<int64_t>(stablehloOp.getWindowDimensions()->size());
    if (!stablehloOp.getWindowStridesAttr())
      addDefaultAttr("window_strides",
                     builder.getI64TensorAttr(
                         SmallVector<int64_t>(numWindowDimensions, 1l)));
    if (!stablehloOp.getPaddingAttr())
      addDefaultAttr("padding",
                     DenseIntElementsAttr::get(
                         RankedTensorType::get({numWindowDimensions, 2},
                                               builder.getI64Type()),
                         SmallVector<int64_t>(numWindowDimensions * 2, 0l)));
  }
  if constexpr (std::is_same<StablehloOpTy, stablehlo::SortOp>::value) {
    if (!stablehloOp.getDimensionAttr())
      addDefaultAttr("dimension", builder.getI64IntegerAttr(-1));
    if (!stablehloOp.getIsStableAttr())
      addDefaultAttr("is_stable", builder.getBoolAttr(false));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StableHLO --> VHLO operations
//===----------------------------------------------------------------------===//

template <typename StablehloOpTy>
class StablehloToVhloOpConverter : public OpConversionPattern<StablehloOpTy> {
 public:
  using OpConversionPattern<StablehloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      StablehloOpTy stablehloOp, typename StablehloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    SmallVector<Type> vhloTypes;
    if (failed(this->getTypeConverter()->convertTypes(
            stablehloOp->getResultTypes(), vhloTypes))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed StableHLO -> VHLO type conversion\n");
      return failure();
    }

    // These operands have already been converted to VHLO by
    // the dialect conversion infrastructure.
    ValueRange vhloOperands = adaptor.getOperands();

    // Convert StableHLO attributes to VHLO equivalents.
    // There are two paths:
    //   1) Generic path where default logic applies, and there is a 1:1
    //      mapping from StableHLO to VHLO.
    //   2) Special cases (currently, about a dozen) where there is not 1:1
    //      mapping from StableHLO to VHLO.
    SmallVector<NamedAttribute> vhloAttrs;
    if (failed(addDefaults(*this, stablehloOp, vhloAttrs))) return failure();
    for (NamedAttribute stablehloAttr : stablehloOp->getAttrs()) {
      auto result = convertSpecial(*this, stablehloAttr.getName(),
                                   stablehloAttr.getValue(), vhloAttrs);
      switch (result) {
        case SpecialResult::SPECIAL_SUCCESS:
          break;
        case SpecialResult::SPECIAL_FAILURE:
          return failure();
        case SpecialResult::NOT_SPECIAL:
          auto vhloAttr = convertGeneric(stablehloAttr.getValue(),
                                         this->getTypeConverter());
          if (!vhloAttr) return failure();
          vhloAttrs.push_back({stablehloAttr.getName(), vhloAttr});
          break;
      }
    }

    // Convert the StableHLO operation to a VHLO equivalent.
    // This can almost be done in a generic fashion, except for
    // stablehlo.case that uses a variadic number of regions which means an
    // additional argument for the generic builder.
    StablehloToVhloOp<StablehloOpTy> vhloOp;
    if constexpr (std::is_same<StablehloOpTy, stablehlo::CaseOp>::value) {
      vhloOp = rewriter.replaceOpWithNewOp<vhlo::CaseOpV1>(
          stablehloOp, vhloTypes, vhloOperands, vhloAttrs,
          stablehloOp.getBranches().size());
    } else {
      vhloOp = rewriter.replaceOpWithNewOp<StablehloToVhloOp<StablehloOpTy>>(
          stablehloOp, vhloTypes, vhloOperands, vhloAttrs);
    }

    for (auto [stablehloRegion, vhloRegion] :
         llvm::zip(stablehloOp->getRegions(), vhloOp->getRegions())) {
      rewriter.inlineRegionBefore(stablehloRegion, vhloRegion,
                                  vhloRegion.end());
      if (failed(rewriter.convertRegionTypes(&vhloRegion,
                                             *this->getTypeConverter(),
                                             /*entryConversion=*/nullptr)))
        return failure();
    }
    return success();
  }
};

template <typename... StablehloOpTypes>
void populateStablehloToVhloPatterns(RewritePatternSet* patterns,
                                     TypeConverter* converter,
                                     MLIRContext* context) {
  patterns->add<StablehloToVhloOpConverter<StablehloOpTypes>...>(*converter,
                                                                 context);
}

}  // namespace

struct StablehloLegalizeToVhloPass
    : public impl::StablehloLegalizeToVhloPassBase<
          StablehloLegalizeToVhloPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<stablehlo::StablehloDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    target.addLegalDialect<vhlo::VhloDialect>();

    StablehloToVhloTypeConverter converter;
    RewritePatternSet patterns(&getContext());
    stablehlo::populateStablehloToVhloPatterns(&patterns, &converter,
                                               &getContext());

    // StableHLO should always be convertible to VHLO.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed partial conversion\n");
      return signalPassFailure();
    }
  }
};

void populateStablehloToVhloPatterns(RewritePatternSet* patterns,
                                     TypeConverter* converter,
                                     MLIRContext* context) {
  populateStablehloToVhloPatterns<
#define GET_OP_LIST
#include "stablehlo/dialect/StablehloOps.cpp.inc"
      , func::CallOp, func::FuncOp, func::ReturnOp>(patterns, converter,
                                                    context);
}
}  // namespace stablehlo
}  // namespace mlir
