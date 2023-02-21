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

///////////////////////////////////////////////
/// StableHLO --> VHLO Types and Attributes ///
///////////////////////////////////////////////

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

#define RETURN_CONVERTED_ENUM_ATTR(Name, Version)                    \
  auto stablehloValue = stablehlo::stringify##Name(attr.getValue()); \
  auto vhloValue = vhlo::symbolize##Name##Version(stablehloValue);   \
  if (!vhloValue.has_value()) return {};                             \
  return vhlo::Name##Version##Attr::get(attr.getContext(), vhloValue.value())

Attribute convertAttrToVhlo(Attribute stablehloAttr,
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
      auto vhloAttr = convertAttrToVhlo(stablehloAttr, typeConverter);
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
      auto vhloName = convertAttrToVhlo(namedAttr.getName(), typeConverter);
      auto vhloValue = convertAttrToVhlo(namedAttr.getValue(), typeConverter);
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

Attribute convertInt(const ConversionPattern& pattern, int64_t stablehloDim) {
  auto stablehloType = IntegerType::get(pattern.getContext(), 64);
  auto stablehloAttr = IntegerAttr::get(stablehloType, stablehloDim);
  return convertAttrToVhlo(stablehloAttr, pattern.getTypeConverter());
}

Attribute convertInts(const ConversionPattern& pattern,
                      ArrayRef<int64_t> stablehloDims) {
  auto stablehloType = RankedTensorType::get(
      stablehloDims.size(), IntegerType::get(pattern.getContext(), 64));
  auto stablehloAttr = DenseIntElementsAttr::get(stablehloType, stablehloDims);
  return convertAttrToVhlo(stablehloAttr, pattern.getTypeConverter());
}

Attribute convertSymbol(const ConversionPattern& pattern,
                        Attribute stablehloAttr) {
  auto stablehloSymbolAttr = stablehloAttr.dyn_cast<FlatSymbolRefAttr>();
  if (!stablehloSymbolAttr) return {};
  return convertAttrToVhlo(stablehloSymbolAttr.getAttr(),
                           pattern.getTypeConverter());
}

Attribute convertCustomCallApiVersion(Attribute stablehloAttr) {
  if (auto attr = stablehloAttr.dyn_cast<CustomCallApiVersionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(CustomCallApiVersion, V1);
  }
  return {};
}

LogicalResult convertChannelHandle(const ConversionPattern& pattern,
                                   Attribute stablehloAttr,
                                   SmallVector<NamedAttribute>& vhloAttrs) {
  auto attr = stablehloAttr.dyn_cast<stablehlo::ChannelHandleAttr>();
  if (!attr) return failure();

  auto vhloChannelId = convertInt(pattern, attr.getHandle());
  if (!vhloChannelId) return failure();
  vhloAttrs.emplace_back(StringAttr::get(pattern.getContext(), "channel_id"),
                         vhloChannelId);

  auto vhloChannelType = convertInt(pattern, attr.getType());
  if (!vhloChannelType) return failure();
  vhloAttrs.emplace_back(StringAttr::get(pattern.getContext(), "channel_type"),
                         vhloChannelType);
  return success();
}

LogicalResult convertChannelId(const ConversionPattern& pattern,
                               Attribute stablehloAttr,
                               SmallVector<NamedAttribute>& vhloAttrs) {
  auto attr = stablehloAttr.dyn_cast<stablehlo::ChannelHandleAttr>();
  if (!attr) return failure();

  auto vhloChannelId = convertInt(pattern, attr.getHandle());
  if (!vhloChannelId) return failure();
  vhloAttrs.emplace_back(StringAttr::get(pattern.getContext(), "channel_id"),
                         vhloChannelId);
  return success();
}

LogicalResult convertConvDimensionNumbers(
    const ConversionPattern& pattern, Attribute stablehloAttr,
    SmallVector<NamedAttribute>& vhloAttrs) {
  auto attr = stablehloAttr.dyn_cast<stablehlo::ConvDimensionNumbersAttr>();
  if (!attr) return failure();

  auto vhloInputBatchDimension =
      convertInt(pattern, attr.getInputBatchDimension());
  if (!vhloInputBatchDimension) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "input_batch_dimension"),
      vhloInputBatchDimension);

  auto vhloInputFeatureDimension =
      convertInt(pattern, attr.getInputFeatureDimension());
  if (!vhloInputFeatureDimension) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "input_feature_dimension"),
      vhloInputFeatureDimension);

  auto vhloInputSpatialDimensions =
      convertInts(pattern, attr.getInputSpatialDimensions());
  if (!vhloInputSpatialDimensions) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "input_spatial_dimensions"),
      vhloInputSpatialDimensions);

  auto vhloKernelInputFeatureDimension =
      convertInt(pattern, attr.getKernelInputFeatureDimension());
  if (!vhloKernelInputFeatureDimension) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "kernel_input_feature_dimension"),
      vhloKernelInputFeatureDimension);

  auto vhloKernelOutputFeatureDimension =
      convertInt(pattern, attr.getKernelOutputFeatureDimension());
  if (!vhloKernelOutputFeatureDimension) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "kernel_output_feature_dimension"),
      vhloKernelOutputFeatureDimension);

  auto vhloKernelSpatialDimensions =
      convertInts(pattern, attr.getKernelSpatialDimensions());
  if (!vhloKernelSpatialDimensions) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "kernel_spatial_dimensions"),
      vhloKernelSpatialDimensions);

  auto vhloOutputBatchDimension =
      convertInt(pattern, attr.getOutputBatchDimension());
  if (!vhloOutputBatchDimension) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "output_batch_dimension"),
      vhloOutputBatchDimension);

  auto vhloOutputFeatureDimension =
      convertInt(pattern, attr.getOutputFeatureDimension());
  if (!vhloOutputFeatureDimension) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "output_feature_dimension"),
      vhloOutputFeatureDimension);

  auto vhloOutputSpatialDimensions =
      convertInts(pattern, attr.getOutputSpatialDimensions());
  if (!vhloOutputSpatialDimensions) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "output_spatial_dimensions"),
      vhloOutputSpatialDimensions);

  return success();
}

Attribute convertCustomCallCalledComputations(const ConversionPattern& pattern,
                                              Attribute stablehloAttr) {
  if (auto stablehloArrayAttr = stablehloAttr.dyn_cast<ArrayAttr>()) {
    SmallVector<Attribute> vhloAttrs;
    for (auto stablehloAttr : stablehloArrayAttr) {
      auto vhloAttr = convertSymbol(pattern, stablehloAttr);
      if (!vhloAttr) return {};
      vhloAttrs.push_back(vhloAttr);
    }
    return vhlo::ArrayV1Attr::get(pattern.getContext(), vhloAttrs);
  }
  return {};
}

LogicalResult convertDotDimensionNumbers(
    const ConversionPattern& pattern, Attribute stablehloAttr,
    SmallVector<NamedAttribute>& vhloAttrs) {
  auto attr = stablehloAttr.dyn_cast<stablehlo::DotDimensionNumbersAttr>();
  if (!attr) return failure();

  auto vhloLhsBatchingDimensions =
      convertInts(pattern, attr.getLhsBatchingDimensions());
  if (!vhloLhsBatchingDimensions) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "lhs_batching_dimensions"),
      vhloLhsBatchingDimensions);

  auto vhloRhsBatchingDimensions =
      convertInts(pattern, attr.getRhsBatchingDimensions());
  if (!vhloRhsBatchingDimensions) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "rhs_batching_dimensions"),
      vhloRhsBatchingDimensions);

  auto vhloLhsContractingDimensions =
      convertInts(pattern, attr.getLhsContractingDimensions());
  if (!vhloLhsContractingDimensions) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "lhs_contracting_dimensions"),
      vhloLhsContractingDimensions);

  auto vhloRhsContractingDimensions =
      convertInts(pattern, attr.getRhsContractingDimensions());
  if (!vhloRhsContractingDimensions) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "rhs_contracting_dimensions"),
      vhloRhsContractingDimensions);
  return success();
}

Attribute convertFuncCallee(const ConversionPattern& pattern,
                            Attribute stablehloAttr) {
  return convertSymbol(pattern, stablehloAttr);
}

LogicalResult convertGatherDimensionNumbers(
    const ConversionPattern& pattern, Attribute stablehloAttr,
    SmallVector<NamedAttribute>& vhloAttrs) {
  auto attr = stablehloAttr.dyn_cast<stablehlo::GatherDimensionNumbersAttr>();
  if (!attr) return failure();

  auto vhloOffsetDims = convertInts(pattern, attr.getOffsetDims());
  if (!vhloOffsetDims) return failure();
  vhloAttrs.emplace_back(StringAttr::get(pattern.getContext(), "offset_dims"),
                         vhloOffsetDims);

  auto vhloCollapsedSliceDims =
      convertInts(pattern, attr.getCollapsedSliceDims());
  if (!vhloCollapsedSliceDims) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "collapsed_slice_dims"),
      vhloCollapsedSliceDims);

  auto vhloStartIndexMap = convertInts(pattern, attr.getStartIndexMap());
  if (!vhloStartIndexMap) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "start_index_map"),
      vhloStartIndexMap);

  auto vhloIndexVectorDim = convertInt(pattern, attr.getIndexVectorDim());
  if (!vhloIndexVectorDim) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "index_vector_dim"),
      vhloIndexVectorDim);
  return success();
}

LogicalResult convertScatterDimensionNumbers(
    const ConversionPattern& pattern, Attribute stablehloAttr,
    SmallVector<NamedAttribute>& vhloAttrs) {
  auto attr = stablehloAttr.dyn_cast<stablehlo::ScatterDimensionNumbersAttr>();
  if (!attr) return failure();

  auto vhloUpdateWindowDims = convertInts(pattern, attr.getUpdateWindowDims());
  if (!vhloUpdateWindowDims) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "update_window_dims"),
      vhloUpdateWindowDims);

  auto vhloInsertedWindowDims =
      convertInts(pattern, attr.getInsertedWindowDims());
  if (!vhloInsertedWindowDims) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "inserted_window_dims"),
      vhloInsertedWindowDims);

  auto vhloScatterDimsToOperandDims =
      convertInts(pattern, attr.getScatterDimsToOperandDims());
  if (!vhloScatterDimsToOperandDims) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "scatter_dims_to_operand_dims"),
      vhloScatterDimsToOperandDims);

  auto vhloIndexVectorDim = convertInt(pattern, attr.getIndexVectorDim());
  if (!vhloIndexVectorDim) return failure();
  vhloAttrs.emplace_back(
      StringAttr::get(pattern.getContext(), "index_vector_dim"),
      vhloIndexVectorDim);
  return success();
}

#undef RETURN_CONVERTED_ENUM_ATTR

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

    SmallVector<NamedAttribute> vhloAttrs;
    for (NamedAttribute stablehloAttr : stablehloOp->getAttrs()) {
      Attribute vhloAttr;
      if (stablehloAttr.getName() == "use_global_device_ids") {
        if (!stablehloAttr.getValue().isa<UnitAttr>()) return failure();
        vhloAttr = vhlo::BooleanV1Attr::get(this->getContext(), true);
      } else if constexpr (
          std::is_same<StablehloOpTy, stablehlo::AllGatherOp>::value ||
          std::is_same<StablehloOpTy, stablehlo::AllReduceOp>::value ||
          std::is_same<StablehloOpTy, stablehlo::AllToAllOp>::value ||
          std::is_same<StablehloOpTy, stablehlo::CollectivePermuteOp>::value ||
          std::is_same<StablehloOpTy, stablehlo::ReduceScatterOp>::value) {
        if (stablehloAttr.getName() == "channel_handle") {
          auto result =
              convertChannelId(*this, stablehloAttr.getValue(), vhloAttrs);
          if (failed(result)) return failure();
          continue;
        }
      } else if constexpr (std::is_same<StablehloOpTy,
                                        stablehlo::CustomCallOp>::value) {
        if (stablehloAttr.getName() == "api_version") {
          vhloAttr = convertCustomCallApiVersion(stablehloAttr.getValue());
          if (!vhloAttr) return failure();
        }
        if (stablehloAttr.getName() == "called_computations") {
          vhloAttr = convertCustomCallCalledComputations(
              *this, stablehloAttr.getValue());
          if (!vhloAttr) return failure();
        }
      } else if constexpr (std::is_same<StablehloOpTy,
                                        stablehlo::DotGeneralOp>::value) {
        if (stablehloAttr.getName() == "dot_dimension_numbers") {
          auto result = convertDotDimensionNumbers(
              *this, stablehloAttr.getValue(), vhloAttrs);
          if (failed(result)) return failure();
          continue;
        }
      } else if constexpr (std::is_same<StablehloOpTy,
                                        stablehlo::DynamicConvOp>::value ||
                           std::is_same<StablehloOpTy,
                                        stablehlo::ConvolutionOp>::value) {
        if (stablehloAttr.getName() == "dimension_numbers") {
          auto result = convertConvDimensionNumbers(
              *this, stablehloAttr.getValue(), vhloAttrs);
          if (failed(result)) return failure();
          continue;
        }
      } else if constexpr (std::is_same<StablehloOpTy,
                                        stablehlo::DynamicGatherOp>::value ||
                           std::is_same<StablehloOpTy,
                                        stablehlo::GatherOp>::value) {
        if (stablehloAttr.getName() == "dimension_numbers") {
          auto result = convertGatherDimensionNumbers(
              *this, stablehloAttr.getValue(), vhloAttrs);
          if (failed(result)) return failure();
          continue;
        }
      } else if constexpr (std::is_same<StablehloOpTy,
                                        stablehlo::RecvOp>::value ||
                           std::is_same<StablehloOpTy,
                                        stablehlo::SendOp>::value) {
        if (stablehloAttr.getName() == "channel_handle") {
          auto result =
              convertChannelHandle(*this, stablehloAttr.getValue(), vhloAttrs);
          if (failed(result)) return failure();
          continue;
        }
      } else if constexpr (std::is_same<StablehloOpTy,
                                        stablehlo::ScatterOp>::value) {
        if (stablehloAttr.getName() == "scatter_dimension_numbers") {
          auto result = convertScatterDimensionNumbers(
              *this, stablehloAttr.getValue(), vhloAttrs);
          if (failed(result)) return failure();
          continue;
        }
      } else if constexpr (std::is_same<StablehloOpTy, func::CallOp>::value) {
        if (stablehloAttr.getName() == "callee") {
          vhloAttr = convertFuncCallee(*this, stablehloAttr.getValue());
          if (!vhloAttr) return failure();
        }
      }
      if (!vhloAttr) {
        vhloAttr = convertAttrToVhlo(stablehloAttr.getValue(),
                                     this->getTypeConverter());
        if (!vhloAttr) return failure();
      }
      vhloAttrs.push_back({stablehloAttr.getName(), vhloAttr});
    }

    // Convert the vhlo operation to a StableHLO equivalent.
    // This can almost be done in a generic fashion, except for
    // vhlo.case that uses a variadic number of regions which means an
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

/////////////////////////////////////
/// StableHLO --> VHLO Operations ///
/////////////////////////////////////

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

    // StableHLO is a subset of VHLO.
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
