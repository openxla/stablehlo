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

///////////////////////////////////////////////
/// VHLO --> StableHLO Types and Attributes ///
///////////////////////////////////////////////

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

#define RETURN_CONVERTED_ENUM_ATTR(Name, Version)                   \
  auto vhloValue = vhlo::stringify##Name##Version(attr.getValue()); \
  auto stablehloValue = stablehlo::symbolize##Name(vhloValue);      \
  if (!stablehloValue.has_value()) return {};                       \
  return stablehlo::Name##Attr::get(attr.getContext(), stablehloValue.value())

Attribute convertAttrToStablehlo(Attribute vhloAttr,
                                 TypeConverter* typeConverter) {
  LLVM_DEBUG(llvm::dbgs() << "Converting attr " << vhloAttr);
  // TODO: ArgResultAliasV1Attr isn't handled yet.
  if (auto vhloAttrs = vhloAttr.dyn_cast<vhlo::ArrayV1Attr>()) {
    SmallVector<Attribute> stablehloAttrs;
    for (auto vhloAttr : vhloAttrs.getValue()) {
      auto stablehloAttr = convertAttrToStablehlo(vhloAttr, typeConverter);
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
      auto builtinName = convertAttrToStablehlo(namedAttr.first, typeConverter)
                             .dyn_cast_or_null<StringAttr>();
      auto builtinValue =
          convertAttrToStablehlo(namedAttr.second, typeConverter);
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
  auto stablehloAttr = convertAttrToStablehlo(vhloAttr, typeConverter)
                           .dyn_cast_or_null<DenseIntElementsAttr>();
  if (!stablehloAttr) return failure();
  llvm::append_range(result, stablehloAttr.getValues<int64_t>());
  return success();
}

Attribute convertSymbol(Attribute vhloAttr, TypeConverter* typeConverter) {
  auto vhloStringAttr = vhloAttr.dyn_cast<vhlo::StringV1Attr>();
  if (!vhloStringAttr) return {};
  auto stablehloStringAttr =
      convertAttrToStablehlo(vhloStringAttr, typeConverter)
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

/////////////////////////////////////
/// VHLO --> StableHLO Operations ///
/////////////////////////////////////

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

    // VHLO should always be convertible to StableHLO if upgraded.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

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

    SmallVector<NamedAttribute> stablehloAttrs;
    if constexpr (std::is_same<VhloOpTy, vhlo::DotGeneralOpV1>::value) {
      auto stablehloAttr =
          convertDotDimensionNumbers(vhloOp, this->getTypeConverter());
      if (!stablehloAttr) return failure();
      stablehloAttrs.emplace_back(
          StringAttr::get(this->getContext(), "dot_dimension_numbers"),
          stablehloAttr);
    } else if constexpr (std::is_same<VhloOpTy, vhlo::DynamicConvOpV1>::value ||
                         std::is_same<VhloOpTy, vhlo::ConvolutionOpV1>::value) {
      auto stablehloAttr =
          convertConvDimensionNumbers(vhloOp, this->getTypeConverter());
      if (!stablehloAttr) return failure();
      stablehloAttrs.emplace_back(
          StringAttr::get(this->getContext(), "dimension_numbers"),
          stablehloAttr);
    } else if constexpr (std::is_same<VhloOpTy,
                                      vhlo::DynamicGatherOpV1>::value ||
                         std::is_same<VhloOpTy, vhlo::GatherOpV1>::value) {
      auto stablehloAttr =
          convertGatherDimensionNumbers(vhloOp, this->getTypeConverter());
      if (!stablehloAttr) return failure();
      stablehloAttrs.emplace_back(
          StringAttr::get(this->getContext(), "dimension_numbers"),
          stablehloAttr);
    } else if constexpr (std::is_same<VhloOpTy, vhlo::ScatterOpV1>::value) {
      auto stablehloAttr =
          convertScatterDimensionNumbers(vhloOp, this->getTypeConverter());
      if (!stablehloAttr) return failure();
      stablehloAttrs.emplace_back(
          StringAttr::get(this->getContext(), "scatter_dimension_numbers"),
          stablehloAttr);
    } else if constexpr (std::is_same<VhloOpTy, vhlo::RecvOpV1>::value ||
                         std::is_same<VhloOpTy, vhlo::SendOpV1>::value) {
      auto stablehloAttr =
          convertChannelHandle(vhloOp, this->getTypeConverter());
      if (!stablehloAttr) return failure();
      stablehloAttrs.emplace_back(
          StringAttr::get(this->getContext(), "channel_handle"), stablehloAttr);
    }

    for (NamedAttribute vhloAttr : vhloOp->getAttrs()) {
      StringAttr stablehloName = vhloAttr.getName();
      Attribute stablehloAttr;
      if (vhloAttr.getName() == "use_global_device_ids") {
        auto vhloBooleanAttr =
            vhloAttr.getValue().dyn_cast<vhlo::BooleanV1Attr>();
        if (!vhloBooleanAttr) return failure();
        if (!vhloBooleanAttr.getValue()) continue;
        stablehloAttr = UnitAttr::get(this->getContext());
      } else if constexpr (std::is_same<VhloOpTy, vhlo::AllGatherOpV1>::value ||
                           std::is_same<VhloOpTy, vhlo::AllReduceOpV1>::value ||
                           std::is_same<VhloOpTy, vhlo::AllToAllOpV1>::value ||
                           std::is_same<VhloOpTy,
                                        vhlo::CollectivePermuteOpV1>::value ||
                           std::is_same<VhloOpTy,
                                        vhlo::ReduceScatterOpV1>::value) {
        if (vhloAttr.getName() == "channel_id") {
          stablehloName = StringAttr::get(this->getContext(), "channel_handle");
          stablehloAttr =
              convertChannelId(vhloAttr.getValue(), this->getTypeConverter());
          if (!stablehloAttr) return failure();
        }
      } else if constexpr (std::is_same<VhloOpTy,
                                        vhlo::CustomCallOpV1>::value) {
        if (vhloAttr.getName() == "called_computations") {
          stablehloAttr = convertCustomCallCalledComputations(
              vhloAttr.getValue(), this->getTypeConverter());
          if (!stablehloAttr) return failure();
        }
      } else if constexpr (std::is_same<VhloOpTy,
                                        vhlo::DotGeneralOpV1>::value) {
        if (vhloAttr.getName() == "lhs_batching_dimensions" ||
            vhloAttr.getName() == "rhs_batching_dimensions" ||
            vhloAttr.getName() == "lhs_contracting_dimensions" ||
            vhloAttr.getName() == "rhs_contracting_dimensions") {
          continue;
        }
      } else if constexpr (std::is_same<VhloOpTy,
                                        vhlo::DynamicConvOpV1>::value ||
                           std::is_same<VhloOpTy,
                                        vhlo::ConvolutionOpV1>::value) {
        if (vhloAttr.getName() == "input_batch_dimension" ||
            vhloAttr.getName() == "input_feature_dimension" ||
            vhloAttr.getName() == "input_spatial_dimensions" ||
            vhloAttr.getName() == "kernel_input_feature_dimension" ||
            vhloAttr.getName() == "kernel_output_feature_dimension" ||
            vhloAttr.getName() == "kernel_spatial_dimensions" ||
            vhloAttr.getName() == "output_batch_dimension" ||
            vhloAttr.getName() == "output_feature_dimension" ||
            vhloAttr.getName() == "output_spatial_dimensions") {
          continue;
        }
      } else if constexpr (std::is_same<VhloOpTy,
                                        vhlo::DynamicGatherOpV1>::value ||
                           std::is_same<VhloOpTy, vhlo::GatherOpV1>::value) {
        if (vhloAttr.getName() == "offset_dims" ||
            vhloAttr.getName() == "collapsed_slice_dims" ||
            vhloAttr.getName() == "start_index_map" ||
            vhloAttr.getName() == "index_vector_dim") {
          continue;
        }
      } else if constexpr (std::is_same<VhloOpTy, vhlo::ScatterOpV1>::value) {
        if (vhloAttr.getName() == "update_window_dims" ||
            vhloAttr.getName() == "inserted_window_dims" ||
            vhloAttr.getName() == "scatter_dims_to_operand_dims" ||
            vhloAttr.getName() == "index_vector_dim") {
          continue;
        }
      } else if constexpr (std::is_same<VhloOpTy, vhlo::RecvOpV1>::value ||
                           std::is_same<VhloOpTy, vhlo::SendOpV1>::value) {
        if (vhloAttr.getName() == "channel_id" ||
            vhloAttr.getName() == "channel_type") {
          continue;
        }
      } else if constexpr (std::is_same<VhloOpTy, vhlo::CallOpV1>::value) {
        if (vhloAttr.getName() == "callee") {
          stablehloAttr =
              convertFuncCallee(vhloAttr.getValue(), this->getTypeConverter());
          if (!stablehloAttr) return failure();
        }
      }
      if (!stablehloAttr) {
        stablehloAttr = convertAttrToStablehlo(vhloAttr.getValue(),
                                               this->getTypeConverter());
        if (!stablehloAttr) return failure();
      }
      stablehloAttrs.push_back({stablehloName, stablehloAttr});
    }

    // Replace vhlo.return --> func.return if direct parent is a func op.
    if constexpr (std::is_same<VhloOpTy, vhlo::ReturnOpV1>::value) {
      if (llvm::isa<vhlo::FuncOpV1, func::FuncOp>(vhloOp->getParentOp())) {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(
            vhloOp, stablehloTypes, stablehloOperands, stablehloAttrs);
        return success();
      }
    }

    // Convert the vhlo operation to a StableHLO equivalent.
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
