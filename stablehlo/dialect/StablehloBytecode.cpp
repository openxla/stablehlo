/* Copyright 2022 The StableHLO Authors.
StablehloBytecode.cpp - StableHLO Bytecode Implementation */

#include "stablehlo/dialect/StablehloBytecode.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Diagnostics.h"
#include "stablehlo/dialect/StablehloOps.h"

//===----------------------------------------------------------------------===//
// Debug Trace Helpers
//===----------------------------------------------------------------------===//

// Enable logging with flag:
//   stablehlo-opt -debug-only=stablehlo-bytecode [...]
//
// Extract after function name, remove namespace.
//   Called: write(mlir::stablehlo::TokenType, mlir::DialectBytecodeWriter ...
//   ***Not Implemened: write(...
#define _EXTRACT_AFTER(a, b) \
  llvm::StringRef(a).substr(llvm::StringRef(a).find(b))

#define _LOG_CALL_TO(func)                                                    \
  DEBUG_WITH_TYPE(                                                            \
      "stablehlo-bytecode",                                                   \
      llvm::errs() << "Called: " << _EXTRACT_AFTER(__PRETTY_FUNCTION__, func) \
                   << '\n')

#define LOG_WRITE_CALL _LOG_CALL_TO("write")
#define LOG_READ_CALL _LOG_CALL_TO(__func__)
#define LOG_NOT_IMPLEMENTED \
  DEBUG_WITH_TYPE(          \
      "stablehlo-bytecode", \
      llvm::errs() << "***Not Implemented: " << __PRETTY_FUNCTION__ << '\n')

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

namespace {
namespace stablehlo_encoding {

/// This enum contains marker codes used to indicate which attribute is
/// currently being decoded, and how it should be decoded. The order of these
/// codes should generally be unchanged, as any changes will inevitably break
/// compatibility with older bytecode.
///
/// To add an attribute, search for "TO ADD ATTRIBUTE" in this file and ensure
/// each location is updated.
enum AttributeCode {
  // TO ADD ATTRIBUTE: Add an enum value with doc string for new attr.

  ///   ArgResultAlias {
  ///     argTupleIndices: svarint[]
  ///     resultIndex: svarint
  ///     resultIndex: svarint[]
  ///     isMustAlias: varint
  ///   }
  kArgResultAlias = 0,

  ///   ChannelHandleAttr {
  ///     handle: svarint
  ///     type: svarint
  ///   }
  kChannelHandleAttr = 1,

  ///   ComparisonDirectionAttr
  ///     ComparisonDirection: varint
  ///   }
  kComparisonDirectionAttr = 2,

  ///   ComparisonTypeAttr
  ///     ComparisonType: varint
  ///   }
  kComparisonTypeAttr = 3,

  ///   ConvDimensionNumbersAttr {
  ///     inputBatchDimension: svarint
  ///     inputFeatureDimension: svarint
  ///     inputSpatialDimensions: svarint[]
  ///     kernelInputFeatureDimension: svarint
  ///     kernelOutputFeatureDimension: svarint
  ///     kernelSpatialDimensions: svarint[]
  ///     outputBatchDimension: svarint
  ///     outputFeatureDimension: svarint
  ///     outputSpatialDimensions: svarint[]
  ///   }
  kConvDimensionNumbersAttr = 4,

  ///   GatherDimensionNumbersAttr {
  ///     lhsBatchingDimensions: svarint[]
  ///     rhsBatchingDimensions: svarint[]
  ///     lhsContractingDimensions: svarint[]
  ///     rhsContractingDimensions: svarint
  ///   }
  kDotDimensionNumbers = 5,

  ///   FftTypeAttr
  ///     FftType: varint
  ///   }
  kFftTypeAttr = 6,

  ///   GatherDimensionNumbersAttr {
  ///     offsetDims: svarint[]
  ///     collapsedSliceDims: svarint[]
  ///     startIndexMap: svarint[]
  ///     indexVectorDim: svarint
  ///   }
  kGatherDimensionNumbers = 7,

  ///   PrecisionAttr {
  ///     Precision: varint
  ///   }
  kPrecisionAttr = 8,

  ///   RngAlgorithmAttr {
  ///     RngAlgorithm: varint
  ///   }
  kRngAlgorithmAttr = 9,

  ///   RngDistributionAttr {
  ///     RngDistribution: varint
  ///   }
  kRngDistributionAttr = 10,

  ///   ScatterDimensionNumbersAttr {
  ///     updateWindowDims: svarint[]
  ///     insertedWindowDims: svarint[]
  ///     scatterDimsToOperandDims: svarint[]
  ///     indexVectorDim: svarint
  ///   }
  kScatterDimensionNumbersAttr = 11,

  ///   TransposeAttr {
  ///     Transpose: varint
  ///   }
  kTransposeAttr = 12,

  ///   TypeExtensionsAttr {
  ///     bounds : svarint[]
  ///   }
  kTypeExtensionsAttr = 13,
};

/// This enum contains marker codes used to indicate which type is currently
/// being decoded, and how it should be decoded. The order of these codes should
/// generally be unchanged, as any changes will inevitably break compatibility
/// with older bytecode.
///
/// To add a type, search for "TO ADD TYPE" in this file and ensure each
/// location is updated.
enum TypeCode {
  // TO ADD TYPE: Add an enum value with doc string for new attr.

  ///   TokenType {
  ///   }
  kTokenType = 0,
};

}  // namespace stablehlo_encoding
}  // namespace

//===----------------------------------------------------------------------===//
// StablehloBytecodeInterface
//===----------------------------------------------------------------------===//

namespace mlir {
namespace stablehlo {

namespace {
/// This class implements the bytecode interface for the stablehlo dialect.
class StablehloBytecodeInterface : public BytecodeDialectInterface {
 public:
  StablehloBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  // These methods are invoked by superclass when an attr from stablehlo dialect
  // is encountered.
  Attribute readAttribute(DialectBytecodeReader &reader) const override;
  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override;

  // TO ADD ATTRIBUTE: Include a read method for each attribute in StableHLO
  // Ex: SomeAttr readSomeAttr(DialectBytecodeReader &reader) const;
  ArgResultAliasAttr readArgResultAliasAttr(
      DialectBytecodeReader &reader) const;
  ChannelHandleAttr readChannelHandleAttr(DialectBytecodeReader &reader) const;
  ComparisonDirectionAttr readComparisonDirectionAttr(
      DialectBytecodeReader &reader) const;
  ComparisonTypeAttr readComparisonTypeAttr(
      DialectBytecodeReader &reader) const;
  ConvDimensionNumbersAttr readConvDimensionNumbersAttr(
      DialectBytecodeReader &reader) const;
  DotDimensionNumbersAttr readDotDimensionNumbersAttr(
      DialectBytecodeReader &reader) const;
  FftTypeAttr readFftTypeAttr(DialectBytecodeReader &reader) const;
  GatherDimensionNumbersAttr readGatherDimensionNumbersAttr(
      DialectBytecodeReader &reader) const;
  PrecisionAttr readPrecisionAttr(DialectBytecodeReader &reader) const;
  RngAlgorithmAttr readRngAlgorithmAttr(DialectBytecodeReader &reader) const;
  RngDistributionAttr readRngDistributionAttr(
      DialectBytecodeReader &reader) const;
  ScatterDimensionNumbersAttr readScatterDimensionNumbersAttr(
      DialectBytecodeReader &reader) const;
  TransposeAttr readTransposeAttr(DialectBytecodeReader &reader) const;
  TypeExtensionsAttr readTypeExtensionsAttr(
      DialectBytecodeReader &reader) const;

  // TO ADD ATTRIBUTE: Include a write method for each attribute in StableHLO
  // Ex: void write(SomeAttr attr, DialectBytecodeWriter &writer) const;
  void write(ArgResultAliasAttr attr, DialectBytecodeWriter &writer) const;
  void write(ChannelHandleAttr attr, DialectBytecodeWriter &writer) const;
  void write(ComparisonDirectionAttr attr, DialectBytecodeWriter &writer) const;
  void write(ComparisonTypeAttr attr, DialectBytecodeWriter &writer) const;
  void write(ConvDimensionNumbersAttr attr,
             DialectBytecodeWriter &writer) const;
  void write(DotDimensionNumbersAttr attr, DialectBytecodeWriter &writer) const;
  void write(FftTypeAttr attr, DialectBytecodeWriter &writer) const;
  void write(GatherDimensionNumbersAttr attr,
             DialectBytecodeWriter &writer) const;
  void write(PrecisionAttr attr, DialectBytecodeWriter &writer) const;
  void write(RngAlgorithmAttr attr, DialectBytecodeWriter &writer) const;
  void write(RngDistributionAttr attr, DialectBytecodeWriter &writer) const;
  void write(ScatterDimensionNumbersAttr attr,
             DialectBytecodeWriter &writer) const;
  void write(TransposeAttr attr, DialectBytecodeWriter &writer) const;
  void write(TypeExtensionsAttr attr, DialectBytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Types

  // These methods are invoked by superclass when a type from stablehlo dialect
  // is encountered.
  Type readType(DialectBytecodeReader &reader) const override;
  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override;

  // TO ADD TYPE: Include a read method for each type in StableHLO
  // Ex: SomeType readSomeType(DialectBytecodeReader &reader) const;
  TokenType readTokenType(DialectBytecodeReader &reader) const;

  // TO ADD TYPE: Include a write method for each type in StableHLO
  // Ex: void write(SomeType attr, DialectBytecodeWriter &writer) const;
  void write(TokenType type, DialectBytecodeWriter &writer) const;

 private:
  //===--------------------------------------------------------------------===//
  // Helper methods

  // Enum reader and writer. Many attrs have a single enum type to serialize.
  // Use the attributes underlying type to get the numeric value.
  // Note this may cause issues if enums use an int64_t and have a large value.
  // All enums in StableHLO currently use int32_t.
  template <typename EnumType, typename EnumTypeAttr, typename SymbolizeFn>
  EnumTypeAttr readEnumAttribute(DialectBytecodeReader &reader,
                                 SymbolizeFn symbolizeFn) const {
    uint64_t code;
    if (failed(reader.readVarInt(code))) return EnumTypeAttr();

    llvm::Optional<EnumType> enumOpt = symbolizeFn(static_cast<uint32_t>(code));
    if (!enumOpt.has_value()) return EnumTypeAttr();

    return EnumTypeAttr::get(getContext(), enumOpt.value());
  }

  template <typename EnumType, typename EnumTypeAttr>
  void writeEnumAttribute(EnumTypeAttr val,
                          DialectBytecodeWriter &writer) const {
    static_assert(
        std::is_same<typename std::underlying_type<EnumType>::type,
                     uint32_t>::value,
        "writeEnumAttribute is only implemented for uint32_t enum values");

    uint32_t enumVal =
        static_cast<typename std::underlying_type<EnumType>::type>(
            val.getValue());
    writer.writeVarInt(enumVal);
  }
};

//===----------------------------------------------------------------------===//
// Implementation for StablehloBytecode

//===----------------------------------------------------------------------===//
// Attributes: Reader

// TO ADD ATTRIBUTE: Update the switch to include a branch for the attr.
Attribute StablehloBytecodeInterface::readAttribute(
    DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code))) return Attribute();
  switch (code) {
    case stablehlo_encoding::kArgResultAlias:
      return readArgResultAliasAttr(reader);
    case stablehlo_encoding::kChannelHandleAttr:
      return readChannelHandleAttr(reader);
    case stablehlo_encoding::kComparisonDirectionAttr:
      return readComparisonDirectionAttr(reader);
    case stablehlo_encoding::kComparisonTypeAttr:
      return readComparisonTypeAttr(reader);
    case stablehlo_encoding::kConvDimensionNumbersAttr:
      return readConvDimensionNumbersAttr(reader);
    case stablehlo_encoding::kDotDimensionNumbers:
      return readDotDimensionNumbersAttr(reader);
    case stablehlo_encoding::kFftTypeAttr:
      return readFftTypeAttr(reader);
    case stablehlo_encoding::kGatherDimensionNumbers:
      return readGatherDimensionNumbersAttr(reader);
    case stablehlo_encoding::kPrecisionAttr:
      return readPrecisionAttr(reader);
    case stablehlo_encoding::kRngAlgorithmAttr:
      return readRngAlgorithmAttr(reader);
    case stablehlo_encoding::kRngDistributionAttr:
      return readRngDistributionAttr(reader);
    case stablehlo_encoding::kScatterDimensionNumbersAttr:
      return readScatterDimensionNumbersAttr(reader);
    case stablehlo_encoding::kTransposeAttr:
      return readTransposeAttr(reader);
    case stablehlo_encoding::kTypeExtensionsAttr:
      return readTypeExtensionsAttr(reader);

    default:
      reader.emitError() << "unknown stablehlo attribute code: " << code;
      return Attribute();
  }
}

ArgResultAliasAttr StablehloBytecodeInterface::readArgResultAliasAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;

  llvm::SmallVector<int64_t> argTupleIndices;
  int64_t resultIndex;
  llvm::SmallVector<int64_t> resultTupleIndices;
  uint64_t isMustAliasUint;

  if (failed(reader.readSignedVarInts(argTupleIndices)) ||
      failed(reader.readSignedVarInt(resultIndex)) ||
      failed(reader.readSignedVarInts(resultTupleIndices)) ||
      failed(reader.readVarInt(isMustAliasUint))) {
    return ArgResultAliasAttr();
  }
  return ArgResultAliasAttr::get(getContext(), argTupleIndices, resultIndex,
                                 resultTupleIndices,
                                 static_cast<bool>(isMustAliasUint));
}

ChannelHandleAttr StablehloBytecodeInterface::readChannelHandleAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  int64_t handle, type;
  if (failed(reader.readSignedVarInt(handle)) ||
      failed(reader.readSignedVarInt(type))) {
    return ChannelHandleAttr();
  }
  return ChannelHandleAttr::get(getContext(), handle, type);
}

ComparisonDirectionAttr StablehloBytecodeInterface::readComparisonDirectionAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return readEnumAttribute<ComparisonDirection, ComparisonDirectionAttr>(
      reader, [](uint32_t val) { return symbolizeComparisonDirection(val); });
}

ComparisonTypeAttr StablehloBytecodeInterface::readComparisonTypeAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return readEnumAttribute<ComparisonType, ComparisonTypeAttr>(
      reader, [](uint32_t val) { return symbolizeComparisonType(val); });
}

ConvDimensionNumbersAttr
StablehloBytecodeInterface::readConvDimensionNumbersAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  int64_t inputBatchDimension, inputFeatureDimension;
  llvm::SmallVector<int64_t> inputSpatialDimensions;

  int64_t kernelInputFeatureDimension, kernelOutputFeatureDimension;
  llvm::SmallVector<int64_t> kernelSpatialDimensions;

  int64_t outputBatchDimension, outputFeatureDimension;
  llvm::SmallVector<int64_t> outputSpatialDimensions;

  if (failed(reader.readSignedVarInt(inputBatchDimension)) ||
      failed(reader.readSignedVarInt(inputFeatureDimension)) ||
      failed(reader.readSignedVarInts(inputSpatialDimensions)) ||
      failed(reader.readSignedVarInt(kernelInputFeatureDimension)) ||
      failed(reader.readSignedVarInt(kernelOutputFeatureDimension)) ||
      failed(reader.readSignedVarInts(kernelSpatialDimensions)) ||
      failed(reader.readSignedVarInt(outputBatchDimension)) ||
      failed(reader.readSignedVarInt(outputFeatureDimension)) ||
      failed(reader.readSignedVarInts(outputSpatialDimensions))) {
    return ConvDimensionNumbersAttr();
  }

  return ConvDimensionNumbersAttr::get(
      getContext(), inputBatchDimension, inputFeatureDimension,
      inputSpatialDimensions, kernelInputFeatureDimension,
      kernelOutputFeatureDimension, kernelSpatialDimensions,
      outputBatchDimension, outputFeatureDimension, outputSpatialDimensions);
}

DotDimensionNumbersAttr StablehloBytecodeInterface::readDotDimensionNumbersAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions;

  if (failed(reader.readSignedVarInts(lhsBatchingDimensions)) ||
      failed(reader.readSignedVarInts(rhsBatchingDimensions)) ||
      failed(reader.readSignedVarInts(lhsContractingDimensions)) ||
      failed(reader.readSignedVarInts(rhsContractingDimensions))) {
    return DotDimensionNumbersAttr();
  }

  return DotDimensionNumbersAttr::get(
      getContext(), lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions);
}

FftTypeAttr StablehloBytecodeInterface::readFftTypeAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return readEnumAttribute<FftType, FftTypeAttr>(
      reader, [](uint32_t val) { return symbolizeFftType(val); });
}

GatherDimensionNumbersAttr
StablehloBytecodeInterface::readGatherDimensionNumbersAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> offsetDims, collapsedSliceDims, startIndexMap;
  int64_t indexVectorDim;

  if (failed(reader.readSignedVarInts(offsetDims)) ||
      failed(reader.readSignedVarInts(collapsedSliceDims)) ||
      failed(reader.readSignedVarInts(startIndexMap)) ||
      failed(reader.readSignedVarInt(indexVectorDim))) {
    return GatherDimensionNumbersAttr();
  }

  return GatherDimensionNumbersAttr::get(getContext(), offsetDims,
                                         collapsedSliceDims, startIndexMap,
                                         indexVectorDim);
}

PrecisionAttr StablehloBytecodeInterface::readPrecisionAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return readEnumAttribute<Precision, PrecisionAttr>(
      reader, [](uint32_t val) { return symbolizePrecision(val); });
}

RngAlgorithmAttr StablehloBytecodeInterface::readRngAlgorithmAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return readEnumAttribute<RngAlgorithm, RngAlgorithmAttr>(
      reader, [](uint32_t val) { return symbolizeRngAlgorithm(val); });
}

RngDistributionAttr StablehloBytecodeInterface::readRngDistributionAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return readEnumAttribute<RngDistribution, RngDistributionAttr>(
      reader, [](uint32_t val) { return symbolizeRngDistribution(val); });
}

ScatterDimensionNumbersAttr
StablehloBytecodeInterface::readScatterDimensionNumbersAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> updateWindowDims, insertedWindowDims,
      scatterDimsToOperandDims;
  int64_t indexVectorDim;

  if (failed(reader.readSignedVarInts(updateWindowDims)) ||
      failed(reader.readSignedVarInts(insertedWindowDims)) ||
      failed(reader.readSignedVarInts(scatterDimsToOperandDims)) ||
      failed(reader.readSignedVarInt(indexVectorDim))) {
    return ScatterDimensionNumbersAttr();
  }

  return ScatterDimensionNumbersAttr::get(
      getContext(), updateWindowDims, insertedWindowDims,
      scatterDimsToOperandDims, indexVectorDim);
}

TransposeAttr StablehloBytecodeInterface::readTransposeAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return readEnumAttribute<Transpose, TransposeAttr>(
      reader, [](uint32_t val) { return symbolizeTranspose(val); });
}

TypeExtensionsAttr StablehloBytecodeInterface::readTypeExtensionsAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> bounds;
  if (failed(reader.readSignedVarInts(bounds))) {
    return TypeExtensionsAttr();
  }
  return TypeExtensionsAttr::get(getContext(), bounds);
}

//===----------------------------------------------------------------------===//
// Attributes: Writer

// TO ADD ATTRIBUTE: Update the case selection to include the new attr.
// If this method returns failure, the string serialization is used in the
// bytecode.
LogicalResult StablehloBytecodeInterface::writeAttribute(
    Attribute attr, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<ArgResultAliasAttr, ComparisonDirectionAttr, ComparisonTypeAttr,
            ConvDimensionNumbersAttr, ChannelHandleAttr,
            DotDimensionNumbersAttr, FftTypeAttr, GatherDimensionNumbersAttr,
            PrecisionAttr, RngAlgorithmAttr, RngDistributionAttr,
            ScatterDimensionNumbersAttr, TransposeAttr, TypeExtensionsAttr>(
          [&](auto attr) {
            LOG_WRITE_CALL;
            write(attr, writer);
            return success();
          })
      .Default([&](Attribute) {
        LOG_NOT_IMPLEMENTED;
        return failure();
      });
}

void StablehloBytecodeInterface::write(ArgResultAliasAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kArgResultAlias);
  writer.writeSignedVarInts(attr.getArgTupleIndices());
  writer.writeSignedVarInt(attr.getResultIndex());
  writer.writeSignedVarInts(attr.getResultTupleIndices());
  writer.writeVarInt(attr.getIsMustAlias());
}

void StablehloBytecodeInterface::write(ChannelHandleAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kChannelHandleAttr);
  writer.writeSignedVarInt(attr.getHandle());
  writer.writeSignedVarInt(attr.getType());
}

void StablehloBytecodeInterface::write(ComparisonDirectionAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kComparisonDirectionAttr);
  writeEnumAttribute<ComparisonDirection>(attr, writer);
}

void StablehloBytecodeInterface::write(ComparisonTypeAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kComparisonTypeAttr);
  writeEnumAttribute<ComparisonType>(attr, writer);
}

void StablehloBytecodeInterface::write(ConvDimensionNumbersAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kConvDimensionNumbersAttr);
  writer.writeSignedVarInt(attr.getInputBatchDimension());
  writer.writeSignedVarInt(attr.getInputFeatureDimension());
  writer.writeSignedVarInts(attr.getInputSpatialDimensions());
  writer.writeSignedVarInt(attr.getKernelInputFeatureDimension());
  writer.writeSignedVarInt(attr.getKernelOutputFeatureDimension());
  writer.writeSignedVarInts(attr.getKernelSpatialDimensions());
  writer.writeSignedVarInt(attr.getOutputBatchDimension());
  writer.writeSignedVarInt(attr.getOutputFeatureDimension());
  writer.writeSignedVarInts(attr.getOutputSpatialDimensions());
}

void StablehloBytecodeInterface::write(DotDimensionNumbersAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kDotDimensionNumbers);
  writer.writeSignedVarInts(attr.getLhsBatchingDimensions());
  writer.writeSignedVarInts(attr.getRhsBatchingDimensions());
  writer.writeSignedVarInts(attr.getLhsContractingDimensions());
  writer.writeSignedVarInts(attr.getRhsContractingDimensions());
}

void StablehloBytecodeInterface::write(FftTypeAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kFftTypeAttr);
  writeEnumAttribute<FftType>(attr, writer);
}

void StablehloBytecodeInterface::write(GatherDimensionNumbersAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kGatherDimensionNumbers);
  writer.writeSignedVarInts(attr.getOffsetDims());
  writer.writeSignedVarInts(attr.getCollapsedSliceDims());
  writer.writeSignedVarInts(attr.getStartIndexMap());
  writer.writeSignedVarInt(attr.getIndexVectorDim());
}

void StablehloBytecodeInterface::write(PrecisionAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kPrecisionAttr);
  writeEnumAttribute<Precision>(attr, writer);
}

void StablehloBytecodeInterface::write(RngAlgorithmAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kRngAlgorithmAttr);
  writeEnumAttribute<RngAlgorithm>(attr, writer);
}

void StablehloBytecodeInterface::write(RngDistributionAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kRngDistributionAttr);
  writeEnumAttribute<RngDistribution>(attr, writer);
}

void StablehloBytecodeInterface::write(ScatterDimensionNumbersAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kScatterDimensionNumbersAttr);
  writer.writeSignedVarInts(attr.getUpdateWindowDims());
  writer.writeSignedVarInts(attr.getInsertedWindowDims());
  writer.writeSignedVarInts(attr.getScatterDimsToOperandDims());
  writer.writeSignedVarInt(attr.getIndexVectorDim());
}

void StablehloBytecodeInterface::write(TransposeAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kTransposeAttr);
  writeEnumAttribute<Transpose>(attr, writer);
}

void StablehloBytecodeInterface::write(TypeExtensionsAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kTypeExtensionsAttr);
  writer.writeSignedVarInts(attr.getBounds());
}

//===----------------------------------------------------------------------===//
// Types: Reader

// TO ADD TYPE: Update the case selection to include the new type.
Type StablehloBytecodeInterface::readType(DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code))) return Type();

  switch (code) {
    case stablehlo_encoding::kTokenType:
      return readTokenType(reader);

    default:
      reader.emitError() << "unknown builtin type code: " << code;
      return Type();
  }
}

TokenType StablehloBytecodeInterface::readTokenType(
    DialectBytecodeReader &) const {
  LOG_READ_CALL;
  return TokenType::get(getContext());
}

//===----------------------------------------------------------------------===//
// Types: Writer

// TO ADD TYPE: Update the case selection to include the new type.
LogicalResult StablehloBytecodeInterface::writeType(
    Type type, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Type, LogicalResult>(type)
      .Case<TokenType>([&](auto type) {
        LOG_WRITE_CALL;
        write(type, writer);
        return success();
      })
      .Default([&](Type) {
        LOG_NOT_IMPLEMENTED;
        return failure();
      });
}

void StablehloBytecodeInterface::write(TokenType type,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kTokenType);
}

}  // namespace

void addBytecodeInterface(StablehloDialect *dialect) {
  dialect->addInterfaces<StablehloBytecodeInterface>();
}
}  // namespace stablehlo
}  // namespace mlir
