/* Copyright 2022 The StableHLO Authors.
StablehloBytecode.cpp - StableHLO Bytecode Implementation */

#include "stablehlo/dialect/StablehloBytecode.h"

#include <iostream>  // FIXME

#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Diagnostics.h"

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

// Remove the `if (0)` to enable logging
#define LOG_CALL \
  if (0) std::cerr << "Called: " << __PRETTY_FUNCTION__ << std::endl

namespace {
namespace stablehlo_encoding {

/// This enum contains marker codes used to indicate which attribute is
/// currently being decoded, and how it should be decoded. The order of these
/// codes should generally be unchanged, as any changes will inevitably break
/// compatibility with older bytecode.
/// TODO: Consider the ordering of these codes before initial submit.
enum AttributeCode {
  ///   FftTypeAttr
  ///     FftType: varint
  ///   }
  kFftTypeAttr = 0,

  ///   ComparisonTypeAttr
  ///     ComparisonType: varint
  ///   }
  kComparisonTypeAttr = 1,

  ///   ComparisonDirectionAttr
  ///     ComparisonDirection: varint
  ///   }
  kComparisonDirectionAttr = 2,

  ///   TransposeAttr {
  ///     Transpose: varint
  ///   }
  kTransposeAttr = 3,

  ///   PrecisionAttr {
  ///     Precision: varint
  ///   }
  kPrecisionAttr = 4,

  ///   RngAlgorithmAttr {
  ///     RngAlgorithm: varint
  ///   }
  kRngAlgorithmAttr = 5,

  ///   RngDistributionAttr {
  ///     RngDistribution: varint
  ///   }
  kRngDistributionAttr = 6,

  ///   ChannelHandleAttr {
  ///     handle: varint
  ///     type: varint
  ///   }
  kChannelHandleAttr = 7,

  ///   ConvDimensionNumbersAttr {
  ///     inputBatchDimension: varint
  ///     inputFeatureDimension: varint
  ///     inputSpatialDimensions: Dim
  ///     kernelInputFeatureDimension: varint
  ///     kernelOutputFeatureDimension: varint
  ///     kernelSpatialDimensions: Dim
  ///     outputBatchDimension: varint
  ///     outputFeatureDimension: varint
  ///     outputSpatialDimensions: Dim
  ///   }
  kConvDimensionNumbersAttr = 8,

  ///   ScatterDimensionNumbersAttr {
  ///     updateWindowDims: Dim
  ///     insertedWindowDims: Dim
  ///     scatterDimsToOperandDims: Dim
  ///     indexVectorDim: varint
  ///   }
  kScatterDimensionNumbersAttr = 9,

  ///   GatherDimensionNumbersAttr {
  ///     offsetDims: Dim
  ///     collapsedSliceDims: Dim
  ///     startIndexMap: Dim
  ///     indexVectorDim: varint
  ///   }
  kGatherDimensionNumbers = 10,

  ///   GatherDimensionNumbersAttr {
  ///     lhsBatchingDimensions: Dim
  ///     rhsBatchingDimensions: Dim
  ///     lhsContractingDimensions: Dim
  ///     rhsContractingDimensions: varint
  ///   }
  kDotDimensionNumbers = 11,
};

/// This enum contains marker codes used to indicate which type is currently
/// being decoded, and how it should be decoded. The order of these codes should
/// generally be unchanged, as any changes will inevitably break compatibility
/// with older bytecode.
/// TODO: Consider the ordering of these codes before initial submit.
enum TypeCode {
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

  Attribute readAttribute(DialectBytecodeReader &reader) const override;
  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override;

  // Include a read method for each attribute in StableHLO
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
  // void read.*Attr(...

  // Include a write method for each attribute in StableHLO
  void write(ChannelHandleAttr attr, DialectBytecodeWriter &writer) const;
  void write(ComparisonDirectionAttr attr, DialectBytecodeWriter &writer) const;
  void write(ComparisonTypeAttr attr, DialectBytecodeWriter &writer) const;
  void write(ConvDimensionNumbersAttr attr,
             DialectBytecodeWriter &writer) const;
  void write(DotDimensionNumbersAttr attr,
             DialectBytecodeWriter &writer) const;
  void write(FftTypeAttr attr, DialectBytecodeWriter &writer) const;
  void write(GatherDimensionNumbersAttr attr,
             DialectBytecodeWriter &writer) const;
  void write(PrecisionAttr attr, DialectBytecodeWriter &writer) const;
  void write(RngAlgorithmAttr attr, DialectBytecodeWriter &writer) const;
  void write(RngDistributionAttr attr, DialectBytecodeWriter &writer) const;
  void write(ScatterDimensionNumbersAttr attr,
             DialectBytecodeWriter &writer) const;
  void write(TransposeAttr attr, DialectBytecodeWriter &writer) const;
  // void write(...

  //===--------------------------------------------------------------------===//
  // Types

  Type readType(DialectBytecodeReader &reader) const override;
  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override;

  // Include a read method for each type in StableHLO
  // IntegerType readIntegerType(DialectBytecodeReader &reader) const;
  // void read.*Type(...

  // Include a write method for each type in StableHLO
  // void write(IntegerType type, DialectBytecodeWriter &writer) const;
  // void write(...

 private:
  //===--------------------------------------------------------------------===//
  // Helper methods

  // Enum reader and writer
  template <typename EnumType, typename EnumTypeAttr, typename SymbolizeFn>
  EnumTypeAttr readEnumAttribute(DialectBytecodeReader &reader,
                             SymbolizeFn symbolizeFn) const {
    uint64_t code;
    if (failed(reader.readVarInt(code))) return EnumTypeAttr();

    llvm::Optional<EnumType> enumOpt = symbolizeFn(static_cast<uint32_t>(code));
    if (!enumOpt.has_value()) return EnumTypeAttr();

    return EnumTypeAttr::get(getContext(), enumOpt.value());
  }

  template <typename EnumTypeAttr>
  void writeEnumAttribute(EnumTypeAttr val,
                          DialectBytecodeWriter &writer) const {
    uint64_t enumVal =
        static_cast<std::underlying_type<FftType>::type>(val.getValue());
    writer.writeVarInt(enumVal);
  }

  // Int64 conversion methods.
  //
  // FIXME: static_cast wont work at the extremes.
  //        How should this encode/decode be done?
  uint64_t encodeInt64(int64_t num) const {
    return static_cast<uint64_t>(num);
  }

  int64_t decodeInt64(uint64_t num) const {
    return static_cast<int64_t>(num);
  }


  // StableHLO_Dim parameters
  LogicalResult readDim(llvm::SmallVector<int64_t> &dim,
                        DialectBytecodeReader &reader) const;
  void writeDim(llvm::ArrayRef<int64_t> attr,
                DialectBytecodeWriter &writer) const;
};

//===----------------------------------------------------------------------===//
// Implementation for StablehloBytecode

//===----------------------------------------------------------------------===//
// Attributes: Reader

// To consider: Could restructure this as a map/array of function pointers.
// Might make enhancements more manageable, won't need to keep adding
// branches for each new attribute, just an entry with an action to use.
Attribute StablehloBytecodeInterface::readAttribute(
    DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code))) return Attribute();
  LOG_CALL << "   ^With code: " << code << std::endl;
  switch (code) {
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

    default:
      reader.emitError() << "unknown stablehlo attribute code: " << code;
      return Attribute();
  }
}

ChannelHandleAttr StablehloBytecodeInterface::readChannelHandleAttr(
    DialectBytecodeReader &reader) const {
  LOG_CALL;
  uint64_t handleUint, typeUint;
  if (failed(reader.readVarInt(handleUint)) ||
      failed(reader.readVarInt(typeUint))) {
    return ChannelHandleAttr();
  }
  return ChannelHandleAttr::get(getContext(), decodeInt64(handleUint),
                                decodeInt64(typeUint));
}

ComparisonDirectionAttr StablehloBytecodeInterface::readComparisonDirectionAttr(
    DialectBytecodeReader &reader) const {
  LOG_CALL;
  return readEnumAttribute<ComparisonDirection, ComparisonDirectionAttr>(
      reader, [](uint32_t val) { return symbolizeComparisonDirection(val); });
}

ComparisonTypeAttr StablehloBytecodeInterface::readComparisonTypeAttr(
    DialectBytecodeReader &reader) const {
  LOG_CALL;
  return readEnumAttribute<ComparisonType, ComparisonTypeAttr>(
      reader, [](uint32_t val) { return symbolizeComparisonType(val); });
}

ConvDimensionNumbersAttr
StablehloBytecodeInterface::readConvDimensionNumbersAttr(
    DialectBytecodeReader &reader) const {
  LOG_CALL;
  uint64_t inputBatchDimension, inputFeatureDimension;
  llvm::SmallVector<int64_t> inputSpatialDimensions;

  uint64_t kernelInputFeatureDimension, kernelOutputFeatureDimension;
  llvm::SmallVector<int64_t> kernelSpatialDimensions;

  uint64_t outputBatchDimension, outputFeatureDimension;
  llvm::SmallVector<int64_t> outputSpatialDimensions;

  if (failed(reader.readVarInt(inputBatchDimension)) ||
      failed(reader.readVarInt(inputFeatureDimension)) ||
      failed(readDim(inputSpatialDimensions, reader)) ||
      failed(reader.readVarInt(kernelInputFeatureDimension)) ||
      failed(reader.readVarInt(kernelOutputFeatureDimension)) ||
      failed(readDim(kernelSpatialDimensions, reader)) ||
      failed(reader.readVarInt(outputBatchDimension)) ||
      failed(reader.readVarInt(outputFeatureDimension)) ||
      failed(readDim(outputSpatialDimensions, reader))) {
    return ConvDimensionNumbersAttr();
  }

  return ConvDimensionNumbersAttr::get(
      getContext(), decodeInt64(inputBatchDimension),
      decodeInt64(inputFeatureDimension), inputSpatialDimensions,
      decodeInt64(kernelInputFeatureDimension),
      decodeInt64(kernelOutputFeatureDimension), kernelSpatialDimensions,
      decodeInt64(outputBatchDimension), decodeInt64(outputFeatureDimension),
      outputSpatialDimensions);
}

DotDimensionNumbersAttr
StablehloBytecodeInterface::readDotDimensionNumbersAttr(
    DialectBytecodeReader &reader) const {
  LOG_CALL;
  llvm::SmallVector<int64_t> lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions;

  if (failed(readDim(lhsBatchingDimensions, reader)) ||
      failed(readDim(rhsBatchingDimensions, reader)) ||
      failed(readDim(lhsContractingDimensions, reader)) ||
      failed(readDim(rhsContractingDimensions, reader))) {
    return DotDimensionNumbersAttr();
  }

  return DotDimensionNumbersAttr::get(
      getContext(), lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions);
}

FftTypeAttr StablehloBytecodeInterface::readFftTypeAttr(
    DialectBytecodeReader &reader) const {
  LOG_CALL;
  return readEnumAttribute<FftType, FftTypeAttr>(
      reader, [](uint32_t val) { return symbolizeFftType(val); });
}

// FIXME: This method and readScatterDimensionNumbersAttr can be generalized.
GatherDimensionNumbersAttr
StablehloBytecodeInterface::readGatherDimensionNumbersAttr(
    DialectBytecodeReader &reader) const {
  LOG_CALL;
  llvm::SmallVector<int64_t> offsetDims, collapsedSliceDims, startIndexMap;
  uint64_t indexVectorDimUint;

  if (failed(readDim(offsetDims, reader)) ||
      failed(readDim(collapsedSliceDims, reader)) ||
      failed(readDim(startIndexMap, reader)) ||
      failed(reader.readVarInt(indexVectorDimUint))) {
    return GatherDimensionNumbersAttr();
  }

  return GatherDimensionNumbersAttr::get(getContext(), offsetDims,
                                         collapsedSliceDims, startIndexMap,
                                         decodeInt64(indexVectorDimUint));
}

PrecisionAttr StablehloBytecodeInterface::readPrecisionAttr(
    DialectBytecodeReader &reader) const {
  LOG_CALL;
  return readEnumAttribute<Precision, PrecisionAttr>(
      reader, [](uint32_t val) { return symbolizePrecision(val); });
}

RngAlgorithmAttr StablehloBytecodeInterface::readRngAlgorithmAttr(
    DialectBytecodeReader &reader) const {
  LOG_CALL;
  return readEnumAttribute<RngAlgorithm, RngAlgorithmAttr>(
      reader, [](uint32_t val) { return symbolizeRngAlgorithm(val); });
}

RngDistributionAttr StablehloBytecodeInterface::readRngDistributionAttr(
    DialectBytecodeReader &reader) const {
  LOG_CALL;
  return readEnumAttribute<RngDistribution, RngDistributionAttr>(
      reader, [](uint32_t val) { return symbolizeRngDistribution(val); });
}

ScatterDimensionNumbersAttr
StablehloBytecodeInterface::readScatterDimensionNumbersAttr(
    DialectBytecodeReader &reader) const {
  LOG_CALL;
  llvm::SmallVector<int64_t> updateWindowDims, insertedWindowDims,
      scatterDimsToOperandDims;
  uint64_t indexVectorDimUint;

  if (failed(readDim(updateWindowDims, reader)) ||
      failed(readDim(insertedWindowDims, reader)) ||
      failed(readDim(scatterDimsToOperandDims, reader)) ||
      failed(reader.readVarInt(indexVectorDimUint))) {
    return ScatterDimensionNumbersAttr();
  }

  return ScatterDimensionNumbersAttr::get(
      getContext(), updateWindowDims, insertedWindowDims,
      scatterDimsToOperandDims, decodeInt64(indexVectorDimUint));
}

TransposeAttr StablehloBytecodeInterface::readTransposeAttr(
    DialectBytecodeReader &reader) const {
  LOG_CALL;
  return readEnumAttribute<Transpose, TransposeAttr>(
      reader, [](uint32_t val) { return symbolizeTranspose(val); });
}

//===----------------------------------------------------------------------===//
// Attributes: Writer

// If this method returns failure, the string serialization is used in the
// bytecode.
LogicalResult StablehloBytecodeInterface::writeAttribute(
    Attribute attr, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<ComparisonDirectionAttr, ComparisonTypeAttr,
            ConvDimensionNumbersAttr, ChannelHandleAttr,
            DotDimensionNumbersAttr, FftTypeAttr, GatherDimensionNumbersAttr,
            PrecisionAttr, RngAlgorithmAttr, RngDistributionAttr,
            ScatterDimensionNumbersAttr, TransposeAttr>([&](auto attr) {
        write(attr, writer);
        return success();
      })
      .Default([&](Attribute) {
        LOG_CALL << "  ^not implemented." << std::endl;
        return failure();
      });
}

void StablehloBytecodeInterface::write(ChannelHandleAttr attr,
                                       DialectBytecodeWriter &writer) const {
  LOG_CALL;
  writer.writeVarInt(stablehlo_encoding::kChannelHandleAttr);
  writer.writeVarInt(encodeInt64(attr.getHandle()));
  writer.writeVarInt(encodeInt64(attr.getType()));
}

void StablehloBytecodeInterface::write(
    ComparisonDirectionAttr attr, DialectBytecodeWriter &writer) const {
  LOG_CALL;
  writer.writeVarInt(stablehlo_encoding::kComparisonDirectionAttr);
  writeEnumAttribute(attr, writer);
}

void StablehloBytecodeInterface::write(
    ComparisonTypeAttr attr, DialectBytecodeWriter &writer) const {
  LOG_CALL;
  writer.writeVarInt(stablehlo_encoding::kComparisonTypeAttr);
  writeEnumAttribute(attr, writer);
}

void StablehloBytecodeInterface::write(
    ConvDimensionNumbersAttr attr, DialectBytecodeWriter &writer) const {
  LOG_CALL;
  writer.writeVarInt(stablehlo_encoding::kConvDimensionNumbersAttr);

  writer.writeVarInt(encodeInt64(attr.getInputBatchDimension()));
  writer.writeVarInt(encodeInt64(attr.getInputFeatureDimension()));
  writeDim(attr.getInputSpatialDimensions(), writer);
  writer.writeVarInt(encodeInt64(attr.getKernelInputFeatureDimension()));
  writer.writeVarInt(encodeInt64(attr.getKernelOutputFeatureDimension()));
  writeDim(attr.getKernelSpatialDimensions(), writer);
  writer.writeVarInt(encodeInt64(attr.getOutputBatchDimension()));
  writer.writeVarInt(encodeInt64(attr.getOutputFeatureDimension()));
  writeDim(attr.getOutputSpatialDimensions(), writer);
}

void StablehloBytecodeInterface::write(
    DotDimensionNumbersAttr attr, DialectBytecodeWriter &writer) const {
  LOG_CALL;
  writer.writeVarInt(stablehlo_encoding::kDotDimensionNumbers);
  writeDim(attr.getLhsBatchingDimensions(), writer);
  writeDim(attr.getRhsBatchingDimensions(), writer);
  writeDim(attr.getLhsContractingDimensions(), writer);
  writeDim(attr.getRhsContractingDimensions(), writer);
}

void StablehloBytecodeInterface::write(
    FftTypeAttr attr, DialectBytecodeWriter &writer) const {
  LOG_CALL;
  writer.writeVarInt(stablehlo_encoding::kFftTypeAttr);
  writeEnumAttribute(attr, writer);
}

void StablehloBytecodeInterface::write(
    GatherDimensionNumbersAttr attr, DialectBytecodeWriter &writer) const {
  LOG_CALL;
  writer.writeVarInt(stablehlo_encoding::kGatherDimensionNumbers);
  writeDim(attr.getOffsetDims(), writer);
  writeDim(attr.getCollapsedSliceDims(), writer);
  writeDim(attr.getStartIndexMap(), writer);
  writer.writeVarInt(encodeInt64(attr.getIndexVectorDim()));
}

void StablehloBytecodeInterface::write(
    PrecisionAttr attr, DialectBytecodeWriter &writer) const {
  LOG_CALL;
  writer.writeVarInt(stablehlo_encoding::kPrecisionAttr);
  writeEnumAttribute(attr, writer);
}

void StablehloBytecodeInterface::write(
    RngAlgorithmAttr attr, DialectBytecodeWriter &writer) const {
  LOG_CALL;
  writer.writeVarInt(stablehlo_encoding::kRngAlgorithmAttr);
  writeEnumAttribute(attr, writer);
}

void StablehloBytecodeInterface::write(
    RngDistributionAttr attr, DialectBytecodeWriter &writer) const {
  LOG_CALL;
  writer.writeVarInt(stablehlo_encoding::kRngDistributionAttr);
  writeEnumAttribute(attr, writer);
}

void StablehloBytecodeInterface::write(
    ScatterDimensionNumbersAttr attr, DialectBytecodeWriter &writer) const {
  LOG_CALL;
  writer.writeVarInt(stablehlo_encoding::kScatterDimensionNumbersAttr);

  writeDim(attr.getUpdateWindowDims(), writer);
  writeDim(attr.getInsertedWindowDims(), writer);
  writeDim(attr.getScatterDimsToOperandDims(), writer);
  writer.writeVarInt(encodeInt64(attr.getIndexVectorDim()));
}

void StablehloBytecodeInterface::write(
    TransposeAttr attr, DialectBytecodeWriter &writer) const {
  LOG_CALL;
  writer.writeVarInt(stablehlo_encoding::kTransposeAttr);
  writeEnumAttribute(attr, writer);
}

//===----------------------------------------------------------------------===//
// Types: Reader

Type StablehloBytecodeInterface::readType(DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code))) return Type();

  switch (code) {
    case stablehlo_encoding::kTokenType:
      return TokenType::get(getContext());

    default:
      reader.emitError() << "unknown builtin type code: " << code;
      return Type();
  }
}

//===----------------------------------------------------------------------===//
// Types: Writer

LogicalResult StablehloBytecodeInterface::writeType(
    Type type, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Type, LogicalResult>(type)
      /* TODO: Implement type cases. Defaults to failure for now. */
      .Case<TokenType>([&](TokenType t) {
        LOG_CALL;
        writer.writeVarInt(stablehlo_encoding::kTokenType);
        return success();
      })
      .Default([&](Type) {
        LOG_CALL << "  ^not implemented." << std::endl;
        return failure();
      });
}


//===----------------------------------------------------------------------===//
// Helper

LogicalResult StablehloBytecodeInterface::readDim(
    llvm::SmallVector<int64_t> &dim, DialectBytecodeReader &reader) const {
  return reader.readList(dim, [&]() -> FailureOr<int64_t> {
    uint64_t valueUint;
    if (failed(reader.readVarInt(valueUint))) {
      return failure();
    }
    return decodeInt64(valueUint);
  });
}

void StablehloBytecodeInterface::writeDim(llvm::ArrayRef<int64_t> dim,
                                          DialectBytecodeWriter &writer) const {
  writer.writeList(dim, [&](int64_t value) {
    writer.writeVarInt(encodeInt64(value));
  });
}

}  // namespace

void addBytecodeInterface(StablehloDialect *dialect) {
  dialect->addInterfaces<StablehloBytecodeInterface>();
}
}  // namespace stablehlo
}  // namespace mlir
