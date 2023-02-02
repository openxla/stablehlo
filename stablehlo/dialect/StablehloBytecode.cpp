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

#include "stablehlo/dialect/StablehloBytecode.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compiler.h"
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

#define _LOG_CALL_TO(func)                                                     \
  DEBUG_WITH_TYPE(                                                             \
      "stablehlo-bytecode",                                                    \
      llvm::errs() << "Called: " << _EXTRACT_AFTER(LLVM_PRETTY_FUNCTION, func) \
                   << '\n')

#define LOG_WRITE_CALL _LOG_CALL_TO("write")
#define LOG_READ_CALL _LOG_CALL_TO(__func__)
#define LOG_NOT_IMPLEMENTED \
  DEBUG_WITH_TYPE(          \
      "stablehlo-bytecode", \
      llvm::errs() << "***Not Implemented: " << LLVM_PRETTY_FUNCTION << '\n')

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

namespace {
namespace stablehlo_encoding {

/// This enum contains marker codes used to indicate which attribute is
/// currently being decoded, and how it should be decoded. The order of these
/// codes must not be changed, as any changes will break compatibility
/// with older bytecode.
///
/// To add an attribute, search for "TO ADD ATTRIBUTE" in this file and ensure
/// each location is updated.
enum AttributeCode {
  // TO ADD ATTRIBUTE: Add an enum value with doc string for new attr.

  ///   ArgResultAliasAttr {
  ///     argTupleIndices: svarint[]
  ///     resultIndex: svarint
  ///     resultIndex: svarint[]
  ///     isMustAlias: varint
  ///   }
  kArgResultAliasAttr = 0,

  ///   ChannelHandleAttr {
  ///     handle: svarint
  ///     type: svarint
  ///   }
  kChannelHandleAttr = 1,

  ///   ComparisonDirectionAttr
  ///     value: varint (encoded enum)
  ///   }
  kComparisonDirectionAttr = 2,

  ///   ComparisonTypeAttr
  ///     value: varint (encoded enum)
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

  ///   DotDimensionNumbersAttr {
  ///     lhsBatchingDimensions: svarint[]
  ///     rhsBatchingDimensions: svarint[]
  ///     lhsContractingDimensions: svarint[]
  ///     rhsContractingDimensions: svarint[]
  ///   }
  kDotDimensionNumbers = 5,

  ///   FftTypeAttr
  ///     value: varint (encoded enum)
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
  ///     value: varint (encoded enum)
  ///   }
  kPrecisionAttr = 8,

  ///   RngAlgorithmAttr {
  ///     value: varint (encoded enum)
  ///   }
  kRngAlgorithmAttr = 9,

  ///   RngDistributionAttr {
  ///     value: varint (encoded enum)
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
  ///     value: varint (encoded enum)
  ///   }
  kTransposeAttr = 12,

  ///   TypeExtensionsAttr {
  ///     bounds : svarint[]
  ///   }
  kTypeExtensionsAttr = 13,

  ///   OutputOperandAliasAttr {
  ///     outputTupleIndices: svarint[]
  ///     operandIndex : svarint
  ///     operandTupleIndices: svarint[]
  ///   }
  kOutputOperandAlias = 14,
};

/// This enum contains marker codes used to indicate which type is
/// currently being decoded, and how it should be decoded. The order of these
/// codes must not be changed, as any changes will break compatibility
/// with older bytecode.
///
/// To add a type, search for "TO ADD TYPE" in this file and ensure each
/// location is updated.
enum TypeCode {
  // TO ADD TYPE: Add an enum value with doc string for new type.

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
/// This class implements the bytecode interface for the StableHLO dialect.
class StablehloBytecodeInterface : public BytecodeDialectInterface {
 public:
  StablehloBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  // These methods are invoked by superclass when an attr from StableHLO dialect
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
  OutputOperandAliasAttr readOutputOperandAliasAttr(
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
  void write(OutputOperandAliasAttr attr, DialectBytecodeWriter &writer) const;
  void write(PrecisionAttr attr, DialectBytecodeWriter &writer) const;
  void write(RngAlgorithmAttr attr, DialectBytecodeWriter &writer) const;
  void write(RngDistributionAttr attr, DialectBytecodeWriter &writer) const;
  void write(ScatterDimensionNumbersAttr attr,
             DialectBytecodeWriter &writer) const;
  void write(TransposeAttr attr, DialectBytecodeWriter &writer) const;
  void write(TypeExtensionsAttr attr, DialectBytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Types

  // These methods are invoked by superclass when a type from StableHLO dialect
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
};

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// TO ADD ATTRIBUTE: Update the switch to include a branch for the attr.
Attribute StablehloBytecodeInterface::readAttribute(
    DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code))) return Attribute();
  switch (code) {
    case stablehlo_encoding::kArgResultAliasAttr:
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
    case stablehlo_encoding::kOutputOperandAlias:
      return readOutputOperandAliasAttr(reader);
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

// TO ADD ATTRIBUTE: Update the case selection to include the new attr.
// If this method returns failure, the string serialization is used in the
// bytecode.
LogicalResult StablehloBytecodeInterface::writeAttribute(
    Attribute attr, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<ArgResultAliasAttr, ChannelHandleAttr, ComparisonDirectionAttr,
            ComparisonTypeAttr, ConvDimensionNumbersAttr,
            DotDimensionNumbersAttr, FftTypeAttr, GatherDimensionNumbersAttr,
            OutputOperandAliasAttr, PrecisionAttr, RngAlgorithmAttr,
            RngDistributionAttr, ScatterDimensionNumbersAttr, TransposeAttr,
            TypeExtensionsAttr>([&](auto attr) {
        LOG_WRITE_CALL;
        write(attr, writer);
        return success();
      })
      .Default([&](Attribute) {
        LOG_NOT_IMPLEMENTED;
        return failure();
      });
}

//===----------------------------------------------------------------------===//
// ArgResultAliasAttr

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
      failed(reader.readVarInt(isMustAliasUint)))
    return ArgResultAliasAttr();

  return ArgResultAliasAttr::get(getContext(), argTupleIndices, resultIndex,
                                 resultTupleIndices,
                                 static_cast<bool>(isMustAliasUint));
}

void StablehloBytecodeInterface::write(ArgResultAliasAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kArgResultAliasAttr);
  writer.writeSignedVarInts(attr.getArgTupleIndices());
  writer.writeSignedVarInt(attr.getResultIndex());
  writer.writeSignedVarInts(attr.getResultTupleIndices());
  writer.writeVarInt(attr.getIsMustAlias());
}

//===----------------------------------------------------------------------===//
// ChannelHandleAttr

ChannelHandleAttr StablehloBytecodeInterface::readChannelHandleAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  int64_t handle, type;
  if (failed(reader.readSignedVarInt(handle)) ||
      failed(reader.readSignedVarInt(type)))
    return ChannelHandleAttr();

  return ChannelHandleAttr::get(getContext(), handle, type);
}

void StablehloBytecodeInterface::write(ChannelHandleAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kChannelHandleAttr);
  writer.writeSignedVarInt(attr.getHandle());
  writer.writeSignedVarInt(attr.getType());
}

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr

ComparisonDirectionAttr StablehloBytecodeInterface::readComparisonDirectionAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<ComparisonDirectionAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeComparisonDirection(val); });
}

void StablehloBytecodeInterface::write(ComparisonDirectionAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kComparisonDirectionAttr);
  hlo::bytecode::writeEnumAttribute<ComparisonDirection>(attr, writer);
}

//===----------------------------------------------------------------------===//
// ComparisonTypeAttr

ComparisonTypeAttr StablehloBytecodeInterface::readComparisonTypeAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<ComparisonTypeAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeComparisonType(val); });
}

void StablehloBytecodeInterface::write(ComparisonTypeAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kComparisonTypeAttr);
  hlo::bytecode::writeEnumAttribute<ComparisonType>(attr, writer);
}

//===----------------------------------------------------------------------===//
// ConvDimensionNumbersAttr

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
      failed(reader.readSignedVarInts(outputSpatialDimensions)))
    return ConvDimensionNumbersAttr();

  return ConvDimensionNumbersAttr::get(
      getContext(), inputBatchDimension, inputFeatureDimension,
      inputSpatialDimensions, kernelInputFeatureDimension,
      kernelOutputFeatureDimension, kernelSpatialDimensions,
      outputBatchDimension, outputFeatureDimension, outputSpatialDimensions);
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

//===----------------------------------------------------------------------===//
// DotDimensionNumbersAttr

DotDimensionNumbersAttr StablehloBytecodeInterface::readDotDimensionNumbersAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions;

  if (failed(reader.readSignedVarInts(lhsBatchingDimensions)) ||
      failed(reader.readSignedVarInts(rhsBatchingDimensions)) ||
      failed(reader.readSignedVarInts(lhsContractingDimensions)) ||
      failed(reader.readSignedVarInts(rhsContractingDimensions)))
    return DotDimensionNumbersAttr();

  return DotDimensionNumbersAttr::get(
      getContext(), lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions);
}

void StablehloBytecodeInterface::write(DotDimensionNumbersAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kDotDimensionNumbers);
  writer.writeSignedVarInts(attr.getLhsBatchingDimensions());
  writer.writeSignedVarInts(attr.getRhsBatchingDimensions());
  writer.writeSignedVarInts(attr.getLhsContractingDimensions());
  writer.writeSignedVarInts(attr.getRhsContractingDimensions());
}

//===----------------------------------------------------------------------===//
// FftTypeAttr

FftTypeAttr StablehloBytecodeInterface::readFftTypeAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<FftTypeAttr>(
      reader, getContext(), [](uint32_t val) { return symbolizeFftType(val); });
}
void StablehloBytecodeInterface::write(FftTypeAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kFftTypeAttr);
  hlo::bytecode::writeEnumAttribute<FftType>(attr, writer);
}

//===----------------------------------------------------------------------===//
// GatherDimensionNumbersAttr

GatherDimensionNumbersAttr
StablehloBytecodeInterface::readGatherDimensionNumbersAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> offsetDims, collapsedSliceDims, startIndexMap;
  int64_t indexVectorDim;

  if (failed(reader.readSignedVarInts(offsetDims)) ||
      failed(reader.readSignedVarInts(collapsedSliceDims)) ||
      failed(reader.readSignedVarInts(startIndexMap)) ||
      failed(reader.readSignedVarInt(indexVectorDim)))
    return GatherDimensionNumbersAttr();

  return GatherDimensionNumbersAttr::get(getContext(), offsetDims,
                                         collapsedSliceDims, startIndexMap,
                                         indexVectorDim);
}

void StablehloBytecodeInterface::write(GatherDimensionNumbersAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kGatherDimensionNumbers);
  writer.writeSignedVarInts(attr.getOffsetDims());
  writer.writeSignedVarInts(attr.getCollapsedSliceDims());
  writer.writeSignedVarInts(attr.getStartIndexMap());
  writer.writeSignedVarInt(attr.getIndexVectorDim());
}

//===----------------------------------------------------------------------===//
// OutputOperandAliasAttr

OutputOperandAliasAttr StablehloBytecodeInterface::readOutputOperandAliasAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> outputTupleIndices, operandTupleIndices;
  int64_t operandIndex;

  if (failed(reader.readSignedVarInts(outputTupleIndices)) ||
      failed(reader.readSignedVarInt(operandIndex)) ||
      failed(reader.readSignedVarInts(operandTupleIndices)))
    return OutputOperandAliasAttr();

  return OutputOperandAliasAttr::get(getContext(), outputTupleIndices,
                                     operandIndex, operandTupleIndices);
}

void StablehloBytecodeInterface::write(OutputOperandAliasAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kOutputOperandAlias);
  writer.writeSignedVarInts(attr.getOutputTupleIndices());
  writer.writeSignedVarInt(attr.getOperandIndex());
  writer.writeSignedVarInts(attr.getOperandTupleIndices());
}

//===----------------------------------------------------------------------===//
// PrecisionAttr

PrecisionAttr StablehloBytecodeInterface::readPrecisionAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<PrecisionAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizePrecision(val); });
}

void StablehloBytecodeInterface::write(PrecisionAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kPrecisionAttr);
  hlo::bytecode::writeEnumAttribute<Precision>(attr, writer);
}

//===----------------------------------------------------------------------===//
// RngAlgorithmAttr

RngAlgorithmAttr StablehloBytecodeInterface::readRngAlgorithmAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<RngAlgorithmAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeRngAlgorithm(val); });
}

void StablehloBytecodeInterface::write(RngAlgorithmAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kRngAlgorithmAttr);
  hlo::bytecode::writeEnumAttribute<RngAlgorithm>(attr, writer);
}

//===----------------------------------------------------------------------===//
// RngDistributionAttr

RngDistributionAttr StablehloBytecodeInterface::readRngDistributionAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<RngDistributionAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeRngDistribution(val); });
}

void StablehloBytecodeInterface::write(RngDistributionAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kRngDistributionAttr);
  hlo::bytecode::writeEnumAttribute<RngDistribution>(attr, writer);
}

//===----------------------------------------------------------------------===//
// ScatterDimensionNumbersAttr

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
      failed(reader.readSignedVarInt(indexVectorDim)))
    return ScatterDimensionNumbersAttr();

  return ScatterDimensionNumbersAttr::get(
      getContext(), updateWindowDims, insertedWindowDims,
      scatterDimsToOperandDims, indexVectorDim);
}

void StablehloBytecodeInterface::write(ScatterDimensionNumbersAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kScatterDimensionNumbersAttr);
  writer.writeSignedVarInts(attr.getUpdateWindowDims());
  writer.writeSignedVarInts(attr.getInsertedWindowDims());
  writer.writeSignedVarInts(attr.getScatterDimsToOperandDims());
  writer.writeSignedVarInt(attr.getIndexVectorDim());
}

//===----------------------------------------------------------------------===//
// TransposeAttr

TransposeAttr StablehloBytecodeInterface::readTransposeAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<TransposeAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeTranspose(val); });
}

void StablehloBytecodeInterface::write(TransposeAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kTransposeAttr);
  hlo::bytecode::writeEnumAttribute<Transpose>(attr, writer);
}

//===----------------------------------------------------------------------===//
// TypeExtensionsAttr

TypeExtensionsAttr StablehloBytecodeInterface::readTypeExtensionsAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> bounds;
  if (failed(reader.readSignedVarInts(bounds))) return TypeExtensionsAttr();
  return TypeExtensionsAttr::get(getContext(), bounds);
}

void StablehloBytecodeInterface::write(TypeExtensionsAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kTypeExtensionsAttr);
  writer.writeSignedVarInts(attr.getBounds());
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// TokenType

TokenType StablehloBytecodeInterface::readTokenType(
    DialectBytecodeReader &) const {
  LOG_READ_CALL;
  return TokenType::get(getContext());
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
