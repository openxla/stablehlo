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

#include "stablehlo/dialect/VhloBytecode.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Diagnostics.h"
#include "stablehlo/dialect/Base.h"  // for readEnumAttribute
#include "stablehlo/dialect/VhloOps.h"

//===----------------------------------------------------------------------===//
// Debug Trace Helpers
//===----------------------------------------------------------------------===//

// Enable logging with flag:
//   stablehlo-opt -debug-only=vhlo-bytecode [...]
//
// Extract after function name, remove namespace.
//   Called: write(mlir::vhlo::TokenType, mlir::DialectBytecodeWriter ...
//   ***Not Implemened: write(...
#define _EXTRACT_AFTER(a, b) \
  llvm::StringRef(a).substr(llvm::StringRef(a).find(b))

#define _LOG_CALL_TO(func)                                                     \
  DEBUG_WITH_TYPE(                                                             \
      "vhlo-bytecode",                                                         \
      llvm::errs() << "Called: " << _EXTRACT_AFTER(LLVM_PRETTY_FUNCTION, func) \
                   << '\n')

#define LOG_WRITE_CALL _LOG_CALL_TO("write")
#define LOG_READ_CALL _LOG_CALL_TO(__func__)
#define LOG_NOT_IMPLEMENTED \
  DEBUG_WITH_TYPE(          \
      "vhlo-bytecode",      \
      llvm::errs() << "***Not Implemented: " << LLVM_PRETTY_FUNCTION << '\n')

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

namespace {
namespace vhlo_encoding {

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

}  // namespace vhlo_encoding
}  // namespace

//===----------------------------------------------------------------------===//
// VhloBytecodeInterface
//===----------------------------------------------------------------------===//

namespace mlir {
namespace vhlo {

namespace {
/// This class implements the bytecode interface for the VHLO dialect.
class VhloBytecodeInterface : public BytecodeDialectInterface {
 public:
  VhloBytecodeInterface(Dialect *dialect) : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  // These methods are invoked by superclass when an attr from VHLO dialect
  // is encountered.
  Attribute readAttribute(DialectBytecodeReader &reader) const override;
  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override;

  // TO ADD ATTRIBUTE: Include a read method for each attribute in VHLO
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

  // TO ADD ATTRIBUTE: Include a write method for each attribute in VHLO
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

  // These methods are invoked by superclass when a type from VHLO dialect
  // is encountered.
  Type readType(DialectBytecodeReader &reader) const override;
  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override;

  // TO ADD TYPE: Include a read method for each type in VHLO
  // Ex: SomeType readSomeType(DialectBytecodeReader &reader) const;
  TokenType readTokenType(DialectBytecodeReader &reader) const;

  // TO ADD TYPE: Include a write method for each type in VHLO
  // Ex: void write(SomeType attr, DialectBytecodeWriter &writer) const;
  void write(TokenType type, DialectBytecodeWriter &writer) const;
};

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// TO ADD ATTRIBUTE: Update the switch to include a branch for the attr.
Attribute VhloBytecodeInterface::readAttribute(
    DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code))) return Attribute();
  switch (code) {
    case vhlo_encoding::kArgResultAliasAttr:
      return readArgResultAliasAttr(reader);
    case vhlo_encoding::kChannelHandleAttr:
      return readChannelHandleAttr(reader);
    case vhlo_encoding::kComparisonDirectionAttr:
      return readComparisonDirectionAttr(reader);
    case vhlo_encoding::kComparisonTypeAttr:
      return readComparisonTypeAttr(reader);
    case vhlo_encoding::kConvDimensionNumbersAttr:
      return readConvDimensionNumbersAttr(reader);
    case vhlo_encoding::kDotDimensionNumbers:
      return readDotDimensionNumbersAttr(reader);
    case vhlo_encoding::kFftTypeAttr:
      return readFftTypeAttr(reader);
    case vhlo_encoding::kGatherDimensionNumbers:
      return readGatherDimensionNumbersAttr(reader);
    case vhlo_encoding::kOutputOperandAlias:
      return readOutputOperandAliasAttr(reader);
    case vhlo_encoding::kPrecisionAttr:
      return readPrecisionAttr(reader);
    case vhlo_encoding::kRngAlgorithmAttr:
      return readRngAlgorithmAttr(reader);
    case vhlo_encoding::kRngDistributionAttr:
      return readRngDistributionAttr(reader);
    case vhlo_encoding::kScatterDimensionNumbersAttr:
      return readScatterDimensionNumbersAttr(reader);
    case vhlo_encoding::kTransposeAttr:
      return readTransposeAttr(reader);
    case vhlo_encoding::kTypeExtensionsAttr:
      return readTypeExtensionsAttr(reader);
    default:
      reader.emitError() << "unknown vhlo attribute code: " << code;
      return Attribute();
  }
}

// TO ADD ATTRIBUTE: Update the case selection to include the new attr.
// If this method returns failure, the string serialization is used in the
// bytecode.
LogicalResult VhloBytecodeInterface::writeAttribute(
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

ArgResultAliasAttr VhloBytecodeInterface::readArgResultAliasAttr(
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

void VhloBytecodeInterface::write(ArgResultAliasAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kArgResultAliasAttr);
  writer.writeSignedVarInts(attr.getArgTupleIndices());
  writer.writeSignedVarInt(attr.getResultIndex());
  writer.writeSignedVarInts(attr.getResultTupleIndices());
  writer.writeVarInt(attr.getIsMustAlias());
}

//===----------------------------------------------------------------------===//
// ChannelHandleAttr

ChannelHandleAttr VhloBytecodeInterface::readChannelHandleAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  int64_t handle, type;
  if (failed(reader.readSignedVarInt(handle)) ||
      failed(reader.readSignedVarInt(type))) {
    return ChannelHandleAttr();
  }
  return ChannelHandleAttr::get(getContext(), handle, type);
}

void VhloBytecodeInterface::write(ChannelHandleAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kChannelHandleAttr);
  writer.writeSignedVarInt(attr.getHandle());
  writer.writeSignedVarInt(attr.getType());
}

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr

ComparisonDirectionAttr VhloBytecodeInterface::readComparisonDirectionAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<ComparisonDirectionAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeComparisonDirection(val); });
}

void VhloBytecodeInterface::write(ComparisonDirectionAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kComparisonDirectionAttr);
  hlo::bytecode::writeEnumAttribute<ComparisonDirection>(attr, writer);
}

//===----------------------------------------------------------------------===//
// ComparisonTypeAttr

ComparisonTypeAttr VhloBytecodeInterface::readComparisonTypeAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<ComparisonTypeAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeComparisonType(val); });
}

void VhloBytecodeInterface::write(ComparisonTypeAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kComparisonTypeAttr);
  hlo::bytecode::writeEnumAttribute<ComparisonType>(attr, writer);
}

//===----------------------------------------------------------------------===//
// ConvDimensionNumbersAttr

ConvDimensionNumbersAttr VhloBytecodeInterface::readConvDimensionNumbersAttr(
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

void VhloBytecodeInterface::write(ConvDimensionNumbersAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kConvDimensionNumbersAttr);
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

DotDimensionNumbersAttr VhloBytecodeInterface::readDotDimensionNumbersAttr(
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

void VhloBytecodeInterface::write(DotDimensionNumbersAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kDotDimensionNumbers);
  writer.writeSignedVarInts(attr.getLhsBatchingDimensions());
  writer.writeSignedVarInts(attr.getRhsBatchingDimensions());
  writer.writeSignedVarInts(attr.getLhsContractingDimensions());
  writer.writeSignedVarInts(attr.getRhsContractingDimensions());
}

//===----------------------------------------------------------------------===//
// FftTypeAttr

FftTypeAttr VhloBytecodeInterface::readFftTypeAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<FftTypeAttr>(
      reader, getContext(), [](uint32_t val) { return symbolizeFftType(val); });
}
void VhloBytecodeInterface::write(FftTypeAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kFftTypeAttr);
  hlo::bytecode::writeEnumAttribute<FftType>(attr, writer);
}

//===----------------------------------------------------------------------===//
// GatherDimensionNumbersAttr

GatherDimensionNumbersAttr
VhloBytecodeInterface::readGatherDimensionNumbersAttr(
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

void VhloBytecodeInterface::write(GatherDimensionNumbersAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kGatherDimensionNumbers);
  writer.writeSignedVarInts(attr.getOffsetDims());
  writer.writeSignedVarInts(attr.getCollapsedSliceDims());
  writer.writeSignedVarInts(attr.getStartIndexMap());
  writer.writeSignedVarInt(attr.getIndexVectorDim());
}

//===----------------------------------------------------------------------===//
// OutputOperandAliasAttr

OutputOperandAliasAttr VhloBytecodeInterface::readOutputOperandAliasAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> outputTupleIndices, operandTupleIndices;
  int64_t operandIndex;

  if (failed(reader.readSignedVarInts(outputTupleIndices)) ||
      failed(reader.readSignedVarInt(operandIndex)) ||
      failed(reader.readSignedVarInts(operandTupleIndices))) {
    return OutputOperandAliasAttr();
  }
  return OutputOperandAliasAttr::get(getContext(), outputTupleIndices,
                                     operandIndex, operandTupleIndices);
}

void VhloBytecodeInterface::write(OutputOperandAliasAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kOutputOperandAlias);
  writer.writeSignedVarInts(attr.getOutputTupleIndices());
  writer.writeSignedVarInt(attr.getOperandIndex());
  writer.writeSignedVarInts(attr.getOperandTupleIndices());
}

//===----------------------------------------------------------------------===//
// PrecisionAttr

PrecisionAttr VhloBytecodeInterface::readPrecisionAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<PrecisionAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizePrecision(val); });
}

void VhloBytecodeInterface::write(PrecisionAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kPrecisionAttr);
  hlo::bytecode::writeEnumAttribute<Precision>(attr, writer);
}

//===----------------------------------------------------------------------===//
// RngAlgorithmAttr

RngAlgorithmAttr VhloBytecodeInterface::readRngAlgorithmAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<RngAlgorithmAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeRngAlgorithm(val); });
}

void VhloBytecodeInterface::write(RngAlgorithmAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kRngAlgorithmAttr);
  hlo::bytecode::writeEnumAttribute<RngAlgorithm>(attr, writer);
}

//===----------------------------------------------------------------------===//
// RngDistributionAttr

RngDistributionAttr VhloBytecodeInterface::readRngDistributionAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<RngDistributionAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeRngDistribution(val); });
}

void VhloBytecodeInterface::write(RngDistributionAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kRngDistributionAttr);
  hlo::bytecode::writeEnumAttribute<RngDistribution>(attr, writer);
}

//===----------------------------------------------------------------------===//
// ScatterDimensionNumbersAttr

ScatterDimensionNumbersAttr
VhloBytecodeInterface::readScatterDimensionNumbersAttr(
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

void VhloBytecodeInterface::write(ScatterDimensionNumbersAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kScatterDimensionNumbersAttr);
  writer.writeSignedVarInts(attr.getUpdateWindowDims());
  writer.writeSignedVarInts(attr.getInsertedWindowDims());
  writer.writeSignedVarInts(attr.getScatterDimsToOperandDims());
  writer.writeSignedVarInt(attr.getIndexVectorDim());
}

//===----------------------------------------------------------------------===//
// TransposeAttr

TransposeAttr VhloBytecodeInterface::readTransposeAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<TransposeAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeTranspose(val); });
}

void VhloBytecodeInterface::write(TransposeAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTransposeAttr);
  hlo::bytecode::writeEnumAttribute<Transpose>(attr, writer);
}

//===----------------------------------------------------------------------===//
// TypeExtensionsAttr

TypeExtensionsAttr VhloBytecodeInterface::readTypeExtensionsAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> bounds;
  if (failed(reader.readSignedVarInts(bounds))) {
    return TypeExtensionsAttr();
  }
  return TypeExtensionsAttr::get(getContext(), bounds);
}

void VhloBytecodeInterface::write(TypeExtensionsAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTypeExtensionsAttr);
  writer.writeSignedVarInts(attr.getBounds());
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// TO ADD TYPE: Update the case selection to include the new type.
Type VhloBytecodeInterface::readType(DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code))) return Type();

  switch (code) {
    case vhlo_encoding::kTokenType:
      return readTokenType(reader);

    default:
      reader.emitError() << "unknown builtin type code: " << code;
      return Type();
  }
}

LogicalResult VhloBytecodeInterface::writeType(
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

TokenType VhloBytecodeInterface::readTokenType(DialectBytecodeReader &) const {
  LOG_READ_CALL;
  return TokenType::get(getContext());
}

void VhloBytecodeInterface::write(TokenType type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTokenType);
}

}  // namespace

void addBytecodeInterface(VhloDialect *dialect) {
  dialect->addInterfaces<VhloBytecodeInterface>();
}

}  // namespace vhlo
}  // namespace mlir
