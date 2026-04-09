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

#include <cstdint>
#include <memory>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"

//===----------------------------------------------------------------------===//
// Debug Trace Helpers
//===----------------------------------------------------------------------===//

// Enable logging with flag:
//   stablehlo-opt -debug-only=stablehlo-bytecode [...]
//
// Extract after function name, remove namespace.
//   Called: write(mlir::stablehlo::TokenType, mlir::DialectBytecodeWriter ...
//   ***Not Implemented: write(...
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

  ///   ArgResultAliasAttr (obsolete)
  // kArgResultAliasAttr = 0,

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

  ///   DotAlgorithmAttr {
  ///     lhsPrecisionType : Type
  ///     rhsPrecisionType : Type
  ///     accumulationType : Type
  ///     lhsComponentCount : svarint
  ///     rhsComponentCount : svarint,
  ///     numPrimitiveOperations : svarint
  ///     allowImpreciseAccumulation : svarint
  ///   }
  kDotAlgorithmAttr = 15,

  // ResultAccuracyModeAttr {
  //   mode: varint (encoded enum)
  // }
  kResultAccuracyModeAttr = 16,

  // ResultAccuracyAttr {
  //   atol: APFloat
  //   rtol: APFloat
  //   ulps: svarint
  // }
  kResultAccuracyAttr = 17,

  // ReplicaGroupMeshAxesAttr {
  //   mesh: Attribute
  //   axes: Attribute[]
  // }
  kReplicaGroupMeshAxesAttr = 18,

  // SubAxisInfoAttr {
  //   preSize: svarint
  //   size: svarint
  // }
  kSubAxisInfoAttr = 19,

  // AxisRefAttr {
  //   name: string
  //   subAxisInfo: Attribute (optional)
  // }
  kAxisRefAttr = 20,

  // MeshAxisAttr {
  //   name: string
  //   size: svarint
  // }
  kMeshAxisAttr = 21,

  // MeshAttr {
  //   axes: Attribute[]
  //   device_ids: DenseIntElementsAttr (optional)
  // }
  kMeshAttr = 22,
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

  ///   FutureType {
  ///     elementType: Type
  ///   }
  kFutureType = 1,
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
  StablehloBytecodeInterface(Dialect* dialect)
      : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  // These methods are invoked by superclass when an attr from StableHLO dialect
  // is encountered.
  Attribute readAttribute(DialectBytecodeReader& reader) const override;
  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter& writer) const override;

  // TO ADD ATTRIBUTE: Include a read method for each attribute in StableHLO
  // Ex: SomeAttr readSomeAttr(DialectBytecodeReader &reader) const;
  ChannelHandleAttr readChannelHandleAttr(DialectBytecodeReader& reader) const;
  ComparisonDirectionAttr readComparisonDirectionAttr(
      DialectBytecodeReader& reader) const;
  ComparisonTypeAttr readComparisonTypeAttr(
      DialectBytecodeReader& reader) const;
  ConvDimensionNumbersAttr readConvDimensionNumbersAttr(
      DialectBytecodeReader& reader) const;
  DotAlgorithmAttr readDotAlgorithmAttr(DialectBytecodeReader& reader) const;
  DotDimensionNumbersAttr readDotDimensionNumbersAttr(
      DialectBytecodeReader& reader) const;
  FftTypeAttr readFftTypeAttr(DialectBytecodeReader& reader) const;
  GatherDimensionNumbersAttr readGatherDimensionNumbersAttr(
      DialectBytecodeReader& reader) const;
  OutputOperandAliasAttr readOutputOperandAliasAttr(
      DialectBytecodeReader& reader) const;
  PrecisionAttr readPrecisionAttr(DialectBytecodeReader& reader) const;
  ResultAccuracyAttr readResultAccuracyAttr(
      DialectBytecodeReader& reader) const;
  ResultAccuracyModeAttr readResultAccuracyModeAttr(
      DialectBytecodeReader& reader) const;
  RngAlgorithmAttr readRngAlgorithmAttr(DialectBytecodeReader& reader) const;
  RngDistributionAttr readRngDistributionAttr(
      DialectBytecodeReader& reader) const;
  ScatterDimensionNumbersAttr readScatterDimensionNumbersAttr(
      DialectBytecodeReader& reader) const;
  TransposeAttr readTransposeAttr(DialectBytecodeReader& reader) const;
  TypeExtensionsAttr readTypeExtensionsAttr(
      DialectBytecodeReader& reader) const;
  ReplicaGroupMeshAxesAttr readReplicaGroupMeshAxesAttr(
      DialectBytecodeReader& reader) const;
  SubAxisInfoAttr readSubAxisInfoAttr(DialectBytecodeReader& reader) const;
  AxisRefAttr readAxisRefAttr(DialectBytecodeReader& reader) const;
  MeshAxisAttr readMeshAxisAttr(DialectBytecodeReader& reader) const;
  MeshAttr readMeshAttr(DialectBytecodeReader& reader) const;

  // TO ADD ATTRIBUTE: Include a write method for each attribute in StableHLO
  // Ex: void write(SomeAttr attr, DialectBytecodeWriter &writer) const;
  void write(ChannelHandleAttr attr, DialectBytecodeWriter& writer) const;
  void write(ComparisonDirectionAttr attr, DialectBytecodeWriter& writer) const;
  void write(ComparisonTypeAttr attr, DialectBytecodeWriter& writer) const;
  void write(ConvDimensionNumbersAttr attr,
             DialectBytecodeWriter& writer) const;
  void write(DotAlgorithmAttr attr, DialectBytecodeWriter& writer) const;
  void write(DotDimensionNumbersAttr attr, DialectBytecodeWriter& writer) const;
  void write(FftTypeAttr attr, DialectBytecodeWriter& writer) const;
  void write(GatherDimensionNumbersAttr attr,
             DialectBytecodeWriter& writer) const;
  void write(OutputOperandAliasAttr attr, DialectBytecodeWriter& writer) const;
  void write(PrecisionAttr attr, DialectBytecodeWriter& writer) const;
  void write(ResultAccuracyAttr attr, DialectBytecodeWriter& writer) const;
  void write(ResultAccuracyModeAttr attr, DialectBytecodeWriter& writer) const;
  void write(RngAlgorithmAttr attr, DialectBytecodeWriter& writer) const;
  void write(RngDistributionAttr attr, DialectBytecodeWriter& writer) const;
  void write(ScatterDimensionNumbersAttr attr,
             DialectBytecodeWriter& writer) const;
  void write(TransposeAttr attr, DialectBytecodeWriter& writer) const;
  void write(TypeExtensionsAttr attr, DialectBytecodeWriter& writer) const;
  void write(ReplicaGroupMeshAxesAttr attr,
             DialectBytecodeWriter& writer) const;
  void write(SubAxisInfoAttr attr, DialectBytecodeWriter& writer) const;
  void write(AxisRefAttr attr, DialectBytecodeWriter& writer) const;
  void write(MeshAxisAttr attr, DialectBytecodeWriter& writer) const;
  void write(MeshAttr attr, DialectBytecodeWriter& writer) const;

  //===--------------------------------------------------------------------===//
  // Types

  // These methods are invoked by superclass when a type from StableHLO dialect
  // is encountered.
  Type readType(DialectBytecodeReader& reader) const override;
  LogicalResult writeType(Type type,
                          DialectBytecodeWriter& writer) const override;

  // TO ADD TYPE: Include a read method for each type in StableHLO
  // Ex: SomeType readSomeType(DialectBytecodeReader &reader) const;
  TokenType readTokenType(DialectBytecodeReader& reader) const;
  FutureType readFutureType(DialectBytecodeReader& reader) const;

  // TO ADD TYPE: Include a write method for each type in StableHLO
  // Ex: void write(SomeType attr, DialectBytecodeWriter &writer) const;
  void write(TokenType type, DialectBytecodeWriter& writer) const;
  void write(FutureType type, DialectBytecodeWriter& writer) const;

  //===--------------------------------------------------------------------===//
  // Version

  std::unique_ptr<DialectVersion> readVersion(
      DialectBytecodeReader& reader) const override final;

  void writeVersion(DialectBytecodeWriter& writer) const override final;
};

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// TO ADD ATTRIBUTE: Update the switch to include a branch for the attr.
Attribute StablehloBytecodeInterface::readAttribute(
    DialectBytecodeReader& reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code))) return Attribute();
  switch (code) {
    case stablehlo_encoding::kChannelHandleAttr:
      return readChannelHandleAttr(reader);
    case stablehlo_encoding::kComparisonDirectionAttr:
      return readComparisonDirectionAttr(reader);
    case stablehlo_encoding::kComparisonTypeAttr:
      return readComparisonTypeAttr(reader);
    case stablehlo_encoding::kConvDimensionNumbersAttr:
      return readConvDimensionNumbersAttr(reader);
    case stablehlo_encoding::kDotAlgorithmAttr:
      return readDotAlgorithmAttr(reader);
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
    case stablehlo_encoding::kResultAccuracyAttr:
      return readResultAccuracyAttr(reader);
    case stablehlo_encoding::kResultAccuracyModeAttr:
      return readResultAccuracyModeAttr(reader);
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
    case stablehlo_encoding::kReplicaGroupMeshAxesAttr:
      return readReplicaGroupMeshAxesAttr(reader);
    case stablehlo_encoding::kSubAxisInfoAttr:
      return readSubAxisInfoAttr(reader);
    case stablehlo_encoding::kAxisRefAttr:
      return readAxisRefAttr(reader);
    case stablehlo_encoding::kMeshAxisAttr:
      return readMeshAxisAttr(reader);
    case stablehlo_encoding::kMeshAttr:
      return readMeshAttr(reader);
    default:
      reader.emitError() << "unknown stablehlo attribute code: " << code;
      return Attribute();
  }
}

// TO ADD ATTRIBUTE: Update the case selection to include the new attr.
// If this method returns failure, the string serialization is used in the
// bytecode.
LogicalResult StablehloBytecodeInterface::writeAttribute(
    Attribute attr, DialectBytecodeWriter& writer) const {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<ChannelHandleAttr, ComparisonDirectionAttr, ComparisonTypeAttr,
            ConvDimensionNumbersAttr, DotAlgorithmAttr, DotDimensionNumbersAttr,
            FftTypeAttr, GatherDimensionNumbersAttr, OutputOperandAliasAttr,
            PrecisionAttr, ResultAccuracyAttr, ResultAccuracyModeAttr,
            RngAlgorithmAttr, RngDistributionAttr, ScatterDimensionNumbersAttr,
            TransposeAttr, TypeExtensionsAttr, ReplicaGroupMeshAxesAttr,
            SubAxisInfoAttr, AxisRefAttr, MeshAxisAttr, MeshAttr>(
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

//===----------------------------------------------------------------------===//
// ChannelHandleAttr

ChannelHandleAttr StablehloBytecodeInterface::readChannelHandleAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  int64_t handle, type;
  if (failed(reader.readSignedVarInt(handle)) ||
      failed(reader.readSignedVarInt(type)))
    return ChannelHandleAttr();

  return ChannelHandleAttr::get(getContext(), handle, type);
}

void StablehloBytecodeInterface::write(ChannelHandleAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kChannelHandleAttr);
  writer.writeSignedVarInt(attr.getHandle());
  writer.writeSignedVarInt(attr.getType());
}

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr

ComparisonDirectionAttr StablehloBytecodeInterface::readComparisonDirectionAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<ComparisonDirectionAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeComparisonDirection(val); });
}

void StablehloBytecodeInterface::write(ComparisonDirectionAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kComparisonDirectionAttr);
  hlo::bytecode::writeEnumAttribute<ComparisonDirection>(attr, writer);
}

//===----------------------------------------------------------------------===//
// ComparisonTypeAttr

ComparisonTypeAttr StablehloBytecodeInterface::readComparisonTypeAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<ComparisonTypeAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeComparisonType(val); });
}

void StablehloBytecodeInterface::write(ComparisonTypeAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kComparisonTypeAttr);
  hlo::bytecode::writeEnumAttribute<ComparisonType>(attr, writer);
}

//===----------------------------------------------------------------------===//
// ConvDimensionNumbersAttr

ConvDimensionNumbersAttr
StablehloBytecodeInterface::readConvDimensionNumbersAttr(
    DialectBytecodeReader& reader) const {
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
                                       DialectBytecodeWriter& writer) const {
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
// DotAlgorithmAttr

DotAlgorithmAttr StablehloBytecodeInterface::readDotAlgorithmAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  Type lhsPrecisionType, rhsPrecisionType, accumulationType;
  int64_t lhsComponentCount, rhsComponentCount, numPrimitiveOperations;
  bool allowImpreciseAccumulation;

  if (failed(reader.readType(lhsPrecisionType)) ||
      failed(reader.readType(rhsPrecisionType)) ||
      failed(reader.readType(accumulationType)) ||
      failed(reader.readSignedVarInt(lhsComponentCount)) ||
      failed(reader.readSignedVarInt(rhsComponentCount)) ||
      failed(reader.readSignedVarInt(numPrimitiveOperations)) ||
      failed(reader.readBool(allowImpreciseAccumulation)))
    return DotAlgorithmAttr();

  return DotAlgorithmAttr::get(getContext(), lhsPrecisionType, rhsPrecisionType,
                               accumulationType, lhsComponentCount,
                               rhsComponentCount, numPrimitiveOperations,
                               allowImpreciseAccumulation);
}

void StablehloBytecodeInterface::write(DotAlgorithmAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kDotAlgorithmAttr);
  writer.writeType(attr.getLhsPrecisionType());
  writer.writeType(attr.getRhsPrecisionType());
  writer.writeType(attr.getAccumulationType());
  writer.writeSignedVarInt(attr.getLhsComponentCount());
  writer.writeSignedVarInt(attr.getRhsComponentCount());
  writer.writeSignedVarInt(attr.getNumPrimitiveOperations());
  writer.writeOwnedBool(attr.getAllowImpreciseAccumulation());
}

//===----------------------------------------------------------------------===//
// DotDimensionNumbersAttr

DotDimensionNumbersAttr StablehloBytecodeInterface::readDotDimensionNumbersAttr(
    DialectBytecodeReader& reader) const {
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
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kDotDimensionNumbers);
  writer.writeSignedVarInts(attr.getLhsBatchingDimensions());
  writer.writeSignedVarInts(attr.getRhsBatchingDimensions());
  writer.writeSignedVarInts(attr.getLhsContractingDimensions());
  writer.writeSignedVarInts(attr.getRhsContractingDimensions());
}

//===----------------------------------------------------------------------===//
// FftTypeAttr

FftTypeAttr StablehloBytecodeInterface::readFftTypeAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<FftTypeAttr>(
      reader, getContext(), [](uint32_t val) { return symbolizeFftType(val); });
}
void StablehloBytecodeInterface::write(FftTypeAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kFftTypeAttr);
  hlo::bytecode::writeEnumAttribute<FftType>(attr, writer);
}

//===----------------------------------------------------------------------===//
// GatherDimensionNumbersAttr

GatherDimensionNumbersAttr
StablehloBytecodeInterface::readGatherDimensionNumbersAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> offsetDims, collapsedSliceDims,
      operandBatchingDims, startIndicesBatchingDims, startIndexMap;
  int64_t indexVectorDim;

  if (failed(reader.readSignedVarInts(offsetDims)) ||
      failed(reader.readSignedVarInts(collapsedSliceDims)) ||
      failed(reader.readSignedVarInts(operandBatchingDims)) ||
      failed(reader.readSignedVarInts(startIndicesBatchingDims)) ||
      failed(reader.readSignedVarInts(startIndexMap)) ||
      failed(reader.readSignedVarInt(indexVectorDim)))
    return GatherDimensionNumbersAttr();

  return GatherDimensionNumbersAttr::get(
      getContext(), offsetDims, collapsedSliceDims, operandBatchingDims,
      startIndicesBatchingDims, startIndexMap, indexVectorDim);
}

void StablehloBytecodeInterface::write(GatherDimensionNumbersAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kGatherDimensionNumbers);
  writer.writeSignedVarInts(attr.getOffsetDims());
  writer.writeSignedVarInts(attr.getCollapsedSliceDims());
  writer.writeSignedVarInts(attr.getOperandBatchingDims());
  writer.writeSignedVarInts(attr.getStartIndicesBatchingDims());
  writer.writeSignedVarInts(attr.getStartIndexMap());
  writer.writeSignedVarInt(attr.getIndexVectorDim());
}

//===----------------------------------------------------------------------===//
// OutputOperandAliasAttr

OutputOperandAliasAttr StablehloBytecodeInterface::readOutputOperandAliasAttr(
    DialectBytecodeReader& reader) const {
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
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kOutputOperandAlias);
  writer.writeSignedVarInts(attr.getOutputTupleIndices());
  writer.writeSignedVarInt(attr.getOperandIndex());
  writer.writeSignedVarInts(attr.getOperandTupleIndices());
}

//===----------------------------------------------------------------------===//
// PrecisionAttr

PrecisionAttr StablehloBytecodeInterface::readPrecisionAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<PrecisionAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizePrecision(val); });
}

void StablehloBytecodeInterface::write(PrecisionAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kPrecisionAttr);
  hlo::bytecode::writeEnumAttribute<Precision>(attr, writer);
}

//===----------------------------------------------------------------------===//
// RngAlgorithmAttr

RngAlgorithmAttr StablehloBytecodeInterface::readRngAlgorithmAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<RngAlgorithmAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeRngAlgorithm(val); });
}

void StablehloBytecodeInterface::write(RngAlgorithmAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kRngAlgorithmAttr);
  hlo::bytecode::writeEnumAttribute<RngAlgorithm>(attr, writer);
}

//===----------------------------------------------------------------------===//
// RngDistributionAttr

RngDistributionAttr StablehloBytecodeInterface::readRngDistributionAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<RngDistributionAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeRngDistribution(val); });
}

void StablehloBytecodeInterface::write(RngDistributionAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kRngDistributionAttr);
  hlo::bytecode::writeEnumAttribute<RngDistribution>(attr, writer);
}

//===----------------------------------------------------------------------===//
// ScatterDimensionNumbersAttr

ScatterDimensionNumbersAttr
StablehloBytecodeInterface::readScatterDimensionNumbersAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> updateWindowDims, insertedWindowDims,
      inputBatchingDims, scatterIndicesBatchingDims, scatterDimsToOperandDims;
  int64_t indexVectorDim;

  if (failed(reader.readSignedVarInts(updateWindowDims)) ||
      failed(reader.readSignedVarInts(insertedWindowDims)) ||
      failed(reader.readSignedVarInts(inputBatchingDims)) ||
      failed(reader.readSignedVarInts(scatterIndicesBatchingDims)) ||
      failed(reader.readSignedVarInts(scatterDimsToOperandDims)) ||
      failed(reader.readSignedVarInt(indexVectorDim)))
    return ScatterDimensionNumbersAttr();

  return ScatterDimensionNumbersAttr::get(
      getContext(), updateWindowDims, insertedWindowDims, inputBatchingDims,
      scatterIndicesBatchingDims, scatterDimsToOperandDims, indexVectorDim);
}

void StablehloBytecodeInterface::write(ScatterDimensionNumbersAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kScatterDimensionNumbersAttr);
  writer.writeSignedVarInts(attr.getUpdateWindowDims());
  writer.writeSignedVarInts(attr.getInsertedWindowDims());
  writer.writeSignedVarInts(attr.getInputBatchingDims());
  writer.writeSignedVarInts(attr.getScatterIndicesBatchingDims());
  writer.writeSignedVarInts(attr.getScatterDimsToOperandDims());
  writer.writeSignedVarInt(attr.getIndexVectorDim());
}

//===----------------------------------------------------------------------===//
// TransposeAttr

TransposeAttr StablehloBytecodeInterface::readTransposeAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<TransposeAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeTranspose(val); });
}

void StablehloBytecodeInterface::write(TransposeAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kTransposeAttr);
  hlo::bytecode::writeEnumAttribute<Transpose>(attr, writer);
}

//===----------------------------------------------------------------------===//
// TypeExtensionsAttr

TypeExtensionsAttr StablehloBytecodeInterface::readTypeExtensionsAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> bounds;
  if (failed(reader.readSignedVarInts(bounds))) return TypeExtensionsAttr();
  return TypeExtensionsAttr::get(getContext(), bounds);
}

void StablehloBytecodeInterface::write(TypeExtensionsAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kTypeExtensionsAttr);
  writer.writeSignedVarInts(attr.getBounds());
}

//===----------------------------------------------------------------------===//
// ReplicaGroupMeshAxesAttr

//===----------------------------------------------------------------------===//
// SubAxisInfoAttr

SubAxisInfoAttr StablehloBytecodeInterface::readSubAxisInfoAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  int64_t preSize, size;
  if (failed(reader.readSignedVarInt(preSize)) ||
      failed(reader.readSignedVarInt(size)))
    return SubAxisInfoAttr();
  return SubAxisInfoAttr::get(getContext(), preSize, size);
}

void StablehloBytecodeInterface::write(SubAxisInfoAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kSubAxisInfoAttr);
  writer.writeSignedVarInt(attr.getPreSize());
  writer.writeSignedVarInt(attr.getSize());
}

//===----------------------------------------------------------------------===//
// AxisRefAttr

AxisRefAttr StablehloBytecodeInterface::readAxisRefAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  StringRef name;
  if (failed(reader.readString(name))) return AxisRefAttr();
  bool hasSubAxisInfo;
  if (failed(reader.readBool(hasSubAxisInfo))) return AxisRefAttr();
  SubAxisInfoAttr subAxisInfo;
  if (hasSubAxisInfo) {
    Attribute attr;
    if (failed(reader.readAttribute(attr))) return AxisRefAttr();
    subAxisInfo = llvm::dyn_cast<SubAxisInfoAttr>(attr);
    if (!subAxisInfo) return AxisRefAttr();
  }
  return AxisRefAttr::get(getContext(), name, subAxisInfo);
}

void StablehloBytecodeInterface::write(AxisRefAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kAxisRefAttr);
  writer.writeOwnedString(attr.getName());
  bool hasSubAxisInfo = (bool)attr.getSubAxisInfo();
  writer.writeOwnedBool(hasSubAxisInfo);
  if (hasSubAxisInfo) {
    writer.writeAttribute(attr.getSubAxisInfo());
  }
}

//===----------------------------------------------------------------------===//
// ReplicaGroupMeshAxesAttr

ReplicaGroupMeshAxesAttr
StablehloBytecodeInterface::readReplicaGroupMeshAxesAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  Attribute meshAttr;
  if (failed(reader.readAttribute(meshAttr))) return ReplicaGroupMeshAxesAttr();

  uint64_t axesSize;
  if (failed(reader.readVarInt(axesSize))) return ReplicaGroupMeshAxesAttr();

  SmallVector<Attribute> axes;
  axes.reserve(axesSize);
  for (uint64_t i = 0; i < axesSize; ++i) {
    Attribute axis;
    if (failed(reader.readAttribute(axis))) return ReplicaGroupMeshAxesAttr();
    axes.push_back(axis);
  }

  return ReplicaGroupMeshAxesAttr::get(getContext(), meshAttr,
                                       ArrayAttr::get(getContext(), axes));
}

void StablehloBytecodeInterface::write(ReplicaGroupMeshAxesAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kReplicaGroupMeshAxesAttr);
  writer.writeAttribute(attr.getMesh());

  auto axes = attr.getAxes();
  writer.writeVarInt(axes.size());
  for (auto axis : axes) {
    writer.writeAttribute(axis);
  }
}

//===----------------------------------------------------------------------===//
// MeshAxisAttr

MeshAxisAttr StablehloBytecodeInterface::readMeshAxisAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  StringRef name;
  int64_t size;
  if (failed(reader.readString(name)) || failed(reader.readSignedVarInt(size)))
    return MeshAxisAttr();
  return MeshAxisAttr::get(getContext(), name, size);
}

void StablehloBytecodeInterface::write(MeshAxisAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kMeshAxisAttr);
  writer.writeOwnedString(attr.getName());
  writer.writeSignedVarInt(attr.getSize());
}

//===----------------------------------------------------------------------===//
// MeshAttr

MeshAttr StablehloBytecodeInterface::readMeshAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  Attribute axesAttr;
  if (failed(reader.readAttribute(axesAttr))) return MeshAttr();
  auto axesArrayAttr = llvm::dyn_cast<ArrayAttr>(axesAttr);
  if (!axesArrayAttr) return MeshAttr();

  SmallVector<stablehlo::MeshAxisAttr> axes;
  for (auto attr : axesArrayAttr) {
    auto axisAttr = llvm::dyn_cast<stablehlo::MeshAxisAttr>(attr);
    if (!axisAttr) return MeshAttr();
    axes.push_back(axisAttr);
  }

  bool hasDeviceIds;
  if (failed(reader.readBool(hasDeviceIds))) return MeshAttr();
  DenseIntElementsAttr deviceIds;
  if (hasDeviceIds) {
    Attribute deviceIdsAttr;
    if (failed(reader.readAttribute(deviceIdsAttr))) return MeshAttr();
    deviceIds = llvm::dyn_cast<DenseIntElementsAttr>(deviceIdsAttr);
    if (!deviceIds) return MeshAttr();
  }

  return MeshAttr::get(getContext(), axes, deviceIds);
}

void StablehloBytecodeInterface::write(MeshAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kMeshAttr);
  SmallVector<Attribute> axesAttrs;
  for (auto axis : attr.getAxes()) axesAttrs.push_back(axis);
  writer.writeAttribute(ArrayAttr::get(getContext(), axesAttrs));

  bool hasDeviceIds = (bool)attr.getDeviceIds();
  writer.writeOwnedBool(hasDeviceIds);
  if (hasDeviceIds) {
    writer.writeAttribute(attr.getDeviceIds());
  }
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// TO ADD TYPE: Update the case selection to include the new type.
Type StablehloBytecodeInterface::readType(DialectBytecodeReader& reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code))) return Type();

  switch (code) {
    case stablehlo_encoding::kTokenType:
      return readTokenType(reader);

    case stablehlo_encoding::kFutureType:
      return readFutureType(reader);

    default:
      reader.emitError() << "unknown builtin type code: " << code;
      return Type();
  }
}

LogicalResult StablehloBytecodeInterface::writeType(
    Type type, DialectBytecodeWriter& writer) const {
  return TypeSwitch<Type, LogicalResult>(type)
      .Case<TokenType, FutureType>([&](auto type) {
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
    DialectBytecodeReader&) const {
  LOG_READ_CALL;
  return TokenType::get(getContext());
}

void StablehloBytecodeInterface::write(TokenType type,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kTokenType);
}

//===----------------------------------------------------------------------===//
// FutureType

FutureType StablehloBytecodeInterface::readFutureType(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<Type> elementTypes;
  if (failed(reader.readTypes(elementTypes))) return FutureType();
  return FutureType::get(getContext(), elementTypes);
}

void StablehloBytecodeInterface::write(FutureType type,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kFutureType);
  writer.writeTypes(type.getTypes());
}

std::unique_ptr<DialectVersion> StablehloBytecodeInterface::readVersion(
    DialectBytecodeReader& reader) const {
  uint64_t major, minor, patch;
  if (failed(reader.readVarInt(major)) || failed(reader.readVarInt(minor)) ||
      failed(reader.readVarInt(patch)))
    return nullptr;

  auto version = std::make_unique<StablehloDialectVersion>(
      /*major=*/major, /*minor=*/minor, /*patch=*/patch);
  if (version && StablehloDialectVersion::getCurrentVersion() < *version) {
    // Note: dialect bytecode reader does not expose emitWarning.
    // TODO(jpienaar): Update when it does.
    mlir::emitWarning(mlir::UnknownLoc::get(getContext()))
        << "reading newer dialect than supported";
    return nullptr;
  }

  return version;
}

void StablehloBytecodeInterface::writeVersion(
    DialectBytecodeWriter& writer) const {
  if (auto version = cast<StablehloDialect>(getDialect())->getVersion()) {
    writer.writeVarInt(static_cast<uint64_t>(version->getMajor()));
    writer.writeVarInt(static_cast<uint64_t>(version->getMinor()));
    writer.writeVarInt(static_cast<uint64_t>(version->getPatch()));
  }
}

//===----------------------------------------------------------------------===//
// ResultAccuracyModeAttr

ResultAccuracyModeAttr StablehloBytecodeInterface::readResultAccuracyModeAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<ResultAccuracyModeAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeResultAccuracyMode(val); });
}

void StablehloBytecodeInterface::write(ResultAccuracyModeAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kResultAccuracyModeAttr);
  hlo::bytecode::writeEnumAttribute<ResultAccuracyMode>(attr, writer);
}

//===----------------------------------------------------------------------===//
// ResultAccuracyAttr

ResultAccuracyAttr StablehloBytecodeInterface::readResultAccuracyAttr(
    DialectBytecodeReader& reader) const {
  LOG_READ_CALL;
  FailureOr<APFloat> atol;
  FailureOr<APFloat> rtol;
  int64_t ulps;
  ResultAccuracyModeAttr mode;
  if (failed(atol =
                 reader.readAPFloatWithKnownSemantics(APFloat::IEEEdouble())) ||
      failed(rtol =
                 reader.readAPFloatWithKnownSemantics(APFloat::IEEEdouble())) ||
      failed(reader.readSignedVarInt(ulps)) ||
      failed(reader.readAttribute(mode))) {
    mlir::emitWarning(mlir::UnknownLoc::get(getContext()))
        << "failed to read APFloat for atol";
    return ResultAccuracyAttr();
  }
  return ResultAccuracyAttr::get(getContext(), *atol, *rtol, ulps, mode);
}

void StablehloBytecodeInterface::write(ResultAccuracyAttr attr,
                                       DialectBytecodeWriter& writer) const {
  writer.writeVarInt(stablehlo_encoding::kResultAccuracyAttr);
  writer.writeAPFloatWithKnownSemantics(attr.getAtol());
  writer.writeAPFloatWithKnownSemantics(attr.getRtol());
  writer.writeSignedVarInt(attr.getUlps());
  writer.writeAttribute(attr.getMode());
}

}  // namespace

void addBytecodeInterface(StablehloDialect* dialect) {
  dialect->addInterfaces<StablehloBytecodeInterface>();
}
}  // namespace stablehlo
}  // namespace mlir
