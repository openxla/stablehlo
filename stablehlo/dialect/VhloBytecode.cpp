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

#include <cassert>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
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

#define DEBUG_TYPE "vhlo-bytecode"

#define _LOG_CALL_TO(func)                                              \
  LLVM_DEBUG(llvm::errs() << "Called: "                                 \
                          << _EXTRACT_AFTER(LLVM_PRETTY_FUNCTION, func) \
                          << '\n')

#define LOG_WRITE_CALL _LOG_CALL_TO("write")
#define LOG_READ_CALL _LOG_CALL_TO(__func__)
#define LOG_NOT_IMPLEMENTED                                                 \
  LLVM_DEBUG(llvm::errs() << "***Not Implemented: " << LLVM_PRETTY_FUNCTION \
                          << '\n')

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

  ///   CustomCallApiVersionAttr
  ///     value: varint (encoded enum)
  ///   }
  kCustomCallApiVersionAttr = 15,

  // Gap in numbers to separate StableHLO types from forked types //

  ///   StringAttr {
  ///     value: string
  ///   }
  kStringAttr = 20,

  ///   FlatSymbolRefAttr {
  ///     rootReference: StringAttr
  ///   }
  /// A variant of SymbolRefAttr with no leaf references.
  kFlatSymbolRefAttr = 21,

  ///   IntegerAttr {
  ///     type: Type
  ///     value: APInt,
  ///   }
  kIntegerAttr = 22,

  ///   FloatAttr {
  ///     type: FloatType
  ///     value: APFloat
  ///   }
  kFloatAttr = 23,

  ///   UnitAttr {
  ///   }
  kUnitAttr = 24,

  ///   ArrayAttr {
  ///     elements: Attribute[]
  ///   }
  ///
  kArrayAttr = 25,

  ///   DenseIntOrFPElementsAttr {
  ///     type: ShapedType,
  ///     data: blob
  ///   }
  kDenseIntOrFPElementsAttr = 26,
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

  // Gap in numbers to separate StableHLO types from forked types //

  ///   IntegerType {
  ///     widthAndSignedness: varint // (width << 2) | (signedness)
  ///   }
  ///
  kIntegerType = 5,

  ///   IndexType {
  ///   }
  ///
  kIndexType = 6,

  ///   FunctionType {
  ///     inputs: Type[],
  ///     results: Type[]
  ///   }
  ///
  // kFunctionType = 3,

  ///   BFloat16Type {
  ///   }
  ///
  kBFloat16Type = 7,

  ///   Float16Type {
  ///   }
  ///
  kFloat16Type = 8,

  ///   Float32Type {
  ///   }
  ///
  kFloat32Type = 9,

  ///   Float64Type {
  ///   }
  ///
  kFloat64Type = 10,

  ///   ComplexType {
  ///     elementType: Type
  ///   }
  ///
  kComplexType = 11,

  ///   UniformQuantizedType {
  ///     flags: varint
  ///     storageType: Type
  ///     expressedType: Type
  ///     scale: APFloat
  ///     zeroPoint: svarint
  ///     storageTypeMin: svarint
  ///     storageTypeMax: svarint
  ///   }
  ///
  kUniformQuantizedType = 12,

  ///   RankedTensorType {
  ///     shape: svarint[],
  ///     elementType: Type,
  ///   }
  ///
  kRankedTensorType = 13,

  ///   RankedTensorTypeWithEncoding {
  ///     encoding: Attribute,
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  /// Variant of RankedTensorType with an encoding.
  kRankedTensorTypeWithEncoding = 14,

  ///   TupleType {
  ///     elementTypes: Type[]
  ///   }
  kTupleType = 15,

  ///   UnrankedTensorType {
  ///     elementType: Type
  ///   }
  ///
  kUnrankedTensorType = 16,

  ///   WitnessType {
  ///   }
  kWitnessType = 17,
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
  CustomCallApiVersionAttr readCustomCallApiVersionAttr(
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
  void write(CustomCallApiVersionAttr attr,
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
  // Wrapped / Forked Attributes
  LogicalResult writeWrappedAttribute(WrappedAttr attr,
                                      DialectBytecodeWriter &writer) const;
  FloatV1Attr readFloatAttr(DialectBytecodeReader &reader) const;
  WrappedAttr readStringAttr(DialectBytecodeReader &reader) const;
  WrappedAttr readIntegerAttr(DialectBytecodeReader &reader) const;
  FlatSymbolRefV1Attr readFlatSymbolRefAttr(
      DialectBytecodeReader &reader) const;

  DenseIntOrFPElementsV1Attr readDenseIntOrFPElementsAttr(
      DialectBytecodeReader &reader) const;
  // WrappedAttr readUnitAttr(DialectBytecodeReader &reader) const; // inlined
  WrappedAttr readArrayAttr(DialectBytecodeReader &reader) const;

  void write(FloatV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(FlatSymbolRefV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(IntegerAttr attr, DialectBytecodeWriter &writer) const;
  void write(StringAttr attr, DialectBytecodeWriter &writer) const;

  void write(DenseIntOrFPElementsV1Attr attr,
             DialectBytecodeWriter &writer) const;
  void write(ArrayAttr attr, DialectBytecodeWriter &writer) const;
  void write(UnitAttr attr, DialectBytecodeWriter &writer) const;

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

  //===--------------------------------------------------------------------===//
  // Wrapped Types
  LogicalResult writeWrappedType(WrappedType type,
                                 DialectBytecodeWriter &writer) const;

  WrappedType readRankedTensorType(DialectBytecodeReader &reader,
                                   bool hasEncoding) const;
  WrappedType readTupleType(DialectBytecodeReader &reader) const;
  WrappedType readUnrankedTensorType(DialectBytecodeReader &reader) const;

  void write(RankedTensorType type, DialectBytecodeWriter &writer) const;
  void write(TupleType type, DialectBytecodeWriter &writer) const;
  void write(UnrankedTensorType type, DialectBytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Element Types
  ComplexV1Type readComplexType(DialectBytecodeReader &reader) const;
  IntegerV1Type readIntegerType(DialectBytecodeReader &reader) const;
  UniformQuantizedV1Type readUniformQuantizedType(
      DialectBytecodeReader &reader) const;

  void write(ComplexV1Type type, DialectBytecodeWriter &writer) const;
  void write(IntegerV1Type type, DialectBytecodeWriter &writer) const;
  void write(UniformQuantizedV1Type type, DialectBytecodeWriter &writer) const;
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
    case vhlo_encoding::kArrayAttr:
      return readArrayAttr(reader);
    case vhlo_encoding::kChannelHandleAttr:
      return readChannelHandleAttr(reader);
    case vhlo_encoding::kComparisonDirectionAttr:
      return readComparisonDirectionAttr(reader);
    case vhlo_encoding::kComparisonTypeAttr:
      return readComparisonTypeAttr(reader);
    case vhlo_encoding::kConvDimensionNumbersAttr:
      return readConvDimensionNumbersAttr(reader);
    case vhlo_encoding::kCustomCallApiVersionAttr:
      return readCustomCallApiVersionAttr(reader);
    case vhlo_encoding::kDenseIntOrFPElementsAttr:
      return readDenseIntOrFPElementsAttr(reader);
    case vhlo_encoding::kDotDimensionNumbers:
      return readDotDimensionNumbersAttr(reader);
    case vhlo_encoding::kFftTypeAttr:
      return readFftTypeAttr(reader);
    case vhlo_encoding::kFlatSymbolRefAttr:
      return readFlatSymbolRefAttr(reader);
    case vhlo_encoding::kFloatAttr:
      return readFloatAttr(reader);
    case vhlo_encoding::kGatherDimensionNumbers:
      return readGatherDimensionNumbersAttr(reader);
    case vhlo_encoding::kIntegerAttr:
      return readIntegerAttr(reader);
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
    case vhlo_encoding::kStringAttr:
      return readStringAttr(reader);
    case vhlo_encoding::kTransposeAttr:
      return readTransposeAttr(reader);
    case vhlo_encoding::kTypeExtensionsAttr:
      return readTypeExtensionsAttr(reader);
    case vhlo_encoding::kUnitAttr:
      return WrappedAttr::get(getContext(), UnitAttr::get(getContext()));
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

CustomCallApiVersionAttr VhloBytecodeInterface::readCustomCallApiVersionAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<CustomCallApiVersionAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeCustomCallApiVersion(val); });
}
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

//===----------------------------------------------------------------------===//
// Attributes: Writer

// TO ADD ATTRIBUTE: Update the case selection to include the new attr.
// If this method returns failure, the string serialization is used in the
// bytecode.
LogicalResult VhloBytecodeInterface::writeAttribute(
    Attribute attr, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<ArgResultAliasAttr, ComparisonDirectionAttr, ComparisonTypeAttr,
            ConvDimensionNumbersAttr, ChannelHandleAttr,
            CustomCallApiVersionAttr, DotDimensionNumbersAttr, FftTypeAttr,
            GatherDimensionNumbersAttr, OutputOperandAliasAttr, PrecisionAttr,
            RngAlgorithmAttr, RngDistributionAttr, ScatterDimensionNumbersAttr,
            TransposeAttr, TypeExtensionsAttr>([&](auto attr) {
        LOG_WRITE_CALL;
        write(attr, writer);
        return success();
      })
      .Case<FloatV1Attr, DenseIntOrFPElementsV1Attr, FlatSymbolRefV1Attr>(
          [&](auto attr) {
            LOG_WRITE_CALL;  // Forked attrs
            write(attr, writer);
            return success();
          })
      .Case<WrappedAttr>([&](WrappedAttr attr) {
        LOG_WRITE_CALL;
        return writeWrappedAttribute(attr, writer);
      })
      .Default([&](Attribute attr) {
        LOG_NOT_IMPLEMENTED;
        LLVM_DEBUG(llvm::dbgs() << attr << '\n');
        return failure();
      });
}

void VhloBytecodeInterface::write(ArgResultAliasAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kArgResultAliasAttr);
  writer.writeSignedVarInts(attr.getArgTupleIndices());
  writer.writeSignedVarInt(attr.getResultIndex());
  writer.writeSignedVarInts(attr.getResultTupleIndices());
  writer.writeVarInt(attr.getIsMustAlias());
}

void VhloBytecodeInterface::write(ChannelHandleAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kChannelHandleAttr);
  writer.writeSignedVarInt(attr.getHandle());
  writer.writeSignedVarInt(attr.getType());
}

void VhloBytecodeInterface::write(ComparisonDirectionAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kComparisonDirectionAttr);
  hlo::bytecode::writeEnumAttribute<ComparisonDirection>(attr, writer);
}

void VhloBytecodeInterface::write(ComparisonTypeAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kComparisonTypeAttr);
  hlo::bytecode::writeEnumAttribute<ComparisonType>(attr, writer);
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

void VhloBytecodeInterface::write(CustomCallApiVersionAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kCustomCallApiVersionAttr);
  hlo::bytecode::writeEnumAttribute<CustomCallApiVersion>(attr, writer);
}

void VhloBytecodeInterface::write(DotDimensionNumbersAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kDotDimensionNumbers);
  writer.writeSignedVarInts(attr.getLhsBatchingDimensions());
  writer.writeSignedVarInts(attr.getRhsBatchingDimensions());
  writer.writeSignedVarInts(attr.getLhsContractingDimensions());
  writer.writeSignedVarInts(attr.getRhsContractingDimensions());
}

void VhloBytecodeInterface::write(FftTypeAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kFftTypeAttr);
  hlo::bytecode::writeEnumAttribute<FftType>(attr, writer);
}

void VhloBytecodeInterface::write(GatherDimensionNumbersAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kGatherDimensionNumbers);
  writer.writeSignedVarInts(attr.getOffsetDims());
  writer.writeSignedVarInts(attr.getCollapsedSliceDims());
  writer.writeSignedVarInts(attr.getStartIndexMap());
  writer.writeSignedVarInt(attr.getIndexVectorDim());
}

void VhloBytecodeInterface::write(OutputOperandAliasAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kOutputOperandAlias);
  writer.writeSignedVarInts(attr.getOutputTupleIndices());
  writer.writeSignedVarInt(attr.getOperandIndex());
  writer.writeSignedVarInts(attr.getOperandTupleIndices());
}

void VhloBytecodeInterface::write(PrecisionAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kPrecisionAttr);
  hlo::bytecode::writeEnumAttribute<Precision>(attr, writer);
}

void VhloBytecodeInterface::write(RngAlgorithmAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kRngAlgorithmAttr);
  hlo::bytecode::writeEnumAttribute<RngAlgorithm>(attr, writer);
}

void VhloBytecodeInterface::write(RngDistributionAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kRngDistributionAttr);
  hlo::bytecode::writeEnumAttribute<RngDistribution>(attr, writer);
}

void VhloBytecodeInterface::write(ScatterDimensionNumbersAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kScatterDimensionNumbersAttr);
  writer.writeSignedVarInts(attr.getUpdateWindowDims());
  writer.writeSignedVarInts(attr.getInsertedWindowDims());
  writer.writeSignedVarInts(attr.getScatterDimsToOperandDims());
  writer.writeSignedVarInt(attr.getIndexVectorDim());
}

void VhloBytecodeInterface::write(TransposeAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTransposeAttr);
  hlo::bytecode::writeEnumAttribute<Transpose>(attr, writer);
}

void VhloBytecodeInterface::write(TypeExtensionsAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTypeExtensionsAttr);
  writer.writeSignedVarInts(attr.getBounds());
}

//===----------------------------------------------------------------------===//
// Attributes: Wrapped

namespace {
template <typename AttrOrType>
bool assertFromVhlo(AttrOrType val) {
  if (val.getDialect().getNamespace() != "vhlo") {
    LLVM_DEBUG(llvm::dbgs() << "Not vhlo: " << val << '\n');
    llvm_unreachable("All types and attributes must be VHLO for bytecode.");
  }
  return true;  // return a value for llvm::all_of use
}
}  // namespace

LogicalResult VhloBytecodeInterface::writeWrappedAttribute(
    WrappedAttr attr, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Attribute, LogicalResult>(attr.getData())
      .Case<IntegerAttr, StringAttr, ArrayAttr>([&](auto attr) {
        LOG_WRITE_CALL;
        write(attr, writer);
        return success();
      })
      .Case<UnitAttr>([&](UnitAttr) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kUnitAttr), success();
      })
      .Default([&]([[maybe_unused]] Attribute attr) {
        LLVM_DEBUG(llvm::dbgs() << "Unsupported element type " << attr << '\n');
        llvm_unreachable("Cannot wrap unsupported VHLO attribute.");
        return failure();
      });
}

WrappedAttr VhloBytecodeInterface::readIntegerAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type type;
  if (failed(reader.readType(type))) return WrappedAttr();
  assertFromVhlo(type);

  // Extract the value storage width from the type.
  unsigned bitWidth;
  if (auto intType = type.dyn_cast<IntegerV1Type>()) {
    bitWidth = intType.getValue().getWidth();
    type = intType.getValue();
  } else if (type.isa<IndexV1Type>()) {
    bitWidth = IndexType::kInternalStorageBitWidth;
    type = IndexType::get(getContext());
  } else {
    reader.emitError()
        << "expected integer or index type for IntegerAttr, but got: " << type;
    return WrappedAttr();
  }

  FailureOr<APInt> value = reader.readAPIntWithKnownWidth(bitWidth);
  if (failed(value)) return WrappedAttr();
  return WrappedAttr::get(getContext(), IntegerAttr::get(type, *value));
}

namespace {
/// Returns the floating semantics for the given type.
const llvm::fltSemantics &getFloatSemantics(Type type) {
  // if (isa<Float8E5M2Type>())
  //   return APFloat::Float8E5M2();
  // if (isa<Float8E4M3FNType>())
  //   return APFloat::Float8E4M3FN();
  if (type.isa<BFloat16V1Type>()) return APFloat::BFloat();
  if (type.isa<Float16V1Type>()) return APFloat::IEEEhalf();
  if (type.isa<Float32V1Type>()) return APFloat::IEEEsingle();
  if (type.isa<Float64V1Type>()) return APFloat::IEEEdouble();
  llvm_unreachable("non-floating point type used");
}
}  // namespace

FloatV1Attr VhloBytecodeInterface::readFloatAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type type;
  if (failed(reader.readType(type))) return FloatV1Attr();
  FailureOr<APFloat> value =
      reader.readAPFloatWithKnownSemantics(getFloatSemantics(type));
  if (failed(value)) return FloatV1Attr();

  assertFromVhlo(type);
  return FloatV1Attr::get(getContext(), type, *value);
}

FlatSymbolRefV1Attr VhloBytecodeInterface::readFlatSymbolRefAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Attribute rootReference;
  if (failed(reader.readAttribute(rootReference))) return FlatSymbolRefV1Attr();

  assertFromVhlo(rootReference);
  return FlatSymbolRefV1Attr::get(getContext(), rootReference);
}

WrappedAttr VhloBytecodeInterface::readStringAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  StringRef string;
  if (failed(reader.readString(string))) return WrappedAttr();
  return WrappedAttr::get(getContext(),
                          StringAttr::get(string, NoneType::get(getContext())));
}

DenseIntOrFPElementsV1Attr VhloBytecodeInterface::readDenseIntOrFPElementsAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type type;
  ArrayRef<char> blob;
  if (failed(reader.readType(type)) || failed(reader.readBlob(blob)))
    return DenseIntOrFPElementsV1Attr();

  assertFromVhlo(type);
  return DenseIntOrFPElementsV1Attr::get(getContext(), type, blob);
}

WrappedAttr VhloBytecodeInterface::readArrayAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  SmallVector<Attribute> elements;
  if (failed(reader.readAttributes(elements))) return WrappedAttr();

  llvm::all_of(elements, assertFromVhlo<Attribute>);
  return WrappedAttr::get(getContext(), ArrayAttr::get(getContext(), elements));
}

void VhloBytecodeInterface::write(ArrayAttr attr,
                                  DialectBytecodeWriter &writer) const {
  llvm::all_of(attr.getValue(), assertFromVhlo<Attribute>);
  writer.writeVarInt(vhlo_encoding::kArrayAttr);
  writer.writeAttributes(attr.getValue());
}

void VhloBytecodeInterface::write(IntegerAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kIntegerAttr);
  LLVM_DEBUG(llvm::dbgs() << "IntegerAttr: " << attr << '\n');
  if (attr.getType().isa<IndexType>()) {
    writer.writeType(IndexV1Type::get(getContext()));
  } else {
    assert(attr.getType().isa<IntegerType>());
    // Wrap integer type in vhlo::IntegerV1Type
    writer.writeType(
        IntegerV1Type::get(getContext(), attr.getType().cast<IntegerType>()));
  }
  writer.writeAPIntWithKnownWidth(attr.getValue());
}

void VhloBytecodeInterface::write(FloatV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  assertFromVhlo(attr.getType());
  writer.writeVarInt(vhlo_encoding::kFloatAttr);
  writer.writeType(attr.getType());
  writer.writeAPFloatWithKnownSemantics(attr.getValue());
}

void VhloBytecodeInterface::write(FlatSymbolRefV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  assertFromVhlo(attr.getRootReference());
  writer.writeVarInt(vhlo_encoding::kFlatSymbolRefAttr);
  writer.writeAttribute(attr.getRootReference());
}

void VhloBytecodeInterface::write(StringAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kStringAttr);
  writer.writeOwnedString(attr.getValue());
}

void VhloBytecodeInterface::write(DenseIntOrFPElementsV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  assertFromVhlo(attr.getType());
  writer.writeVarInt(vhlo_encoding::kDenseIntOrFPElementsAttr);
  writer.writeType(attr.getType());
  writer.writeOwnedBlob(attr.getRawData());
}

//===----------------------------------------------------------------------===//
// Types: Reader

// TO ADD TYPE: Update the case selection to include the new type.
Type VhloBytecodeInterface::readType(DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code))) return Type();

  switch (code) {
    case vhlo_encoding::kTokenType:
      return readTokenType(reader);
    case vhlo_encoding::kRankedTensorType:
      return readRankedTensorType(reader, /*hasEncoding=*/false);
    case vhlo_encoding::kRankedTensorTypeWithEncoding:
      return readRankedTensorType(reader, /*hasEncoding=*/true);
    case vhlo_encoding::kTupleType:
      return readTupleType(reader);
    case vhlo_encoding::kUnrankedTensorType:
      return readUnrankedTensorType(reader);
    case vhlo_encoding::kWitnessType:
      return WrappedType::get(getContext(),
                              shape::WitnessType::get(getContext()));
    case vhlo_encoding::kIntegerType:
      return readIntegerType(reader);
    case vhlo_encoding::kIndexType:
      return IndexV1Type::get(getContext());
    case vhlo_encoding::kBFloat16Type:
      return BFloat16V1Type::get(getContext());
    case vhlo_encoding::kFloat16Type:
      return Float16V1Type::get(getContext());
    case vhlo_encoding::kFloat32Type:
      return Float32V1Type::get(getContext());
    case vhlo_encoding::kFloat64Type:
      return Float64V1Type::get(getContext());
    case vhlo_encoding::kComplexType:
      return readComplexType(reader);
    case vhlo_encoding::kUniformQuantizedType:
      return readUniformQuantizedType(reader);
    default:
      reader.emitError() << "unknown vhlo type code: " << code;
      return Type();
  }
}


WrappedType VhloBytecodeInterface::readRankedTensorType(
    DialectBytecodeReader &reader, bool hasEncoding) const {
  LOG_READ_CALL;
  Attribute encoding;
  if (hasEncoding && failed(reader.readAttribute(encoding)))
    return WrappedType();
  SmallVector<int64_t> shape;
  Type elementType;
  if (failed(reader.readSignedVarInts(shape)) ||
      failed(reader.readType(elementType)))
    return WrappedType();

  if (hasEncoding) assertFromVhlo(encoding);
  assertFromVhlo(elementType);
  return WrappedType::get(getContext(),
                          RankedTensorType::get(shape, elementType, encoding));
}

WrappedType VhloBytecodeInterface::readTupleType(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  SmallVector<Type> elements;
  if (failed(reader.readTypes(elements))) return WrappedType();

  llvm::all_of(elements, assertFromVhlo<Type>);
  return WrappedType::get(getContext(), TupleType::get(getContext(), elements));
}

WrappedType VhloBytecodeInterface::readUnrankedTensorType(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type elementType;
  if (failed(reader.readType(elementType))) return WrappedType();

  assertFromVhlo(elementType);
  return WrappedType::get(getContext(), UnrankedTensorType::get(elementType));
}

ComplexV1Type VhloBytecodeInterface::readComplexType(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type elementType;
  if (failed(reader.readType(elementType))) return ComplexV1Type();
  assertFromVhlo(elementType);
  return ComplexV1Type::get(getContext(), elementType);
}

IntegerV1Type VhloBytecodeInterface::readIntegerType(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  uint64_t encoding;
  if (failed(reader.readVarInt(encoding))) return IntegerV1Type();
  return IntegerV1Type::get(
      getContext(),
      IntegerType::get(
          getContext(), encoding >> 2,
          static_cast<IntegerType::SignednessSemantics>(encoding & 0x3)));
}

UniformQuantizedV1Type VhloBytecodeInterface::readUniformQuantizedType(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  uint64_t flags;
  Type storageType, expressedType;
  FailureOr<APFloat> scale;
  int64_t zeroPoint, storageTypeMin, storageTypeMax;
  if (failed(reader.readVarInt(flags)) ||
      failed(reader.readType(storageType)) ||
      failed(reader.readType(expressedType)) ||
      failed(scale = reader.readAPFloatWithKnownSemantics(
                 llvm::APFloat::IEEEdouble())) ||
      failed(reader.readSignedVarInt(zeroPoint)) ||
      failed(reader.readSignedVarInt(storageTypeMin)) ||
      failed(reader.readSignedVarInt(storageTypeMax)))
    return reader.emitError("invalid UniformQuantizedType"),
           UniformQuantizedV1Type();

  assertFromVhlo(storageType);
  assertFromVhlo(expressedType);
  return UniformQuantizedV1Type::get(getContext(), flags, storageType,
                                     expressedType, scale.value(), zeroPoint,
                                     storageTypeMin, storageTypeMax);
}

//===----------------------------------------------------------------------===//
// Types: Writer

// TO ADD TYPE: Update the case selection to include the new type.
LogicalResult VhloBytecodeInterface::writeType(
    Type type, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Type, LogicalResult>(type)
      .Case<TokenType>([&](auto type) {
        LOG_WRITE_CALL;
        write(type, writer);
        return success();
      })
      .Case<WrappedType>([&](WrappedType type) {
        LOG_WRITE_CALL;
        return writeWrappedType(type, writer);
      })
      .Case<ComplexV1Type, IntegerV1Type, UniformQuantizedV1Type>(
          [&](auto type) {
            LOG_WRITE_CALL;
            return write(type, writer), success();
          })
      .Case([&](IndexV1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIndexType), success();
      })
      .Case([&](BFloat16V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kBFloat16Type), success();
      })
      .Case([&](Float16V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kFloat16Type), success();
      })
      .Case([&](Float32V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kFloat32Type), success();
      })
      .Case([&](Float64V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kFloat64Type), success();
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

///////////////////
// Wrapped Types //
///////////////////

LogicalResult VhloBytecodeInterface::writeWrappedType(
    WrappedType type, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Type, LogicalResult>(type.getData())
      .Case<RankedTensorType, UnrankedTensorType, TupleType>([&](auto type) {
        LOG_WRITE_CALL;
        write(type, writer);
        return success();
      })
      .Case([&](shape::WitnessType) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kWitnessType), success();
      })
      .Default([&](Type) {
        LOG_NOT_IMPLEMENTED;
        LLVM_DEBUG(llvm::dbgs() << "Unsupported element type " << type << '\n');
        llvm_unreachable("Cannot wrap unsupported VHLO type.");
        return failure();
      });
}

void VhloBytecodeInterface::write(RankedTensorType type,
                                  DialectBytecodeWriter &writer) const {
  assertFromVhlo(type.getElementType());
  if (Attribute encoding = type.getEncoding()) {
    assertFromVhlo(encoding);
    writer.writeVarInt(vhlo_encoding::kRankedTensorTypeWithEncoding);
    writer.writeAttribute(encoding);
  } else {
    writer.writeVarInt(vhlo_encoding::kRankedTensorType);
  }
  writer.writeSignedVarInts(type.getShape());
  writer.writeType(type.getElementType());
}

void VhloBytecodeInterface::write(TupleType type,
                                  DialectBytecodeWriter &writer) const {
  llvm::all_of(type.getTypes(), assertFromVhlo<Type>);
  writer.writeVarInt(vhlo_encoding::kTupleType);
  writer.writeTypes(type.getTypes());
}

void VhloBytecodeInterface::write(UnrankedTensorType type,
                                  DialectBytecodeWriter &writer) const {
  assertFromVhlo(type.getElementType());
  writer.writeVarInt(vhlo_encoding::kUnrankedTensorType);
  writer.writeType(type.getElementType());
}

void VhloBytecodeInterface::write(ComplexV1Type type,
                                  DialectBytecodeWriter &writer) const {
  assertFromVhlo(type.getElementType());
  writer.writeVarInt(vhlo_encoding::kComplexType);
  writer.writeType(type.getElementType());
}

void VhloBytecodeInterface::write(IntegerV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kIntegerType);
  writer.writeVarInt((type.getValue().getWidth() << 2) |
                     type.getValue().getSignedness());
}

void VhloBytecodeInterface::write(UniformQuantizedV1Type type,
                                  DialectBytecodeWriter &writer) const {
  assertFromVhlo(type.getStorageType());
  assertFromVhlo(type.getExpressedType());
  writer.writeVarInt(vhlo_encoding::kUniformQuantizedType);
  writer.writeVarInt(type.getFlags());
  writer.writeType(type.getStorageType());
  writer.writeType(type.getExpressedType());
  writer.writeAPFloatWithKnownSemantics(APFloat(type.getScale()));
  writer.writeSignedVarInt(type.getZeroPoint());
  writer.writeSignedVarInt(type.getStorageTypeMin());
  writer.writeSignedVarInt(type.getStorageTypeMax());
}

}  // namespace

void addBytecodeInterface(VhloDialect *dialect) {
  dialect->addInterfaces<VhloBytecodeInterface>();
}

}  // namespace vhlo
}  // namespace mlir
