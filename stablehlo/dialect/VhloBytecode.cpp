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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
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
//   Called: write(mlir::vhlo::TokenV1Type, mlir::DialectBytecodeWriter ...
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

  ///   ArrayAttr {
  ///     elements: Attribute[]
  ///   }
  kArrayAttr = 16,

  ///   DenseIntOrFPElementsAttr {
  ///     type: Type
  ///     data: blob
  ///   }
  kDenseIntOrFPElementsAttr = 17,

  ///   FlatSymbolRefAttr {
  ///     rootReference: StringAttr
  ///   }
  kFlatSymbolRefAttr = 18,

  ///   FloatAttr {
  ///     type: Type
  ///     value: APFloat
  ///   }
  kFloatAttr = 19,

  ///   IntegerAttr {
  ///     type: Type
  ///     value: APInt
  ///   }
  kIntegerAttr = 20,

  ///   StringAttr {
  ///     value: string
  ///   }
  kStringAttr = 21,

  ///   UnitAttr {
  ///   }
  kUnitAttr = 22,
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

  ///   BFloat16Type {
  ///   }
  kBFloat16Type = 1,

  ///   ComplexType {
  ///     elementType: Type
  ///   }
  kComplexType = 2,

  ///   Float16Type {
  ///   }
  kFloat16Type = 3,

  ///   Float32Type {
  ///   }
  kFloat32Type = 4,

  ///   Float64Type {
  ///   }
  kFloat64Type = 5,

  ///   FunctionType {
  ///     inputs: Type[]
  ///     results: Type[]
  ///   }
  kFunctionType = 6,

  ///   IntegerI1Type {
  ///   }
  kIntegerI1Type = 7,

  ///   IntegerI4Type {
  ///   }
  kIntegerI4Type = 8,

  ///   IntegerI8Type {
  ///   }
  kIntegerI8Type = 9,

  ///   IntegerI16Type {
  ///   }
  kIntegerI16Type = 10,

  ///   IntegerI32Type {
  ///   }
  kIntegerI32Type = 11,

  ///   IntegerI64Type {
  ///   }
  kIntegerI64Type = 12,

  ///   IntegerUI4Type {
  ///   }
  kIntegerUI4Type = 13,

  ///   IntegerUI8Type {
  ///   }
  kIntegerUI8Type = 14,

  ///   IntegerUI16Type {
  ///   }
  kIntegerUI16Type = 15,

  ///   IntegerUI32Type {
  ///   }
  kIntegerUI32Type = 16,

  ///   IntegerUI64Type {
  ///   }
  kIntegerUI64Type = 17,

  ///   IndexType {
  ///   }
  kIndexType = 18,

  ///   RankedTensorType {
  ///     shape: svarint[]
  ///     elementType: Type,
  ///   }
  kRankedTensorType = 19,

  ///   RankedTensorTypeWithEncoding {
  ///     encoding: Attribute
  ///     shape: svarint[]
  ///     elementType: Type
  ///   }
  /// Variant of RankedTensorType with an encoding.
  kRankedTensorTypeWithEncoding = 20,

  ///   TupleType {
  ///     elementTypes: Type[]
  ///   }
  kTupleType = 21,

  ///   UniformQuantizedType {
  ///     flags: varint
  ///     storageType: Type
  ///     expressedType: Type
  ///     scale: APFloat
  ///     zeroPoint: svarint
  ///     storageTypeMin: svarint
  ///     storageTypeMax: svarint
  ///   }
  kUniformQuantizedType = 22,

  ///   UnrankedTensorType {
  ///     elementType: Type
  ///   }
  kUnrankedTensorType = 23,

  ///   WitnessType {
  ///   }
  kWitnessType = 24,
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
  ArgResultAliasV1Attr readArgResultAliasV1Attr(
      DialectBytecodeReader &reader) const;
  ChannelHandleV1Attr readChannelHandleV1Attr(
      DialectBytecodeReader &reader) const;
  ComparisonDirectionV1Attr readComparisonDirectionV1Attr(
      DialectBytecodeReader &reader) const;
  ComparisonTypeV1Attr readComparisonTypeV1Attr(
      DialectBytecodeReader &reader) const;
  ConvDimensionNumbersV1Attr readConvDimensionNumbersV1Attr(
      DialectBytecodeReader &reader) const;
  CustomCallApiVersionV1Attr readCustomCallApiVersionV1Attr(
      DialectBytecodeReader &reader) const;
  DotDimensionNumbersV1Attr readDotDimensionNumbersV1Attr(
      DialectBytecodeReader &reader) const;
  FftTypeV1Attr readFftTypeV1Attr(DialectBytecodeReader &reader) const;
  GatherDimensionNumbersV1Attr readGatherDimensionNumbersV1Attr(
      DialectBytecodeReader &reader) const;
  OutputOperandAliasV1Attr readOutputOperandAliasV1Attr(
      DialectBytecodeReader &reader) const;
  PrecisionV1Attr readPrecisionV1Attr(DialectBytecodeReader &reader) const;
  RngAlgorithmV1Attr readRngAlgorithmV1Attr(
      DialectBytecodeReader &reader) const;
  RngDistributionV1Attr readRngDistributionV1Attr(
      DialectBytecodeReader &reader) const;
  ScatterDimensionNumbersV1Attr readScatterDimensionNumbersV1Attr(
      DialectBytecodeReader &reader) const;
  TransposeV1Attr readTransposeV1Attr(DialectBytecodeReader &reader) const;
  TypeExtensionsV1Attr readTypeExtensionsV1Attr(
      DialectBytecodeReader &reader) const;

  // TO ADD ATTRIBUTE: Include a write method for each attribute in VHLO
  // Ex: void write(SomeAttr attr, DialectBytecodeWriter &writer) const;
  void write(ArgResultAliasV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(ChannelHandleV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(ComparisonDirectionV1Attr attr,
             DialectBytecodeWriter &writer) const;
  void write(ComparisonTypeV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(ConvDimensionNumbersV1Attr attr,
             DialectBytecodeWriter &writer) const;
  void write(CustomCallApiVersionV1Attr attr,
             DialectBytecodeWriter &writer) const;
  void write(DotDimensionNumbersV1Attr attr,
             DialectBytecodeWriter &writer) const;
  void write(FftTypeV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(GatherDimensionNumbersV1Attr attr,
             DialectBytecodeWriter &writer) const;
  void write(OutputOperandAliasV1Attr attr,
             DialectBytecodeWriter &writer) const;
  void write(PrecisionV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(RngAlgorithmV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(RngDistributionV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(ScatterDimensionNumbersV1Attr attr,
             DialectBytecodeWriter &writer) const;
  void write(TransposeV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(TypeExtensionsV1Attr attr, DialectBytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Forked Attributes
  ArrayV1Attr readArrayV1Attr(DialectBytecodeReader &reader) const;
  DenseIntOrFPElementsV1Attr readDenseIntOrFPElementsV1Attr(
      DialectBytecodeReader &reader) const;
  FlatSymbolRefV1Attr readFlatSymbolRefV1Attr(
      DialectBytecodeReader &reader) const;
  FloatV1Attr readFloatV1Attr(DialectBytecodeReader &reader) const;
  IntegerV1Attr readIntegerV1Attr(DialectBytecodeReader &reader) const;
  StringV1Attr readStringV1Attr(DialectBytecodeReader &reader) const;
  // UnitV1Attr readUnitV1Attr(DialectBytecodeReader &reader) const; // inlined

  void write(ArrayV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(DenseIntOrFPElementsV1Attr attr,
             DialectBytecodeWriter &writer) const;
  void write(FlatSymbolRefV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(FloatV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(IntegerV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(StringV1Attr attr, DialectBytecodeWriter &writer) const;
  // void write(UnitV1Attr attr, DialectBytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Types

  // These methods are invoked by superclass when a type from VHLO dialect
  // is encountered.
  Type readType(DialectBytecodeReader &reader) const override;
  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override;

  // TO ADD TYPE: Include a read method for each type in VHLO
  // Ex: SomeType readSomeType(DialectBytecodeReader &reader) const;
  TokenV1Type readTokenV1Type(DialectBytecodeReader &reader) const;

  // TO ADD TYPE: Include a write method for each type in VHLO
  // Ex: void write(SomeType attr, DialectBytecodeWriter &writer) const;
  void write(TokenV1Type type, DialectBytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Forked Types
  ComplexV1Type readComplexType(DialectBytecodeReader &reader) const;
  RankedTensorV1Type readRankedTensorType(DialectBytecodeReader &reader,
                                          bool hasEncoding) const;
  TupleV1Type readTupleType(DialectBytecodeReader &reader) const;
  UniformQuantizedV1Type readUniformQuantizedType(
      DialectBytecodeReader &reader) const;
  UnrankedTensorV1Type readUnrankedTensorType(
      DialectBytecodeReader &reader) const;

  void write(ComplexV1Type type, DialectBytecodeWriter &writer) const;
  void write(RankedTensorV1Type type, DialectBytecodeWriter &writer) const;
  void write(TupleV1Type type, DialectBytecodeWriter &writer) const;
  void write(UniformQuantizedV1Type type, DialectBytecodeWriter &writer) const;
  void write(UnrankedTensorV1Type type, DialectBytecodeWriter &writer) const;
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
      return readArgResultAliasV1Attr(reader);
    case vhlo_encoding::kChannelHandleAttr:
      return readChannelHandleV1Attr(reader);
    case vhlo_encoding::kComparisonDirectionAttr:
      return readComparisonDirectionV1Attr(reader);
    case vhlo_encoding::kComparisonTypeAttr:
      return readComparisonTypeV1Attr(reader);
    case vhlo_encoding::kConvDimensionNumbersAttr:
      return readConvDimensionNumbersV1Attr(reader);
    case vhlo_encoding::kCustomCallApiVersionAttr:
      return readCustomCallApiVersionV1Attr(reader);
    case vhlo_encoding::kDotDimensionNumbers:
      return readDotDimensionNumbersV1Attr(reader);
    case vhlo_encoding::kFftTypeAttr:
      return readFftTypeV1Attr(reader);
    case vhlo_encoding::kGatherDimensionNumbers:
      return readGatherDimensionNumbersV1Attr(reader);
    case vhlo_encoding::kOutputOperandAlias:
      return readOutputOperandAliasV1Attr(reader);
    case vhlo_encoding::kPrecisionAttr:
      return readPrecisionV1Attr(reader);
    case vhlo_encoding::kRngAlgorithmAttr:
      return readRngAlgorithmV1Attr(reader);
    case vhlo_encoding::kRngDistributionAttr:
      return readRngDistributionV1Attr(reader);
    case vhlo_encoding::kScatterDimensionNumbersAttr:
      return readScatterDimensionNumbersV1Attr(reader);
    case vhlo_encoding::kTransposeAttr:
      return readTransposeV1Attr(reader);
    case vhlo_encoding::kTypeExtensionsAttr:
      return readTypeExtensionsV1Attr(reader);
    // Forked Attributes
    case vhlo_encoding::kArrayAttr:
      return readArrayV1Attr(reader);
    case vhlo_encoding::kDenseIntOrFPElementsAttr:
      return readDenseIntOrFPElementsV1Attr(reader);
    case vhlo_encoding::kFlatSymbolRefAttr:
      return readFlatSymbolRefV1Attr(reader);
    case vhlo_encoding::kFloatAttr:
      return readFloatV1Attr(reader);
    case vhlo_encoding::kIntegerAttr:
      return readIntegerV1Attr(reader);
    case vhlo_encoding::kStringAttr:
      return readStringV1Attr(reader);
    case vhlo_encoding::kUnitAttr:
      return UnitV1Attr::get(getContext());
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
      .Case<
          ArgResultAliasV1Attr, ChannelHandleV1Attr, ComparisonDirectionV1Attr,
          ComparisonTypeV1Attr, ConvDimensionNumbersV1Attr,
          CustomCallApiVersionV1Attr, DotDimensionNumbersV1Attr, FftTypeV1Attr,
          GatherDimensionNumbersV1Attr, OutputOperandAliasV1Attr,
          PrecisionV1Attr, RngAlgorithmV1Attr, RngDistributionV1Attr,
          ScatterDimensionNumbersV1Attr, TransposeV1Attr, TypeExtensionsV1Attr>(
          [&](auto attr) {
            LOG_WRITE_CALL;
            write(attr, writer);
            return success();
          })
      .Case<ArrayV1Attr, DenseIntOrFPElementsV1Attr, FlatSymbolRefV1Attr,
            FloatV1Attr, IntegerV1Attr, StringV1Attr>([&](auto attr) {
        LOG_WRITE_CALL;  // Forked attrs
        write(attr, writer);
        return success();
      })
      .Case([&](UnitV1Attr) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kUnitAttr), success();
      })
      .Default([&](Attribute) {
        LOG_NOT_IMPLEMENTED;
        return failure();
      });
}

//===----------------------------------------------------------------------===//
// ArgResultAliasAttr

ArgResultAliasV1Attr VhloBytecodeInterface::readArgResultAliasV1Attr(
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
    return ArgResultAliasV1Attr();
  }
  return ArgResultAliasV1Attr::get(getContext(), argTupleIndices, resultIndex,
                                   resultTupleIndices,
                                   static_cast<bool>(isMustAliasUint));
}

void VhloBytecodeInterface::write(ArgResultAliasV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kArgResultAliasAttr);
  writer.writeSignedVarInts(attr.getArgTupleIndices());
  writer.writeSignedVarInt(attr.getResultIndex());
  writer.writeSignedVarInts(attr.getResultTupleIndices());
  writer.writeVarInt(attr.getIsMustAlias());
}

//===----------------------------------------------------------------------===//
// ChannelHandleAttr

ChannelHandleV1Attr VhloBytecodeInterface::readChannelHandleV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  int64_t handle, type;
  if (failed(reader.readSignedVarInt(handle)) ||
      failed(reader.readSignedVarInt(type))) {
    return ChannelHandleV1Attr();
  }
  return ChannelHandleV1Attr::get(getContext(), handle, type);
}

void VhloBytecodeInterface::write(ChannelHandleV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kChannelHandleAttr);
  writer.writeSignedVarInt(attr.getHandle());
  writer.writeSignedVarInt(attr.getType());
}

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr

ComparisonDirectionV1Attr VhloBytecodeInterface::readComparisonDirectionV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<ComparisonDirectionV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeComparisonDirectionV1(val); });
}

void VhloBytecodeInterface::write(ComparisonDirectionV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kComparisonDirectionAttr);
  hlo::bytecode::writeEnumAttribute<ComparisonDirectionV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// ComparisonTypeAttr

ComparisonTypeV1Attr VhloBytecodeInterface::readComparisonTypeV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<ComparisonTypeV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeComparisonTypeV1(val); });
}

void VhloBytecodeInterface::write(ComparisonTypeV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kComparisonTypeAttr);
  hlo::bytecode::writeEnumAttribute<ComparisonTypeV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// ConvDimensionNumbersAttr

ConvDimensionNumbersV1Attr
VhloBytecodeInterface::readConvDimensionNumbersV1Attr(
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
    return ConvDimensionNumbersV1Attr();
  }

  return ConvDimensionNumbersV1Attr::get(
      getContext(), inputBatchDimension, inputFeatureDimension,
      inputSpatialDimensions, kernelInputFeatureDimension,
      kernelOutputFeatureDimension, kernelSpatialDimensions,
      outputBatchDimension, outputFeatureDimension, outputSpatialDimensions);
}

void VhloBytecodeInterface::write(ConvDimensionNumbersV1Attr attr,
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
// CustomCallApiVersionAttr

CustomCallApiVersionV1Attr
VhloBytecodeInterface::readCustomCallApiVersionV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<CustomCallApiVersionV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeCustomCallApiVersionV1(val); });
}

void VhloBytecodeInterface::write(CustomCallApiVersionV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kCustomCallApiVersionAttr);
  hlo::bytecode::writeEnumAttribute<CustomCallApiVersionV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// DotDimensionNumbersAttr

DotDimensionNumbersV1Attr VhloBytecodeInterface::readDotDimensionNumbersV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions;

  if (failed(reader.readSignedVarInts(lhsBatchingDimensions)) ||
      failed(reader.readSignedVarInts(rhsBatchingDimensions)) ||
      failed(reader.readSignedVarInts(lhsContractingDimensions)) ||
      failed(reader.readSignedVarInts(rhsContractingDimensions))) {
    return DotDimensionNumbersV1Attr();
  }

  return DotDimensionNumbersV1Attr::get(
      getContext(), lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions);
}

void VhloBytecodeInterface::write(DotDimensionNumbersV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kDotDimensionNumbers);
  writer.writeSignedVarInts(attr.getLhsBatchingDimensions());
  writer.writeSignedVarInts(attr.getRhsBatchingDimensions());
  writer.writeSignedVarInts(attr.getLhsContractingDimensions());
  writer.writeSignedVarInts(attr.getRhsContractingDimensions());
}

//===----------------------------------------------------------------------===//
// FftTypeAttr

FftTypeV1Attr VhloBytecodeInterface::readFftTypeV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<FftTypeV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeFftTypeV1(val); });
}
void VhloBytecodeInterface::write(FftTypeV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kFftTypeAttr);
  hlo::bytecode::writeEnumAttribute<FftTypeV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// GatherDimensionNumbersAttr

GatherDimensionNumbersV1Attr
VhloBytecodeInterface::readGatherDimensionNumbersV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> offsetDims, collapsedSliceDims, startIndexMap;
  int64_t indexVectorDim;

  if (failed(reader.readSignedVarInts(offsetDims)) ||
      failed(reader.readSignedVarInts(collapsedSliceDims)) ||
      failed(reader.readSignedVarInts(startIndexMap)) ||
      failed(reader.readSignedVarInt(indexVectorDim))) {
    return GatherDimensionNumbersV1Attr();
  }

  return GatherDimensionNumbersV1Attr::get(getContext(), offsetDims,
                                           collapsedSliceDims, startIndexMap,
                                           indexVectorDim);
}

void VhloBytecodeInterface::write(GatherDimensionNumbersV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kGatherDimensionNumbers);
  writer.writeSignedVarInts(attr.getOffsetDims());
  writer.writeSignedVarInts(attr.getCollapsedSliceDims());
  writer.writeSignedVarInts(attr.getStartIndexMap());
  writer.writeSignedVarInt(attr.getIndexVectorDim());
}

//===----------------------------------------------------------------------===//
// OutputOperandAliasAttr

OutputOperandAliasV1Attr VhloBytecodeInterface::readOutputOperandAliasV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> outputTupleIndices, operandTupleIndices;
  int64_t operandIndex;

  if (failed(reader.readSignedVarInts(outputTupleIndices)) ||
      failed(reader.readSignedVarInt(operandIndex)) ||
      failed(reader.readSignedVarInts(operandTupleIndices))) {
    return OutputOperandAliasV1Attr();
  }
  return OutputOperandAliasV1Attr::get(getContext(), outputTupleIndices,
                                       operandIndex, operandTupleIndices);
}

void VhloBytecodeInterface::write(OutputOperandAliasV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kOutputOperandAlias);
  writer.writeSignedVarInts(attr.getOutputTupleIndices());
  writer.writeSignedVarInt(attr.getOperandIndex());
  writer.writeSignedVarInts(attr.getOperandTupleIndices());
}

//===----------------------------------------------------------------------===//
// PrecisionAttr

PrecisionV1Attr VhloBytecodeInterface::readPrecisionV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<PrecisionV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizePrecisionV1(val); });
}

void VhloBytecodeInterface::write(PrecisionV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kPrecisionAttr);
  hlo::bytecode::writeEnumAttribute<PrecisionV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// RngAlgorithmAttr

RngAlgorithmV1Attr VhloBytecodeInterface::readRngAlgorithmV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<RngAlgorithmV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeRngAlgorithmV1(val); });
}

void VhloBytecodeInterface::write(RngAlgorithmV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kRngAlgorithmAttr);
  hlo::bytecode::writeEnumAttribute<RngAlgorithmV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// RngDistributionAttr

RngDistributionV1Attr VhloBytecodeInterface::readRngDistributionV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<RngDistributionV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeRngDistributionV1(val); });
}

void VhloBytecodeInterface::write(RngDistributionV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kRngDistributionAttr);
  hlo::bytecode::writeEnumAttribute<RngDistributionV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// ScatterDimensionNumbersAttr

ScatterDimensionNumbersV1Attr
VhloBytecodeInterface::readScatterDimensionNumbersV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> updateWindowDims, insertedWindowDims,
      scatterDimsToOperandDims;
  int64_t indexVectorDim;

  if (failed(reader.readSignedVarInts(updateWindowDims)) ||
      failed(reader.readSignedVarInts(insertedWindowDims)) ||
      failed(reader.readSignedVarInts(scatterDimsToOperandDims)) ||
      failed(reader.readSignedVarInt(indexVectorDim))) {
    return ScatterDimensionNumbersV1Attr();
  }

  return ScatterDimensionNumbersV1Attr::get(
      getContext(), updateWindowDims, insertedWindowDims,
      scatterDimsToOperandDims, indexVectorDim);
}

void VhloBytecodeInterface::write(ScatterDimensionNumbersV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kScatterDimensionNumbersAttr);
  writer.writeSignedVarInts(attr.getUpdateWindowDims());
  writer.writeSignedVarInts(attr.getInsertedWindowDims());
  writer.writeSignedVarInts(attr.getScatterDimsToOperandDims());
  writer.writeSignedVarInt(attr.getIndexVectorDim());
}

//===----------------------------------------------------------------------===//
// TransposeAttr

TransposeV1Attr VhloBytecodeInterface::readTransposeV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<TransposeV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeTransposeV1(val); });
}

void VhloBytecodeInterface::write(TransposeV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTransposeAttr);
  hlo::bytecode::writeEnumAttribute<TransposeV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// TypeExtensionsAttr

TypeExtensionsV1Attr VhloBytecodeInterface::readTypeExtensionsV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> bounds;
  if (failed(reader.readSignedVarInts(bounds))) {
    return TypeExtensionsV1Attr();
  }
  return TypeExtensionsV1Attr::get(getContext(), bounds);
}

void VhloBytecodeInterface::write(TypeExtensionsV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTypeExtensionsAttr);
  writer.writeSignedVarInts(attr.getBounds());
}

//===----------------------------------------------------------------------===//
// Forked Attributes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ArrayV1Attr

ArrayV1Attr VhloBytecodeInterface::readArrayV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  SmallVector<Attribute> elements;
  if (failed(reader.readAttributes(elements))) return ArrayV1Attr();
  return ArrayV1Attr::get(getContext(), elements);
}

void VhloBytecodeInterface::write(ArrayV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kArrayAttr);
  writer.writeAttributes(attr.getValue());
}

//===----------------------------------------------------------------------===//
// DenseIntOrFPElementsV1Attr

DenseIntOrFPElementsV1Attr
VhloBytecodeInterface::readDenseIntOrFPElementsV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type type;
  ArrayRef<char> blob;
  if (failed(reader.readType(type)) || failed(reader.readBlob(blob)))
    return DenseIntOrFPElementsV1Attr();
  return DenseIntOrFPElementsV1Attr::get(getContext(), type, blob);
}

void VhloBytecodeInterface::write(DenseIntOrFPElementsV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kDenseIntOrFPElementsAttr);
  writer.writeType(attr.getType());
  writer.writeOwnedBlob(attr.getRawData());
}

//===----------------------------------------------------------------------===//
// FloatV1Attr

namespace {
/// Returns the floating semantics for the given type.
const llvm::fltSemantics &getFloatSemantics(Type type) {
  if (type.isa<BFloat16V1Type>()) return APFloat::BFloat();
  if (type.isa<Float16V1Type>()) return APFloat::IEEEhalf();
  if (type.isa<Float32V1Type>()) return APFloat::IEEEsingle();
  if (type.isa<Float64V1Type>()) return APFloat::IEEEdouble();
  llvm_unreachable("non-floating point type used");
}
}  // namespace

FloatV1Attr VhloBytecodeInterface::readFloatV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type type;
  if (failed(reader.readType(type))) return FloatV1Attr();

  FailureOr<APFloat> value =
      reader.readAPFloatWithKnownSemantics(getFloatSemantics(type));
  if (failed(value)) return FloatV1Attr();

  return FloatV1Attr::get(getContext(), type, *value);
}

void VhloBytecodeInterface::write(FloatV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kFloatAttr);
  writer.writeType(attr.getType());
  writer.writeAPFloatWithKnownSemantics(attr.getValue());
}

//===----------------------------------------------------------------------===//
// FlatSymbolRefV1Attr

FlatSymbolRefV1Attr VhloBytecodeInterface::readFlatSymbolRefV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Attribute rootReference;
  if (failed(reader.readAttribute(rootReference))) return FlatSymbolRefV1Attr();
  return FlatSymbolRefV1Attr::get(getContext(), rootReference);
}

void VhloBytecodeInterface::write(FlatSymbolRefV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kFlatSymbolRefAttr);
  writer.writeAttribute(attr.getRootReference());
}

//===----------------------------------------------------------------------===//
// IntegerV1Attr

namespace {
unsigned getBitWidthForIntegerType(Type type) {
  if (type.isa<IntegerI1V1Type>()) return 1;
  if (type.isa<IntegerI4V1Type>() || type.isa<IntegerUI4V1Type>()) return 4;
  if (type.isa<IntegerI8V1Type>() || type.isa<IntegerUI8V1Type>()) return 8;
  if (type.isa<IntegerI16V1Type>() || type.isa<IntegerUI16V1Type>()) return 16;
  if (type.isa<IntegerI32V1Type>() || type.isa<IntegerUI32V1Type>()) return 32;
  if (type.isa<IntegerI64V1Type>() || type.isa<IntegerUI64V1Type>()) return 64;
  llvm_unreachable("unsupported integer type used in IntegerV1Attr");
}
}  // namespace

IntegerV1Attr VhloBytecodeInterface::readIntegerV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type type;
  if (failed(reader.readType(type))) return IntegerV1Attr();

  // Extract the value storage width from the type.
  unsigned bitWidth;
  if (type.isa<IndexV1Type>()) {
    bitWidth = IndexType::kInternalStorageBitWidth;
  } else {
    bitWidth = getBitWidthForIntegerType(type);
  }

  FailureOr<APInt> value = reader.readAPIntWithKnownWidth(bitWidth);
  if (failed(value)) return IntegerV1Attr();
  return IntegerV1Attr::get(getContext(), type, *value);
}

void VhloBytecodeInterface::write(IntegerV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kIntegerAttr);
  writer.writeType(attr.getType());
  writer.writeAPIntWithKnownWidth(attr.getValue());
}

//===----------------------------------------------------------------------===//
// StringV1Attr

StringV1Attr VhloBytecodeInterface::readStringV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  StringRef string;
  if (failed(reader.readString(string))) return StringV1Attr();
  return StringV1Attr::get(getContext(), string);
}

void VhloBytecodeInterface::write(StringV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kStringAttr);
  writer.writeOwnedString(attr.getValue());
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
      return readTokenV1Type(reader);
    // Forked Types:
    case vhlo_encoding::kBFloat16Type:
      return BFloat16V1Type::get(getContext());
    case vhlo_encoding::kComplexType:
      return readComplexType(reader);
    case vhlo_encoding::kFloat16Type:
      return Float16V1Type::get(getContext());
    case vhlo_encoding::kFloat32Type:
      return Float32V1Type::get(getContext());
    case vhlo_encoding::kFloat64Type:
      return Float64V1Type::get(getContext());
    case vhlo_encoding::kIndexType:
      return IndexV1Type::get(getContext());
    case vhlo_encoding::kIntegerI1Type:
      return IntegerI1V1Type::get(getContext());
    case vhlo_encoding::kIntegerI4Type:
      return IntegerI4V1Type::get(getContext());
    case vhlo_encoding::kIntegerI8Type:
      return IntegerI8V1Type::get(getContext());
    case vhlo_encoding::kIntegerI16Type:
      return IntegerI16V1Type::get(getContext());
    case vhlo_encoding::kIntegerI32Type:
      return IntegerI32V1Type::get(getContext());
    case vhlo_encoding::kIntegerI64Type:
      return IntegerI64V1Type::get(getContext());
    case vhlo_encoding::kIntegerUI4Type:
      return IntegerUI4V1Type::get(getContext());
    case vhlo_encoding::kIntegerUI8Type:
      return IntegerUI8V1Type::get(getContext());
    case vhlo_encoding::kIntegerUI16Type:
      return IntegerUI16V1Type::get(getContext());
    case vhlo_encoding::kIntegerUI32Type:
      return IntegerUI32V1Type::get(getContext());
    case vhlo_encoding::kIntegerUI64Type:
      return IntegerUI64V1Type::get(getContext());
    case vhlo_encoding::kRankedTensorType:
      return readRankedTensorType(reader, /*hasEncoding=*/false);
    case vhlo_encoding::kRankedTensorTypeWithEncoding:
      return readRankedTensorType(reader, /*hasEncoding=*/true);
    case vhlo_encoding::kTupleType:
      return readTupleType(reader);
    case vhlo_encoding::kUniformQuantizedType:
      return readUniformQuantizedType(reader);
    case vhlo_encoding::kUnrankedTensorType:
      return readUnrankedTensorType(reader);
    case vhlo_encoding::kWitnessType:
      return WitnessV1Type::get(getContext());
    default:
      reader.emitError() << "unknown vhlo type code: " << code;
      return Type();
  }
}

// TO ADD TYPE: Update the case selection to include the new type.
LogicalResult VhloBytecodeInterface::writeType(
    Type type, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Type, LogicalResult>(type)
      .Case<TokenV1Type>([&](auto type) {
        LOG_WRITE_CALL;
        write(type, writer);
        return success();
      })
      .Case<ComplexV1Type, RankedTensorV1Type, TupleV1Type,
            UnrankedTensorV1Type, UniformQuantizedV1Type>([&](auto type) {
        LOG_WRITE_CALL;
        return write(type, writer), success();
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
      .Case([&](IndexV1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIndexType), success();
      })
      .Case([&](IntegerI1V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerI1Type), success();
      })
      .Case([&](IntegerI4V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerI4Type), success();
      })
      .Case([&](IntegerI8V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerI8Type), success();
      })
      .Case([&](IntegerI16V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerI16Type), success();
      })
      .Case([&](IntegerI32V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerI32Type), success();
      })
      .Case([&](IntegerI64V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerI64Type), success();
      })
      .Case([&](IntegerUI4V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI4Type), success();
      })
      .Case([&](IntegerUI8V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI8Type), success();
      })
      .Case([&](IntegerUI16V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI16Type), success();
      })
      .Case([&](IntegerUI32V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI32Type), success();
      })
      .Case([&](IntegerUI64V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI64Type), success();
      })
      .Case([&](WitnessV1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kWitnessType), success();
      })
      .Default([&](Type) {
        LOG_NOT_IMPLEMENTED;
        return failure();
      });
}

//===----------------------------------------------------------------------===//
// TokenV1Type

TokenV1Type VhloBytecodeInterface::readTokenV1Type(
    DialectBytecodeReader &) const {
  LOG_READ_CALL;
  return TokenV1Type::get(getContext());
}

void VhloBytecodeInterface::write(TokenV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTokenType);
}

//===----------------------------------------------------------------------===//
// Forked Types
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ComplexV1Type

ComplexV1Type VhloBytecodeInterface::readComplexType(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type elementType;
  if (failed(reader.readType(elementType))) return ComplexV1Type();
  return ComplexV1Type::get(getContext(), elementType);
}

void VhloBytecodeInterface::write(ComplexV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kComplexType);
  writer.writeType(type.getElementType());
}

//===----------------------------------------------------------------------===//
// RankedTensorV1Type

RankedTensorV1Type VhloBytecodeInterface::readRankedTensorType(
    DialectBytecodeReader &reader, bool hasEncoding) const {
  LOG_READ_CALL;
  Attribute encoding;
  if (hasEncoding && failed(reader.readAttribute(encoding)))
    return RankedTensorV1Type();

  SmallVector<int64_t> shape;
  Type elementType;
  if (failed(reader.readSignedVarInts(shape)) ||
      failed(reader.readType(elementType)))
    return RankedTensorV1Type();

  return RankedTensorV1Type::get(getContext(), shape, elementType, encoding);
}

void VhloBytecodeInterface::write(RankedTensorV1Type type,
                                  DialectBytecodeWriter &writer) const {
  if (Attribute encoding = type.getEncoding()) {
    writer.writeVarInt(vhlo_encoding::kRankedTensorTypeWithEncoding);
    writer.writeAttribute(encoding);
  } else {
    writer.writeVarInt(vhlo_encoding::kRankedTensorType);
  }
  writer.writeSignedVarInts(type.getShape());
  writer.writeType(type.getElementType());
}

//===----------------------------------------------------------------------===//
// TupleV1Type

TupleV1Type VhloBytecodeInterface::readTupleType(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  SmallVector<Type> elements;
  if (failed(reader.readTypes(elements))) return TupleV1Type();

  return TupleV1Type::get(getContext(), elements);
}

void VhloBytecodeInterface::write(TupleV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTupleType);
  writer.writeTypes(type.getTypes());
}

//===----------------------------------------------------------------------===//
// UniformQuantizedV1Type

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

  return UniformQuantizedV1Type::get(getContext(), flags, storageType,
                                     expressedType, scale.value(), zeroPoint,
                                     storageTypeMin, storageTypeMax);
}

void VhloBytecodeInterface::write(UniformQuantizedV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kUniformQuantizedType);
  writer.writeVarInt(type.getFlags());
  writer.writeType(type.getStorageType());
  writer.writeType(type.getExpressedType());
  writer.writeAPFloatWithKnownSemantics(APFloat(type.getScale()));
  writer.writeSignedVarInt(type.getZeroPoint());
  writer.writeSignedVarInt(type.getStorageTypeMin());
  writer.writeSignedVarInt(type.getStorageTypeMax());
}

//===----------------------------------------------------------------------===//
// UnrankedTensorV1Type

UnrankedTensorV1Type VhloBytecodeInterface::readUnrankedTensorType(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type elementType;
  if (failed(reader.readType(elementType))) return UnrankedTensorV1Type();

  return UnrankedTensorV1Type::get(getContext(), elementType);
}

void VhloBytecodeInterface::write(UnrankedTensorV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kUnrankedTensorType);
  writer.writeType(type.getElementType());
}

}  // namespace

void addBytecodeInterface(VhloDialect *dialect) {
  dialect->addInterfaces<VhloBytecodeInterface>();
}

}  // namespace vhlo
}  // namespace mlir
