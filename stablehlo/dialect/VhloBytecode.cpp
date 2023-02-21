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
#include "mlir/IR/Attributes.h"
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

  ///   ArgResultAliasV1Attr {
  ///     argTupleIndices: svarint[]
  ///     resultIndex: svarint
  ///     resultTupleIndices: svarint[]
  ///     isMustAlias: varint
  ///   }
  kArgResultAliasV1Attr = 0,

  ///   ArrayV1Attr {
  ///     elements: Attribute[]
  ///   }
  kArrayV1Attr = 1,

  ///   BooleanV1Attr {
  ///     value: varint
  ///   }
  kBooleanV1Attr = 2,

  ///   ComparisonDirectionV1Attr
  ///     value: varint (encoded enum)
  ///   }
  kComparisonDirectionV1Attr = 3,

  ///   ComparisonTypeV1Attr
  ///     value: varint (encoded enum)
  ///   }
  kComparisonTypeV1Attr = 4,

  ///   CustomCallApiVersionV1Attr
  ///     value: varint (encoded enum)
  ///   }
  kCustomCallApiVersionV1Attr = 5,

  ///   DictionaryV1Attr {
  ///     attrs: <Attribute, Attribute>[]
  ///   }
  kDictionaryV1Attr = 6,

  ///   FftTypeV1Attr
  ///     value: varint (encoded enum)
  ///   }
  kFftTypeV1Attr = 7,

  ///   FloatV1Attr {
  ///     type: Type
  ///     value: APFloat
  ///   }
  kFloatV1Attr = 8,

  ///   IntegerV1Attr {
  ///     type: Type
  ///     value: APInt
  ///   }
  kIntegerV1Attr = 9,

  ///   OutputOperandAliasV1Attr {
  ///     outputTupleIndices: svarint[]
  ///     operandIndex : svarint
  ///     operandTupleIndices: svarint[]
  ///   }
  kOutputOperandAliasV1Attr = 10,

  ///   PrecisionV1Attr {
  ///     value: varint (encoded enum)
  ///   }
  kPrecisionV1Attr = 11,

  ///   RngAlgorithmV1Attr {
  ///     value: varint (encoded enum)
  ///   }
  kRngAlgorithmV1Attr = 12,

  ///   RngDistributionV1Attr {
  ///     value: varint (encoded enum)
  ///   }
  kRngDistributionV1Attr = 13,

  ///   StringV1Attr {
  ///     value: string
  ///   }
  kStringV1Attr = 14,

  ///   TensorV1Attr {
  ///     type: Type
  ///     data: blob
  ///   }
  kTensorV1Attr = 15,

  ///   TransposeV1Attr {
  ///     value: varint (encoded enum)
  ///   }
  kTransposeV1Attr = 16,

  ///   TypeV1Attr {
  ///     value: Type
  ///   }
  kTypeV1Attr = 17,

  ///   TypeExtensionsV1Attr {
  ///     bounds : svarint[]
  ///   }
  kTypeExtensionsV1Attr = 18,
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

  ///   BooleanV1Type {
  ///   }
  kBooleanV1Type = 0,

  ///   ComplexV1Type {
  ///     elementType: Type
  ///   }
  kComplexV1Type = 1,

  ///   FloatBF16V1Type {
  ///   }
  kFloatBF16V1Type = 2,

  ///   FloatF16V1Type {
  ///   }
  kFloatF16V1Type = 3,

  ///   FloatF32V1Type {
  ///   }
  kFloatF32V1Type = 4,

  ///   FloatF64V1Type {
  ///   }
  kFloatF64V1Type = 5,

  ///   FloatF8E4M3FNV1Type {
  ///   }
  kFloatF8E4M3FNV1Type = 6,

  ///   FloatF8E5M2V1Type {
  ///   }
  kFloatF8E5M2V1Type = 7,

  ///   FunctionV1Type {
  ///     inputs: Type[]
  ///     outputs: Type[]
  ///   }
  kFunctionV1Type = 8,

  ///   IndexV1Type {
  ///   }
  kIndexV1Type = 9,

  ///   IntegerSI4V1Type {
  ///   }
  kIntegerSI4V1Type = 10,

  ///   IntegerSI8V1Type {
  ///   }
  kIntegerSI8V1Type = 11,

  ///   IntegerSI16V1Type {
  ///   }
  kIntegerSI16V1Type = 12,

  ///   IntegerSI32V1Type {
  ///   }
  kIntegerSI32V1Type = 13,

  ///   IntegerSI64V1Type {
  ///   }
  kIntegerSI64V1Type = 14,

  ///   IntegerUI4V1Type {
  ///   }
  kIntegerUI4V1Type = 15,

  ///   IntegerUI8V1Type {
  ///   }
  kIntegerUI8V1Type = 16,

  ///   IntegerUI16V1Type {
  ///   }
  kIntegerUI16V1Type = 17,

  ///   IntegerUI32V1Type {
  ///   }
  kIntegerUI32V1Type = 18,

  ///   IntegerUI64V1Type {
  ///   }
  kIntegerUI64V1Type = 19,

  ///   RankedTensorV1Type {
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  kRankedTensorV1Type = 20,

  ///   RankedTensorV1TypeWithEncoding {
  ///     encoding: Attribute,
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  kRankedTensorV1TypeWithEncoding = 21,

  ///   TokenV1Type {
  ///   }
  kTokenV1Type = 22,

  ///   TupleV1Type {
  ///     elementTypes: Type[]
  ///   }
  kTupleV1Type = 23,

  ///   UniformQuantizedV1Type {
  ///     flags: varint
  ///     storageType: Type
  ///     expressedType: Type
  ///     scale: APFloat
  ///     zeroPoint: svarint
  ///     storageTypeMin: svarint
  ///     storageTypeMax: svarint
  ///   }
  kUniformQuantizedV1Type = 24,

  ///   UnrankedTensorV1Type {
  ///     elementType: Type
  ///   }
  kUnrankedTensorV1Type = 25,

  ///   WitnessV1Type {
  ///   }
  kWitnessV1Type = 26,
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
  ArrayV1Attr readArrayV1Attr(DialectBytecodeReader &reader) const;
  BooleanV1Attr readBooleanV1Attr(DialectBytecodeReader &reader) const;
  ComparisonDirectionV1Attr readComparisonDirectionV1Attr(
      DialectBytecodeReader &reader) const;
  ComparisonTypeV1Attr readComparisonTypeV1Attr(
      DialectBytecodeReader &reader) const;
  CustomCallApiVersionV1Attr readCustomCallApiVersionV1Attr(
      DialectBytecodeReader &reader) const;
  DictionaryV1Attr readDictionaryV1Attr(DialectBytecodeReader &reader) const;
  FftTypeV1Attr readFftTypeV1Attr(DialectBytecodeReader &reader) const;
  FloatV1Attr readFloatV1Attr(DialectBytecodeReader &reader) const;
  IntegerV1Attr readIntegerV1Attr(DialectBytecodeReader &reader) const;
  OutputOperandAliasV1Attr readOutputOperandAliasV1Attr(
      DialectBytecodeReader &reader) const;
  PrecisionV1Attr readPrecisionV1Attr(DialectBytecodeReader &reader) const;
  RngAlgorithmV1Attr readRngAlgorithmV1Attr(
      DialectBytecodeReader &reader) const;
  RngDistributionV1Attr readRngDistributionV1Attr(
      DialectBytecodeReader &reader) const;
  StringV1Attr readStringV1Attr(DialectBytecodeReader &reader) const;
  TensorV1Attr readTensorV1Attr(DialectBytecodeReader &reader) const;
  TransposeV1Attr readTransposeV1Attr(DialectBytecodeReader &reader) const;
  TypeV1Attr readTypeV1Attr(DialectBytecodeReader &reader) const;
  TypeExtensionsV1Attr readTypeExtensionsV1Attr(
      DialectBytecodeReader &reader) const;

  // TO ADD ATTRIBUTE: Include a write method for each attribute in VHLO
  // Ex: void write(SomeAttr attr, DialectBytecodeWriter &writer) const;
  void write(ArgResultAliasV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(ArrayV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(BooleanV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(ComparisonDirectionV1Attr attr,
             DialectBytecodeWriter &writer) const;
  void write(ComparisonTypeV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(CustomCallApiVersionV1Attr attr,
             DialectBytecodeWriter &writer) const;
  void write(DictionaryV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(FftTypeV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(FloatV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(IntegerV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(OutputOperandAliasV1Attr attr,
             DialectBytecodeWriter &writer) const;
  void write(PrecisionV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(RngAlgorithmV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(RngDistributionV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(StringV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(TensorV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(TransposeV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(TypeV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(TypeExtensionsV1Attr attr, DialectBytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Types

  // These methods are invoked by superclass when a type from VHLO dialect
  // is encountered.
  Type readType(DialectBytecodeReader &reader) const override;
  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override;

  // TO ADD TYPE: Include a read method for each type in VHLO
  // Ex: SomeType readSomeType(DialectBytecodeReader &reader) const;
  ComplexV1Type readComplexV1Type(DialectBytecodeReader &reader) const;
  FunctionV1Type readFunctionV1Type(DialectBytecodeReader &reader) const;
  RankedTensorV1Type readRankedTensorV1Type(DialectBytecodeReader &reader,
                                            bool hasEncoding) const;
  TokenV1Type readTokenV1Type(DialectBytecodeReader &reader) const;
  TupleV1Type readTupleV1Type(DialectBytecodeReader &reader) const;
  UniformQuantizedV1Type readUniformQuantizedV1Type(
      DialectBytecodeReader &reader) const;
  UnrankedTensorV1Type readUnrankedTensorV1Type(
      DialectBytecodeReader &reader) const;

  // TO ADD TYPE: Include a write method for each type in VHLO
  // Ex: void write(SomeType attr, DialectBytecodeWriter &writer) const;
  void write(ComplexV1Type type, DialectBytecodeWriter &writer) const;
  void write(FunctionV1Type type, DialectBytecodeWriter &writer) const;
  void write(RankedTensorV1Type type, DialectBytecodeWriter &writer) const;
  void write(TokenV1Type type, DialectBytecodeWriter &writer) const;
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
    case vhlo_encoding::kArgResultAliasV1Attr:
      return readArgResultAliasV1Attr(reader);
    case vhlo_encoding::kArrayV1Attr:
      return readArrayV1Attr(reader);
    case vhlo_encoding::kBooleanV1Attr:
      return readBooleanV1Attr(reader);
    case vhlo_encoding::kComparisonDirectionV1Attr:
      return readComparisonDirectionV1Attr(reader);
    case vhlo_encoding::kComparisonTypeV1Attr:
      return readComparisonTypeV1Attr(reader);
    case vhlo_encoding::kCustomCallApiVersionV1Attr:
      return readCustomCallApiVersionV1Attr(reader);
    case vhlo_encoding::kDictionaryV1Attr:
      return readDictionaryV1Attr(reader);
    case vhlo_encoding::kFftTypeV1Attr:
      return readFftTypeV1Attr(reader);
    case vhlo_encoding::kFloatV1Attr:
      return readFloatV1Attr(reader);
    case vhlo_encoding::kIntegerV1Attr:
      return readIntegerV1Attr(reader);
    case vhlo_encoding::kOutputOperandAliasV1Attr:
      return readOutputOperandAliasV1Attr(reader);
    case vhlo_encoding::kPrecisionV1Attr:
      return readPrecisionV1Attr(reader);
    case vhlo_encoding::kRngAlgorithmV1Attr:
      return readRngAlgorithmV1Attr(reader);
    case vhlo_encoding::kRngDistributionV1Attr:
      return readRngDistributionV1Attr(reader);
    case vhlo_encoding::kStringV1Attr:
      return readStringV1Attr(reader);
    case vhlo_encoding::kTensorV1Attr:
      return readTensorV1Attr(reader);
    case vhlo_encoding::kTransposeV1Attr:
      return readTransposeV1Attr(reader);
    case vhlo_encoding::kTypeV1Attr:
      return readTypeV1Attr(reader);
    case vhlo_encoding::kTypeExtensionsV1Attr:
      return readTypeExtensionsV1Attr(reader);
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
      .Case<ArgResultAliasV1Attr, ArrayV1Attr, BooleanV1Attr,
            ComparisonDirectionV1Attr, ComparisonTypeV1Attr,
            CustomCallApiVersionV1Attr, DictionaryV1Attr, FftTypeV1Attr,
            FloatV1Attr, IntegerV1Attr, OutputOperandAliasV1Attr,
            PrecisionV1Attr, RngAlgorithmV1Attr, RngDistributionV1Attr,
            StringV1Attr, TensorV1Attr, TransposeV1Attr, TypeV1Attr,
            TypeExtensionsV1Attr>([&](auto attr) {
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
// ArgResultAliasV1Attr
//===----------------------------------------------------------------------===//

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
      failed(reader.readVarInt(isMustAliasUint)))
    return ArgResultAliasV1Attr();
  return ArgResultAliasV1Attr::get(getContext(), argTupleIndices, resultIndex,
                                   resultTupleIndices,
                                   static_cast<bool>(isMustAliasUint));
}

void VhloBytecodeInterface::write(ArgResultAliasV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kArgResultAliasV1Attr);
  writer.writeSignedVarInts(attr.getArgTupleIndices());
  writer.writeSignedVarInt(attr.getResultIndex());
  writer.writeSignedVarInts(attr.getResultTupleIndices());
  writer.writeVarInt(attr.getIsMustAlias());
}

//===----------------------------------------------------------------------===//
// ArrayV1Attr
//===----------------------------------------------------------------------===//

ArrayV1Attr VhloBytecodeInterface::readArrayV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  SmallVector<Attribute> elements;
  if (failed(reader.readAttributes(elements))) return ArrayV1Attr();
  return ArrayV1Attr::get(getContext(), elements);
}

void VhloBytecodeInterface::write(ArrayV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kArrayV1Attr);
  writer.writeAttributes(attr.getValue());
}

//===----------------------------------------------------------------------===//
// BooleanV1Attr
//===----------------------------------------------------------------------===//

BooleanV1Attr VhloBytecodeInterface::readBooleanV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  uint64_t int_value;
  if (failed(reader.readVarInt(int_value))) return BooleanV1Attr();
  if (int_value != 0 && int_value != 1) {
    reader.emitError() << "unsupported value: " << int_value;
    return BooleanV1Attr();
  }
  return BooleanV1Attr::get(getContext(), int_value == 1);
}

void VhloBytecodeInterface::write(BooleanV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kBooleanV1Attr);
  writer.writeVarInt(attr.getValue() ? 1 : 0);
}

//===----------------------------------------------------------------------===//
// ComparisonDirectionV1Attr
//===----------------------------------------------------------------------===//

ComparisonDirectionV1Attr VhloBytecodeInterface::readComparisonDirectionV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<ComparisonDirectionV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeComparisonDirectionV1(val); });
}

void VhloBytecodeInterface::write(ComparisonDirectionV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kComparisonDirectionV1Attr);
  hlo::bytecode::writeEnumAttribute<ComparisonDirectionV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// ComparisonTypeV1Attr
//===----------------------------------------------------------------------===//

ComparisonTypeV1Attr VhloBytecodeInterface::readComparisonTypeV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<ComparisonTypeV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeComparisonTypeV1(val); });
}

void VhloBytecodeInterface::write(ComparisonTypeV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kComparisonTypeV1Attr);
  hlo::bytecode::writeEnumAttribute<ComparisonTypeV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// CustomCallApiVersionV1Attr
//===----------------------------------------------------------------------===//

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
  writer.writeVarInt(vhlo_encoding::kCustomCallApiVersionV1Attr);
  hlo::bytecode::writeEnumAttribute<CustomCallApiVersionV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// DictionaryV1Attr
//===----------------------------------------------------------------------===//

DictionaryV1Attr VhloBytecodeInterface::readDictionaryV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  auto readNamedAttr = [&]() -> FailureOr<std::pair<Attribute, Attribute>> {
    Attribute name;
    Attribute value;
    if (failed(reader.readAttribute(name)) ||
        failed(reader.readAttribute(value)))
      return failure();
    return {{name, value}};
  };
  SmallVector<std::pair<Attribute, Attribute>> attrs;
  if (failed(reader.readList(attrs, readNamedAttr))) return DictionaryV1Attr();

  return DictionaryV1Attr::get(getContext(), attrs);
}

void VhloBytecodeInterface::write(DictionaryV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kDictionaryV1Attr);
  writer.writeList(attr.getValue(), [&](auto attrPair) {
    writer.writeAttribute(attrPair.first);
    writer.writeAttribute(attrPair.second);
  });
}

//===----------------------------------------------------------------------===//
// FftTypeV1Attr
//===----------------------------------------------------------------------===//

FftTypeV1Attr VhloBytecodeInterface::readFftTypeV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<FftTypeV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeFftTypeV1(val); });
}
void VhloBytecodeInterface::write(FftTypeV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kFftTypeV1Attr);
  hlo::bytecode::writeEnumAttribute<FftTypeV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// FloatV1Attr
//===----------------------------------------------------------------------===//

namespace {
/// Returns the floating semantics for the given type.
const llvm::fltSemantics &getFloatSemantics(Type type) {
  if (type.isa<FloatBF16V1Type>()) return APFloat::BFloat();
  if (type.isa<FloatF16V1Type>()) return APFloat::IEEEhalf();
  if (type.isa<FloatF32V1Type>()) return APFloat::IEEEsingle();
  if (type.isa<FloatF64V1Type>()) return APFloat::IEEEdouble();
  if (type.isa<FloatF8E4M3FNV1Type>()) return APFloat::Float8E4M3FN();
  if (type.isa<FloatF8E5M2V1Type>()) return APFloat::Float8E5M2();
  llvm::report_fatal_error("unsupported floating-point type");
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
  writer.writeVarInt(vhlo_encoding::kFloatV1Attr);
  writer.writeType(attr.getType());
  writer.writeAPFloatWithKnownSemantics(attr.getValue());
}

//===----------------------------------------------------------------------===//
// IntegerV1Attr
//===----------------------------------------------------------------------===//

namespace {
unsigned getBitWidthForIntegerType(Type type) {
  if (type.isa<IntegerSI4V1Type>() || type.isa<IntegerUI4V1Type>()) return 4;
  if (type.isa<IntegerSI8V1Type>() || type.isa<IntegerUI8V1Type>()) return 8;
  if (type.isa<IntegerSI16V1Type>() || type.isa<IntegerUI16V1Type>()) return 16;
  if (type.isa<IntegerSI32V1Type>() || type.isa<IntegerUI32V1Type>()) return 32;
  if (type.isa<IntegerSI64V1Type>() || type.isa<IntegerUI64V1Type>()) return 64;
  llvm::report_fatal_error("unsupported integer type");
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
  writer.writeVarInt(vhlo_encoding::kIntegerV1Attr);
  writer.writeType(attr.getType());
  writer.writeAPIntWithKnownWidth(attr.getValue());
}

//===----------------------------------------------------------------------===//
// OutputOperandAliasV1Attr
//===----------------------------------------------------------------------===//

OutputOperandAliasV1Attr VhloBytecodeInterface::readOutputOperandAliasV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> outputTupleIndices, operandTupleIndices;
  int64_t operandIndex;

  if (failed(reader.readSignedVarInts(outputTupleIndices)) ||
      failed(reader.readSignedVarInt(operandIndex)) ||
      failed(reader.readSignedVarInts(operandTupleIndices)))
    return OutputOperandAliasV1Attr();

  return OutputOperandAliasV1Attr::get(getContext(), outputTupleIndices,
                                       operandIndex, operandTupleIndices);
}

void VhloBytecodeInterface::write(OutputOperandAliasV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kOutputOperandAliasV1Attr);
  writer.writeSignedVarInts(attr.getOutputTupleIndices());
  writer.writeSignedVarInt(attr.getOperandIndex());
  writer.writeSignedVarInts(attr.getOperandTupleIndices());
}

//===----------------------------------------------------------------------===//
// PrecisionV1Attr
//===----------------------------------------------------------------------===//

PrecisionV1Attr VhloBytecodeInterface::readPrecisionV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<PrecisionV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizePrecisionV1(val); });
}

void VhloBytecodeInterface::write(PrecisionV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kPrecisionV1Attr);
  hlo::bytecode::writeEnumAttribute<PrecisionV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// RngAlgorithmV1Attr
//===----------------------------------------------------------------------===//

RngAlgorithmV1Attr VhloBytecodeInterface::readRngAlgorithmV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<RngAlgorithmV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeRngAlgorithmV1(val); });
}

void VhloBytecodeInterface::write(RngAlgorithmV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kRngAlgorithmV1Attr);
  hlo::bytecode::writeEnumAttribute<RngAlgorithmV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// RngDistributionV1Attr
//===----------------------------------------------------------------------===//

RngDistributionV1Attr VhloBytecodeInterface::readRngDistributionV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<RngDistributionV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeRngDistributionV1(val); });
}

void VhloBytecodeInterface::write(RngDistributionV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kRngDistributionV1Attr);
  hlo::bytecode::writeEnumAttribute<RngDistributionV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// StringV1Attr
//===----------------------------------------------------------------------===//

StringV1Attr VhloBytecodeInterface::readStringV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  StringRef string;
  if (failed(reader.readString(string))) return StringV1Attr();
  return StringV1Attr::get(getContext(), string);
}

void VhloBytecodeInterface::write(StringV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kStringV1Attr);
  writer.writeOwnedString(attr.getValue());
}

//===----------------------------------------------------------------------===//
// TensorV1Attr
//===----------------------------------------------------------------------===//

TensorV1Attr VhloBytecodeInterface::readTensorV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type type;
  ArrayRef<char> blob;
  if (failed(reader.readType(type)) || failed(reader.readBlob(blob)))
    return TensorV1Attr();
  return TensorV1Attr::get(getContext(), type, blob);
}

void VhloBytecodeInterface::write(TensorV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTensorV1Attr);
  writer.writeType(attr.getType());
  writer.writeOwnedBlob(attr.getData());
}

//===----------------------------------------------------------------------===//
// TransposeV1Attr
//===----------------------------------------------------------------------===//

TransposeV1Attr VhloBytecodeInterface::readTransposeV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<TransposeV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeTransposeV1(val); });
}

void VhloBytecodeInterface::write(TransposeV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTransposeV1Attr);
  hlo::bytecode::writeEnumAttribute<TransposeV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// TypeV1Attr
//===----------------------------------------------------------------------===//

TypeV1Attr VhloBytecodeInterface::readTypeV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type type;
  if (failed(reader.readType(type))) return TypeV1Attr();

  return TypeV1Attr::get(getContext(), type);
}

void VhloBytecodeInterface::write(TypeV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTypeV1Attr);
  writer.writeType(attr.getValue());
}

//===----------------------------------------------------------------------===//
// TypeExtensionsV1Attr
//===----------------------------------------------------------------------===//

TypeExtensionsV1Attr VhloBytecodeInterface::readTypeExtensionsV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> bounds;
  if (failed(reader.readSignedVarInts(bounds))) return TypeExtensionsV1Attr();
  return TypeExtensionsV1Attr::get(getContext(), bounds);
}

void VhloBytecodeInterface::write(TypeExtensionsV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTypeExtensionsV1Attr);
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
    case vhlo_encoding::kBooleanV1Type:
      return BooleanV1Type::get(getContext());
    case vhlo_encoding::kComplexV1Type:
      return readComplexV1Type(reader);
    case vhlo_encoding::kFloatBF16V1Type:
      return FloatBF16V1Type::get(getContext());
    case vhlo_encoding::kFloatF16V1Type:
      return FloatF16V1Type::get(getContext());
    case vhlo_encoding::kFloatF32V1Type:
      return FloatF32V1Type::get(getContext());
    case vhlo_encoding::kFloatF64V1Type:
      return FloatF64V1Type::get(getContext());
    case vhlo_encoding::kFloatF8E5M2V1Type:
      return FloatF8E5M2V1Type::get(getContext());
    case vhlo_encoding::kFloatF8E4M3FNV1Type:
      return FloatF8E4M3FNV1Type::get(getContext());
    case vhlo_encoding::kFunctionV1Type:
      return readFunctionV1Type(reader);
    case vhlo_encoding::kIndexV1Type:
      return IndexV1Type::get(getContext());
    case vhlo_encoding::kIntegerSI4V1Type:
      return IntegerSI4V1Type::get(getContext());
    case vhlo_encoding::kIntegerSI8V1Type:
      return IntegerSI8V1Type::get(getContext());
    case vhlo_encoding::kIntegerSI16V1Type:
      return IntegerSI16V1Type::get(getContext());
    case vhlo_encoding::kIntegerSI32V1Type:
      return IntegerSI32V1Type::get(getContext());
    case vhlo_encoding::kIntegerSI64V1Type:
      return IntegerSI64V1Type::get(getContext());
    case vhlo_encoding::kIntegerUI4V1Type:
      return IntegerUI4V1Type::get(getContext());
    case vhlo_encoding::kIntegerUI8V1Type:
      return IntegerUI8V1Type::get(getContext());
    case vhlo_encoding::kIntegerUI16V1Type:
      return IntegerUI16V1Type::get(getContext());
    case vhlo_encoding::kIntegerUI32V1Type:
      return IntegerUI32V1Type::get(getContext());
    case vhlo_encoding::kIntegerUI64V1Type:
      return IntegerUI64V1Type::get(getContext());
    case vhlo_encoding::kRankedTensorV1Type:
      return readRankedTensorV1Type(reader, /*hasEncoding=*/false);
    case vhlo_encoding::kRankedTensorV1TypeWithEncoding:
      return readRankedTensorV1Type(reader, /*hasEncoding=*/true);
    case vhlo_encoding::kTokenV1Type:
      return readTokenV1Type(reader);
    case vhlo_encoding::kTupleV1Type:
      return readTupleV1Type(reader);
    case vhlo_encoding::kUniformQuantizedV1Type:
      return readUniformQuantizedV1Type(reader);
    case vhlo_encoding::kUnrankedTensorV1Type:
      return readUnrankedTensorV1Type(reader);
    case vhlo_encoding::kWitnessV1Type:
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
      .Case<ComplexV1Type, FunctionV1Type, RankedTensorV1Type, TokenV1Type,
            TupleV1Type, UnrankedTensorV1Type, UniformQuantizedV1Type>(
          [&](auto type) {
            LOG_WRITE_CALL;
            return write(type, writer), success();
          })
      .Case([&](BooleanV1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kBooleanV1Type), success();
      })
      .Case([&](FloatBF16V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kFloatBF16V1Type), success();
      })
      .Case([&](FloatF16V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kFloatF16V1Type), success();
      })
      .Case([&](FloatF32V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kFloatF32V1Type), success();
      })
      .Case([&](FloatF64V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kFloatF64V1Type), success();
      })
      .Case([&](FloatF8E4M3FNV1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kFloatF8E4M3FNV1Type),
               success();
      })
      .Case([&](FloatF8E5M2V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kFloatF8E5M2V1Type), success();
      })
      .Case([&](IndexV1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIndexV1Type), success();
      })
      .Case([&](IntegerSI4V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerSI4V1Type), success();
      })
      .Case([&](IntegerSI8V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerSI8V1Type), success();
      })
      .Case([&](IntegerSI16V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerSI16V1Type), success();
      })
      .Case([&](IntegerSI32V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerSI32V1Type), success();
      })
      .Case([&](IntegerSI64V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerSI64V1Type), success();
      })
      .Case([&](IntegerUI4V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI4V1Type), success();
      })
      .Case([&](IntegerUI8V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI8V1Type), success();
      })
      .Case([&](IntegerUI16V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI16V1Type), success();
      })
      .Case([&](IntegerUI32V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI32V1Type), success();
      })
      .Case([&](IntegerUI64V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI64V1Type), success();
      })
      .Case([&](WitnessV1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kWitnessV1Type), success();
      })
      .Default([&](Type) {
        LOG_NOT_IMPLEMENTED;
        return failure();
      });
}

//===----------------------------------------------------------------------===//
// ComplexV1Type
//===----------------------------------------------------------------------===//

ComplexV1Type VhloBytecodeInterface::readComplexV1Type(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type elementType;
  if (failed(reader.readType(elementType))) return ComplexV1Type();
  return ComplexV1Type::get(getContext(), elementType);
}

void VhloBytecodeInterface::write(ComplexV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kComplexV1Type);
  writer.writeType(type.getElementType());
}

//===----------------------------------------------------------------------===//
// FunctionV1Type
//===----------------------------------------------------------------------===//

FunctionV1Type VhloBytecodeInterface::readFunctionV1Type(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  SmallVector<Type> inputs;
  SmallVector<Type> outputs;
  if (failed(reader.readTypes(inputs)) || failed(reader.readTypes(outputs)))
    return FunctionV1Type();

  return FunctionV1Type::get(getContext(), inputs, outputs);
}

void VhloBytecodeInterface::write(FunctionV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kFunctionV1Type);
  writer.writeTypes(type.getInputs());
  writer.writeTypes(type.getOutputs());
}

//===----------------------------------------------------------------------===//
// RankedTensorV1Type
//===----------------------------------------------------------------------===//

RankedTensorV1Type VhloBytecodeInterface::readRankedTensorV1Type(
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
    writer.writeVarInt(vhlo_encoding::kRankedTensorV1TypeWithEncoding);
    writer.writeAttribute(encoding);
  } else {
    writer.writeVarInt(vhlo_encoding::kRankedTensorV1Type);
  }
  writer.writeSignedVarInts(type.getShape());
  writer.writeType(type.getElementType());
}

//===----------------------------------------------------------------------===//
// TokenV1Type
//===----------------------------------------------------------------------===//

TokenV1Type VhloBytecodeInterface::readTokenV1Type(
    DialectBytecodeReader &) const {
  LOG_READ_CALL;
  return TokenV1Type::get(getContext());
}

void VhloBytecodeInterface::write(TokenV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTokenV1Type);
}

//===----------------------------------------------------------------------===//
// TupleV1Type
//===----------------------------------------------------------------------===//

TupleV1Type VhloBytecodeInterface::readTupleV1Type(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  SmallVector<Type> elements;
  if (failed(reader.readTypes(elements))) return TupleV1Type();

  return TupleV1Type::get(getContext(), elements);
}

void VhloBytecodeInterface::write(TupleV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTupleV1Type);
  writer.writeTypes(type.getTypes());
}

//===----------------------------------------------------------------------===//
// UniformQuantizedV1Type
//===----------------------------------------------------------------------===//

UniformQuantizedV1Type VhloBytecodeInterface::readUniformQuantizedV1Type(
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
  writer.writeVarInt(vhlo_encoding::kUniformQuantizedV1Type);
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
//===----------------------------------------------------------------------===//

UnrankedTensorV1Type VhloBytecodeInterface::readUnrankedTensorV1Type(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type elementType;
  if (failed(reader.readType(elementType))) return UnrankedTensorV1Type();

  return UnrankedTensorV1Type::get(getContext(), elementType);
}

void VhloBytecodeInterface::write(UnrankedTensorV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kUnrankedTensorV1Type);
  writer.writeType(type.getElementType());
}

}  // namespace

void addBytecodeInterface(VhloDialect *dialect) {
  dialect->addInterfaces<VhloBytecodeInterface>();
}

}  // namespace vhlo
}  // namespace mlir
