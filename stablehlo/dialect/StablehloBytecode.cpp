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

namespace {
namespace stablehlo_encoding {

/// This enum contains marker codes used to indicate which attribute is
/// currently being decoded, and how it should be decoded. The order of these
/// codes should generally be unchanged, as any changes will inevitably break
/// compatibility with older bytecode.
enum AttributeCode {
  ///   FftTypeAttr
  ///     varint FftType
  ///   }
  kFftTypeAttr = 0,
};

/// This enum contains marker codes used to indicate which type is currently
/// being decoded, and how it should be decoded. The order of these codes should
/// generally be unchanged, as any changes will inevitably break compatibility
/// with older bytecode.
enum TypeCode {
   // TODO
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
struct StablehloBytecodeInterface : public BytecodeDialectInterface {
  StablehloBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  Attribute readAttribute(DialectBytecodeReader &reader) const override;
  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override;

  // Include a read method for each attribute in StableHLO
  FftTypeAttr readFftTypeAttr(DialectBytecodeReader &reader) const;
  // void read.*Attr(...

  // Include a write method for each attribute in StableHLO
  void write(FftTypeAttr attr, DialectBytecodeWriter &writer) const;
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
};

//===----------------------------------------------------------------------===//
// Implementation for StablehloBytecode

//===----------------------------------------------------------------------===//
// Attributes: Reader

Attribute StablehloBytecodeInterface::readAttribute(
    DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code)))
    return Attribute();
  switch (code) {
  case stablehlo_encoding::kFftTypeAttr:
    return readFftTypeAttr(reader);
  default:
    reader.emitError() << "unknown stablehlo attribute code: " << code;
    return Attribute();
  }
}

FftTypeAttr StablehloBytecodeInterface::readFftTypeAttr(
    DialectBytecodeReader &reader) const {

  uint64_t code;
  if (failed(reader.readVarInt(code)))
    return FftTypeAttr();

  llvm::Optional<FftType> fftTypeOpt =
      symbolizeFftType(static_cast<uint32_t>(code));
  if (!fftTypeOpt.has_value()) {
    return FftTypeAttr();  // <-- Guessing empty makes the infrastructure error?
  }

  return FftTypeAttr::get(getContext(), fftTypeOpt.value());
}


//===----------------------------------------------------------------------===//
// Attributes: Writer

// If this method returns failure, the string serialization is used in the
// bytecode.
LogicalResult StablehloBytecodeInterface::writeAttribute(
    Attribute attr, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<FftTypeAttr>([&](auto attr) {
        write(attr, writer);
        return success();
      })
      .Default([&](Attribute) { return failure(); });
}

void StablehloBytecodeInterface::write(
    FftTypeAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kFftTypeAttr);
  uint32_t fft_type_int =
      static_cast<std::underlying_type<FftType>::type>(attr.getValue());
  writer.writeVarInt(fft_type_int);
}

//===----------------------------------------------------------------------===//
// Types: Reader

Type StablehloBytecodeInterface::readType(
    DialectBytecodeReader &reader) const {

  reader.emitError() << "readType not implemented.";
  return Type();

  /*
  uint64_t code;
  if (failed(reader.readVarInt(code)))
    return Type();
  switch (code) {
  case stablehlo_encoding::kIntegerType:
    return readIntegerType(reader);
    // ...
  default:
    reader.emitError() << "unknown stablehlo type code: " << code;
    return Type();
  }
  */
}

//===----------------------------------------------------------------------===//
// Types: Writer

LogicalResult StablehloBytecodeInterface::writeType(
    Type type, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Type, LogicalResult>(type)
      /* TODO: Implement type cases. Defaults to failure for now. */
      .Default([&](Type) { return failure(); });
}

}  // namespace

void addBytecodeInterface(StablehloDialect *dialect) {
  dialect->addInterfaces<StablehloBytecodeInterface>();
}
}  // namespace stablehlo
}  // namespace mlir


///////////////////////////////////////////////////
// Reference code from Builtin Dialect. To delete.
////////////////////////////////////////////////////
/*
ArrayAttr StablehloBytecodeInterface::readArrayAttr(
    DialectBytecodeReader &reader) const {
  SmallVector<Attribute> elements;
  if (failed(reader.readAttributes(elements)))
    return ArrayAttr();
  return ArrayAttr::get(getContext(), elements);
}

DictionaryAttr StablehloBytecodeInterface::readDictionaryAttr(
    DialectBytecodeReader &reader) const {
  auto readNamedAttr = [&]() -> FailureOr<NamedAttribute> {
    StringAttr name;
    Attribute value;
    if (failed(reader.readAttribute(name)) ||
        failed(reader.readAttribute(value)))
      return failure();
    return NamedAttribute(name, value);
  };
  SmallVector<NamedAttribute> attrs;
  if (failed(reader.readList(attrs, readNamedAttr)))
    return DictionaryAttr();
  return DictionaryAttr::get(getContext(), attrs);
}

StringAttr StablehloBytecodeInterface::readStringAttr(
    DialectBytecodeReader &reader) const {
  StringRef string;
  if (failed(reader.readString(string)))
    return StringAttr();
  return StringAttr::get(getContext(), string);
}
*/

/*
IntegerType StablehloBytecodeInterface::readIntegerType(
    DialectBytecodeReader &reader) const {
  uint64_t encoding;
  if (failed(reader.readVarInt(encoding)))
    return IntegerType();
  return IntegerType::get(
      getContext(), encoding >> 2,
      static_cast<IntegerType::SignednessSemantics>(encoding & 0x3));
}

FunctionType StablehloBytecodeInterface::readFunctionType(
    DialectBytecodeReader &reader) const {
  SmallVector<Type> inputs, results;
  if (failed(reader.readTypes(inputs)) || failed(reader.readTypes(results)))
    return FunctionType();
  return FunctionType::get(getContext(), inputs, results);
}
*/


/*
void StablehloBytecodeInterface::write(
    ArrayAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kArrayAttr);
  writer.writeAttributes(attr.getValue());
}

void StablehloBytecodeInterface::write(
    DictionaryAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kDictionaryAttr);
  writer.writeList(attr.getValue(), [&](NamedAttribute attr) {
    writer.writeAttribute(attr.getName());
    writer.writeAttribute(attr.getValue());
  });
}

void StablehloBytecodeInterface::write(
    StringAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kStringAttr);
  writer.writeOwnedString(attr.getValue());
}
*/



/*
void StablehloBytecodeInterface::write(
    IntegerType type, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kIntegerType);
  writer.writeVarInt((type.getWidth() << 2) | type.getSignedness());
}

void StablehloBytecodeInterface::write(
    FunctionType type, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kFunctionType);
  writer.writeTypes(type.getInputs());
  writer.writeTypes(type.getResults());
}
*/
