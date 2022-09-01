/* Copyright 2022 The CHLO Authors.
ChloBytecode.cpp - CHLO Bytecode Implementation */

#include "stablehlo/dialect/ChloBytecode.h"

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Diagnostics.h"
#include "stablehlo/dialect/ChloOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// Debug Trace Helpers
//===----------------------------------------------------------------------===//

// Enable logging with flag:
//   stablehlo-opt -debug-only=chlo-bytecode [...]
//
// Extract after function name, remove namespace.
//   Called: write(mlir::chlo::TokenType, mlir::DialectBytecodeWriter ...
//   ***Not Implemened: write(...
#define _EXTRACT_AFTER(a, b)                                                   \
  llvm::StringRef(a).substr(llvm::StringRef(a).find(b))

#define _LOG_CALL_TO(func)                                                     \
  DEBUG_WITH_TYPE("chlo-bytecode",                                             \
                  llvm::errs()                                                 \
                      << "Called: "                                            \
                      << _EXTRACT_AFTER(__PRETTY_FUNCTION__, func) << '\n')

#define LOG_WRITE_CALL _LOG_CALL_TO("write")
#define LOG_READ_CALL _LOG_CALL_TO(__func__)
#define LOG_NOT_IMPLEMENTED                                                    \
  DEBUG_WITH_TYPE("chlo-bytecode", llvm::errs()                                \
                                       << "***Not Implemented: "               \
                                       << __PRETTY_FUNCTION__ << '\n')

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

namespace {
namespace chlo_encoding {

/// This enum contains marker codes used to indicate which attribute is
/// currently being decoded, and how it should be decoded. The order of these
/// codes should generally be unchanged, as any changes will inevitably break
/// compatibility with older bytecode.
///
/// To add an attribute, search for "TO ADD ATTRIBUTE" in this file and ensure
/// each location is updated.
enum AttributeCode {
  // TO ADD ATTRIBUTE: Add an enum value with doc string for new attr.

  ///   ComparisonDirectionAttr
  ///     ComparisonDirection: varint
  ///   }
  kComparisonDirectionAttr = 0,

  ///   ComparisonTypeAttr
  ///     ComparisonType: varint
  ///   }
  kComparisonTypeAttr = 1,
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

};

} // namespace chlo_encoding
} // namespace

//===----------------------------------------------------------------------===//
// ChloBytecodeInterface
//===----------------------------------------------------------------------===//

namespace mlir {
namespace chlo {

namespace {
/// This class implements the bytecode interface for the chlo dialect.
class ChloBytecodeInterface : public BytecodeDialectInterface {
public:
  ChloBytecodeInterface(Dialect *dialect) : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  // These methods are invoked by superclass when an attr from chlo dialect
  // is encountered.
  Attribute readAttribute(DialectBytecodeReader &reader) const override;
  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override;

  // TO ADD ATTRIBUTE: Include a read method for each attribute in CHLO
  // Ex: SomeAttr readSomeAttr(DialectBytecodeReader &reader) const;
  ComparisonDirectionAttr
  readComparisonDirectionAttr(DialectBytecodeReader &reader) const;
  ComparisonTypeAttr
  readComparisonTypeAttr(DialectBytecodeReader &reader) const;

  // TO ADD ATTRIBUTE: Include a write method for each attribute in CHLO
  // Ex: void write(SomeAttr attr, DialectBytecodeWriter &writer) const;
  void write(ComparisonDirectionAttr attr, DialectBytecodeWriter &writer) const;
  void write(ComparisonTypeAttr attr, DialectBytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Types

  // These methods are invoked by superclass when a type from chlo dialect
  // is encountered.
  Type readType(DialectBytecodeReader &reader) const override;
  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override;

  // TO ADD TYPE: Include a read method for each type in CHLO
  // Ex: SomeType readSomeType(DialectBytecodeReader &reader) const;

  // TO ADD TYPE: Include a write method for each type in CHLO
  // Ex: void write(SomeType attr, DialectBytecodeWriter &writer) const;

private:
  //===--------------------------------------------------------------------===//
  // Helper methods

  // Enum reader and writer. Many attrs have a single enum type to serialize.
  // Use the attributes underlying type to get the numeric value.
  // Note this may cause issues if enums use an int64_t and have a large value.
  // All enums in CHLO currently use int32_t.
  template <typename EnumType, typename EnumTypeAttr, typename SymbolizeFn>
  EnumTypeAttr readEnumAttribute(DialectBytecodeReader &reader,
                                 SymbolizeFn symbolizeFn) const {
    uint64_t code;
    if (failed(reader.readVarInt(code)))
      return EnumTypeAttr();

    llvm::Optional<EnumType> enumOpt = symbolizeFn(static_cast<uint32_t>(code));
    if (!enumOpt.has_value())
      return EnumTypeAttr();

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
// Implementation for ChloBytecode

//===----------------------------------------------------------------------===//
// Attributes: Reader

// TO ADD ATTRIBUTE: Update the switch to include a branch for the attr.
Attribute
ChloBytecodeInterface::readAttribute(DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code)))
    return Attribute();
  switch (code) {
  case chlo_encoding::kComparisonDirectionAttr:
    return readComparisonDirectionAttr(reader);
  case chlo_encoding::kComparisonTypeAttr:
    return readComparisonTypeAttr(reader);
  default:
    reader.emitError() << "unknown chlo attribute code: " << code;
    return Attribute();
  }
}

ComparisonDirectionAttr ChloBytecodeInterface::readComparisonDirectionAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return readEnumAttribute<ComparisonDirection, ComparisonDirectionAttr>(
      reader, [](uint32_t val) { return symbolizeComparisonDirection(val); });
}

ComparisonTypeAttr ChloBytecodeInterface::readComparisonTypeAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return readEnumAttribute<ComparisonType, ComparisonTypeAttr>(
      reader, [](uint32_t val) { return symbolizeComparisonType(val); });
}

//===----------------------------------------------------------------------===//
// Attributes: Writer

// TO ADD ATTRIBUTE: Update the case selection to include the new attr.
// If this method returns failure, the string serialization is used in the
// bytecode.
LogicalResult
ChloBytecodeInterface::writeAttribute(Attribute attr,
                                      DialectBytecodeWriter &writer) const {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<ComparisonDirectionAttr, ComparisonTypeAttr>([&](auto attr) {
        LOG_WRITE_CALL;
        write(attr, writer);
        return success();
      })
      .Default([&](Attribute) {
        LOG_NOT_IMPLEMENTED;
        return failure();
      });
}

void ChloBytecodeInterface::write(ComparisonDirectionAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(chlo_encoding::kComparisonDirectionAttr);
  writeEnumAttribute<ComparisonDirection>(attr, writer);
}

void ChloBytecodeInterface::write(ComparisonTypeAttr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(chlo_encoding::kComparisonTypeAttr);
  writeEnumAttribute<ComparisonType>(attr, writer);
}

//===----------------------------------------------------------------------===//
// Types: Reader

// TO ADD TYPE: Update the case selection to include the new type.
Type ChloBytecodeInterface::readType(DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code)))
    return Type();

  switch (code) {
  default:
    reader.emitError() << "unknown builtin type code: " << code;
    return Type();
  }
}

//===----------------------------------------------------------------------===//
// Types: Writer

// TO ADD TYPE: Update the case selection to include the new type.
LogicalResult
ChloBytecodeInterface::writeType(Type type,
                                 DialectBytecodeWriter &writer) const {
  return TypeSwitch<Type, LogicalResult>(type).Default([&](Type) {
    LOG_NOT_IMPLEMENTED;
    return failure();
  });
}

} // namespace

void addBytecodeInterface(ChloDialect *dialect) {
  dialect->addInterfaces<ChloBytecodeInterface>();
}
} // namespace chlo
} // namespace mlir
