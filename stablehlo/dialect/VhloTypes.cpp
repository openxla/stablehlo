/* Copyright 2023 The StableHLO Authors.

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

#include "stablehlo/dialect/VhloTypes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "stablehlo/dialect/AssemblyFormat.h"

namespace mlir {
namespace vhlo {
namespace {
Type convertBuiltinIntegerType(IntegerType type) {
  if (!type.isSignless() && !type.isUnsigned()) return {};

  if (type.getWidth() == 1 && type.isSignless()) {  // Predicate
    return BooleanV1Type::get(type.getContext());
  }

  // Has valid signedness, check for valid widths
  // NOTE: Signless builtin types correspond to signed VHLO types.
  bool isSignless = type.isSignless();
  auto ctx = type.getContext();
  switch (type.getWidth()) {
    case 4:
      return isSignless ? IntegerSI4V1Type::get(ctx).cast<Type>()
                        : IntegerUI4V1Type::get(ctx).cast<Type>();
    case 8:
      return isSignless ? IntegerSI8V1Type::get(ctx).cast<Type>()
                        : IntegerUI8V1Type::get(ctx).cast<Type>();
    case 16:
      return isSignless ? IntegerSI16V1Type::get(ctx).cast<Type>()
                        : IntegerUI16V1Type::get(ctx).cast<Type>();
    case 32:
      return isSignless ? IntegerSI32V1Type::get(ctx).cast<Type>()
                        : IntegerUI32V1Type::get(ctx).cast<Type>();
    case 64:
      return isSignless ? IntegerSI64V1Type::get(ctx).cast<Type>()
                        : IntegerUI64V1Type::get(ctx).cast<Type>();
  }
  return {};
}
}  // namespace

void VhloTypeConverter::addBuiltinToVhloConversions() {
  addConversion([&](BFloat16Type type) {
    return FloatBF16V1Type::get(type.getContext());
  });
  addConversion([&](ComplexType type) {
    return ComplexV1Type::get(type.getContext(),
                              convertType(type.getElementType()));
  });
  addConversion(
      [&](Float16Type type) { return FloatF16V1Type::get(type.getContext()); });
  addConversion(
      [&](Float32Type type) { return FloatF32V1Type::get(type.getContext()); });
  addConversion(
      [&](Float64Type type) { return FloatF64V1Type::get(type.getContext()); });
  addConversion([&](Float8E4M3FNType type) {
    return FloatF8E4M3FNV1Type::get(type.getContext());
  });
  addConversion([&](Float8E5M2Type type) {
    return FloatF8E5M2V1Type::get(type.getContext());
  });
  addConversion([&](FunctionType type) -> Type {
    SmallVector<Type> convertedInputs;
    SmallVector<Type> convertedResults;
    if (failed(convertTypes(type.getInputs(), convertedInputs))) return {};
    if (failed(convertTypes(type.getResults(), convertedResults))) return {};
    return FunctionV1Type::get(type.getContext(), convertedInputs,
                               convertedResults);
  });
  addConversion(
      [&](IndexType type) { return IndexV1Type::get(type.getContext()); });
  addConversion(
      [&](IntegerType type) { return convertBuiltinIntegerType(type); });
  addConversion([&](RankedTensorType type) -> Type {
    auto encoding = type.getEncoding();
    auto convertedEncoding = encoding ? convertEncoding(encoding) : encoding;
    auto convertedElementType = convertType(type.getElementType());
    if ((encoding && !convertedEncoding) || !convertedElementType) return {};
    return RankedTensorV1Type::get(type.getContext(), type.getShape(),
                                   convertedElementType, convertedEncoding);
  });
  addConversion([&](TupleType type) -> Type {
    SmallVector<Type> convertedTypes;
    if (failed(convertTypes(type.getTypes(), convertedTypes))) return {};
    return vhlo::TupleV1Type::get(type.getContext(), convertedTypes);
  });
  addConversion([&](quant::UniformQuantizedType type) -> Type {
    Type convertedStorageType = convertType(type.getStorageType());
    Type convertedExpressedType = convertType(type.getExpressedType());
    if (!convertedStorageType || !convertedExpressedType) return {};
    return vhlo::UniformQuantizedV1Type::get(
        type.getContext(), type.getFlags(), convertedStorageType,
        convertedExpressedType, APFloat(type.getScale()), type.getZeroPoint(),
        type.getStorageTypeMin(), type.getStorageTypeMax());
  });
  addConversion([&](UnrankedTensorType type) -> Type {
    auto convertedElementType = convertType(type.getElementType());
    if (!convertedElementType) return {};
    return UnrankedTensorV1Type::get(type.getContext(), convertedElementType);
  });
  addConversion([&](shape::WitnessType type) -> Type {
    return vhlo::WitnessV1Type::get(type.getContext());
  });
}

void VhloTypeConverter::addVhloToBuiltinConversions() {
  addConversion([&](BooleanV1Type type) {
    return IntegerType::get(type.getContext(), 1);
  });
  addConversion([&](ComplexV1Type type) {
    return ComplexType::get(convertType(type.getElementType()));
  });
  addConversion([&](FloatBF16V1Type type) {
    return BFloat16Type::get(type.getContext());
  });
  addConversion(
      [&](FloatF16V1Type type) { return Float16Type::get(type.getContext()); });
  addConversion(
      [&](FloatF32V1Type type) { return Float32Type::get(type.getContext()); });
  addConversion(
      [&](FloatF64V1Type type) { return Float64Type::get(type.getContext()); });
  addConversion([&](FloatF8E4M3FNV1Type type) {
    return Float8E4M3FNType::get(type.getContext());
  });
  addConversion([&](FloatF8E5M2V1Type type) {
    return Float8E5M2Type::get(type.getContext());
  });
  addConversion([&](FunctionV1Type type) -> Type {
    SmallVector<Type> convertedInputs;
    SmallVector<Type> convertedOutputs;
    if (failed(convertTypes(type.getInputs(), convertedInputs))) return {};
    if (failed(convertTypes(type.getOutputs(), convertedOutputs))) return {};
    return FunctionType::get(type.getContext(), convertedInputs,
                             convertedOutputs);
  });
  addConversion(
      [&](IndexV1Type type) { return IndexType::get(type.getContext()); });
  addConversion([&](IntegerSI4V1Type type) {
    return IntegerType::get(type.getContext(), 4);
  });
  addConversion([&](IntegerSI8V1Type type) {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](IntegerSI16V1Type type) {
    return IntegerType::get(type.getContext(), 16);
  });
  addConversion([&](IntegerSI32V1Type type) {
    return IntegerType::get(type.getContext(), 32);
  });
  addConversion([&](IntegerSI64V1Type type) {
    return IntegerType::get(type.getContext(), 64);
  });
  addConversion([&](IntegerUI4V1Type type) {
    return IntegerType::get(type.getContext(), 4, IntegerType::Unsigned);
  });
  addConversion([&](IntegerUI8V1Type type) {
    return IntegerType::get(type.getContext(), 8, IntegerType::Unsigned);
  });
  addConversion([&](IntegerUI16V1Type type) {
    return IntegerType::get(type.getContext(), 16, IntegerType::Unsigned);
  });
  addConversion([&](IntegerUI32V1Type type) {
    return IntegerType::get(type.getContext(), 32, IntegerType::Unsigned);
  });
  addConversion([&](IntegerUI64V1Type type) {
    return IntegerType::get(type.getContext(), 64, IntegerType::Unsigned);
  });
  addConversion([&](RankedTensorV1Type type) -> Type {
    auto encoding = type.getEncoding();
    auto convertedEncoding = encoding ? convertEncoding(encoding) : encoding;
    auto convertedElementType = convertType(type.getElementType());
    if ((encoding && !convertedEncoding) || !convertedElementType) return {};
    return RankedTensorType::get(type.getShape(), convertedElementType,
                                 convertedEncoding);
  });
  addConversion([&](TupleV1Type type) -> Type {
    SmallVector<Type> convertedTypes;
    if (failed(convertTypes(type.getTypes(), convertedTypes))) return {};
    return TupleType::get(type.getContext(), convertedTypes);
  });
  addConversion([&](UniformQuantizedV1Type type) -> Type {
    Type convertedStorageType = convertType(type.getStorageType());
    Type convertedExpressedType = convertType(type.getExpressedType());
    if (!convertedStorageType || !convertedExpressedType) return {};
    return quant::UniformQuantizedType::get(
        type.getFlags(), convertedStorageType, convertedExpressedType,
        type.getScale().convertToDouble(), type.getZeroPoint(),
        type.getStorageTypeMin(), type.getStorageTypeMax());
  });
  addConversion([&](UnrankedTensorV1Type type) -> Type {
    auto convertedElementType = convertType(type.getElementType());
    if (!convertedElementType) return {};
    return UnrankedTensorType::get(convertedElementType);
  });
  addConversion([&](WitnessV1Type type) -> Type {
    return shape::WitnessType::get(type.getContext());
  });
}

namespace {
// Helper functions for VHLO verifiers
template <typename TypeOrAttr>
bool isFromVhlo(TypeOrAttr t) {
  return t.getDialect().getNamespace() == "vhlo";
}

template <typename TypeOrAttr>
bool allFromVhlo(ArrayRef<TypeOrAttr> range) {
  return llvm::all_of(range, isFromVhlo<TypeOrAttr>);
}
}  // namespace

// Helper functions for VHLO type printers and parsers.
void printEncoding(AsmPrinter& os, Attribute encoding) {
  if (!encoding) return;
  os << ", " << encoding;
}

ParseResult parseEncoding(AsmParser& parser, Attribute& encoding) {
  if (failed(parser.parseOptionalComma())) {
    return success();
  }
  if (failed(parser.parseAttribute(encoding))) return failure();
  return success();
}

void printShape(AsmPrinter& os, ArrayRef<int64_t> dimSizes) {
  if (dimSizes.empty()) return;
  for (int64_t dimSize : dimSizes) {
    os << hlo::dimSizeToString(dimSize) << 'x';
  }
}

ParseResult parseShape(AsmParser& parser, SmallVector<int64_t>& dimSizes) {
  if (failed(parser.parseDimensionList(dimSizes))) {
    return failure();
  }
  return success();
}

// Print types in parentheses: (!vhlo.type, !vhlo.type)
static void printTypeArray(AsmPrinter& os, ArrayRef<Type> typeArray) {
  if (typeArray.empty()) os << "()";
  os << typeArray;
}

// Parse types in parentheses: (!vhlo.type, !vhlo.type)
ParseResult parseTypeArray(AsmParser& parser, SmallVector<Type>& typeArray) {
  if (succeeded(parser.parseOptionalLParen()) &&
      succeeded(parser.parseOptionalRParen())) {
    return success();
  }

  auto parseEle = [&]() { return parser.parseType(typeArray.emplace_back()); };
  if (failed(parser.parseCommaSeparatedList(parseEle))) {
    return failure();
  }
  return success();
}

}  // namespace vhlo
}  // namespace mlir

// Include order matters
#include "stablehlo/dialect/VhloTypeInterfaces.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "stablehlo/dialect/VhloTypeDefs.cpp.inc"

namespace mlir {
namespace vhlo {

LogicalResult printVhloType(Type type, AsmPrinter& printer) {
  return generatedTypePrinter(type, printer);
}

OptionalParseResult parseVhloType(mlir::AsmParser& parser,
                                  llvm::StringRef* mnemonic, mlir::Type& type) {
  return generatedTypeParser(parser, mnemonic, type);
}

namespace {
template <typename... Types>
void registerVhloTypes(MLIRContext* context) {
  (mlir::detail::TypeUniquer::registerType<Types>(context), ...);
}
}  // namespace

void registerVhloTypes(MLIRContext* context) {
  registerVhloTypes<
#define GET_TYPEDEF_LIST
#include "stablehlo/dialect/VhloTypeDefs.cpp.inc"
      >(context);
}

}  // namespace vhlo
}  // namespace mlir
