/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_BASE_H
#define STABLEHLO_DIALECT_BASE_H

#include <algorithm>
#include <optional>

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"

// Include order matters
#include "stablehlo/dialect/BaseAttrInterfaces.h.inc"

namespace mlir {
namespace hlo {

// TODO(zhouxin) change to a better name as it's used by both of size and bound
// Check if the dimension size is dynamic.
inline static bool isDynamicDimSize(int64_t val) {
  return ShapedType::isDynamic(val);
}

inline static bool isStaticDimSize(int64_t val) {
  return !isDynamicDimSize(val);
}

//  Verifies that the two types have compatible shape with bounds but allows
//  different element types.
LogicalResult verifyCompatibleShapeWithBounds(Type type1, Type type2);

// Returns true if the given types are the same for the purposes of HLO type
// inference, accounting for special properties of quantization and sparsity.
bool isCompatibleForHloTypeInference(Type tp1, Type tp2);

// Returns true if the given type ranges have same types for the purposes of HLO
// type inference, accounting for special properties of quantization and
// sparsity.
bool isCompatibleForHloTypeInference(TypeRange tp1, TypeRange tp2);

// TODO(zhouxin) Move type inference related methods to TypeInference.cpp

std::pair<int64_t, int64_t> inferConcatenatedDimAndBound(int64_t leftSize,
                                                         int64_t rightSize,
                                                         int64_t leftBound,
                                                         int64_t rightBound);

FailureOr<std::pair<int64_t, int64_t>> inferMostSpecificDimAndBound(
    Optional<Location> location, int64_t dim, int64_t leftSize,
    int64_t rightSize, int64_t leftBound, int64_t rightBound);

FailureOr<std::pair<int64_t, int64_t>> inferLeastSpecificDimAndBound(
    Optional<Location> location, int64_t dim, int64_t leftSize,
    int64_t rightSize, int64_t leftBound, int64_t rightBound);

// Infer single least specific return type from inputTypes with support for
// bounds. (Size, bound) of each dimension of the return type will be merged
// from corresponding dimensions of every inputType by extracting the least
// specific one. Return unranked tensor if any input is unranked.
FailureOr<Type> inferLeastSpecificType(Optional<Location> location,
                                       TypeRange inputTypes);

// Infer single most specific return type from inputTypes with support for
// bounds. (Size, bound) of each dimension of the return type will be merged
// from corresponding dimensions of every inputType by extracting the most
// specific one. Return unranked tensor if all inputs are unranked.
FailureOr<Type> inferMostSpecificType(Optional<Location> location,
                                      TypeRange inputTypes);

LogicalResult inferMostSpecificTypeComponents(
    std::optional<Location> location, TypeRange inputTypes,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

// Shape derivation function that computes the shape of the result based on an
// operand. For a 2-dimensional input tensor, this produces IR of the form
//
//  %0 = dim %arg0, 0 : memref<?x?xf32>
//  %1 = index_cast %0 : index to i64
//  %2 = dim %arg0, 1 : memref<?x?xf32>
//  %3 = index_cast %2 : index to i64
//  %4 = "shape.shape_of"(%1, %3)
//    : (i64, i64) -> tensor<2xi64>
//
// and returns %4 as the shape value.
LogicalResult deriveShapeFromOperand(
    OpBuilder *builder, Operation *op, Value operand,
    SmallVectorImpl<Value> *reifiedReturnShapes);

// Type derivation function that returns a tensor type with a new element type.
TensorType getSameShapeTensorType(TensorType tensorType, Type elementType);

// Takes a tensor type that may have complex elements and returns a type that
// maintains the shape, but with real numeric data types.
//   Ex: tensor<4xcomplex<f32>>  -->  tensor<4xf32>
Type createRealType(TensorType type);

// Verify bounds expressed by HLO_BoundedAttrInterface against the provided
// type. See documentation for HLO_BoundedAttrInterface for the list of checks.
LogicalResult verifyBounds(ArrayRef<int64_t> bounds, RankedTensorType type,
                           function_ref<InFlightDiagnostic()> emitError);

// If an encoding attribute conforms to HLO_BoundedAttrInterface, return the
// bounds that it carries. Otherwise, return an empty ArrayRef.
ArrayRef<int64_t> encodingToBounds(Attribute encoding);

// Create an HLO_BoundedAttrInterface encoding attribute that carries the given
// bounds. Requires a prototype - an existing encoding attribute - to obtain
// the underlying dialect that knows how to create these attributes.
Attribute boundsToEncoding(Attribute prototype, ArrayRef<int64_t> bounds);

// This interface is implemented by both StableHLO and MHLO dialects
// and is used as the foundation for sharing verification, type inference and
// prettyprinting logic between them.
class HloDialectInterface : public DialectInterface::Base<HloDialectInterface> {
 public:
  HloDialectInterface(Dialect *dialect) : Base(dialect) {}

  // Creates a TokenType type, specific to this dialect.
  // See docs for the particular type in the corresponding dialect.
  virtual Type createTokenType() const = 0;

  // Check whether the type is of TokenType in the corresponding dialect.
  virtual bool isTokenType(Type type) const = 0;

  // Creates a TypeExtensions attribute, specific to this dialect.
  // See docs for the particular attribute in the corresponding dialect.
  virtual Attribute createTypeExtensions(ArrayRef<int64_t> bounds) const = 0;
};

namespace bytecode {
// Helper methods for bytecode
// Enum reader and writer. Many attrs have a single enum type to serialize.
// Use the attributes underlying type to get the numeric value.
// Note this may cause issues if enums use an int64_t and have a large value.
// All enums in StableHLO and CHLO currently use uint32_t.
template <typename EnumTypeAttr, typename SymbolizeFn>
EnumTypeAttr readEnumAttribute(DialectBytecodeReader &reader,
                               MLIRContext *context, SymbolizeFn symbolizeFn) {
  uint64_t code;
  if (failed(reader.readVarInt(code))) return EnumTypeAttr();

  auto enumOpt = symbolizeFn(static_cast<uint32_t>(code));
  if (!enumOpt.has_value()) return EnumTypeAttr();

  return EnumTypeAttr::get(context, enumOpt.value());
}

template <typename EnumType, typename EnumTypeAttr>
void writeEnumAttribute(EnumTypeAttr val, DialectBytecodeWriter &writer) {
  static_assert(
      std::is_same<typename std::underlying_type<EnumType>::type,
                   uint32_t>::value,
      "writeEnumAttribute is only implemented for uint32_t enum values");

  uint32_t enumVal = static_cast<typename std::underlying_type<EnumType>::type>(
      val.getValue());
  writer.writeVarInt(enumVal);
}
}  // namespace bytecode

namespace OpTrait {

template <typename ConcreteType>
class BroadcastingElementwise
    : public mlir::OpTrait::TraitBase<ConcreteType, BroadcastingElementwise> {};

template <typename ConcreteType>
class PairwiseSameOperandAndResultType
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      PairwiseSameOperandAndResultType> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    const int numOperands = op->getNumOperands();
    const int numResults = op->getNumResults();
    if (numOperands != numResults) {
      return op->emitOpError()
             << "requires the same number of operands and results";
    }

    for (int idx : llvm::seq<int>(0, numOperands)) {
      if (op->getOperand(idx).getType() != op->getResult(idx).getType()) {
        return op->emitOpError()
               << "requires the same type for operand and result at index "
               << idx;
      }
    }
    return success();
  }
};

template <typename ConcreteType>
class CompatibleOperandsAndResultType
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      CompatibleOperandsAndResultType> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    Type expected;
    if (op->getNumResults() != 0) expected = op->getResult(0).getType();
    if (op->getNumOperands() != 0) expected = op->getOperand(0).getType();
    if (!expected) return failure();

    auto typeMatch = [&](Type actual) {
      return isCompatibleForHloTypeInference(actual, expected);
    };
    auto allMatch = llvm::all_of(op->getOperandTypes(), typeMatch) &&
                    llvm::all_of(op->getResultTypes(), typeMatch);
    if (!allMatch) {
      return op->emitOpError(
          "requires compatible types for all operands and results");
    }

    return success(allMatch);
  }

  static LogicalResult inferReturnTypes(
      MLIRContext * /*context*/, std::optional<Location> location,
      ValueRange operands, DictionaryAttr /*attributes*/,
      RegionRange /*regions*/, SmallVectorImpl<Type> &inferredReturnTypes) {
    // TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
    // support quantization or sparsity.
    if (operands.empty())
      return emitOptionalError(
          location,
          "Expected non-empty operands for [CompatibleOperandsAndResultType]");

    auto inferredTypeOrErr =
        inferMostSpecificType(location, operands.getTypes());
    if (failed(inferredTypeOrErr)) return failure();
    inferredReturnTypes.emplace_back(*inferredTypeOrErr);
    return success();
  }

  // This function is not going to be called automatically.
  // It needs to be paired with INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS
  // (see examples in StablehloOps.cpp).
  static LogicalResult inferReturnTypeComponentsFromOperands(
      MLIRContext *context, std::optional<Location> location,
      ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
    SmallVector<Type> inferredReturnTypes;
    if (failed(inferReturnTypes(context, location, operands.getValues(),
                                attributes, regions, inferredReturnTypes)))
      return failure();
    auto inferredReturnType = inferredReturnTypes[0].cast<ShapedType>();
    inferredReturnShapes.push_back(inferredReturnType);
    return success();
  }
};

}  // namespace OpTrait
}  // namespace hlo
}  // namespace mlir

#endif
