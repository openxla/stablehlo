/* Copyright 2025 The OpenXLA Authors.

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

#ifndef STABLEHLO_BUILDER_STABLEHLOBUILDER_H_
#define STABLEHLO_BUILDER_STABLEHLOBUILDER_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace mlir {
namespace stablehlo {

///////////////
// Dialect Helpers
///////////////

// Appends or overwrites an entry in the `mhlo.frontend_attributes` attribute
//
// of the given op.
// Ex:
//   stablehlo.abs %0 { mhlo.frontend_attributes = { "foo" = 123 } }
MlirOp AttachFrontendAttribute(MlirBuilder& builder, MlirOp op, StringRef name,
                               Attribute value);

/////////////////
// MANUAL APIs
/////////////////
// There should be some manual APIs for each dialect for the APIs that require
// max usability like constant.
// I.e. sugar for int64_t -> tensor<i64>
// Or std::vector<int64_t> -> tensor<Nxi64>

MlirOp ConvertElementType(MlirOp input, ElementType resultElementTypeKind);
MlirOp ConvertElementType(MlirOp input, Type resultElementType);

MlirOp Constant(MlirBuilder& builder, int64_t value);
MlirOp Constant(MlirBuilder& builder, std::vector<int64_t> value);

// IotaLike is a sugar API for iota that accounts for bounded dynamism in the
// input tensor. Eventually this should be a chlo.iota_like op with a StableHLO
// decomposition, but for now it will be housed as a builder API.
MlirOp IotaLike(MlirOp input, int64_t dim, ElementType elementType);
MlirOp IotaLike(MlirOp input, int64_t dim, Type elementType);

// Better Dot / DotGeneral builders.
// These ops don't support full type inference because the result element type
// cannot be inferred from operands, however the result shape can be.
//
// The generated APIs require specifying the full result type, which requires
// callsite to compute the proper shape, these APIs take a preferred result
// type and will infer the result shape from the operands.
MlirOp Dot(MlirOp lhs, MlirOp rhs, ArrayAttr precisionConfig = ArrayAttr(),
           std::optional<ElementType> preferredElementType = std::nullopt);
MlirOp DotGeneral(
    MlirOp lhs, MlirOp rhs, DotDimensionNumbersAttr dotDimsAttr,
    ArrayAttr precisionConfig = ArrayAttr(),
    std::optional<ElementType> preferredElementType = std::nullopt);

MlirOp Reshape(MlirOp input, ArrayRef<int64_t> newShape);

// Get all arguments for a given region of a while op
// Initializes the region arguments given the WhileOp operands.
SmallVector<MlirOp> Arguments(RegionBuilder& rb, WhileOp op);

/////////////////
// GENERATED APIs
/////////////////

// stablehlo::While - UX issue, all regions need to declare all arguments, even
//   if unused
// stablehlo::Case - Can't generate builder for op with variadic region yet, add
//   manually.

#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h.inc"

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_BUILDER_STABLEHLOBUILDER_H_
