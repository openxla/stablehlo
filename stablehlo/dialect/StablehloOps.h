/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef STABLEHLO_DIALECT_STABLEHLO_OPS_H
#define STABLEHLO_DIALECT_STABLEHLO_OPS_H

#include <algorithm>
#include <optional>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/Version.h"

#define GET_TYPEDEF_CLASSES
#include "stablehlo/dialect/StablehloTypeDefs.h.inc"

// Include order matters.
#include "stablehlo/dialect/StablehloEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "stablehlo/dialect/StablehloAttrs.h.inc"

namespace mlir {
namespace stablehlo {

struct StablehloDialectVersion : public mlir::DialectVersion {
  StablehloDialectVersion(int64_t major, int64_t minor, int64_t patch)
      : dialectVersion(major, minor, patch) {}

  int64_t getMajor() const { return dialectVersion.getMajor(); }
  int64_t getMinor() const { return dialectVersion.getMinor(); }
  int64_t getPatch() const { return dialectVersion.getPatch(); }

  static StablehloDialectVersion getCurrentVersion() {
    // The same version as VHLO as this is serialization related only.
    auto vhloVer = vhlo::Version::getCurrentVersion();
    return {vhloVer.getMajor(), vhloVer.getMinor(), vhloVer.getPatch()};
  }

  bool operator<(const StablehloDialectVersion &other) const {
    return this->dialectVersion < other.dialectVersion;
  }

 private:
  // The dialect version read from bytecode.
  vhlo::Version dialectVersion;
};

class StablehloDialect : public Dialect {
 public:
  explicit StablehloDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "stablehlo"; }

  // Registered hook to materialize a constant operation from a given attribute
  // value with the desired resultant type.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  // Parses a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;

  // Parses an attribute registered to this dialect.
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  // Prints an attribute registered to this dialect.
  void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;

  // Get the set dialect version.
  std::optional<StablehloDialectVersion> getVersion() const;

  // Set dialect version.
  // Note: there is currently no validation.
  void setVersion(std::optional<StablehloDialectVersion> version);

 private:
  std::optional<StablehloDialectVersion> version;
};

// Verifies the source target pairs attached to collective permute.
LogicalResult verifyCollectivePermuteSourceTargetPairs(
    Operation *op, DenseIntElementsAttr attr);

void printConvolutionDimensions(AsmPrinter &p,
                                ConvDimensionNumbersAttr dimNums);
void printConvolutionDimensions(AsmPrinter &p, Operation *,
                                ConvDimensionNumbersAttr dimNums);
ParseResult parseConvolutionDimensions(AsmParser &parser,
                                       ConvDimensionNumbersAttr &dimNums);

void printWindowAttributes(OpAsmPrinter &p, Operation *op,
                           std::optional<DenseI64ArrayAttr> windowStrides,
                           std::optional<DenseIntElementsAttr> padding,
                           std::optional<DenseI64ArrayAttr> lhsDilation,
                           std::optional<DenseI64ArrayAttr> rhsDilation,
                           std::optional<DenseBoolArrayAttr> windowReversal);

ParseResult parseWindowAttributes(OpAsmParser &parser,
                                  DenseI64ArrayAttr &windowStrides,
                                  DenseIntElementsAttr &padding,
                                  DenseI64ArrayAttr &lhsDilation,
                                  DenseI64ArrayAttr &rhsDilation,
                                  DenseBoolArrayAttr &windowReversal);

namespace side_effects {

struct SendResource : ::mlir::SideEffects::Resource::Base<SendResource> {
  StringRef getName() final { return "SendResource"; }
};
struct RecvResource : ::mlir::SideEffects::Resource::Base<RecvResource> {
  StringRef getName() final { return "RecvResource"; }
};
struct InfeedResource : ::mlir::SideEffects::Resource::Base<InfeedResource> {
  StringRef getName() final { return "InfeedResource"; }
};
struct OutfeedResource : ::mlir::SideEffects::Resource::Base<OutfeedResource> {
  StringRef getName() final { return "OutfeedResource"; }
};

}  // end namespace side_effects

}  // end namespace stablehlo
}  // end namespace mlir

#define GET_OP_CLASSES
#include "stablehlo/dialect/StablehloOps.h.inc"

namespace mlir {
namespace stablehlo {

// Returns the broadcast_dimensions for a BroadcastInDimOp from the
// result_type and broadcast_sizes from a BroadcastOp.
DenseI64ArrayAttr getBroadcastDimensionsFromBroadcastSizes(
    RankedTensorType resultType, DenseI64ArrayAttr broadcastSizes);

// Returns the dimension numbers for a DotGeneral op that can be expressed as
// a DotOp, given the LHS of such an operation.
DotDimensionNumbersAttr getDefaultDotDimensionNumbers(mlir::Value lhs);

SortOp createSortOp(PatternRewriter *rewriter, const Location &loc,
                    const llvm::ArrayRef<Value> &operands,
                    const llvm::ArrayRef<Type> &elementTypes, int64_t dimension,
                    bool isStable, ComparisonDirection direction);

template <typename OpTy>
void buildReduceBody(Type elementType, Region &body, OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  if (body.getBlocks().empty()) builder.createBlock(&body);
  Block *block = &body.getBlocks().front();

  // Block arguments are scalars of the given element type.
  Type type = RankedTensorType::get(/*shape=*/{}, elementType);
  Location loc = body.getLoc();
  block->addArguments({type, type}, {loc, loc});

  auto reducer =
      builder.create<OpTy>(loc, block->getArgument(0), block->getArgument(1));
  builder.create<stablehlo::ReturnOp>(loc, reducer.getResult());
}

// PrecisionConfigAttr is a constraint attribute on ArrayAttrs.
// Create this class to allow for building this attr similar to other
// attributes.
struct PrecisionConfigAttr : public ArrayAttr {
  static ArrayAttr get(MLIRContext *context, ArrayRef<Precision> precisions);
};

}  // end namespace stablehlo
}  // end namespace mlir

#endif  // STABLEHLO_DIALECT_STABLEHLO_OPS_H
