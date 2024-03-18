/* Copyright 2022 The IREE Authors
   Copyright 2023 OpenXLA Authors. All Rights Reserved.

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

// Implements utilities for lowering StableHLO dialect to Linalg dialect.

#include "stablehlo/conversions/linalg/transforms/LegalizeToLinalgUtils.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::stablehlo {
namespace {
bool hasIntegralShapeType(Operation *op) {
  auto stp = llvm::dyn_cast<ShapedType>(op->getOperand(0).getType());
  return stp && stp.getElementType().isIntOrIndex();
}

}  // namespace

SmallVector<utils::IteratorType, 3> getParallelAndReductionIterators(
    unsigned nLoops, unsigned nReduction) {
  SmallVector<utils::IteratorType, 3> res(nLoops - nReduction,
                                          utils::IteratorType::parallel);
  res.append(nReduction, utils::IteratorType::reduction);
  return res;
}

SmallVector<utils::IteratorType, 3> getNParallelLoopsAttrs(
    unsigned nParallelLoops) {
  return getParallelAndReductionIterators(nParallelLoops, 0);
}

Value getEmptySparseTensor(OpBuilder &b, Location loc, ShapedType type,
                           ArrayRef<Value> dynSizes) {
  return b.create<bufferization::AllocTensorOp>(
      loc, llvm::cast<TensorType>(type), dynSizes,
      /*copy=*/Value(),
      /*memory_space=*/IntegerAttr());
}

Value getEmptyTensor(OpBuilder &b, Location loc, ShapedType type,
                     ArrayRef<Value> dynSizes) {
  return b.create<tensor::EmptyOp>(
      loc, type.getShape(), type.getElementType(), dynSizes,
      llvm::cast<RankedTensorType>(type).getEncoding());
}

Value getEmptyTensorFor(OpBuilder &b, Location loc, ShapedType resultType,
                        Operation *op, ValueRange operands) {
  bool isSparse = sparse_tensor::getSparseTensorEncoding(resultType) != nullptr;
  // Collect the sizes for a ranked tensor to be passed as parameter to a
  // new tensor initialization operation. This operation only needs the
  // dynamic sizes.
  SmallVector<Value> sizes;
  if (resultType.hasRank() && !resultType.hasStaticShape()) {
    // Ask the op for its output shape.
    auto shapeSource = cast<InferShapedTypeOpInterface>(op);
    SmallVector<Value, 1> reifiedShapes;
    assert(succeeded(
               shapeSource.reifyReturnTypeShapes(b, operands, reifiedShapes)) &&
           "could not reify");
    assert(reifiedShapes.size() == 1 && "Expected one reified result");
    // Construct sizes for the required dimensions.
    for (const auto &en : llvm::enumerate(resultType.getShape())) {
      if (en.value() != ShapedType::kDynamic) continue;
      sizes.push_back(b.create<tensor::ExtractOp>(
          loc, reifiedShapes[0],
          ValueRange{b.create<arith::ConstantIndexOp>(loc, en.index())}));
    }
  }
  return isSparse ? getEmptySparseTensor(b, loc, resultType, sizes)
                  : getEmptyTensor(b, loc, resultType, sizes);
}

Value coerceTensorShape(OpBuilder &builder, Location loc,
                        TypedValue<ShapedType> value, ShapedType targetType) {
  return builder.createOrFold<tensor::CastOp>(
      loc, targetType.cloneWith(std::nullopt, value.getType().getElementType()),
      value);
}

LogicalResult verifyHloOpBufferOrTensorSemantics(Operation *op) {
  auto isRankedTensor = [](Value val) {
    return isa<RankedTensorType>(val.getType());
  };
  if (!llvm::all_of(op->getOperands(), isRankedTensor)) return failure();
  return success(llvm::all_of(op->getResults(), isRankedTensor));
}

Value fillTensorWithZeros(OpBuilder &builder, Location loc, Value tensor) {
  auto type = cast<ShapedType>(tensor.getType());
  Value zero;
  // Complex numbers are a special case.
  if (auto complexType = llvm::dyn_cast<ComplexType>(type.getElementType())) {
    auto zeroElement = builder.getZeroAttr(complexType.getElementType());
    auto zeroAttr = builder.getArrayAttr({zeroElement, zeroElement});
    zero = builder.create<complex::ConstantOp>(loc, complexType, zeroAttr);
  } else {
    auto zeroAttr = builder.getZeroAttr(type.getElementType());
    zero = builder.create<arith::ConstantOp>(loc, zeroAttr);
  }
  return builder.create<linalg::FillOp>(loc, zero, tensor).result();
}

Value preSparsify(Operation *op, llvm::SmallVector<Value, 2> &values, Type rtp,
                  OpBuilder *b) {
  // Apply for semi-ring operations that lower to elaborate code
  // (any sign-op, or an integral abs-op).
  // TODO(peiming, ajcbik): these all can potentially be optimized by applying
  // value transform on sparse_tenosr.value memref
  if (isa<mlir::stablehlo::SignOp>(op) || isa<mlir::stablehlo::NegOp>(op) ||
      (isa<mlir::stablehlo::AbsOp>(op) && hasIntegralShapeType(op)) ||
      isa<chlo::AsinOp>(op) || isa<chlo::AsinhOp>(op) ||
      isa<chlo::AtanOp>(op) || isa<chlo::AtanhOp>(op) ||
      isa<chlo::BesselI1eOp>(op) || isa<chlo::SinhOp>(op) ||
      isa<chlo::TanOp>(op)) {
    if (!sparse_tensor::getSparseTensorEncoding(op->getResult(0).getType()) &&
        !sparse_tensor::getSparseTensorEncoding(op->getOperand(0).getType()))
      return Value();
    Location loc = op->getLoc();
    auto semiring = b->create<sparse_tensor::UnaryOp>(loc, rtp, values[0]);
    Type itp = values[0].getType();
    Block *present = b->createBlock(&semiring.getPresentRegion(), {}, itp, loc);
    b->setInsertionPointToStart(&semiring.getPresentRegion().front());
    values[0] = present->getArgument(0);
    return semiring;
  }
  return Value();
}

Value postSparsify(Operation *op, Value semiring, Value result, OpBuilder *b) {
  if (semiring) {
    b->create<sparse_tensor::YieldOp>(op->getLoc(), result);
    b->setInsertionPointAfter(semiring.getDefiningOp());
    return semiring;
  }
  return result;
}

bool allOperandsAreScalarTensors(Operation *op) {
  return llvm::all_of(op->getOperands(), [](Value operand) {
    auto operandTy = llvm::dyn_cast<ShapedType>(operand.getType());
    return operandTy && operandTy.getRank() == 0;
  });
}

bool isInBodyOfLinalgOps(Operation *op) {
  auto *parentOp = op->getParentRegion()->getParentOp();
  return parentOp->getDialect() ==
         parentOp->getContext()->getLoadedDialect<linalg::LinalgDialect>();
}

SmallVector<int64_t> extract1DVector(DenseIntElementsAttr elements) {
  SmallVector<int64_t> ret;
  for (const APInt &element : elements) {
    ret.push_back(element.getLimitedValue());
  }
  return ret;
}

}  // namespace mlir::stablehlo
