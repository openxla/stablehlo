/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include <optional>

#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {
namespace {

struct SortBuildResult {
  bool isValid;
  std::optional<ComparisonType> comparisonType;
};

SortBuildResult buildAndVerifySort(Type keyElementType,
                                   Type payloadElementType) {
  MLIRContext* context = keyElementType.getContext();
  context->getOrLoadDialect<func::FuncDialect>();
  context->getOrLoadDialect<StablehloDialect>();

  PatternRewriter rewriter(context);
  Location loc = rewriter.getUnknownLoc();
  auto keyType = RankedTensorType::get({2}, keyElementType);
  auto payloadType = RankedTensorType::get({2}, payloadElementType);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  auto function = func::FuncOp::create(
      rewriter, loc, "sort",
      rewriter.getFunctionType({keyType, payloadType}, {}));
  module->push_back(function);

  Block* entry = function.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);
  SortOp sort = createSortOp(
      &rewriter, loc, {entry->getArgument(0), entry->getArgument(1)},
      {keyElementType, payloadElementType}, /*dimension=*/0,
      /*isStable=*/false, ComparisonDirection::LT);
  func::ReturnOp::create(rewriter, loc);

  auto compare = cast<CompareOp>(sort.getComparator().front().front());
  return {succeeded(verify(*module)), compare.getCompareType()};
}

TEST(CreateSortOpTest, UsesDefaultComparisonForIntegerKeyWithFloatPayload) {
  MLIRContext context;
  SortBuildResult result = buildAndVerifySort(IntegerType::get(&context, 32),
                                              Float32Type::get(&context));

  EXPECT_TRUE(result.isValid);
  EXPECT_FALSE(result.comparisonType.has_value());
}

TEST(CreateSortOpTest, UsesTotalOrderComparisonForFloatKey) {
  MLIRContext context;
  SortBuildResult result = buildAndVerifySort(Float32Type::get(&context),
                                              IntegerType::get(&context, 32));

  EXPECT_TRUE(result.isValid);
  ASSERT_TRUE(result.comparisonType.has_value());
  EXPECT_EQ(*result.comparisonType, ComparisonType::TOTALORDER);
}

}  // namespace
}  // namespace stablehlo
}  // namespace mlir
