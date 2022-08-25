#include "gtest/gtest.h"

#include "reference/Interpreter.h"
#include "reference/Tensor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {

void runTestCase(StringRef leftTyStr, ArrayRef<StringRef> leftData,
                 StringRef rightTyStr, ArrayRef<StringRef> rightData,
                 StringRef expectedTyStr, ArrayRef<StringRef> expectedData) {
  DialectRegistry registry;
  registry.insert<func::FuncDialect>();
  registry.insert<stablehlo::StablehloDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  OpBuilder builder(&context);

  // Builds the test module.
  Location loc = builder.getUnknownLoc();

  auto leftType = parseType(leftTyStr, &context).cast<ShapedType>();
  auto rightType = parseType(rightTyStr, &context).cast<ShapedType>();
  auto expectedType = parseType(expectedTyStr, &context).cast<ShapedType>();
  auto funcType =
      builder.getFunctionType({leftType, rightType}, {expectedType});
  auto funcOp = func::FuncOp::create(loc, "main", funcType, {});

  OwningOpRef<ModuleOp> moduleOp = ModuleOp::create(loc);
  moduleOp->push_back(funcOp);

  Block *block = funcOp.addEntryBlock();
  builder.setInsertionPointToEnd(block);
  auto addOp =
      builder.create<AddOp>(loc, funcOp.getArgument(0), funcOp.getArgument(1));
  builder.create<func::ReturnOp>(loc, addOp.getResult());

  // Provide Inputs.
  Tensor left = makeTensor(leftType, leftData);
  Tensor right = makeTensor(rightType, rightData);
  Tensor expectedResult = makeTensor(expectedType, expectedData);

  // Run the test model.
  auto results = eval(funcOp, {left, right});
  ASSERT_TRUE((bool)results) << toString(results.takeError());

  // Check results
  ASSERT_EQ(results->size(), 1);
  Tensor result = (*results)[0];
  ASSERT_EQ(result.getType(), expectedType);

  for (int i = 0; i < expectedType.getNumElements(); ++i) {
    EXPECT_EQ(result.get(i), expectedResult.get(i));
  }
}

// For each operand or expected result, the input is as follows:
// 1. A shaped type represented as string.
// 2. Numeric data, represented as an string-array, following the
//    default minor-to-major dimension order of N-1 down to 0 for an N
//    dimensional array.
TEST(AddOpInterpreterTest, F64) {
  runTestCase(/*operand 0*/
              "tensor<2x3xf64>", {"1.0", "2.0", "3.0", "4.0", "5.0", "6.0"},
              /*operand 1*/
              "tensor<2x3xf64>", {"7.0", "8.0", "9.0", "10.0", "11.0", "12.0"},
              /*expected result*/
              "tensor<2x3xf64>",
              {"8.0", "10.0", "12.0", "14.0", "16.0", "18.0"});
}

TEST(AddOpInterpreterTest, F32) {
  runTestCase(/*operand 0*/
              "tensor<1x34xf32>",
              {
                  "0.0",          "-0.0",         "0.125",        "0.25",
                  "0.5",          "0.375",        "0.75",         "1.5",
                  "0.1",          "0.2",          "0.4",          "0.8",
                  "1.0e+24",      "1.0e+36",      "1.17549e-38",  "3.14159",
                  "0.333333",     "12.375",       "68.123",       "2.0000002",
                  "0.87747e-39",  "-5.87747e-39", "-5.87747e-39", "1.17549e-38",
                  "-1.17549e-38", "-1.17549e-38", "1.17549e-38",  "1.76324e-38",
                  "2.35099e-38",  "0.0",          "inf",          "inf",
                  "-inf",         "inf",
              },
              /*operand 1*/
              "tensor<1x34xf32>",
              {
                  "0.0",          "-0.0",         "0.125",        "0.25",
                  "0.5",          "0.375",        "0.75",         "1.5",
                  "0.1",          "0.2",          "0.4",          "0.8",
                  "1.0e+24",      "1.0e+36",      "1.17549e-38",  "3.14159",
                  "0.333333",     "12.375",       "68.123",       "2.0000002",
                  "5.87747e-39",  "-5.87747e-39", "-5.87747e-39", "1.17549e-38",
                  "-1.17549e-38", "-1.17549e-38", "1.17549e-38",  "1.76324e-38",
                  "2.35099e-38",  "-0.0",         "0.0",          "inf",
                  "-inf",         "-inf",
              },
              /*expected result*/
              "tensor<1x34xf32>",
              {
                  "0.0",          "-0.0",         "0.25",         "0.5",
                  "1.0",          "0.75",         "1.5",          "3.0",
                  "0.2",          "0.4",          "0.8",          "1.6",
                  "2.0e+24",      "2.0e+36",      "2.35099e-38",  "6.28319",
                  "0.666667",     "24.75",        "136.246",      "4.0",
                  "1.17549e-38",  "-1.17549e-38", "-1.17549e-38", "2.35099e-38",
                  "-2.35099e-38", "-2.35099e-38", "2.35099e-38",  "3.52648e-38",
                  "4.70198e-38",  "0.0",          "inf",          "inf",
                  "-inf",         "nan",
              });
}

TEST(AddOpInterpreterTest, F16) {
  runTestCase(/*operand 0*/
              "tensor<2x3xf16>", {"1.0", "2.0", "3.0", "4.0", "5.0", "6.0"},
              /*operand 1*/
              "tensor<2x3xf16>", {"7.0", "8.0", "9.0", "10.0", "11.0", "12.0"},
              /*expected result*/
              "tensor<2x3xf16>",
              {"8.0", "10.0", "12.0", "14.0", "16.0", "18.0"});
}

TEST(AddOpInterpreterTest, BF16) {
  runTestCase(/*operand 0*/
              "tensor<2x3xbf16>", {"1.0", "2.0", "3.0", "4.0", "5.0", "6.0"},
              /*operand 1*/
              "tensor<2x3xbf16>", {"7.0", "8.0", "9.0", "10.0", "11.0", "12.0"},
              /*expected result*/
              "tensor<2x3xbf16>",
              {"8.0", "10.0", "12.0", "14.0", "16.0", "18.0"});
}

TEST(AddOpInterpreterTest, SInt8WithOv) {
  runTestCase(
      /*operand 0*/
      "tensor<1x9x1xi8>",
      {"1", "2", "-128", "-128", "16", "-16", "16", "127", "-127"},
      /*operand 1*/
      "tensor<1x9x1xi8>",
      {"127", "127", "-1", "-128", "16", "-16", "-16", "0", "0"},
      /*expected result*/
      "tensor<1x9x1xi8>",
      {"-128", "-127", "127", "0", "32", "-32", "0", "127", "-127"});
}

TEST(AddOpInterpreterTest, UInt8WithOv) {
  runTestCase(
      /*operand 0*/
      "tensor<2x2xui8>", {"1", "2", "255", "4"},
      /*operand 1*/
      "tensor<2x2xui8>", {"255", "255", "255", "10"},
      /*expected result*/
      "tensor<2x2xui8>", {"0", "1", "254", "14"});
}

TEST(AddOpInterpreterTest, SInt4WithOv) {
  runTestCase(
      /*operand 0*/
      "tensor<2x2x2xi4>", {"0", "1", "2", "-3", "7", "-8", "7", "-8"},
      /*operand 1*/
      "tensor<2x2x2xi4>", {"-8", "-1", "2", "-3", "1", "-1", "7", "-8"},
      /*expected result*/
      "tensor<2x2x2xi4>", {"-8", "0", "4", "-6", "-8", "7", "-2", "0"});
}

TEST(AddOpInterpreterTest, UInt4WithOv) {
  runTestCase(
      /*operand 0*/
      "tensor<2x2xui4>", {"0", "7", "15", "15"},
      /*operand 1*/
      "tensor<2x2xui4>", {"15", "7", "1", "15"},
      /*expected result*/
      "tensor<2x2xui4>", {"15", "14", "0", "14"});
}

TEST(AddOpInterpreterTest, Complex32) {
  runTestCase(
      /*operand 0*/
      "tensor<2xcomplex<f32>>", {"1.5", "2.5", "7.5", "5.5"},
      /*operand 1*/
      "tensor<2xcomplex<f32>>", {"1.5", "2.5", "7.5", "5.5"},
      /*expected result*/
      "tensor<2xcomplex<f32>>", {"3.0", "5.0", "15.0", "11.0"});
}

}  // namespace stablehlo
}  // namespace mlir
