/* Copyright 2023 The JAX Authors.

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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOCHECKSHAPEASSERTIONSPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

constexpr llvm::StringRef shapeAssertionName = "shape_assertion";
constexpr llvm::StringRef errorMessageAttrName = "error_message";
// We bound the number of error_message_inputs for using llvm::formatv
constexpr int maxErrorMessageInputs = 32;  // TODO(necula): Remove this bound

// This pass is needed when we have shape assertions. A shape assertion is
// represented via the `stablehlo.custom_call @shape_assertion`
// custom call, and represents an assertion that the first operand
// (`assert_what`) evaluates to `true`. The custom call also has an
// `error_message` string attribute, and a variadic number
// of integer scalar operands that may be used to format the error message.
// The `error_message` may contain format specifiers `{0}`, `{1}`, ..., that
// are replaced with the values of the error message inputs. The formatting is
// done with the `llvm::formatv` function
// (https://llvm.org/docs/ProgrammersManual.html#formatting-strings-the-formatv-function).
//
struct CheckShapeAssertionsPass
    : public impl::StablehloCheckShapeAssertionsPassBase<
          CheckShapeAssertionsPass> {
  using StablehloCheckShapeAssertionsPassBase::
      StablehloCheckShapeAssertionsPassBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([this](CustomCallOp op) {
      if (op.getCallTargetName() != shapeAssertionName) return;
      if (!enable_shape_assertions) {
        op.erase();
        return;
      }
      // Check first for ill-formed assertions, rather than silently fail.
      if (failed(verifyShapeAssertion(op))) {
        signalPassFailure();
        return;
      }
      OperandRange inputs = op.getInputs();
      SmallVector<int64_t> assertWhat;
      if (failed(hlo::matchInts(inputs[0], assertWhat))) {
        op.emitError() << "expects static assert_what (operand #0)";
        signalPassFailure();
        return;
      }
      if (assertWhat[0] != 0) {
        op.erase();
        return;
      }
      StringRef errorMessage = getErrorMessage(op);
      SmallVector<int64_t> errorMessageInputs;
      for (int i = 1; i < inputs.size(); ++i) {
        SmallVector<int64_t> input;
        if (failed(hlo::matchInts(inputs[i], input))) {
          op.emitError() << "expects static error_message_input (operand #" << i
                         << ")";
          signalPassFailure();
          return;
        }
        errorMessageInputs.push_back(input[0]);
      }
      op.emitError(formatErrorMessage(errorMessage, errorMessageInputs));
      signalPassFailure();
    });
  }

 private:
  LogicalResult verifyShapeAssertion(CustomCallOp op) {
    if (!(1 <= op->getNumOperands() &&
          op->getNumOperands() <= 1 + maxErrorMessageInputs))
      return op.emitError() << "expects 1 <= size(operands) <= "
                            << (1 + maxErrorMessageInputs);
    int nrErrorMessageInputs = op.getNumOperands() - 1;
    if (op->getNumResults() != 0)
      return op.emitError("expects size(results) = 0");
    for (const auto& attr : op->getAttrs()) {
      if (attr.getName() != "api_version" &&
          attr.getName() != "backend_config" &&
          attr.getName() != "call_target_name" &&
          attr.getName() != "error_message" &&
          attr.getName() != "has_side_effect")
        return op.emitError()
               << attr.getName() << " is not a supported attribute";
    }
    if (!op.hasEmptyBackendConfig())
      return op.emitError() << "expects an empty backend_config";
    if (op.getCallTargetName() != shapeAssertionName)
      return op.emitError() << "expects @shape_assertion";

    // input[0] (assert_what) : tensor<i1>
    auto assertWhatType = dyn_cast<ShapedType>(op.getInputs()[0].getType());
    if (!assertWhatType || !assertWhatType.hasRank() ||
        assertWhatType.getRank() != 0 ||
        !assertWhatType.getElementType().isSignlessInteger() ||
        assertWhatType.getElementTypeBitWidth() != 1)
      return op.emitError() << "expects assert_what (operand #0) "
                            << "to be a constant of type tensor<i1>";

    // input[1:] (error_message_inputs) : tensor<i32> or tensor<i64>
    for (int i = 0; i < nrErrorMessageInputs; ++i) {
      auto errorMessageInputType =
          dyn_cast<ShapedType>(op.getInputs()[i + 1].getType());
      if (!errorMessageInputType || !errorMessageInputType.hasRank() ||
          errorMessageInputType.getRank() != 0 ||
          !errorMessageInputType.getElementType().isSignlessInteger() ||
          (errorMessageInputType.getElementTypeBitWidth() != 32 &&
           errorMessageInputType.getElementTypeBitWidth() != 64))
        return op.emitError()
               << "expects error_message_input (operand #" << (i + 1) << ") "
               << "to be a constant of type tensor<i32> or tensor<i64>";
    }

    if (!op->hasAttr(errorMessageAttrName))
      return op.emitError() << "expects an error_message attribute";

    // error_message contains valid format specifiers.
    StringRef errorMessage = getErrorMessage(op);

    // format specs: "{" index ["," layout] [":" format] "}"
    size_t spec_begin = errorMessage.find_first_of('{');
    size_t spec_end = errorMessage.find_first_of(",:}", spec_begin);

    // Check that all specs reference valid input indices.
    while (spec_begin != StringRef::npos && spec_end != StringRef::npos) {
      StringRef index_str =
          errorMessage.substr(spec_begin + 1, spec_end - spec_begin - 1);

      int32_t index;
      if (!index_str.getAsInteger(10, index) &&
          !(0 <= index && index < nrErrorMessageInputs)) {
        return op.emitError()
               << "expects error_message to contain format specifiers with "
               << "error_message_input index less than " << nrErrorMessageInputs
               << ". Found specifier "
               << errorMessage.substr(spec_begin, spec_end - spec_begin + 1);
      }

      spec_begin = errorMessage.find_first_of('{', spec_begin + 1);
      spec_end = errorMessage.find_first_of(",:}", spec_begin);
    }
    return success();
  }

  StringRef getErrorMessage(CustomCallOp op) const {
    return cast<StringAttr>(op->getAttr(errorMessageAttrName)).getValue();
  }

  std::string formatErrorMessage(
      StringRef errorMessage,
      const SmallVector<int64_t>& errorMessageInputs) const {
    int nrErrorMessageInputs = errorMessageInputs.size();
    auto errorMessageFormat = errorMessage.data();
    if (nrErrorMessageInputs == 0) return errorMessageFormat;
    auto errInput = [nrErrorMessageInputs, &errorMessageInputs](int idx) {
      return (idx < nrErrorMessageInputs ? errorMessageInputs[idx] : -1);
    };
    return llvm::formatv(
        false, errorMessageFormat, errInput(0), errInput(1), errInput(2),
        errInput(3), errInput(4), errInput(5), errInput(6), errInput(7),
        errInput(8), errInput(9), errInput(10), errInput(11), errInput(12),
        errInput(13), errInput(14), errInput(15), errInput(16), errInput(17),
        errInput(18), errInput(19), errInput(20), errInput(21), errInput(22),
        errInput(23), errInput(24), errInput(25), errInput(26), errInput(27),
        errInput(28), errInput(29), errInput(30), errInput(31));
  }
};

}  // namespace

}  // namespace stablehlo
}  // namespace mlir
