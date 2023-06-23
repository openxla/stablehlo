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

#include "stablehlo/reference/InterpreterValue.h"

#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Types.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/reference/Token.h"

namespace mlir {
namespace stablehlo {

InterpreterValue::InterpreterValue(const Tensor &tensor) : value_(tensor) {}
InterpreterValue::InterpreterValue(const Token &token) : value_(token) {}

Tensor InterpreterValue::getTensor() const {
  if (!isTensor())
    llvm::report_fatal_error("InterpreterValue is not a Tensor.");

  return std::get<Tensor>(value_);
}

Token InterpreterValue::getToken() const {
  if (!isToken()) llvm::report_fatal_error("InterpreterValue is not a Token.");

  return std::get<Token>(value_);
}

Type InterpreterValue::getType() const {
  if (isTensor()) return getTensor().getType();
  if (isToken()) return getToken().getType();

  report_fatal_error(invalidArgument("Unsupported interpreter value."));
}

bool InterpreterValue::isTensor() const {
  return std::holds_alternative<Tensor>(value_);
}

bool InterpreterValue::isToken() const {
  return std::holds_alternative<Token>(value_);
}

void InterpreterValue::print(raw_ostream &os) const {
  if (isTensor())
    getTensor().print(os);
  else if (isToken())
    getToken().print(os);
  else
    report_fatal_error(invalidArgument("Unsupported interpreter value."));
}

void InterpreterValue::dump() const { print(llvm::errs()); }

}  // namespace stablehlo
}  // namespace mlir
