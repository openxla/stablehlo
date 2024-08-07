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

#include "stablehlo/reference/Value.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/reference/Token.h"

namespace mlir {
namespace stablehlo {

//===----------------------------------------------------------------------===//
// Tuple.
//===----------------------------------------------------------------------===//

Tuple::Tuple(ArrayRef<InterpreterValue> val, TupleType type) : type_(type) {
  for (auto value : val)
    values_.push_back(std::make_shared<InterpreterValue>(std::move(value)));
}

InterpreterValue Tuple::get(int32_t index) const { return *values_[index]; }

TupleType Tuple::getType() const { return type_; }

void Tuple::print(raw_ostream &os) const {
  getType().print(os);
  os << " (\n";
  for (size_t i = 0; i < values_.size(); ++i) {
    values_[i]->dump();
    if (i != values_.size() - 1) os << ",";
    os << "\n";
  }
  os << ")";
}

void Tuple::dump() const { print(llvm::errs()); }

//===----------------------------------------------------------------------===//
// InterpreterValue.
//===----------------------------------------------------------------------===//

InterpreterValue::InterpreterValue(const Tensor &tensor) : value_(tensor) {}
InterpreterValue::InterpreterValue(const Token &token) : value_(token) {}
InterpreterValue::InterpreterValue(const Tuple &tuple) : value_(tuple) {}

Tensor InterpreterValue::getTensor() const {
  if (!isTensor())
    report_fatal_error(invalidArgument("InterpreterValue is not a Tensor."));

  return std::get<Tensor>(value_);
}

Token InterpreterValue::getToken() const {
  if (!isToken())
    report_fatal_error(invalidArgument("InterpreterValue is not a Token."));

  return std::get<Token>(value_);
}

Tuple InterpreterValue::getTuple() const {
  if (!isTuple())
    report_fatal_error(invalidArgument("InterpreterValue is not a Tuple."));

  return std::get<Tuple>(value_);
}

Type InterpreterValue::getType() const {
  if (isTensor()) return getTensor().getType();
  if (isToken()) return getToken().getType();
  if (isTuple()) return getTuple().getType();

  report_fatal_error(invalidArgument("Unsupported interpreter value."));
}

bool InterpreterValue::isTensor() const {
  return std::holds_alternative<Tensor>(value_);
}

bool InterpreterValue::isToken() const {
  return std::holds_alternative<Token>(value_);
}

bool InterpreterValue::isTuple() const {
  return std::holds_alternative<Tuple>(value_);
}

void InterpreterValue::print(raw_ostream &os) const {
  if (isTensor())
    getTensor().print(os);
  else if (isToken())
    getToken().print(os);
  else if (isTuple())
    getTuple().print(os);
  else
    report_fatal_error(invalidArgument("Unsupported interpreter value."));
}

void InterpreterValue::dump() const { print(llvm::errs()); }

}  // namespace stablehlo
}  // namespace mlir
