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

#include "StablehloValue.h"

#include "mlir/IR/Types.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/reference/Token.h"

namespace mlir {
namespace stablehlo {

StablehloValue::StablehloValue(const Tensor &tensor)
    : type_(tensor.getType()), value_(tensor) {}
StablehloValue::StablehloValue(const Token &token)
    : type_(token.getType()), value_(token) {}

Tensor StablehloValue::getTensor() const { return std::get<Tensor>(value_); }

Token StablehloValue::getToken() const { return std::get<Token>(value_); }

Type StablehloValue::getType() const { return type_; }

bool StablehloValue::isTensor() const {
  return std::holds_alternative<Tensor>(value_);
}

bool StablehloValue::isToken() const {
  return std::holds_alternative<Token>(value_);
}

void StablehloValue::print(raw_ostream &os) const {
  if (isTensor())
    getTensor().print(os);
  else if (isToken())
    getToken().print(os);
}

void StablehloValue::dump() const { print(llvm::errs()); }

}  // namespace stablehlo
}  // namespace mlir
