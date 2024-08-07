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

#include "stablehlo/reference/Token.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {

Token::Token(MLIRContext *context) : type_(TokenType::get(context)) {}

TokenType Token::getType() const { return type_; }

void Token::print(raw_ostream &os) const { getType().print(os); }

void Token::dump() const { print(llvm::errs()); }

}  // namespace stablehlo
}  // namespace mlir
