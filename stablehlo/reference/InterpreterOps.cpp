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

#include "stablehlo/reference/InterpreterOps.h"

#define GET_OP_CLASSES
#include "stablehlo/reference/InterpreterOps.cpp.inc"

namespace mlir {
namespace stablehlo {
namespace interpreter {

//===----------------------------------------------------------------------===//
// Interpreter Dialect Constructor
//===----------------------------------------------------------------------===//

InterpreterDialect::InterpreterDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              TypeID::get<InterpreterDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "stablehlo/reference/InterpreterOps.cpp.inc"
      >();
}

}  // namespace interpreter
}  // namespace stablehlo
}  // namespace mlir
