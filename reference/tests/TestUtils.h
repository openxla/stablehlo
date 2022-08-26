/* Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_REFERENCE_TESTS_RUNTESTUTILS_H_
#define STABLEHLO_REFERENCE_TESTS_RUNTESTUTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace stablehlo {

/// Helper function to setup an mlir function, provided as a string, evaluate it
/// using the provided tensor inputs, and finally match with the provided
/// expected result tensor(s). The tensor provided as input or expected result
/// is a one-dimensional flattened format of the tensor data values. The
/// flattening follows the minor-to-major dimension order of N-1 down to 0 for
/// an N-D Tensor.
void runTestCase(llvm::StringRef sourceStr,
                 llvm::ArrayRef<llvm::ArrayRef<llvm::StringRef>>
                     operandsAndexpectedResultValues);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_TESTS_RUNTESTUTILS_H_
