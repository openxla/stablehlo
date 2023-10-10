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

#include "stablehlo/tests/CheckOps.h"

#include <fstream>

#define GET_OP_CLASSES
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/NumPy.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/tests/CheckOps.cpp.inc"

namespace mlir {
namespace stablehlo {
namespace check {

//===----------------------------------------------------------------------===//
// Check Dialect Constructor
//===----------------------------------------------------------------------===//

CheckDialect::CheckDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<CheckDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "stablehlo/tests/CheckOps.cpp.inc"
      >();
}

llvm::Error evalExpectAlmostEqConstOp(const Tensor &lhs, ElementsAttr value) {
  auto rhs = makeTensor(value.cast<DenseElementsAttr>());
  return evalExpectAlmostEqOp(lhs, rhs);
}

llvm::Error evalExpectAlmostEqOp(const Tensor &lhs, const Tensor &rhs) {
  for (auto lhsIt = lhs.index_begin(), rhsIt = rhs.index_begin();
       lhsIt != lhs.index_end(); ++lhsIt, ++rhsIt)
    if (!areApproximatelyEqual(lhs.get(*lhsIt), rhs.get(*rhsIt))
             .getBooleanValue())
      return invalidArgument(
          "Element values don't match: %s (actual) vs %s (expected) at index "
          "%s\n",
          debugString(lhs.get(*lhsIt)).c_str(),
          debugString(rhs.get(*rhsIt)).c_str(), debugString((*lhsIt)).c_str());

  return llvm::Error::success();
}

llvm::Error evalExpectEqConstOp(const Tensor &lhs, ElementsAttr value) {
  auto rhs = makeTensor(value.cast<DenseElementsAttr>());
  return evalExpectEqOp(lhs, rhs);
}

llvm::Error evalExpectEqOp(const Tensor &lhs, const Tensor &rhs) {
  for (auto lhsIt = lhs.index_begin(), rhsIt = rhs.index_begin();
       lhsIt != lhs.index_end(); ++lhsIt, ++rhsIt)
    if ((lhs.get(*lhsIt) != rhs.get(*rhsIt)).getBooleanValue())
      return invalidArgument(
          "Element values don't match: %s (actual) vs %s (expected) at index "
          "%s\n",
          debugString(lhs.get(*lhsIt)).c_str(),
          debugString(rhs.get(*rhsIt)).c_str(), debugString((*lhsIt)).c_str());

  return llvm::Error::success();
}

// Fetch a previously serialized filepath given a `probeId` and a `probeDir` for
// a specified `iteration` value from an `index.csv` metadata file. If no data
// is found, returns an error.
static llvm::ErrorOr<std::string> getSerializedTensorPath(StringRef probeId,
                                                          StringRef probeDir,
                                                          uint32_t iteration) {
  if (probeDir.empty()) return llvm::errc::invalid_argument;

  llvm::SmallString<128> instrumentationMetadataFile(probeDir);
  llvm::sys::path::append(instrumentationMetadataFile,
                          numpy::kInstrumentationMetadataFilename);
  std::ifstream metadataFile(instrumentationMetadataFile.str().str());

  if (!metadataFile.is_open()) return llvm::errc::io_error;

  std::string line;

  for (uint32_t match = 0; metadataFile >> line && match <= iteration;
       ++match) {
    auto pos = line.find(probeId);

    if (pos != std::string::npos && match == iteration)
      return line.substr(pos + probeId.size() + 1);
  }

  return llvm::errc::bad_address;
}

llvm::Error evalExpectSerializedEqOp(const Tensor &expected, StringRef probeId,
                                     StringRef probeDir, uint32_t iteration) {
  auto serializedFilePathOrError =
      getSerializedTensorPath(probeId, probeDir, iteration);

  if (!serializedFilePathOrError)
    return llvm::createStringError(serializedFilePathOrError.getError(),
                                   "Failed to find serialized data for probe");

  auto tensorOrError =
      numpy::deserializeTensor(*serializedFilePathOrError, expected.getType());

  if (!tensorOrError)
    return llvm::createStringError(tensorOrError.getError(),
                                   "Failed to verify serialized tensor.");

  return evalExpectEqOp(expected, *tensorOrError);
}

}  // namespace check
}  // namespace stablehlo
}  // namespace mlir
