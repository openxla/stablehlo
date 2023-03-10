/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#include "stablehlo/dialect/Version.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir {
namespace vhlo {
namespace {
// Helper function for number to string.
// Precondition that numRef is a valid decimal digit.
static int64_t parseNumber(llvm::StringRef numRef) {
  int64_t num;
  if (numRef.getAsInteger(/*radix=*/10, num)) {
    llvm::report_fatal_error("failed to parse version number");
  }
  return num;
}

/// Validate version argument is `#.#.#` (ex: 0.9.0, 0.99.0, 1.2.3)
/// Returns the vector of 3 matches (major, minor, patch) if successful,
/// else returns failure.
static FailureOr<std::array<int64_t, 3>> extractVersionNumbers(
    llvm::StringRef versionRef) {
  llvm::Regex versionRegex("^([0-9]+)\\.([0-9]+)\\.([0-9]+)$");
  llvm::SmallVector<llvm::StringRef> matches;
  if (!versionRegex.match(versionRef, &matches)) return failure();
  return std::array<int64_t, 3>{parseNumber(matches[1]),
                                parseNumber(matches[2]),
                                parseNumber(matches[3])};
}

template <typename OutputT>
OutputT& dump(OutputT& out, const Version& version) {
  return out << version.getMajor() << '.' << version.getMinor() << '.'
             << version.getPatch();
}
}  // namespace

FailureOr<Version> Version::fromString(llvm::StringRef versionRef) {
  auto failOrVersionArray = extractVersionNumbers(versionRef);
  if (failed(failOrVersionArray)) return failure();
  auto versionArr = *failOrVersionArray;
  return Version(versionArr[0], versionArr[1], versionArr[2]);
}

mlir::Diagnostic& operator<<(mlir::Diagnostic& diag, const Version& version) {
  return dump<mlir::Diagnostic>(diag, version);
}
llvm::raw_ostream& operator<<(llvm::raw_ostream& diag, const Version& version) {
  return dump<llvm::raw_ostream>(diag, version);
}

}  // namespace vhlo
}  // namespace mlir
