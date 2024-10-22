/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.

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

#include <array>
#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"

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

}  // namespace

FailureOr<Version> Version::fromString(llvm::StringRef versionRef) {
  auto failOrVersionArray = extractVersionNumbers(versionRef);
  if (failed(failOrVersionArray)) return failure();
  auto versionArr = *failOrVersionArray;
  return Version(versionArr[0], versionArr[1], versionArr[2]);
}

FailureOr<int64_t> Version::getBytecodeVersion() const {
  if (*this < Version(0, 9, 0)) return failure();
  if (*this < Version(0, 10, 0)) return 0;
  if (*this < Version(0, 12, 0)) return 1;
  if (*this < Version(0, 14, 0)) return 3;
  if (*this < Version(0, 15, 0)) return 4;  // (revised from 5 to 4 in #1827)
  if (*this <= getCurrentVersion()) return 6;
  return failure();
}

Version Version::fromCompatibilityRequirement(
    CompatibilityRequirement requirement) {
  // Compatibility requirement versions can be updated as needed, as long as the
  // version satisifies the requirement.
  // The time frames used are from the date that the release was tagged on, not
  // merged. The tag date is when the version has been verified and exported to
  // XLA. See: https://github.com/openxla/stablehlo/tags
  switch (requirement) {
    case CompatibilityRequirement::NONE:
      return Version::getCurrentVersion();
    case CompatibilityRequirement::WEEK_4:
      return Version(1, 7, 3);  // v1.7.3 - Sept 23, 2024
    case CompatibilityRequirement::WEEK_12:
      return Version(1, 4, 2);  // v1.4.2 - Jul 25, 2024
    case CompatibilityRequirement::MAX:
      return Version::getMinimumVersion();
  }
  llvm::report_fatal_error("Unhandled compatibility requirement");
}

mlir::Diagnostic& operator<<(mlir::Diagnostic& diag, const Version& version) {
  return diag << version.toString();
}
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Version& version) {
  return os << version.toString();
}

}  // namespace vhlo
}  // namespace mlir
