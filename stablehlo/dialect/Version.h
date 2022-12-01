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

#ifndef STABLEHLO_DIALECT_VERSION_H
#define STABLEHLO_DIALECT_VERSION_H

#include <algorithm>
#include <cstdint>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace vhlo {

class Version {
 public:
  /// Convenience method to extract major, minor, patch and create a Version
  /// from a StringRef. Returns failure if invalid string.
  static FailureOr<Version> get(llvm::StringRef versionRef) {
    auto failOrVersionArray = Version::extractVersionNumbers(versionRef);
    if (failed(failOrVersionArray)) {
      return failure();
    }

    auto versionArr = *failOrVersionArray;
    return Version(versionArr);
  }

  /// Construct Version from major, minor, patch integers.
  Version(std::array<int64_t, 3> majorMinorPatch)
      : majorMinorPatch(majorMinorPatch) {}

  std::array<int64_t, 3> getAsArray() const {
    return majorMinorPatch;
  }

  bool operator<(Version const& other) {
    // Uses lexicographical_compare
    return majorMinorPatch < other.majorMinorPatch;
  }
  bool operator==(Version const& other) {
    return majorMinorPatch == other.majorMinorPatch;
  }
  bool operator<=(Version const& other) {
    return majorMinorPatch <= other.majorMinorPatch;
  }

 private:
  /// Validate version argument is `#.#.#` (ex: 0.1.0, 1.2.3, 0.123.0) 
  /// Returns the vector of 3 matches (major, minor, patch) if successful,
  /// else returns failure.  
  static FailureOr<std::array<int64_t, 3>> extractVersionNumbers(llvm::StringRef versionRef) {
    llvm::Regex versionRegex("^([0-9]+)\\.([0-9]+)\\.([0-9]+)$");
    llvm::SmallVector<llvm::StringRef> matches;
    if (!versionRegex.match(versionRef, &matches)) {
      return failure();
    }
    return std::array<int64_t, 3>{parseNumber(matches[1]),
                                  parseNumber(matches[2]),
                                  parseNumber(matches[3])};
  }

  static int64_t parseNumber(llvm::StringRef numRef) {
    int64_t num;
    if (numRef.getAsInteger(/*radix=*/10, num)) {
      llvm_unreachable("failed to parse version number");
    }
    return num;
  }

  std::array<int64_t,3> majorMinorPatch;
};

}  // namespace vhlo
}  // namespace mlir

#endif  // STABLEHLO_DIALECT_VERSION_H