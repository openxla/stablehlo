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

#ifndef STABLEHLO_DIALECT_VERSION_H
#define STABLEHLO_DIALECT_VERSION_H

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace vhlo {

class Version {
 public:
  /// Convenience method to extract major, minor, patch and create a Version
  /// from a StringRef of the form `#.#.#`. Returns failure if invalid string.
  static FailureOr<Version> fromString(llvm::StringRef versionRef);

  /// Return a Version representing the current VHLO dialect version.
  static Version getCurrentVersion() { return Version(0, 14, 4); }

  /// Return a Version representing the minimum supported VHLO dialect version.
  static Version getMinimumVersion() { return Version(0, 9, 0); }

  /// Return the MLIR Bytecode Format associated with the version instance.
  /// Returns failure if version is not in compatibility window.
  FailureOr<int64_t> getBytecodeVersion() const;

  /// Construct Version from major, minor, patch integers.
  Version(int64_t major, int64_t minor, int64_t patch)
      : majorMinorPatch({major, minor, patch}) {}

  int64_t getMajor() const { return majorMinorPatch[0]; }
  int64_t getMinor() const { return majorMinorPatch[1]; }
  int64_t getPatch() const { return majorMinorPatch[2]; }

  bool operator<(const Version& other) const {
    // Uses lexicographical_compare
    return majorMinorPatch < other.majorMinorPatch;
  }
  bool operator==(const Version& other) const {
    return majorMinorPatch == other.majorMinorPatch;
  }
  bool operator<=(const Version& other) const {
    return majorMinorPatch <= other.majorMinorPatch;
  }
  std::string toString() const {
    std::ostringstream os;
    os << getMajor() << '.' << getMinor() << '.' << getPatch();
    return os.str();
  }

 private:
  std::array<int64_t, 3> majorMinorPatch;
};

mlir::Diagnostic& operator<<(mlir::Diagnostic& diag, const Version& version);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Version& version);

}  // namespace vhlo
}  // namespace mlir

#endif  // STABLEHLO_DIALECT_VERSION_H
