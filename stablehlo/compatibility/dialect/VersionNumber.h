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

#ifndef STABLEHLO_COMPATIBILITY_VERSION_NUMBER_H
#define STABLEHLO_COMPATIBILITY_VERSION_NUMBER_H

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace vhlo {

class VersionNumber {
  static constexpr llvm::StringLiteral CURRENT_VERSION = "0.1.0";
  static constexpr llvm::StringLiteral MINIMUM_VERSION = "0.0.0";

 public:
  static FailureOr<VersionNumber> get(llvm::StringRef versionRef) {
    if (failed(VersionNumber::validateVersionString(versionRef))) {
      return failure();
    }
    return VersionNumber(versionRef);
  }

  static VersionNumber getCurrent() { return VersionNumber(CURRENT_VERSION); }
  static VersionNumber getMinimumSupported() {
    return VersionNumber(MINIMUM_VERSION);
  }

  /// Validate version argument is one of {current, minimum, #.#.#}
  static LogicalResult validateVersionString(llvm::StringRef versionRef) {
    if (versionRef == "current" || versionRef == "minimum") {
      return success();
    }

    llvm::Regex versionRegex("^([0-9]+)\\.([0-9]+)\\.([0-9]+)$");
    llvm::SmallVector<llvm::StringRef> matches;
    return success(/*isSuccess=*/versionRegex.match(versionRef, &matches));
  }

  int64_t getMinorVersion() const {
    if (version == "current")
      return parseMinorVersion(VersionNumber::CURRENT_VERSION);
    if (version == "minimum")
      return parseMinorVersion(VersionNumber::MINIMUM_VERSION);
    return parseMinorVersion(version);
  }

  bool operator<(VersionNumber const& other) {
    return getMinorVersion() < other.getMinorVersion();
  }
  bool operator==(VersionNumber const& other) {
    return getMinorVersion() == other.getMinorVersion();
  }
  bool operator<=(VersionNumber const& other) {
    return *this < other || *this == other;
  }

 private:
  VersionNumber(llvm::StringRef versionRef) : version(versionRef) {
    assert(succeeded(VersionNumber::validateVersionString(versionRef)));
  }

  int64_t parseMinorVersion(llvm::StringRef versionRef) const {
    // Precondition: must be x.y.z
    auto isDot = [](char c) { return c == '.'; };
    auto minorS = versionRef.drop_until(isDot).drop_front(1).take_until(isDot);
    int64_t minor;
    if (minorS.getAsInteger(/*radix=*/10, minor)) {
      llvm_unreachable("failed to parse minor version");
    }
    return minor;
  }

  std::string version;
};

}  // namespace vhlo
}  // namespace mlir

#endif  // STABLEHLO_COMPATIBILITY_VERSION_NUMBER_H