/* Copyright 2026 The OpenXLA Authors.

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

#include "stablehlo/dialect/Revision.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "stablehlo/dialect/Version.h"

// Generated at build time, defines STABLEHLO_REVISION when available.
#include "StablehloRevision.h"

namespace mlir {
namespace stablehlo {

llvm::StringRef getRevision() {
#ifdef STABLEHLO_REVISION
  return STABLEHLO_REVISION;
#else
  return "";
#endif
}

void printVersion(llvm::raw_ostream& os) {
  llvm::StringRef revision = getRevision();
  os << "StableHLO revision " << (revision.empty() ? "(unknown)" : revision)
     << "\n";
  os << "StableHLO VHLO opset version "
     << vhlo::Version::getCurrentVersion().toString() << "\n";
}

}  // namespace stablehlo
}  // namespace mlir
