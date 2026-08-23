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

#ifndef STABLEHLO_DIALECT_REVISION_H
#define STABLEHLO_DIALECT_REVISION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace stablehlo {

// Returns the StableHLO source revision embedded at build time, or an empty
// string if no revision was embedded (e.g. built from a source archive).
llvm::StringRef getRevision();

// Prints StableHLO version information to `os`. Intended for use with
// llvm::cl::AddExtraVersionPrinter so tools report the StableHLO revision in
// addition to the LLVM version.
void printVersion(llvm::raw_ostream& os);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_DIALECT_REVISION_H
