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

#ifndef STABLEHLO_INTEGRATIONS_PYTHON_API_STABLEHLOAPI_H
#define STABLEHLO_INTEGRATIONS_PYTHON_API_STABLEHLOAPI_H

#include "pybind11/pybind11.h"

namespace mlir {
namespace stablehlo {

// Add StableHLO APIs to the pybind11 module.
// Signatures of these APIs have no dependency on C++ MLIR types and all must
// use C API passthrough.
void AddStablehloApi(pybind11::module& m);

// Adds a subset of the StableHLO API that doesn't use MLIR in any definitions,
// and is methods only, introducing no new objects / enums to avoid potential
// redefinition issues in complex build environments.
void AddPortableApi(pybind11::module& m);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_INTEGRATIONS_PYTHON_API_STABLEHLOAPI_H
