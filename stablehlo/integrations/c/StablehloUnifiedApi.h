/* Copyright 2024 The StableHLO Authors.
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

#ifndef STABLEHLO_INTEGRATIONS_C_STABLEHLOREFERENCEAPI_H_
#define STABLEHLO_INTEGRATIONS_C_STABLEHLOREFERENCEAPI_H_

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

// Re-exports dialect capi
#include "stablehlo/integrations/c/StablehloDialectApi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Entrypoint for calling the StableHLO reference interpreter.
// Returns an array attribute of dense element attributes for results.
// Sets error code to non-zero on failure.
MLIR_CAPI_EXPORTED MlirAttribute stablehloEvalModule(MlirModule module,
                                                     int nArgs,
                                                     MlirAttribute const* args,
                                                     int* errorCode);

#ifdef __cplusplus
}
#endif

#endif  // STABLEHLO_INTEGRATIONS_C_STABLEHLOREFERENCEAPI_H_
