/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#include "stablehlo/integrations/c/StablehloUnifiedApi.h"

#include <vector>

#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/api/PortableApi.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/Version.h"

#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"

MlirAttribute stablehloEvalModule(MlirModule module, int nArgs,
                                  MlirAttribute const *args, int *errorCode) {
  std::vector<mlir::DenseElementsAttr> inputs;
  inputs.reserve(nArgs);
  for (int i = 0; i < nArgs; ++i) {
    inputs.push_back(llvm::cast<mlir::DenseElementsAttr>(unwrap(args[i])));
  }
  mlir::stablehlo::InterpreterConfiguration config;
  mlir::FailureOr<llvm::SmallVector<mlir::DenseElementsAttr>> results =
      mlir::stablehlo::evalModule(unwrap(module), inputs, config);
  if (mlir::failed(results)) {
    *errorCode = 1;
    return MlirAttribute{nullptr};
  }
  std::vector<MlirAttribute> resultsVec;
  for (const auto &result : results.value()) {
    resultsVec.push_back(wrap(result));
  }
  return mlirArrayAttrGet(mlirModuleGetContext(module), resultsVec.size(),
                          resultsVec.data());
}
