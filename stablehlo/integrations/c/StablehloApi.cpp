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

#include "stablehlo/integrations/c/StablehloApi.h"

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

int stablehloGetApiVersion() { return mlir::stablehlo::getApiVersion(); }

mlir::vhlo::Version::CompatibilityRequirement unwrapCompatibilityRequirement(
    MlirStablehloCompatibilityRequirement requirement) {
  switch (requirement) {
    case MlirStablehloCompatibilityRequirement::NONE:
      return mlir::vhlo::Version::CompatibilityRequirement::NONE;
    case MlirStablehloCompatibilityRequirement::WEEK_4:
      return mlir::vhlo::Version::CompatibilityRequirement::WEEK_4;
    case MlirStablehloCompatibilityRequirement::WEEK_12:
      return mlir::vhlo::Version::CompatibilityRequirement::WEEK_12;
    case MlirStablehloCompatibilityRequirement::MAX:
      return mlir::vhlo::Version::CompatibilityRequirement::MAX;
  }
  llvm::report_fatal_error("unhandled compatibility requirement");
}

void stablehloVersionFromCompatibilityRequirement(
    MlirStablehloCompatibilityRequirement requirement,
    MlirStringCallback callback, void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  stream << mlir::vhlo::Version::fromCompatibilityRequirement(
      unwrapCompatibilityRequirement(requirement));
}

void stablehloGetCurrentVersion(MlirStringCallback callback, void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  stream << mlir::stablehlo::getCurrentVersion();
}

void stablehloGetMinimumVersion(MlirStringCallback callback, void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  stream << mlir::stablehlo::getMinimumVersion();
}

MlirLogicalResult stablehloGetSmallerVersion(MlirStringRef version1,
                                             MlirStringRef version2,
                                             MlirStringCallback callback,
                                             void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  auto result =
      mlir::stablehlo::getSmallerVersion(unwrap(version1), unwrap(version2));
  if (mlir::failed(result)) return mlirLogicalResultFailure();
  stream << result.value();
  return mlirLogicalResultSuccess();
}

MlirLogicalResult stablehloSerializePortableArtifactFromModule(
    MlirModule moduleStr, MlirStringRef targetVersion,
    MlirStringCallback callback, void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  if (failed(mlir::stablehlo::serializePortableArtifact(
          unwrap(moduleStr), unwrap(targetVersion), stream)))
    return mlirLogicalResultFailure();
  return mlirLogicalResultSuccess();
}

MlirLogicalResult stablehloSerializePortableArtifactFromStringRef(
    MlirStringRef moduleStr, MlirStringRef targetVersion,
    MlirStringCallback callback, void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  if (failed(mlir::stablehlo::serializePortableArtifact(
          unwrap(moduleStr), unwrap(targetVersion), stream)))
    return mlirLogicalResultFailure();
  return mlirLogicalResultSuccess();
}

MlirLogicalResult stablehloDeserializePortableArtifact(
    MlirStringRef artifactStr, MlirStringCallback callback, void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  if (failed(mlir::stablehlo::deserializePortableArtifact(unwrap(artifactStr),
                                                          stream)))
    return mlirLogicalResultFailure();
  return mlirLogicalResultSuccess();
}

MlirModule stablehloDeserializePortableArtifactNoError(
    MlirStringRef artifactStr, MlirContext ctx) {
  return wrap(mlir::stablehlo::deserializePortableArtifact(unwrap(artifactStr),
                                                           unwrap(ctx))
                  .release());
}

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
