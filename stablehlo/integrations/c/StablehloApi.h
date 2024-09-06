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

#ifndef STABLEHLO_INTEGRATIONS_C_STABLEHLOAPI_H_
#define STABLEHLO_INTEGRATIONS_C_STABLEHLOAPI_H_

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Get the current StableHLO API version.
//
// This value is incremented as needed to help integrate API changes.
MLIR_CAPI_EXPORTED int stablehloGetApiVersion();

typedef enum MlirStablehloCompatibilityRequirement {
  NONE = 0,
  WEEK_4 = 1,
  WEEK_12 = 2,
  MAX = 3
} MlirStablehloCompatibilityRequirement;

// Returns a StringAtt with the version of StableHLO that satisfies the
// compatibility requirement, which is owned by ctx.
MLIR_CAPI_EXPORTED void stablehloVersionFromCompatibilityRequirement(
    MlirStablehloCompatibilityRequirement requirement,
    MlirStringCallback callback, void* userData);

// Get the current StableHLO version.
//
// This value can be used as the `targetVersion` argument to
// `serializePortableArtifact`.
MLIR_CAPI_EXPORTED void stablehloGetCurrentVersion(MlirStringCallback callback,
                                                   void* userData);

// Get the minimum supported StableHLO version.
//
// This value can be used as the `targetVersion` argument to
// `serializePortableArtifact`.
//
// Each StableHLO version `producer_version` has a compatibility window,
// i.e. range of versions [`consumer_version_min`, `consumer_version_max`],
// where StableHLO portable artifacts serialized by `producer_version`
// can be deserialized by `consumer_version` within the window.
// See https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md
// for the exact extent of these compatibility guarantees.
//
// This function returns `consumer_version_min` for the current StableHLO
// version. It can be used maximize forward compatibility, i.e. to maximize how
// far into the past we can go and still have the payloads produced by
// `serializePortableArtifact` compatible with potential consumers from the past
MLIR_CAPI_EXPORTED void stablehloGetMinimumVersion(MlirStringCallback callback,
                                                   void* userData);

// For two given version strings, return the smaller version.
// Returns failure if either version is not a valid version string.
MLIR_CAPI_EXPORTED MlirLogicalResult
stablehloGetSmallerVersion(MlirStringRef version1, MlirStringRef version2,
                           MlirStringCallback callback, void* userData);

// Write a StableHLO program expressed as a string (either prettyprinted MLIR
// module or MLIR bytecode) to a portable artifact.
// Can fail if `moduleStr` cannot be parsed, or if it cannot be expressed in the
// `targetVersion` version of StableHLO, e.g. if it's using new or removed
// features, or if it involves unsupported dialects.
// Returns false on failure.
MLIR_CAPI_EXPORTED MlirLogicalResult
stablehloSerializePortableArtifactFromStringRef(MlirStringRef moduleStr,
                                                MlirStringRef targetVersion,
                                                MlirStringCallback callback,
                                                void* userData);

// Write a StableHLO program expressed as a string (either prettyprinted MLIR
// module or MLIR bytecode) to a portable artifact.
// Can fail if `moduleStr` cannot be parsed, or if it cannot be expressed in the
// `targetVersion` version of StableHLO, e.g. if it's using new or removed
// features, or if it involves unsupported dialects.
// Returns false on failure.
MLIR_CAPI_EXPORTED MlirLogicalResult
stablehloSerializePortableArtifactFromModule(MlirModule moduleStr,
                                             MlirStringRef targetVersion,
                                             MlirStringCallback callback,
                                             void* userData);

// Read a StableHLO program from a portable artifact, returning the module as
// MLIR bytecode. Note, this bytecode returned is not a portable artifact,
// and has the stability of returning textual assembly format. Bytecode is
// returned here since it is more compact and faster to read and write.
// Can fail if `artifactStr` cannot be expressed in the current version of
// StableHLO, e.g. if it's using incompatible features.
// Returns false on failure.
MLIR_CAPI_EXPORTED MlirLogicalResult stablehloDeserializePortableArtifact(
    MlirStringRef artifactStr, MlirStringCallback callback, void* userData);

// Read a StableHLO program from a portable artifact, returning the module as
// MLIR bytecode. Note, this bytecode returned is not a portable artifact,
// and has the stability of returning textual assembly format. Bytecode is
// returned here since it is more compact and faster to read and write.
// Can fail if `artifactStr` cannot be expressed in the current version of
// StableHLO, e.g. if it's using incompatible features.
//
// Returns empty module on failure.
MLIR_CAPI_EXPORTED MlirModule stablehloDeserializePortableArtifactNoError(
    MlirStringRef artifactStr, MlirContext ctx);

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

#endif  // STABLEHLO_INTEGRATIONS_C_STABLEHLOAPI_H_
