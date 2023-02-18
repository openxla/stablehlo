/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef STABLEHLO_DIALECT_VHLO_ENUMS
#define STABLEHLO_DIALECT_VHLO_ENUMS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"
include "stablehlo/dialect/VhloBase.td"
include "stablehlo/dialect/VhloAttrs.td"

class VHLO_I32EnumAttr<string name, list<I32EnumAttrCase> cases> :
    I32EnumAttr<name, name, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::vhlo";
}

class VHLO_EnumAttr<EnumAttrInfo enumInfo, string name, string minVersion, string maxVersion>
  : EnumAttr<VHLO_Dialect, enumInfo, name, [VHLO_VersionedAttrInterface]> {
  let extraClassDeclaration = [{
    mlir::vhlo::Version getMinVersion() {
      auto version = mlir::vhlo::Version::fromString("}] #  minVersion # [{");
      if (failed(version)) llvm::report_fatal_error("invalid version }] # minVersion # [{ in }] # name # [{");
      return *version;
    }
    mlir::vhlo::Version getMaxVersion() {
      if (!strcmp("}] # maxVersion # [{", "current")) return Version::getCurrentVersion();
      auto version = mlir::vhlo::Version::fromString("}] #  maxVersion # [{");
      if (failed(version)) llvm::report_fatal_error("invalid version }] # maxVersion # [{ in }] # name # [{");
      return *version;
    }
  }];
}

//===----------------------------------------------------------------------===//
// ComparisonDirection
//===----------------------------------------------------------------------===//

def VHLO_COMPARISON_DIRECTION_EQ : I32EnumAttrCase<"EQ", 0>;
def VHLO_COMPARISON_DIRECTION_NE : I32EnumAttrCase<"NE", 1>;
def VHLO_COMPARISON_DIRECTION_GE : I32EnumAttrCase<"GE", 2>;
def VHLO_COMPARISON_DIRECTION_GT : I32EnumAttrCase<"GT", 3>;
def VHLO_COMPARISON_DIRECTION_LE : I32EnumAttrCase<"LE", 4>;
def VHLO_COMPARISON_DIRECTION_LT : I32EnumAttrCase<"LT", 5>;

def VHLO_ComparisonDirectionV1 : VHLO_I32EnumAttr<"ComparisonDirectionV1", [
  VHLO_COMPARISON_DIRECTION_EQ,
  VHLO_COMPARISON_DIRECTION_NE,
  VHLO_COMPARISON_DIRECTION_GE,
  VHLO_COMPARISON_DIRECTION_GT,
  VHLO_COMPARISON_DIRECTION_LE,
  VHLO_COMPARISON_DIRECTION_LT
]> {}

def VHLO_ComparisonDirectionAttrV1
  : VHLO_EnumAttr<VHLO_ComparisonDirectionV1, "comparison_direction", "0.3.0", "current">;

//===----------------------------------------------------------------------===//
// ComparisonType
//===----------------------------------------------------------------------===//

// TODO(#1186): NOTYPE is not part of the StableHLO spec.
def VHLO_COMPARISON_TYPE_NOTYPE : I32EnumAttrCase<"NOTYPE", 0>;
def VHLO_COMPARISON_TYPE_FLOAT : I32EnumAttrCase<"FLOAT", 1>;
def VHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER : I32EnumAttrCase<"TOTALORDER", 2>;
def VHLO_COMPARISON_TYPE_SIGNED : I32EnumAttrCase<"SIGNED", 3>;
def VHLO_COMPARISON_TYPE_UNSIGNED : I32EnumAttrCase<"UNSIGNED", 4>;

def VHLO_ComparisonTypeV1 : VHLO_I32EnumAttr<"ComparisonTypeV1", [
  VHLO_COMPARISON_TYPE_NOTYPE,
  VHLO_COMPARISON_TYPE_FLOAT,
  VHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER,
  VHLO_COMPARISON_TYPE_SIGNED,
  VHLO_COMPARISON_TYPE_UNSIGNED
]> {}

def VHLO_ComparisonTypeAttrV1
  : VHLO_EnumAttr<VHLO_ComparisonTypeV1, "comparison_type", "0.3.0", "current">;

//===----------------------------------------------------------------------===//
// CustomCallApiVersion
//===----------------------------------------------------------------------===//

// TODO(#1187): CustomCallApiVersion is not part of the StableHLO spec.
def VHLO_CUSTOM_CALL_API_VERSION_UNSPECIFIED : I32EnumAttrCase<"API_VERSION_UNSPECIFIED", 0>;
def VHLO_CUSTOM_CALL_API_VERSION_ORIGINAL : I32EnumAttrCase<"API_VERSION_ORIGINAL", 1>;
def VHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING : I32EnumAttrCase<"API_VERSION_STATUS_RETURNING", 2>;
def VHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING_UNIFIED : I32EnumAttrCase<"API_VERSION_STATUS_RETURNING_UNIFIED", 3>;

def VHLO_CustomCallApiVersionV1 : VHLO_I32EnumAttr<"CustomCallApiVersionV1", [
    VHLO_CUSTOM_CALL_API_VERSION_UNSPECIFIED,
    VHLO_CUSTOM_CALL_API_VERSION_ORIGINAL,
    VHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING,
    VHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING_UNIFIED
]> {}

def VHLO_CustomCallApiVersionAttrV1
  : VHLO_EnumAttr<VHLO_CustomCallApiVersionV1, "api_version", "0.3.0", "current">;

//===----------------------------------------------------------------------===//
// FftType
//===----------------------------------------------------------------------===//

def VHLO_FFT_TYPE_FFT : I32EnumAttrCase<"FFT", 0>;
def VHLO_FFT_TYPE_IFFT : I32EnumAttrCase<"IFFT", 1>;
def VHLO_FFT_TYPE_RFFT : I32EnumAttrCase<"RFFT", 2>;
def VHLO_FFT_TYPE_IRFFT : I32EnumAttrCase<"IRFFT", 3>;

def VHLO_FftTypeV1 : VHLO_I32EnumAttr<"FftTypeV1", [
  VHLO_FFT_TYPE_FFT,
  VHLO_FFT_TYPE_IFFT,
  VHLO_FFT_TYPE_RFFT,
  VHLO_FFT_TYPE_IRFFT
]> {}

def VHLO_FftTypeAttrV1
  : VHLO_EnumAttr<VHLO_FftTypeV1, "fft_type", "0.3.0", "current">;

//===----------------------------------------------------------------------===//
// Precision
//===----------------------------------------------------------------------===//

def VHLO_PRECISION_DEFAULT : I32EnumAttrCase<"DEFAULT", 0>;
def VHLO_PRECISION_HIGH : I32EnumAttrCase<"HIGH", 1>;
def VHLO_PRECISION_HIGHEST : I32EnumAttrCase<"HIGHEST", 2>;

def VHLO_PrecisionV1 : VHLO_I32EnumAttr<"PrecisionV1", [
  VHLO_PRECISION_DEFAULT,
  VHLO_PRECISION_HIGH,
  VHLO_PRECISION_HIGHEST
]> {}

def VHLO_PrecisionAttrV1
  : VHLO_EnumAttr<VHLO_PrecisionV1, "precision", "0.3.0", "current">;

//===----------------------------------------------------------------------===//
// RngAlgorithm
//===----------------------------------------------------------------------===//

def VHLO_RNG_ALGORITHM_DEFAULT : I32EnumAttrCase<"DEFAULT", 0>;
def VHLO_RNG_ALGORITHM_THREE_FRY : I32EnumAttrCase<"THREE_FRY", 1>;
def VHLO_RNG_ALGORITHM_PHILOX : I32EnumAttrCase<"PHILOX", 2>;

def VHLO_RngAlgorithmV1 : VHLO_I32EnumAttr<"RngAlgorithmV1", [
  VHLO_RNG_ALGORITHM_DEFAULT,
  VHLO_RNG_ALGORITHM_THREE_FRY,
  VHLO_RNG_ALGORITHM_PHILOX
]> {}

def VHLO_RngAlgorithmAttrV1
  : VHLO_EnumAttr<VHLO_RngAlgorithmV1, "rng_algorithm", "0.3.0", "current">;

//===----------------------------------------------------------------------===//
// RngDistribution
//===----------------------------------------------------------------------===//

def VHLO_RNG_DISTRIBUTION_UNIFORM : I32EnumAttrCase<"UNIFORM", 1>;
def VHLO_RNG_DISTRIBUTION_NORMAL : I32EnumAttrCase<"NORMAL", 2>;

def VHLO_RngDistributionV1 : VHLO_I32EnumAttr<"RngDistributionV1", [
  VHLO_RNG_DISTRIBUTION_UNIFORM,
  VHLO_RNG_DISTRIBUTION_NORMAL
]> {}

def VHLO_RngDistributionAttrV1
  : VHLO_EnumAttr<VHLO_RngDistributionV1, "rng_distribution", "0.3.0", "current">;

//===----------------------------------------------------------------------===//
// Transpose
//===----------------------------------------------------------------------===//

// TODO(#1186): TRANSPOSE_INVALID is not part of the StableHLO spec.
def VHLO_TRANSPOSE_INVALID : I32EnumAttrCase<"TRANSPOSE_INVALID", 0>;
def VHLO_NO_TRANSPOSE : I32EnumAttrCase<"NO_TRANSPOSE", 1>;
def VHLO_TRANSPOSE : I32EnumAttrCase<"TRANSPOSE", 2>;
def VHLO_ADJOINT : I32EnumAttrCase<"ADJOINT", 3>;

def VHLO_TransposeV1 : VHLO_I32EnumAttr<"TransposeV1", [
  VHLO_TRANSPOSE_INVALID,
  VHLO_NO_TRANSPOSE,
  VHLO_TRANSPOSE,
  VHLO_ADJOINT
]> {}

def VHLO_TransposeAttrV1
  : VHLO_EnumAttr<VHLO_TransposeV1, "transpose", "0.3.0", "current">;

#endif // STABLEHLO_DIALECT_VHLO_ENUMS
