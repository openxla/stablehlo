/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The VersionedHLO Authors.

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

#ifndef STABLEHLO_COMPATIBILITY_VERSIONEDHLO_ENUMS
#define STABLEHLO_COMPATIBILITY_VERSIONEDHLO_ENUMS

include "stablehlo/compatibility/dialect/VersionedStablehloBase.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// Enumerations
//===----------------------------------------------------------------------===//

// These mirror the XLA PrecisionConfig proto enum.
def VERSIONEDHLO_PRECISION_DEFAULT : I32EnumAttrCase<"DEFAULT", 0>;
def VERSIONEDHLO_PRECISION_HIGH    : I32EnumAttrCase<"HIGH", 1>;
def VERSIONEDHLO_PRECISION_HIGHEST : I32EnumAttrCase<"HIGHEST", 2>;

def VersionedHLO_Precision : I32EnumAttr<"Precision",
    "XLA precision for an operand. Has backend specific meaning.",
    [
      VERSIONEDHLO_PRECISION_DEFAULT,
      VERSIONEDHLO_PRECISION_HIGH,
      VERSIONEDHLO_PRECISION_HIGHEST
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::versionedhlo";
}

def VersionedHLO_PrecisionAttr : EnumAttr<VersionedHLO_Dialect, VersionedHLO_Precision, "precision">;

// TODO(b/129153247) See if it's possible to also validate the size.
def VersionedHLO_PrecisionConfigAttr:
    OptionalAttr<
          TypedArrayAttrBase<VersionedHLO_PrecisionAttr, "Precision Config attribute">>;

//===----------------------------------------------------------------------===//
// Fast Fourier Transform Type enum definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA FftType proto enum.
def VERSIONEDHLO_FFT_TYPE_FFT : I32EnumAttrCase<"FFT", 0>;
def VERSIONEDHLO_FFT_TYPE_IFFT : I32EnumAttrCase<"IFFT", 1>;
def VERSIONEDHLO_FFT_TYPE_RFFT : I32EnumAttrCase<"RFFT", 2>;
def VERSIONEDHLO_FFT_TYPE_IRFFT : I32EnumAttrCase<"IRFFT", 3>;

def VersionedHLO_FftType : I32EnumAttr<"FftType",
    "XLA fast fourier transform type.",
    [
      VERSIONEDHLO_FFT_TYPE_FFT,
      VERSIONEDHLO_FFT_TYPE_IFFT,
      VERSIONEDHLO_FFT_TYPE_RFFT,
      VERSIONEDHLO_FFT_TYPE_IRFFT
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::versionedhlo";
}

def VersionedHLO_FftTypeAttr : EnumAttr<VersionedHLO_Dialect, VersionedHLO_FftType, "fft_type">;

//===----------------------------------------------------------------------===//
// Custom call enum definitions.
//===----------------------------------------------------------------------===//

def VERSIONEDHLO_CUSTOM_CALL_API_VERISON_UNSPECIFIED :
    I32EnumAttrCase<"API_VERSION_UNSPECIFIED", 0>;
def VERSIONEDHLO_CUSTOM_CALL_API_VERSION_ORIGINAL :
    I32EnumAttrCase<"API_VERSION_ORIGINAL", 1>;
def VERSIONEDHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING :
    I32EnumAttrCase<"API_VERSION_STATUS_RETURNING", 2>;
def VERSIONEDHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING_UNIFIED :
    I32EnumAttrCase<"API_VERSION_STATUS_RETURNING_UNIFIED", 3>;
def VersionedHLO_CustomCallApiVersionAttr :
    I32EnumAttr<"CustomCallApiVersion", "Custom call API version", [
        VERSIONEDHLO_CUSTOM_CALL_API_VERISON_UNSPECIFIED,
        VERSIONEDHLO_CUSTOM_CALL_API_VERSION_ORIGINAL,
        VERSIONEDHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING,
        VERSIONEDHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING_UNIFIED
    ]> {
  let cppNamespace = "::mlir::versionedhlo";
}

//===----------------------------------------------------------------------===//
// Comparison op definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA ComparisonDirection enum.
def VERSIONEDHLO_COMPARISON_DIRECTION_EQ : I32EnumAttrCase<"EQ", 0>;
def VERSIONEDHLO_COMPARISON_DIRECTION_NE : I32EnumAttrCase<"NE", 1>;
def VERSIONEDHLO_COMPARISON_DIRECTION_GE : I32EnumAttrCase<"GE", 2>;
def VERSIONEDHLO_COMPARISON_DIRECTION_GT : I32EnumAttrCase<"GT", 3>;
def VERSIONEDHLO_COMPARISON_DIRECTION_LE : I32EnumAttrCase<"LE", 4>;
def VERSIONEDHLO_COMPARISON_DIRECTION_LT : I32EnumAttrCase<"LT", 5>;

def VersionedHLO_ComparisonDirection : I32EnumAttr<"ComparisonDirection",
    "Which comparison operation to perform.",
    [
      VERSIONEDHLO_COMPARISON_DIRECTION_EQ,
      VERSIONEDHLO_COMPARISON_DIRECTION_NE,
      VERSIONEDHLO_COMPARISON_DIRECTION_GE,
      VERSIONEDHLO_COMPARISON_DIRECTION_GT,
      VERSIONEDHLO_COMPARISON_DIRECTION_LE,
      VERSIONEDHLO_COMPARISON_DIRECTION_LT
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::versionedhlo";
}

def VersionedHLO_ComparisonDirectionAttr : EnumAttr<VersionedHLO_Dialect, VersionedHLO_ComparisonDirection, "comparison_direction">;

def VERSIONEDHLO_DEFAULT_COMPARISON_TYPE : NativeCodeCall<"::mlir::versionedhlo::ComparisonTypeAttr()">;
def VERSIONEDHLO_COMPARISON_TYPE_NOTYPE : I32EnumAttrCase<"NOTYPE", 0>;
def VERSIONEDHLO_COMPARISON_TYPE_FLOAT : I32EnumAttrCase<"FLOAT", 1>;
def VERSIONEDHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER : I32EnumAttrCase<"TOTALORDER", 2>;
def VERSIONEDHLO_COMPARISON_TYPE_SIGNED : I32EnumAttrCase<"SIGNED", 3>;
def VERSIONEDHLO_COMPARISON_TYPE_UNSIGNED : I32EnumAttrCase<"UNSIGNED", 4>;

def VersionedHLO_ComparisonType : I32EnumAttr<"ComparisonType",
    "Which comparison type to use.",
    [
      VERSIONEDHLO_COMPARISON_TYPE_NOTYPE,
      VERSIONEDHLO_COMPARISON_TYPE_FLOAT,
      VERSIONEDHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER,
      VERSIONEDHLO_COMPARISON_TYPE_SIGNED,
      VERSIONEDHLO_COMPARISON_TYPE_UNSIGNED
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::versionedhlo";
}

def VersionedHLO_ComparisonTypeAttr 
  : EnumAttr<VersionedHLO_Dialect, VersionedHLO_ComparisonType, "comparison_type">;

// These mirror the XLA Transpose enum in Triangular Solve options.
def VERSIONEDHLO_TRANSPOSE_INVALID : I32EnumAttrCase<"TRANSPOSE_INVALID", 0>;
def VERSIONEDHLO_NO_TRANSPOSE : I32EnumAttrCase<"NO_TRANSPOSE", 1>;
def VERSIONEDHLO_TRANSPOSE : I32EnumAttrCase<"TRANSPOSE", 2>;
def VERSIONEDHLO_ADJOINT : I32EnumAttrCase<"ADJOINT", 3>;

def VersionedHLO_Transpose : I32EnumAttr<"Transpose",
    "Transpose options",
    [
      VERSIONEDHLO_TRANSPOSE_INVALID,
      VERSIONEDHLO_NO_TRANSPOSE,
      VERSIONEDHLO_TRANSPOSE,
      VERSIONEDHLO_ADJOINT
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::versionedhlo";
}

def VersionedHLO_TransposeAttr : EnumAttr<VersionedHLO_Dialect, VersionedHLO_Transpose, "transpose">;

def VERSIONEDHLO_RNG_DISTRIBUTION_UNIFORM : I32EnumAttrCase<"UNIFORM", 1>;
def VERSIONEDHLO_RNG_DISTRIBUTION_NORMAL : I32EnumAttrCase<"NORMAL", 2>;

def VersionedHLO_RngDistribution : I32EnumAttr<"RngDistribution",
    "XLA PRNG distribution to be used.",
    [
      VERSIONEDHLO_RNG_DISTRIBUTION_UNIFORM,
      VERSIONEDHLO_RNG_DISTRIBUTION_NORMAL
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::versionedhlo";
}

def VersionedHLO_RngDistributionAttr : EnumAttr<VersionedHLO_Dialect, VersionedHLO_RngDistribution, "rng_distribution">;

def VERSIONEDHLO_RNG_ALGORITHM_DEFAULT : I32EnumAttrCase<"DEFAULT", 0>;
def VERSIONEDHLO_RNG_ALGORITHM_THREE_FRY : I32EnumAttrCase<"THREE_FRY", 1>;
def VERSIONEDHLO_RNG_ALGORITHM_PHILOX : I32EnumAttrCase<"PHILOX", 2>;

def VersionedHLO_RngAlgorithm : I32EnumAttr<"RngAlgorithm",
    "XLA PRNG algorithm to be used.",
    [
      VERSIONEDHLO_RNG_ALGORITHM_DEFAULT,
      VERSIONEDHLO_RNG_ALGORITHM_THREE_FRY,
      VERSIONEDHLO_RNG_ALGORITHM_PHILOX
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::versionedhlo";
}

def VersionedHLO_RngAlgorithmAttr : EnumAttr<VersionedHLO_Dialect, VersionedHLO_RngAlgorithm, "rng_algorithm">;

#endif // STABLEHLO_COMPATIBILITY_VERSIONEDHLO_ENUMS