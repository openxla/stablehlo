// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file | FileCheck %s
// RUN: %if asserts %{ stablehlo-opt %s -verify-diagnostics -split-input-file -emit-bytecode -debug-only=stablehlo-bytecode 2>&1 | FileCheck --check-prefix=CHECK-WARN %s %}
// RUN: stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.24.0' --vhlo-legalize-to-stablehlo -verify-diagnostics -split-input-file | FileCheck %s

// CHECK-WARN-NOT: Not Implemented

// Tests for the StableHLO complex element type domain {f16, bf16, f32, f64}.
// The round trip pins target=1.24.0, the version that introduced these
// component types.

// CHECK-LABEL: func @complex_construct_f16
func.func @complex_construct_f16(%lhs: tensor<4xf16>, %rhs: tensor<4xf16>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.complex"(%lhs, %rhs) : (tensor<4xf16>, tensor<4xf16>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @complex_extract_f16
func.func @complex_extract_f16(%arg0: tensor<4xcomplex<f16>>) -> (tensor<4xf16>, tensor<4xf16>) {
  %real = "stablehlo.real"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xf16>
  %imag = "stablehlo.imag"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xf16>
  func.return %real, %imag : tensor<4xf16>, tensor<4xf16>
}

// -----

// CHECK-LABEL: func @add_f16
func.func @add_f16(%arg0: tensor<4xcomplex<f16>>, %arg1: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @multiply_f16
func.func @multiply_f16(%arg0: tensor<4xcomplex<f16>>, %arg1: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<4xcomplex<f16>>, tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @negate_f16
func.func @negate_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.negate"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @sqrt_f16
func.func @sqrt_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.sqrt"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @cosine_f16
func.func @cosine_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.cosine"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @tanh_f16
func.func @tanh_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.tanh"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @abs_f16
func.func @abs_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xf16> {
  %0 = "stablehlo.abs"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xf16>
  func.return %0 : tensor<4xf16>
}

// -----

// CHECK-LABEL: func @atan2_f16
func.func @atan2_f16(%arg0: tensor<4xcomplex<f16>>, %arg1: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.atan2"(%arg0, %arg1) : (tensor<4xcomplex<f16>>, tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @logistic_f16
func.func @logistic_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.logistic"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @cbrt_f16
func.func @cbrt_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.cbrt"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @exponential_f16
func.func @exponential_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.exponential"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @exponential_minus_one_f16
func.func @exponential_minus_one_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.exponential_minus_one"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @log_f16
func.func @log_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.log"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @log_plus_one_f16
func.func @log_plus_one_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @rsqrt_f16
func.func @rsqrt_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.rsqrt"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @sign_f16
func.func @sign_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.sign"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @sine_f16
func.func @sine_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.sine"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @tan_f16
func.func @tan_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.tan"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @divide_f16
func.func @divide_f16(%arg0: tensor<4xcomplex<f16>>, %arg1: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.divide"(%arg0, %arg1) : (tensor<4xcomplex<f16>>, tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @power_f16
func.func @power_f16(%arg0: tensor<4xcomplex<f16>>, %arg1: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.power"(%arg0, %arg1) : (tensor<4xcomplex<f16>>, tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @remainder_f16
func.func @remainder_f16(%arg0: tensor<4xcomplex<f16>>, %arg1: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.remainder"(%arg0, %arg1) : (tensor<4xcomplex<f16>>, tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @subtract_f16
func.func @subtract_f16(%arg0: tensor<4xcomplex<f16>>, %arg1: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.subtract"(%arg0, %arg1) : (tensor<4xcomplex<f16>>, tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @constant_f16
func.func @constant_f16() -> tensor<2xcomplex<f16>> {
  %0 = stablehlo.constant dense<[(1.0, 2.0), (3.0, 4.0)]> : tensor<2xcomplex<f16>>
  // CHECK: stablehlo.constant
  // CHECK-SAME: tensor<2xcomplex<f16>>
  func.return %0 : tensor<2xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @convert_f16_to_complex_f16
func.func @convert_f16_to_complex_f16(%arg0: tensor<4xf16>) -> tensor<4xcomplex<f16>> {
  %0 = stablehlo.convert %arg0 : (tensor<4xf16>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @convert_complex_f16_to_f16
func.func @convert_complex_f16_to_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xf16> {
  %0 = stablehlo.convert %arg0 : (tensor<4xcomplex<f16>>) -> tensor<4xf16>
  func.return %0 : tensor<4xf16>
}

// -----

// CHECK-LABEL: func @iota_complex_f16
func.func @iota_complex_f16() -> tensor<2x2xcomplex<f16>> {
  %0 = stablehlo.iota dim = 0 : tensor<2x2xcomplex<f16>>
  func.return %0 : tensor<2x2xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @cholesky_complex_f16
func.func @cholesky_complex_f16(%arg0: tensor<2x4x4xcomplex<f16>>) -> tensor<2x4x4xcomplex<f16>> {
  %0 = "stablehlo.cholesky"(%arg0) { lower = true } : (tensor<2x4x4xcomplex<f16>>) -> tensor<2x4x4xcomplex<f16>>
  func.return %0 : tensor<2x4x4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @triangular_solve_complex_f16
func.func @triangular_solve_complex_f16(%arg0: tensor<4x4xcomplex<f16>>, %arg1: tensor<4x4xcomplex<f16>>) -> tensor<4x4xcomplex<f16>> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) { left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true } : (tensor<4x4xcomplex<f16>>, tensor<4x4xcomplex<f16>>) -> tensor<4x4xcomplex<f16>>
  func.return %0 : tensor<4x4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @tuple_complex_f16
func.func @tuple_complex_f16(%arg0: tensor<2xcomplex<f16>>, %arg1: tensor<4xf16>) -> tuple<tensor<2xcomplex<f16>>, tensor<4xf16>> {
  %0 = "stablehlo.tuple"(%arg0, %arg1) : (tensor<2xcomplex<f16>>, tensor<4xf16>) -> tuple<tensor<2xcomplex<f16>>, tensor<4xf16>>
  func.return %0 : tuple<tensor<2xcomplex<f16>>, tensor<4xf16>>
}

// -----

// CHECK-LABEL: func @custom_call_complex_f16
func.func @custom_call_complex_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.custom_call"(%arg0) { call_target_name = "foo", has_side_effect = false } : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @real_dynamic_f16
func.func @real_dynamic_f16(%arg0: tensor<?xcomplex<f16>>) -> tensor<?xf16> {
  %0 = "stablehlo.real"(%arg0) : (tensor<?xcomplex<f16>>) -> tensor<?xf16>
  func.return %0 : tensor<?xf16>
}

// -----

// CHECK-LABEL: func @imag_dynamic_f16
func.func @imag_dynamic_f16(%arg0: tensor<?xcomplex<f16>>) -> tensor<?xf16> {
  %0 = "stablehlo.imag"(%arg0) : (tensor<?xcomplex<f16>>) -> tensor<?xf16>
  func.return %0 : tensor<?xf16>
}

// -----

// CHECK-LABEL: func @complex_construct_bf16
func.func @complex_construct_bf16(%lhs: tensor<4xbf16>, %rhs: tensor<4xbf16>) -> tensor<4xcomplex<bf16>> {
  %0 = "stablehlo.complex"(%lhs, %rhs) : (tensor<4xbf16>, tensor<4xbf16>) -> tensor<4xcomplex<bf16>>
  func.return %0 : tensor<4xcomplex<bf16>>
}

// -----

// CHECK-LABEL: func @complex_extract_bf16
func.func @complex_extract_bf16(%arg0: tensor<4xcomplex<bf16>>) -> (tensor<4xbf16>, tensor<4xbf16>) {
  %real = "stablehlo.real"(%arg0) : (tensor<4xcomplex<bf16>>) -> tensor<4xbf16>
  %imag = "stablehlo.imag"(%arg0) : (tensor<4xcomplex<bf16>>) -> tensor<4xbf16>
  func.return %real, %imag : tensor<4xbf16>, tensor<4xbf16>
}

// -----

// CHECK-LABEL: func @add_bf16
func.func @add_bf16(%arg0: tensor<4xcomplex<bf16>>, %arg1: tensor<4xcomplex<bf16>>) -> tensor<4xcomplex<bf16>> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xcomplex<bf16>>
  func.return %0 : tensor<4xcomplex<bf16>>
}

// -----

// CHECK-LABEL: func @abs_bf16
func.func @abs_bf16(%arg0: tensor<4xcomplex<bf16>>) -> tensor<4xbf16> {
  %0 = "stablehlo.abs"(%arg0) : (tensor<4xcomplex<bf16>>) -> tensor<4xbf16>
  func.return %0 : tensor<4xbf16>
}

// -----

// CHECK-LABEL: func @constant_bf16
func.func @constant_bf16() -> tensor<1xcomplex<bf16>> {
  %0 = stablehlo.constant dense<[(1.0, 2.0)]> : tensor<1xcomplex<bf16>>
  // CHECK: stablehlo.constant
  // CHECK-SAME: tensor<1xcomplex<bf16>>
  func.return %0 : tensor<1xcomplex<bf16>>
}

// -----

// CHECK-LABEL: func @fft_f16
func.func @fft_f16(%arg0: tensor<3x9xcomplex<f16>>) -> tensor<3x9xcomplex<f16>> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type FFT> } : (tensor<3x9xcomplex<f16>>) -> tensor<3x9xcomplex<f16>>
  func.return %0 : tensor<3x9xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @ifft_f16
func.func @ifft_f16(%arg0: tensor<3x9xcomplex<f16>>) -> tensor<3x9xcomplex<f16>> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type IFFT> } : (tensor<3x9xcomplex<f16>>) -> tensor<3x9xcomplex<f16>>
  func.return %0 : tensor<3x9xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @rfft_f16
func.func @rfft_f16(%arg0: tensor<3x9xf16>) -> tensor<3x5xcomplex<f16>> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type RFFT> } : (tensor<3x9xf16>) -> tensor<3x5xcomplex<f16>>
  func.return %0 : tensor<3x5xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @irfft_f16
func.func @irfft_f16(%arg0: tensor<3x9xcomplex<f16>>) -> tensor<3x16xf16> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 16>, fft_type = #stablehlo<fft_type IRFFT> } : (tensor<3x9xcomplex<f16>>) -> tensor<3x16xf16>
  func.return %0 : tensor<3x16xf16>
}

// -----

// CHECK-LABEL: func @fft_bf16
func.func @fft_bf16(%arg0: tensor<3x9xcomplex<bf16>>) -> tensor<3x9xcomplex<bf16>> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type FFT> } : (tensor<3x9xcomplex<bf16>>) -> tensor<3x9xcomplex<bf16>>
  func.return %0 : tensor<3x9xcomplex<bf16>>
}

// -----

// CHECK-LABEL: func @ifft_bf16
func.func @ifft_bf16(%arg0: tensor<3x9xcomplex<bf16>>) -> tensor<3x9xcomplex<bf16>> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type IFFT> } : (tensor<3x9xcomplex<bf16>>) -> tensor<3x9xcomplex<bf16>>
  func.return %0 : tensor<3x9xcomplex<bf16>>
}

// -----

// CHECK-LABEL: func @rfft_bf16
func.func @rfft_bf16(%arg0: tensor<3x9xbf16>) -> tensor<3x5xcomplex<bf16>> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type RFFT> } : (tensor<3x9xbf16>) -> tensor<3x5xcomplex<bf16>>
  func.return %0 : tensor<3x5xcomplex<bf16>>
}

// -----

// CHECK-LABEL: func @irfft_bf16
func.func @irfft_bf16(%arg0: tensor<3x9xcomplex<bf16>>) -> tensor<3x16xbf16> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 16>, fft_type = #stablehlo<fft_type IRFFT> } : (tensor<3x9xcomplex<bf16>>) -> tensor<3x16xbf16>
  func.return %0 : tensor<3x16xbf16>
}

// -----

func.func @complex_construct_int32(%lhs: tensor<4xi32>, %rhs: tensor<4xi32>) -> tensor<4xcomplex<i32>> {
  // expected-error-re@+1 {{operand #0 must be ranked tensor of {{.*}}, but got 'tensor<4xi32>'}}
  %0 = "stablehlo.complex"(%lhs, %rhs) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xcomplex<i32>>
  func.return %0 : tensor<4xcomplex<i32>>
}

// -----

func.func @complex_construct_f8(%lhs: tensor<4xf8E4M3FN>, %rhs: tensor<4xf8E4M3FN>) -> tensor<4xcomplex<f8E4M3FN>> {
  // expected-error-re@+1 {{operand #0 must be ranked tensor of {{.*}}, but got 'tensor<4xf8E4M3FN>'}}
  %0 = "stablehlo.complex"(%lhs, %rhs) : (tensor<4xf8E4M3FN>, tensor<4xf8E4M3FN>) -> tensor<4xcomplex<f8E4M3FN>>
  func.return %0 : tensor<4xcomplex<f8E4M3FN>>
}

// -----

func.func @complex_construct_mismatched_component_types(%lhs: tensor<4xf16>, %rhs: tensor<4xbf16>) -> tensor<4xcomplex<f16>> {
  // expected-error@+1 {{'stablehlo.complex' op requires the same element type for all operands}}
  %0 = "stablehlo.complex"(%lhs, %rhs) : (tensor<4xf16>, tensor<4xbf16>) -> tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

func.func @complex_construct_result_component_mismatch(%lhs: tensor<4xf16>, %rhs: tensor<4xf16>) -> tensor<4xcomplex<f32>> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.complex' op inferred type(s) 'tensor<4xcomplex<f16>>' are incompatible with return type(s) of operation 'tensor<4xcomplex<f32>>'}}
  %0 = "stablehlo.complex"(%lhs, %rhs) : (tensor<4xf16>, tensor<4xf16>) -> tensor<4xcomplex<f32>>
  func.return %0 : tensor<4xcomplex<f32>>
}

// -----

func.func @rfft_f8(%arg0: tensor<3x9xf8E4M3FN>) -> tensor<3x5xcomplex<f32>> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{RFFT requires f16, bf16, f32 or f64 input type, but is given 'f8E4M3FN'.}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type RFFT> } : (tensor<3x9xf8E4M3FN>) -> tensor<3x5xcomplex<f32>>
  func.return %0 : tensor<3x5xcomplex<f32>>
}

// -----

func.func @rfft_result_component_mismatch(%arg0: tensor<3x9xf16>) -> tensor<3x5xcomplex<bf16>> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.fft' op inferred type(s) 'tensor<3x5xcomplex<f16>>' are incompatible with return type(s) of operation 'tensor<3x5xcomplex<bf16>>'}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type RFFT> } : (tensor<3x9xf16>) -> tensor<3x5xcomplex<bf16>>
  func.return %0 : tensor<3x5xcomplex<bf16>>
}

// -----

func.func @irfft_result_component_mismatch(%arg0: tensor<3x9xcomplex<bf16>>) -> tensor<3x16xf16> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.fft' op inferred type(s) 'tensor<3x16xbf16>' are incompatible with return type(s) of operation 'tensor<3x16xf16>'}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 16>, fft_type = #stablehlo<fft_type IRFFT> } : (tensor<3x9xcomplex<bf16>>) -> tensor<3x16xf16>
  func.return %0 : tensor<3x16xf16>
}
