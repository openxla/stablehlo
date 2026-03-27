// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// CHECK-LABEL: func private @future_type
// CHECK-SAME: !stablehlo.future<tensor<f32>>
func.func private @future_type() -> !stablehlo.future<tensor<f32>>

// -----

// CHECK-LABEL: func private @future_type_ranked
// CHECK-SAME: !stablehlo.future<tensor<2x3xf32>>
func.func private @future_type_ranked() -> !stablehlo.future<tensor<2x3xf32>>

// -----

// CHECK-LABEL: func private @future_type_dynamic
// CHECK-SAME: !stablehlo.future<tensor<?xf32>>
func.func private @future_type_dynamic() -> !stablehlo.future<tensor<?xf32>>

// -----

// CHECK-LABEL: func private @future_type_int
// CHECK-SAME: !stablehlo.future<tensor<4xi32>>
func.func private @future_type_int() -> !stablehlo.future<tensor<4xi32>>

// -----

// CHECK-LABEL: func private @future_type_complex
// CHECK-SAME: !stablehlo.future<tensor<2xcomplex<f32>>>
func.func private @future_type_complex() -> !stablehlo.future<tensor<2xcomplex<f32>>>

// -----

// CHECK-LABEL: func private @future_type_multiple
// CHECK-SAME: !stablehlo.future<tensor<f32>, tensor<i32>>
func.func private @future_type_multiple() -> !stablehlo.future<tensor<f32>, tensor<i32>>

// -----

// CHECK-LABEL: func private @future_type_multiple_ranked
// CHECK-SAME: !stablehlo.future<tensor<2x3xf32>, tensor<4xi32>>
func.func private @future_type_multiple_ranked() -> !stablehlo.future<tensor<2x3xf32>, tensor<4xi32>>

// -----

// expected-error@+1 {{future element type must be a ranked tensor, but got 'f32'}}
func.func private @future_type_invalid_scalar() -> !stablehlo.future<f32>

// -----

// expected-error@+1 {{future element type must be a ranked tensor, but got 'i32'}}
func.func private @future_type_invalid_integer() -> !stablehlo.future<i32>

// -----

// expected-error@+1 {{future element type must be a ranked tensor, but got 'f32'}}
func.func private @future_type_invalid_multiple() -> !stablehlo.future<tensor<f32>, f32>
