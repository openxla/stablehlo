// RUN: stablehlo-opt %s --stablehlo-legalize-to-tosa | FileCheck %s

// CHECK-LABEL: @abs
func.func @abs(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.abs
  %0 = "stablehlo.abs"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @ceil
func.func @ceil(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.ceil
  %0 = "stablehlo.ceil"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @convert
func.func @convert(%arg : tensor<10xi32>) -> tensor<10xf32> {
  // CHECK: tosa.cast
  %0 = "stablehlo.convert"(%arg) : (tensor<10xi32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @exponential
func.func @exponential(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.exp
  %0 = "stablehlo.exponential"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @exponential_minus_one
func.func @exponential_minus_one(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{value = dense<1.000000e+00>
  // CHECK-DAG: %[[VAR1:.*]] = tosa.exp %arg0
  // CHECK-DAG: %[[VAR2:.*]] = tosa.sub %[[VAR1]], %[[VAR0]]
  %0 = "stablehlo.exponential_minus_one"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @floor
func.func @floor(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.floor
  %0 = "stablehlo.floor"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @is_finite
func.func @is_finite(%arg : tensor<10xf32>) -> tensor<10xi1> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{value = dense<0x7F800000>
  // CHECK-DAG: %[[VAR1:.*]] = tosa.abs %arg0
  // CHECK-DAG: %[[VAR2:.*]] = tosa.equal %[[VAR1]], %[[VAR0]]
  // CHECK-DAG: %[[VAR3:.*]] = tosa.logical_not %[[VAR2]]
  %0 = "stablehlo.is_finite"(%arg) : (tensor<10xf32>) -> tensor<10xi1>
  return %0 : tensor<10xi1>
}

// CHECK-LABEL: @log
func.func @log(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.log
  %0 = "stablehlo.log"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @log_plus_one
func.func @log_plus_one(%arg : tensor<10xf16>) -> tensor<10xf16> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{value = dense<1.000000e+00>
  // CHECK-DAG: %[[VAR1:.*]] = tosa.add %arg0, %[[VAR0]]
  // CHECK-DAG: %[[VAR2:.*]] = tosa.log %[[VAR1]]
  %0 = "stablehlo.log_plus_one"(%arg) : (tensor<10xf16>) -> tensor<10xf16>
  return %0 : tensor<10xf16>
}

// CHECK-LABEL: @negate
func.func @negate(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.negate
  %0 = "stablehlo.negate"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @slice
func.func @slice(%arg : tensor<4x3xf32>) -> tensor<2x2xf32> {
  // CHECK: tosa.slice %arg0 {size = array<i64: 2, 2>, start = array<i64: 2, 1>}
  %0 = "stablehlo.slice"(%arg) {
    start_indices = array<i64: 2, 1>,
    limit_indices = array<i64: 4, 3>,
    strides = array<i64: 1, 1>
  } : (tensor<4x3xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: @slice_stride_not_one
func.func @slice_stride_not_one(%arg : tensor<4x3xf32>) -> tensor<2x1xf32> {
  // tosa.slice only supports strides of 1, so this should not legalize.
  // CHECK: stablehlo.slice
  %0 = "stablehlo.slice"(%arg) {
    start_indices = array<i64: 2, 1>,
    limit_indices = array<i64: 4, 3>,
    strides = array<i64: 1, 2>
  } : (tensor<4x3xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// CHECK-LABEL: @slice_rank_seven
func.func @slice_rank_seven(%arg : tensor<2x3x4x5x6x7x8xf32>) -> tensor<1x2x3x4x5x6x7xf32> {
  // tosa.slice only supports 1D to 6D tensors, so this should not legalize.
  // CHECK: stablehlo.slice
  %0 = "stablehlo.slice"(%arg) {
    start_indices = array<i64: 1, 1, 1, 1, 1, 1, 1>,
    limit_indices = array<i64: 2, 3, 4, 5, 6, 7, 8>,
    strides = array<i64: 1, 1, 1, 1, 1, 1, 1>
  } : (tensor<2x3x4x5x6x7x8xf32>) -> tensor<1x2x3x4x5x6x7xf32>
  return %0 : tensor<1x2x3x4x5x6x7xf32>
}

// CHECK-LABEL: @tanh
func.func @tanh(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.tanh
  %0 = "stablehlo.tanh"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @transpose
func.func @transpose(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  // CHECK: %[[VAR0:.*]] = "tosa.const"() <{value = dense<[2, 1, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
  // CHECK: %[[VAR1:.*]] = tosa.transpose %arg0, %[[VAR0]]
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 2, 1, 0>} : (tensor<1x2x3xf32>) -> tensor<3x2x1xf32>
  return %0 : tensor<3x2x1xf32>
}

// CHECK-LABEL: @while
func.func @while(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{value = dense<3> : tensor<i32>}
  // CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{value = dense<1> : tensor<i32>}
  // CHECK:     %[[VAR2:.*]] = tosa.while_loop (%[[ARG1:.+]] = %arg0) : (tensor<i32>) -> tensor<i32> {
  // CHECK:       %[[VAR3:.*]] = tosa.equal %[[ARG1]], %[[VAR0]]
  // CHECK:       tosa.yield %[[VAR3]]
  // CHECK:     } do {
  // CHECK:     ^bb0(%[[ARG1:.+]]: tensor<i32>):
  // CHECK:       %[[VAR4:.*]] = tosa.add %[[ARG1]], %[[VAR1]]
  // CHECK:       tosa.yield %[[VAR4]]
  // CHECK:     }
  // CHECK:     return %[[VAR2]] : tensor<i32>
  // CHECK:   }
  %0 = "stablehlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<i32>):
    %1 = "stablehlo.constant"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
    %2 = "stablehlo.compare"(%arg1, %1) {comparison_direction = #stablehlo<comparison_direction EQ>}: (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i32>):
    %1 = "stablehlo.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %2 = "stablehlo.add"(%arg1, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%2) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> (tensor<i32>)
  return %0 : tensor<i32>
}
