// RUN: stablehlo-opt %s --stablehlo-legalize-to-tosa | FileCheck %s

// CHECK-LABEL: @constant
func.func @constant() -> tensor<10xf32> {
  // CHECK: tosa.const
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @constant_f64
func.func @constant_f64() -> tensor<10xf64> {
  // CHECK: tosa.const
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
  return %0 : tensor<10xf64>
}

// CHECK-LABEL: @iota_dimension_0
func.func @iota_dimension_0() -> tensor<4x8xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.const"()
  // CHECK-SAME{LITERAL}: <{values = dense<[[0.000000e+00], [1.000000e+00], [2.000000e+00], [3.000000e+00]]> : tensor<4x1xf32>}> : () -> tensor<4x1xf32>
  // CHECK-DAG: %[[VAR1:.*]] = tosa.const_shape  {values = dense<[1, 8]> : vector<2xindex>} : () -> !tosa.shape<2>
  // CHECK-DAG: %[[VAR2:.*]] = tosa.tile %[[VAR0]], %[[VAR1]]
  %0 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> (tensor<4x8xf32>)
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: @iota_dimension_1
func.func @iota_dimension_1() -> tensor<4x8xi32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.const"()
  // CHECK-SAME{LITERAL}: <{values = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi32>}> : () -> tensor<1x8xi32>
  // CHECK-DAG: %[[VAR1:.*]] = tosa.const_shape  {values = dense<[4, 1]> : vector<2xindex>} : () -> !tosa.shape<2>
  // CHECK-DAG: %[[VAR2:.*]] = tosa.tile %[[VAR0]], %[[VAR1]]
  %0 = "stablehlo.iota"() {iota_dimension = 1 : i64} : () -> (tensor<4x8xi32>)
  return %0 : tensor<4x8xi32>
}
