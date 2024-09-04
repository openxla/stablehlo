// RUN: stablehlo-opt --stablehlo-legalize-deprecated-ops --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: op_dot
func.func @op_dot(%arg0: tensor<2x3xf32>,
                 %arg1: tensor<3x?xf32>) -> tensor<2x?xf32> {
  // CHECK: stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x?xf32>) -> tensor<2x?xf32>
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3x?xf32>) -> tensor<2x?xf32>
  func.return %0 : tensor<2x?xf32>
}

// -----

// CHECK-LABEL: op_unary_einsum
func.func @op_unary_einsum(%arg0: tensor<8x16xf32>) -> tensor<8xf32> {
  // CHECK:      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-NEXT: stablehlo.einsum %cst, %arg0, config = ",ab->a" : (tensor<f32>, tensor<8x16xf32>) -> tensor<8xf32>
  %0 = "stablehlo.unary_einsum"(%arg0) {einsum_config = "ab->a"} : (tensor<8x16xf32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
}

// -----

// CHECK-LABEL: op_broadcast
func.func @op_broadcast(%arg: tensor<1xf32>) -> tensor<4x3x2x1xf32> {
  // CHECK: stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<1xf32>) -> tensor<4x3x2x1xf32>
  %0 = "stablehlo.broadcast"(%arg) {broadcast_sizes = array<i64: 4, 3, 2>} : (tensor<1xf32>) -> tensor<4x3x2x1xf32>
  func.return %0: tensor<4x3x2x1xf32>
}

// -----

// CHECK-LABEL: op_create_token
func.func @op_create_token() -> !stablehlo.token {
  // CHECK: stablehlo.after_all  : !stablehlo.token
  %0 = "stablehlo.create_token"() : () -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

// -----

// CHECK-LABEL: op_cross_replica_sum
func.func @op_cross_replica_sum(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK{LITERAL}: "stablehlo.all_reduce"(%arg0) <{replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>}>
  // CHECK:   stablehlo.add %arg1, %arg2 : tensor<f32>
  %0 = "stablehlo.cross-replica-sum"(%arg0) {replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>} : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: op_einsum
func.func @op_einsum(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  // TODO CHECK
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "ijk,ikm->ijm"}: (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>
}

// -----

// CHECK-LABEL: op_torch_index_select
func.func @op_torch_index_select(%arg0: tensor<5x1x5xi32>,
                         %arg1: tensor<2xi32>) ->  tensor<2x1x5xi32> {
  // TODO CHECK
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) {dim = 0 : i64, batch_dims = 0 : i64} : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}

// -----

func.func @op_rng(%min: tensor<f32>, %max: tensor<f32>) -> tensor<10xf32> {
  %shape = arith.constant dense<[10]>  : tensor<1xi32>
  // expected-error @+1 {{failed to legalize operation 'stablehlo.rng' that was explicitly marked illegal}}
  %0 = "stablehlo.rng"(%min, %max, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<1xi32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @op_map(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  // expected-error @+1 {{failed to legalize operation 'stablehlo.map' that was explicitly marked illegal}}
  %0 = "stablehlo.map"(%arg0) ({
    ^bb0(%arg1: tensor<f32>):
      %1 = "stablehlo.abs"(%arg1) : (tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    dimensions = array<i64: 0>
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
