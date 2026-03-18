// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file

// -----

func.func @all_reduce_empty_operands() {
  // expected-error@+1 {{must have at least 1 operand}}
  "stablehlo.all_reduce"() ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %0 : tensor<f32>
  }) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : () -> ()
  func.return
}

// -----

func.func @all_gather_empty_operands() {
  // expected-error@+1 {{must have at least 1 operand}}
  "stablehlo.all_gather"() {
    all_gather_dim = 0 : i64,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : () -> ()
  func.return
}

// -----

func.func @all_to_all_empty_operands() {
  // expected-error@+1 {{must have at least 1 operand}}
  "stablehlo.all_to_all"() {
    split_dimension = 0 : i64,
    concat_dimension = 0 : i64,
    split_count = 2 : i64,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : () -> ()
  func.return
}
