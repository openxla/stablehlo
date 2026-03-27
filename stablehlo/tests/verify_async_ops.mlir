// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// Tests that async_start can successfully wrap an all_reduce operation.
// CHECK-LABEL: func @async_start_all_reduce
func.func @async_start_all_reduce(%arg0: tensor<4x4xf32>) -> !stablehlo.future<tensor<4x4xf32>> {
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<4x4xf32>):
    %1 = "stablehlo.all_reduce"(%barg0) ({
    ^bb1(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    "stablehlo.return"(%1) : (tensor<4x4xf32>) -> ()
  }) : (tensor<4x4xf32>) -> !stablehlo.future<tensor<4x4xf32>>
  func.return %0: !stablehlo.future<tensor<4x4xf32>>
}

// -----

// Tests that async_start can successfully wrap an all_gather operation.
// CHECK-LABEL: func @async_start_all_gather
func.func @async_start_all_gather(%arg0: tensor<8x2xf32>) -> !stablehlo.future<tensor<8x8xf32>> {
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<8x2xf32>):
    %1 = "stablehlo.all_gather"(%barg0) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
    } : (tensor<8x2xf32>) -> tensor<8x8xf32>
    "stablehlo.return"(%1) : (tensor<8x8xf32>) -> ()
  }) : (tensor<8x2xf32>) -> !stablehlo.future<tensor<8x8xf32>>
  func.return %0: !stablehlo.future<tensor<8x8xf32>>
}

// -----

// Tests that async_start can successfully wrap an all_to_all operation.
// CHECK-LABEL: func @async_start_all_to_all
func.func @async_start_all_to_all(%arg0: tensor<4x16xf32>) -> !stablehlo.future<tensor<16x4xf32>> {
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<4x16xf32>):
    %1 = "stablehlo.all_to_all"(%barg0) {
      split_dimension = 1 : i64,
      concat_dimension = 0 : i64,
      split_count = 4 : i64,
      replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
    } : (tensor<4x16xf32>) -> tensor<16x4xf32>
    "stablehlo.return"(%1) : (tensor<16x4xf32>) -> ()
  }) : (tensor<4x16xf32>) -> !stablehlo.future<tensor<16x4xf32>>
  func.return %0: !stablehlo.future<tensor<16x4xf32>>
}

// -----

// Tests that async_start can successfully wrap a collective_permute operation.
// CHECK-LABEL: func @async_start_collective_permute
func.func @async_start_collective_permute(%arg0: tensor<128x32xf32>) -> !stablehlo.future<tensor<128x32xf32>> {
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<128x32xf32>):
    %1 = "stablehlo.collective_permute"(%barg0) {
      source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
    } : (tensor<128x32xf32>) -> tensor<128x32xf32>
    "stablehlo.return"(%1) : (tensor<128x32xf32>) -> ()
  }) : (tensor<128x32xf32>) -> !stablehlo.future<tensor<128x32xf32>>
  func.return %0: !stablehlo.future<tensor<128x32xf32>>
}

// -----

// Tests that async_start can successfully wrap a collective_broadcast operation.
// CHECK-LABEL: func @async_start_collective_broadcast
func.func @async_start_collective_broadcast(%arg0: tensor<16x8xf32>) -> !stablehlo.future<tensor<16x8xf32>> {
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<16x8xf32>):
    %1 = "stablehlo.collective_broadcast"(%barg0) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<16x8xf32>) -> tensor<16x8xf32>
    "stablehlo.return"(%1) : (tensor<16x8xf32>) -> ()
  }) : (tensor<16x8xf32>) -> !stablehlo.future<tensor<16x8xf32>>
  func.return %0: !stablehlo.future<tensor<16x8xf32>>
}

// -----

// Tests that async_start can successfully wrap a reduce_scatter operation.
// CHECK-LABEL: func @async_start_reduce_scatter
func.func @async_start_reduce_scatter(%arg0: tensor<4x16xf32>) -> !stablehlo.future<tensor<4x4xf32>> {
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<4x16xf32>):
    %1 = "stablehlo.reduce_scatter"(%barg0) ({
      ^bb1(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
        scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
    "stablehlo.return"(%1) : (tensor<4x4xf32>) -> ()
  }) : (tensor<4x16xf32>) -> !stablehlo.future<tensor<4x4xf32>>
  func.return %0: !stablehlo.future<tensor<4x4xf32>>
}

// -----

// Tests that async_start fails if the inner operation is not a collective.
func.func @async_start_invalid_op(%arg0: tensor<4x4xf32>) -> !stablehlo.future<tensor<4x4xf32>> {
  // expected-error@+2 {{'stablehlo.async_start' op region must contain a collective or slice operation}}
  // expected-error@+1 {{'stablehlo.async_start' op failed to infer returned types}}
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<4x4xf32>):
    %1 = "stablehlo.add"(%barg0, %barg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    "stablehlo.return"(%1) : (tensor<4x4xf32>) -> ()
  }) : (tensor<4x4xf32>) -> !stablehlo.future<tensor<4x4xf32>>
  func.return %0: !stablehlo.future<tensor<4x4xf32>>
}

// -----

// Tests that async_start fails if the region contains more than one operation.
func.func @async_start_invalid_num_ops(%arg0: tensor<4x4xf32>) -> !stablehlo.future<tensor<4x4xf32>> {
  // expected-error@+2 {{'stablehlo.async_start' op region must contain exactly one operation and a return}}
  // expected-error@+1 {{'stablehlo.async_start' op failed to infer returned types}}
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<4x4xf32>):
    %1 = "stablehlo.all_reduce"(%barg0) ({
    ^bb1(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = "stablehlo.all_reduce"(%1) ({
    ^bb1(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%4) : (tensor<f32>) -> ()
    }) {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    "stablehlo.return"(%3) : (tensor<4x4xf32>) -> ()
  }) : (tensor<4x4xf32>) -> !stablehlo.future<tensor<4x4xf32>>
  func.return %0: !stablehlo.future<tensor<4x4xf32>>
}

// -----


// Tests that async_start fails if the inner collective operation fails to verify.
func.func @async_start_all_gather_invalid(%arg0: tensor<8x2xf32>) -> !stablehlo.future<tensor<8x8xf32>> {
  // expected-error@+3 {{all_gather_dim must be a valid index of operand}}
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<8x2xf32>):
    %1 = "stablehlo.all_gather"(%barg0) {
      all_gather_dim = 2 : i64,
      replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
    } : (tensor<8x2xf32>) -> tensor<8x8xf32>
    "stablehlo.return"(%1) : (tensor<8x8xf32>) -> ()
  }) : (tensor<8x2xf32>) -> !stablehlo.future<tensor<8x8xf32>>
  func.return %0: !stablehlo.future<tensor<8x8xf32>>
}

// -----

// Tests that async_start fails if it has no regions.
func.func @async_start_no_regions(%arg0: tensor<4x4xf32>) -> !stablehlo.future<tensor<4x4xf32>> {
  // expected-error@+1 {{op requires one region}}
  %0 = "stablehlo.async_start"(%arg0) : (tensor<4x4xf32>) -> !stablehlo.future<tensor<4x4xf32>>
  func.return %0: !stablehlo.future<tensor<4x4xf32>>
}

// -----

// Tests that async_start fails if it has more than one region.
func.func @async_start_multiple_regions(%arg0: tensor<4x4xf32>) -> !stablehlo.future<tensor<4x4xf32>> {
  // expected-error@+1 {{op requires one region}}
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<4x4xf32>):
    %1 = "stablehlo.all_reduce"(%barg0) ({
    ^bb1(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    "stablehlo.return"(%1) : (tensor<4x4xf32>) -> ()
  }, {
  ^bb0(%barg0: tensor<4x4xf32>):
    %1 = "stablehlo.all_reduce"(%barg0) ({
    ^bb1(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    "stablehlo.return"(%1) : (tensor<4x4xf32>) -> ()
  }) : (tensor<4x4xf32>) -> !stablehlo.future<tensor<4x4xf32>>
  func.return %0: !stablehlo.future<tensor<4x4xf32>>
}

// -----

// Tests that async_done can successfully wait for an async_start operation.
// CHECK-LABEL: func @async_done
func.func @async_done(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<4x4xf32>):
    %2 = "stablehlo.all_reduce"(%barg0) ({
    ^bb1(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %3 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%3) : (tensor<f32>) -> ()
    }) {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    "stablehlo.return"(%2) : (tensor<4x4xf32>) -> ()
  }) : (tensor<4x4xf32>) -> !stablehlo.future<tensor<4x4xf32>>
  %1 = "stablehlo.async_done"(%0) : (!stablehlo.future<tensor<4x4xf32>>) -> tensor<4x4xf32>
  func.return %1: tensor<4x4xf32>
}

// -----

// Tests that async_start fails if the body return type doesn't match the
// op return type (body returns tensor<8x8xf32> but future wraps tensor<8x2xf32>).
func.func @async_start_body_return_type_mismatch(%arg0: tensor<8x2xf32>) -> !stablehlo.future<tensor<8x2xf32>> {
  // expected-error@+2 {{'stablehlo.async_start' op inferred type(s) '!stablehlo.future<tensor<8x8xf32>>' are incompatible with return type(s) of operation '!stablehlo.future<tensor<8x2xf32>>'}}
  // expected-error@+1 {{'stablehlo.async_start' op failed to infer returned types}}
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<8x2xf32>):
    %1 = "stablehlo.all_gather"(%barg0) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
    } : (tensor<8x2xf32>) -> tensor<8x8xf32>
    "stablehlo.return"(%1) : (tensor<8x8xf32>) -> ()
  }) : (tensor<8x2xf32>) -> !stablehlo.future<tensor<8x2xf32>>
  func.return %0: !stablehlo.future<tensor<8x2xf32>>
}

// -----

// Tests that async_start fails if the body region input arg types don't match
// the async op input arg types.
func.func @async_start_body_input_type_mismatch(%arg0: tensor<128x32xf32>) -> !stablehlo.future<tensor<128x32xf32>> {
  // expected-error@+2 {{'stablehlo.async_start' op operand type 'tensor<128x32xf32>' at index 0 must match region argument type 'tensor<128x32xi32>'}}
  // expected-error@+1 {{'stablehlo.async_start' op failed to infer returned types}}
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<128x32xi32>):
    %1 = "stablehlo.collective_permute"(%barg0) {
      source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
    } : (tensor<128x32xi32>) -> tensor<128x32xi32>
    "stablehlo.return"(%1) : (tensor<128x32xi32>) -> ()
  }) : (tensor<128x32xf32>) -> !stablehlo.future<tensor<128x32xf32>>
  func.return %0: !stablehlo.future<tensor<128x32xf32>>
}

// -----

// Tests that async_start fails if the body region input count doesn't match
// the async op input count.
func.func @async_start_body_input_count_mismatch(%arg0: tensor<128x32xf32>) -> !stablehlo.future<tensor<128x32xf32>> {
  // expected-error@+2 {{'stablehlo.async_start' op number of operands (1) must match number of region arguments (2)}}
  // expected-error@+1 {{'stablehlo.async_start' op failed to infer returned types}}
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<128x32xf32>, %barg1: tensor<128x32xf32>):
    %1 = "stablehlo.collective_permute"(%barg0) {
      source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
    } : (tensor<128x32xf32>) -> tensor<128x32xf32>
    "stablehlo.return"(%1) : (tensor<128x32xf32>) -> ()
  }) : (tensor<128x32xf32>) -> !stablehlo.future<tensor<128x32xf32>>
  func.return %0: !stablehlo.future<tensor<128x32xf32>>
}

// -----

// Tests that async_start fails if the body return element type doesn't match
// the async op return type (body returns f32 but future wraps i32).
func.func @async_start_body_return_element_type_mismatch(%arg0: tensor<128x32xf32>) -> !stablehlo.future<tensor<128x32xi32>> {
  // expected-error@+2 {{'stablehlo.async_start' op inferred type(s) '!stablehlo.future<tensor<128x32xf32>>' are incompatible with return type(s) of operation '!stablehlo.future<tensor<128x32xi32>>'}}
  // expected-error@+1 {{'stablehlo.async_start' op failed to infer returned types}}
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<128x32xf32>):
    %1 = "stablehlo.collective_permute"(%barg0) {
      source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
    } : (tensor<128x32xf32>) -> tensor<128x32xf32>
    "stablehlo.return"(%1) : (tensor<128x32xf32>) -> ()
  }) : (tensor<128x32xf32>) -> !stablehlo.future<tensor<128x32xi32>>
  func.return %0: !stablehlo.future<tensor<128x32xi32>>
}

// -----

// Tests that async_done fails if the result type doesn't match the future's
// inner type.
func.func @async_done_result_type_mismatch(%arg0: tensor<4x4xf32>) -> tensor<4x4xi32> {
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%barg0: tensor<4x4xf32>):
    %1 = "stablehlo.all_reduce"(%barg0) ({
    ^bb1(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    "stablehlo.return"(%1) : (tensor<4x4xf32>) -> ()
  }) : (tensor<4x4xf32>) -> !stablehlo.future<tensor<4x4xf32>>
  // expected-error@+2 {{'stablehlo.async_done' op inferred type(s) 'tensor<4x4xf32>' are incompatible with return type(s) of operation 'tensor<4x4xi32>'}}
  // expected-error@+1 {{'stablehlo.async_done' op failed to infer returned types}}
  %1 = "stablehlo.async_done"(%0) : (!stablehlo.future<tensor<4x4xf32>>) -> tensor<4x4xi32>
  func.return %1: tensor<4x4xi32>
}
