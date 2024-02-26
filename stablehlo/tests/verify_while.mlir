// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file

// CHECK: func @while
func.func @while(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<4xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  %cst = arith.constant dense<-1> : tensor<i32>
  %cst_0 = arith.constant dense<1> : tensor<i32>
  %cst_1 = arith.constant dense<0> : tensor<i32>
  %cst_2 = arith.constant dense<1000> : tensor<i32>
  %1:3 = "stablehlo.while"(%cst_1, %cst, %cst_2) ({
  ^bb0(%arg9: tensor<i32>, %arg10: tensor<i32>, %arg11: tensor<i32>):
    %4 = "stablehlo.compare"(%arg9, %arg11) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%4) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg9: tensor<i32>, %arg10: tensor<i32>, %arg11: tensor<i32>):
    %3 = stablehlo.add %arg9, %cst_0 : tensor<i32>
    "stablehlo.return"(%3, %arg10, %arg11) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
  }) : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  func.return %1#0, %1#2, %1#2: tensor<i32>, tensor<i32>, tensor<i32>
}

// -----

// CHECK-LABEL: while_dynamic
func.func @while_dynamic(%arg0: tensor<3xf32>) -> tensor<?xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  %1:4 = "stablehlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<?xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "stablehlo.slice"(%arg2) {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "stablehlo.compare"(%arg1, %3) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "stablehlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "stablehlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<?xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "stablehlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = array<i64: 0>} : (tensor<1xf32>) -> tensor<3xf32>
    %4 = stablehlo.add %3, %arg4 : tensor<3xf32>
    "stablehlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<?xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<?xf32>)
  func.return %1#3: tensor<?xf32>
}

// -----

// CHECK-LABEL: while_with_different_types
func.func @while_with_different_types(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  %1:4 = "stablehlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "stablehlo.slice"(%arg2) {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "stablehlo.compare"(%arg1, %3) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "stablehlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "stablehlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "stablehlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = array<i64: 0>} : (tensor<1xf32>) -> tensor<3xf32>
    %4 = stablehlo.add %3, %arg4 : tensor<3xf32>
    "stablehlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_c1(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect operands to be compatible with condition block arguments but got 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>' vs 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<3xf32>', 'tensor<3xf32>'}}
  %1:4 = "stablehlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "stablehlo.slice"(%arg2) {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "stablehlo.compare"(%arg1, %3) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "stablehlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "stablehlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "stablehlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = array<i64: 0>} : (tensor<1xf32>) -> tensor<3xf32>
    %4 = stablehlo.add %3, %arg4 : tensor<3xf32>
    "stablehlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_c1(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect operands to be compatible with condition block arguments but got 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>' vs 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>'}}
  %1:4 = "stablehlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "stablehlo.slice"(%arg2) {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "stablehlo.compare"(%arg1, %3) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "stablehlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "stablehlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<3xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "stablehlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = array<i64: 0>} : (tensor<1xf32>) -> tensor<3xf32>
    %4 = stablehlo.add %3, %arg4 : tensor<3xf32>
    "stablehlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<3xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_c1(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect condition body returns a single value but got 2}}
  %1:4 = "stablehlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "stablehlo.slice"(%arg2) {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "stablehlo.compare"(%arg1, %3) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "stablehlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "stablehlo.return"(%5, %5) : (tensor<i1>, tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "stablehlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = array<i64: 0>} : (tensor<1xf32>) -> tensor<3xf32>
    %4 = stablehlo.add %3, %arg4 : tensor<3xf32>
    "stablehlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_c1(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect condition block return a zero-ranked tensor of i1 but got 'tensor<i32>'}}
  %1:4 = "stablehlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = "stablehlo.reshape"(%arg1) : (tensor<1xi32>) -> tensor<i32>
    "stablehlo.return"(%2) : (tensor<i32>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "stablehlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = array<i64: 0>} : (tensor<1xf32>) -> tensor<3xf32>
    %4 = stablehlo.add %3, %arg4 : tensor<3xf32>
    "stablehlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_c1(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect condition block return a zero-ranked tensor of i1 but got 'tensor<1xi1>'}}
  %1:4 = "stablehlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "stablehlo.slice"(%arg2) {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "stablehlo.compare"(%arg1, %3) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    "stablehlo.return"(%4) : (tensor<1xi1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "stablehlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = array<i64: 0>} : (tensor<1xf32>) -> tensor<3xf32>
    %4 = stablehlo.add %3, %arg4 : tensor<3xf32>
    "stablehlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_c2(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect operands to be compatible with body block arguments but got 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>' vs 'tensor<1xi32>', 'tensor<3xi32>', 'tensor<1xf32>', 'tensor<3xf32>'}}
  %1:4 = "stablehlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "stablehlo.slice"(%arg2) {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "stablehlo.compare"(%arg1, %3) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "stablehlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "stablehlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<3xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "stablehlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = array<i64: 0>} : (tensor<1xf32>) -> tensor<3xf32>
    %4 = stablehlo.add %3, %arg4 : tensor<3xf32>
    "stablehlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<3xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_c2(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect operands to be compatible with body block arguments but got 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>' vs 'tensor<1xi32>', 'tensor<3xi32>', 'tensor<1xf32>'}}
  %1:4 = "stablehlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "stablehlo.slice"(%arg2) {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "stablehlo.compare"(%arg1, %3) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "stablehlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "stablehlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<3xi32>, %arg3: tensor<1xf32>):
    %3 = "stablehlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = array<i64: 0>} : (tensor<1xf32>) -> tensor<3xf32>
    "stablehlo.return"(%arg1, %arg2, %arg3, %3) : (tensor<1xi32>, tensor<3xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_c2(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect operands to be compatible with body block return types but got 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>' vs 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<1xf32>'}}
  %1:4 = "stablehlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "stablehlo.slice"(%arg2) {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "stablehlo.compare"(%arg1, %3) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "stablehlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "stablehlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    "stablehlo.return"(%arg1, %arg2, %arg3, %arg3) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<1xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_c2(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect operands to be compatible with body block return types but got 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>' vs 'tensor<1xf32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>'}}
  %1:4 = "stablehlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "stablehlo.slice"(%arg2) {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "stablehlo.compare"(%arg1, %3) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "stablehlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "stablehlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    "stablehlo.return"(%arg3, %arg2, %arg3, %arg4) : (tensor<1xf32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_c2(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect operands to be compatible with body block return types but got 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>' vs 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>'}}
  %1:4 = "stablehlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "stablehlo.slice"(%arg2) {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "stablehlo.compare"(%arg1, %3) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "stablehlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "stablehlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "stablehlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = array<i64: 0>} : (tensor<1xf32>) -> tensor<3xf32>
    %4 = stablehlo.add %3, %arg4 : tensor<3xf32>
    "stablehlo.return"(%arg1, %arg2, %arg3) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}
