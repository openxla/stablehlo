// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=0.16.0' --verify-diagnostics --split-input-file %s

// expected-error @-3 {{failed to convert VHLO to v0.16.0}}
func.func @reduce_with_promotable_types(%arg0: tensor<4x4xf32>, %arg1 : tensor<f32>)
    -> (tensor<4xf64>) {

  // expected-error @+1 {{failed to legalize operation 'vhlo.reduce_v1' that was explicitly marked illegal}}
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64> ):
    %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    "stablehlo.return"(%1) : (tensor<f64>) -> ()

  }) {dimensions = array<i64: 0>} : (tensor<4x4xf32>, tensor<f32>) -> tensor<4xf64>

  func.return %0: tensor<4xf64>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v0.16.0}}
func.func @all_reduce_with_promotable_types(%operand: tensor<f32>) -> tensor<f64> {

  // expected-error @+1 {{failed to legalize operation 'vhlo.all_reduce_v2' that was explicitly marked illegal}}
  %result = "stablehlo.all_reduce"(%operand) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%0) : (tensor<f64>) -> ()
  }) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<f32>) -> tensor<f64>

  func.return %result : tensor<f64>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v0.16.0}}
func.func @reduce_scatter_with_promotable_types(%data: tensor<4x16xf32>) -> tensor<4x4xf64> {

  // expected-error @+1 {{failed to legalize operation 'vhlo.reduce_scatter_v1' that was explicitly marked illegal}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f64>
    "stablehlo.return"(%1) : (tensor<f64>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids} : (tensor<4x16xf32>) -> tensor<4x4xf64>
  func.return %0 : tensor<4x4xf64>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v0.16.0}}
func.func @reduce_window_with_promotable_types(%arg0: tensor<4x2xf32>,
    %arg1: tensor<4x2xf32>, %init0: tensor<f32>, %init1: tensor<f32>) ->
    (tensor<2x2xf64>, tensor<2x2xf32>) {

  // expected-error @+1 {{failed to legalize operation 'vhlo.reduce_window_v1' that was explicitly marked illegal}}
  %0:2 = "stablehlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f64>, %a1: tensor<f32>, %b0: tensor<f64>,
                %b1: tensor<f32>):
              %2 = stablehlo.add %a0, %b0 : tensor<f64>
              %3 = stablehlo.add %a1, %b1 : tensor<f32>
              "stablehlo.return"(%2,%3) : (tensor<f64>, tensor<f32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = array<i64: 5, 1>,
           window_strides = array<i64: 3, 1> }
         : (tensor<4x2xf32>, tensor<4x2xf32>, tensor<f32>, tensor<f32>) ->
              (tensor<2x2xf64>, tensor<2x2xf32>)
  func.return %0#0, %0#1 : tensor<2x2xf64>, tensor<2x2xf32>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v0.16.0}}
func.func @scatter_with_promotable_types(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf64> {

  // expected-error @+1 {{failed to legalize operation 'vhlo.scatter_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f64>, %rhs: tensor<f64>):
    %add = stablehlo.add %lhs, %rhs : tensor<f64>
    "stablehlo.return"(%add) : (tensor<f64>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf64>
  func.return %0 : tensor<200x100x300xf64>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v0.16.0}}
func.func @select_and_scatter_with_promotable_types(
    %arg0: tensor<10x24x24x64xf32>,
    %arg1: tensor<10x12x12x64xf32>) -> () {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // expected-error @+1 {{failed to legalize operation 'vhlo.select_and_scatter_v1' that was explicitly marked illegal}}
  %1 = "stablehlo.select_and_scatter"(%arg0, %arg1, %0) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = "stablehlo.compare"(%arg3, %arg4) {
      comparison_direction = #stablehlo<comparison_direction GE>
      } : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
    %2 = stablehlo.add %arg3, %arg4 : tensor<f64>
    "stablehlo.return"(%2) : (tensor<f64>) -> ()
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>,
    window_strides = array<i64: 1, 2, 2, 1>,
    padding = dense<0> : tensor<4x2xi64>
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
        tensor<10x24x24x64xf64>
  func.return
}
