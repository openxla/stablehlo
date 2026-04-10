// RUN: stablehlo-opt %s -stablehlo-compatibility-expander="target=1.14.0" | FileCheck %s

module {


  // CHECK-LABEL: @all_reduce_rgv3
  func.func @all_reduce_rgv3(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: replica_groups = dense<{{\[\[}}0, 2], [1, 3]]> : tensor<2x2xi64>
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }) {
      replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = "x", size = 2>, #stablehlo.mesh_axis<name = "y", size = 2>]>, axes = [#stablehlo.axis_ref<name = "x">]>
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }


  // CHECK-LABEL: @all_gather_rgv3
  func.func @all_gather_rgv3(%arg0: tensor<4xf32>) -> tensor<8xf32> {
    // CHECK: replica_groups = dense<{{\[\[}}0, 1], [2, 3]]> : tensor<2x2xi64>
    %0 = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 0 : i64,
      replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = "x", size = 2>, #stablehlo.mesh_axis<name = "y", size = 2>]>, axes = [#stablehlo.axis_ref<name = "y">]>
    } : (tensor<4xf32>) -> tensor<8xf32>

    return %0 : tensor<8xf32>
  }

  // CHECK-LABEL: @all_to_all_rgv3
  func.func @all_to_all_rgv3(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: replica_groups = dense<{{\[\[}}0, 2], [1, 3]]> : tensor<2x2xi64>
    %0 = "stablehlo.all_to_all"(%arg0) {
      concat_dimension = 0 : i64,
      split_dimension = 0 : i64,
      split_count = 2 : i64,
      replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = "x", size = 2>, #stablehlo.mesh_axis<name = "y", size = 2>]>, axes = [#stablehlo.axis_ref<name = "x">]>
    } : (tensor<4xf32>) -> tensor<4xf32>

    return %0 : tensor<4xf32>
  }
}
