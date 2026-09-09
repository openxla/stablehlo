// RUN: stablehlo-opt %s -stablehlo-compatibility-expander="target=1.14.0" -allow-unregistered-dialect | FileCheck %s

module {


  // CHECK-LABEL: @all_reduce_rgv3
  func.func @all_reduce_rgv3(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK{LITERAL}: replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>
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
    // CHECK{LITERAL}: replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
    %0 = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 0 : i64,
      replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = "x", size = 2>, #stablehlo.mesh_axis<name = "y", size = 2>]>, axes = [#stablehlo.axis_ref<name = "y">]>
    } : (tensor<4xf32>) -> tensor<8xf32>

    return %0 : tensor<8xf32>
  }

  // CHECK-LABEL: @all_to_all_rgv3
  func.func @all_to_all_rgv3(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK{LITERAL}: replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>
    %0 = "stablehlo.all_to_all"(%arg0) {
      concat_dimension = 0 : i64,
      split_dimension = 0 : i64,
      split_count = 2 : i64,
      replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = "x", size = 2>, #stablehlo.mesh_axis<name = "y", size = 2>]>, axes = [#stablehlo.axis_ref<name = "x">]>
    } : (tensor<4xf32>) -> tensor<4xf32>

    return %0 : tensor<4xf32>
  }
  "sdy.mesh"() {sym_name = "sdy_mesh", stablehlo.mesh = {axes = [{name = "x", size = 2 : i64}, {name = "y", size = 2 : i64}]}} : () -> ()

  // CHECK-LABEL: @all_reduce_sdy_mesh
  func.func @all_reduce_sdy_mesh(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK{LITERAL}: replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }) {
      replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @sdy_mesh, axes = [#stablehlo.axis_ref<name = "x">]>
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  "sdy.mesh"() {sym_name = "sdy_mesh_dev", stablehlo.mesh = {axes = [{name = "x", size = 2 : i64}, {name = "y", size = 2 : i64}], device_ids = dense<[0, 2, 1, 3]> : tensor<4xi64>}} : () -> ()

  // CHECK-LABEL: @all_reduce_sdy_mesh_dev
  func.func @all_reduce_sdy_mesh_dev(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK{LITERAL}: replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }) {
      replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @sdy_mesh_dev, axes = [#stablehlo.axis_ref<name = "x">]>
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  "sdy.mesh"() {sym_name = "sdy_mesh_max", stablehlo.mesh = {axes = [], device_ids = dense<0> : tensor<1xi64>}} : () -> ()

  // CHECK-LABEL: @all_reduce_sdy_mesh_max
  func.func @all_reduce_sdy_mesh_max(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: replica_groups = dense<0> : tensor<1x1xi64>
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }) {
      replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @sdy_mesh_max, axes = []>
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @all_reduce_subaxis
  func.func @all_reduce_subaxis(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK{LITERAL}: replica_groups = dense<[[0, 2], [1, 3], [4, 6], [5, 7]]> : tensor<4x2xi64>
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }) {
      replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = "x", size = 2>, #stablehlo.mesh_axis<name = "y", size = 4>]>, axes = [#stablehlo.axis_ref<name = "y", sub_axis_info = (1)2>]>
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @all_reduce_subaxis_order_1
  func.func @all_reduce_subaxis_order_1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK{LITERAL}: replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]> : tensor<3x10xi64>
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }) {
      replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = "foo", size = 6>, #stablehlo.mesh_axis<name = "bar", size = 5>]>, axes = [#stablehlo.axis_ref<name = "foo", sub_axis_info = (3)2>, #stablehlo.axis_ref<name = "bar">]>
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @all_reduce_subaxis_order_2
  func.func @all_reduce_subaxis_order_2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK{LITERAL}: replica_groups = dense<[[0, 5, 1, 6, 2, 7, 3, 8, 4, 9], [10, 15, 11, 16, 12, 17, 13, 18, 14, 19], [20, 25, 21, 26, 22, 27, 23, 28, 24, 29]]> : tensor<3x10xi64>
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }) {
      replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = "foo", size = 6>, #stablehlo.mesh_axis<name = "bar", size = 5>]>, axes = [#stablehlo.axis_ref<name = "bar">, #stablehlo.axis_ref<name = "foo", sub_axis_info = (3)2>]>
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @all_reduce_size_one_axis
  func.func @all_reduce_size_one_axis(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }) {
      replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = "i", size = 1>, #stablehlo.mesh_axis<name = "j", size = 2>]>, axes = [#stablehlo.axis_ref<name = "i">, #stablehlo.axis_ref<name = "j">]>
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
