// RUN: not stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.15.0' %s 2>&1 | FileCheck %s

// Future types cannot be downgraded to v1.15.0 (v1).

// CHECK: failed to legalize operation
func.func @attr_replica_group_mesh_axes(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    replica_groups = #stablehlo.replica_group_mesh_axes<
      mesh = @mesh,
      axes = [
        #stablehlo.axis_ref<name = "x", sub_axis_info = (1)2>,
        #stablehlo.axis_ref<name = "y">
      ]
    >
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
