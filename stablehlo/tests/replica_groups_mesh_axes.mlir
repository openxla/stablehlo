// RUN: stablehlo-opt %s | stablehlo-opt | FileCheck %s
// RUN: stablehlo-opt -emit-bytecode %s | stablehlo-opt | FileCheck %s

// CHECK-LABEL: @test_replica_groups_mesh_axes
func.func @test_replica_groups_mesh_axes(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh, axes = [#stablehlo.axis_ref<name = "foo">, #stablehlo.axis_ref<name = "bar">]>
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "test",
    has_side_effect = false,
    replica_groups = #stablehlo.replica_group_mesh_axes<
      mesh = @mesh,
      axes = [#stablehlo.axis_ref<name = "foo">, #stablehlo.axis_ref<name = "bar">]
    >
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
