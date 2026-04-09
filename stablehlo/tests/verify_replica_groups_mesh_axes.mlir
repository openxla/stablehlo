// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// CHECK-LABEL: func @all_gather_replica_group_mesh_axes

func.func @all_gather_replica_group_mesh_axes(%operand: tensor<16x8xf32>) -> tensor<16x16xf32> {
  %result = "stablehlo.all_gather"(%operand) {
    all_gather_dim = 1 : i64,
    replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh, axes = [#stablehlo.axis_ref<name = "foo">, #stablehlo.axis_ref<name = "bar">]>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
  } : (tensor<16x8xf32>) -> tensor<16x16xf32>
  func.return %result : tensor<16x16xf32>
}

