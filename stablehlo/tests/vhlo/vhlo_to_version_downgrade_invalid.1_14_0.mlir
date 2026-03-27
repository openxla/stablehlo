// RUN: not stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.14.0' --split-input-file %s 2>&1 | FileCheck %s

// Future types cannot be downgraded to v1.14.0 (v1).

// CHECK: failed to legalize operation 'vhlo.func_v1'
func.func private @type_future() -> !stablehlo.future<tensor<f32>>

// -----

// CHECK: failed to legalize operation 'vhlo.async_start_v1'
func.func @op_async_start(%arg0: tensor<4x4xf32>) {
  %0 = "stablehlo.async_start"(%arg0) ({
    ^bb0(%barg0: tensor<4x4xf32>):
    %1 = "stablehlo.all_reduce"(%barg0) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    "stablehlo.return"(%1) : (tensor<4x4xf32>) -> ()
  }) : (tensor<4x4xf32>) -> !stablehlo.future<tensor<4x4xf32>>
  return
}

// -----

// CHECK: failed to legalize operation 'vhlo.func_v1'
func.func @op_async_done(%arg0: !stablehlo.future<tensor<4x4xf32>>) -> tensor<4x4xf32> {
  %0 = "stablehlo.async_done"(%arg0) : (!stablehlo.future<tensor<4x4xf32>>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}
