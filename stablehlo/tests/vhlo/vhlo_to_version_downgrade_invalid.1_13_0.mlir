// RUN: not stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.13.0' %s 2>&1 | FileCheck %s

// CompositeOp with regions cannot be downgraded to v1.13.0 (v1).

module {
  func.func @composite_with_regions(%arg0: tensor<f32>) -> tensor<f32> {
    // CHECK: failed to legalize operation 'vhlo.composite_v2'
    %0 = "stablehlo.composite"(%arg0) ({
      ^bb0(%arg1: tensor<f32>):
        "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
    }) {
      name = "test.composite",
      composite_attributes = {},
      decomposition = @decomposition,
      version = 1 : i32
    } : (tensor<f32>) -> tensor<f32>
    func.return %0 : tensor<f32>
  }

  func.func @decomposition(%arg0: tensor<f32>) -> tensor<f32> {
    func.return %arg0 : tensor<f32>
  }
}
