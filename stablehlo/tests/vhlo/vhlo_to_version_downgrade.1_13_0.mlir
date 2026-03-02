// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.13.0' %s | FileCheck %s

// CompositeOp was changed in v1.14.0 to have regions.
// Ensure that serializing for 1.13.0 uses composite_v1 (no regions).

// CHECK-LABEL: vhlo.func_v1 @composite_op
// CHECK-NEXT: "vhlo.composite_v1"(%arg0) <{
// CHECK-NOT: regions
// CHECK-SAME: }> : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
func.func @composite_op(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.composite"(%arg0) {
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
