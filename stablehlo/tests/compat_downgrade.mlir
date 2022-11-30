// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=0.0.0' %s | FileCheck %s

// CHECK-LABEL: @test_downgrade
func.func @test_downgrade(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  // CHECK:      %0 = "vhlo.add"(%arg0, %arg0)
  // CHECK-NEXT: %1 = "vhlo.custom_call"(%arg0)
  %0 = stablehlo.add %arg0, %arg0 : tensor<2xi1>
  %1 = stablehlo.custom_call @foo(%arg0) : (tensor<2xi1>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}
