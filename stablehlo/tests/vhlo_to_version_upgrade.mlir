// RUN: stablehlo-opt --vhlo-to-version='target=0.4.0' %s | FileCheck %s
// RUN: stablehlo-opt --vhlo-to-version='target=current' %s | FileCheck %s

// CHECK-LABEL: @test_upgrade
func.func @test_upgrade(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  // CHECK:      %0 = "vhlo.add"(%arg0, %arg0)
  // CHECK-NEXT: %1 = "vhlo.custom_call_v2"(%arg0)
  %0 = "vhlo.add"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %1 = "vhlo.custom_call"(%arg0) {backend_config = "", call_target_name = "foo"} : (tensor<2xi1>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}