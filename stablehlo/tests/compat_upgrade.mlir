// RUN: stablehlo-opt --versionedhlo-to-version='target=0.1.0' %s | FileCheck %s
// RUN: stablehlo-opt --versionedhlo-to-version='target=current' %s | FileCheck %s

// CHECK-LABEL: @test_upgrade
func.func @test_upgrade(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  // CHECK:      %0 = "versionedhlo.add"(%arg0, %arg0)
  // CHECK-NEXT: %1 = "versionedhlo.custom_call_v2"(%arg0)
  %0 = "versionedhlo.add"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %1 = "versionedhlo.custom_call"(%arg0) {backend_config = "", call_target_name = "foo"} : (tensor<2xi1>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}