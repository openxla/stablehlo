// RUN: stablehlo-opt --stablehlo-legalize-to-versionedhlo --versionedhlo-to-version='target=minimum' %s | FileCheck %s

// CHECK-LABEL: @test_downgrade
func.func @test_downgrade(%arg0: tensor<2xi1>) -> tensor<2xi1> 
  attributes {stablehlo.compat_version = 40 : i32} {
  // CHECK:      %0 = "versionedhlo.add"(%arg0, %arg0)
  // CHECK-NEXT: %1 = "versionedhlo.custom_call"(%arg0)
  %0 = stablehlo.add %arg0, %arg0 {version_39_attr = 1 : i64} : tensor<2xi1>
  %1 = stablehlo.custom_call @foo(%arg0) : (tensor<2xi1>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}
