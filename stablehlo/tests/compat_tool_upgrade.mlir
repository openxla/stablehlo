// RUN: stablehlo-translate --compat --use-test-converter %s --target=38 | FileCheck %s
// RUN: stablehlo-translate --compat --use-test-converter %s --target=40 | stablehlo-translate --compat --use-test-converter --target=38 | FileCheck %s

func.func private @test_upgrade(%arg0: tensor<2xi1>) -> tensor<2xi1> 
  attributes {compat_version = 37 : i32} {
  // CHECK:      %0 = "stablehlo.add"(%arg0, %arg0) {version_38_attr = 1 : i64} : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  // CHECK-NEXT: %1 = "stablehlo.sub"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %1 = "stablehlo.sub"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
  // CHECK: compat_version = 38
}
