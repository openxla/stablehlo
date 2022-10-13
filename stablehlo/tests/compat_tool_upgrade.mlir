// RUN: stablehlo-translate --compat --emit-assembly --use-test-converter %s --target=40 | FileCheck %s
// RUN: stablehlo-translate --compat --use-test-converter %s --target=38 | stablehlo-translate --compat --emit-assembly --use-test-converter --target=40 | FileCheck %s
// RUN: stablehlo-translate --compat --use-test-converter %s | stablehlo-translate --compat --emit-assembly --use-test-converter --target=40 | FileCheck %s

func.func private @test_upgrade(%arg0: tensor<2xi1>) -> tensor<2xi1> 
  attributes {stablehlo.compat_version = 37 : i32} {
  // CHECK:      attributes {stablehlo.compat_version = 40 : i64}
  // CHECK-NEXT: %0 = stablehlo.add %arg0, %arg0 {version_39_attr = 1 : i64} : tensor<2xi1>
  // CHECK-NEXT: %1 = stablehlo.subtract %arg0, %arg0 : tensor<2xi1>
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %1 = "stablehlo.sub"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

