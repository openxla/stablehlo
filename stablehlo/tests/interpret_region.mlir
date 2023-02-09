// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: region_test
func.func @region_test() -> tensor<i64> {
  %input = stablehlo.constant dense<[10]> : tensor<1xi64>
  %init_values = stablehlo.constant dense<20> : tensor<i64>
  %v0 = stablehlo.constant dense<30> : tensor<i64>

  %result = "stablehlo.reduce"(%input, %init_values) ({

    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %v1 = stablehlo.constant dense<40> : tensor<i64>
      %v2 = stablehlo.add %v0, %v1 :  tensor<i64>  // 30 + 40 = 70
      %v3 = stablehlo.add %v2, %arg1 :  tensor<i64> // 70 + 20 = 90

      %v4 = "stablehlo.reduce"(%input, %v3) ({

        ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
          %v4 = stablehlo.add %v2, %arg3 :  tensor<i64> // 70 + 90 = 160
          "stablehlo.return"(%v4) : (tensor<i64>) -> ()
      }) {
        dimensions = dense<[0]> : tensor<1xi64>
      } : (tensor<1xi64>, tensor<i64>) -> tensor<i64>

      "stablehlo.return"(%v4) : (tensor<i64>) -> ()
  }) {
    dimensions = dense<[0]> : tensor<1xi64>
  } : (tensor<1xi64>, tensor<i64>) -> tensor<i64>

  func.return %result : tensor<i64>
  // CHECK-NEXT: tensor<i64>
  // CHECK-NEXT: 160 : i64
}
