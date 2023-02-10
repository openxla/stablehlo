// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: region_test
func.func @region_test() -> tensor<i64> {
  %cond = stablehlo.constant dense<true> : tensor<i1>
  %limit = stablehlo.constant dense<2> : tensor<i64>
  %iter = stablehlo.constant dense<0> : tensor<i64>
  %v0 = stablehlo.constant dense<30> : tensor<i64>

  %results0, %results1 = "stablehlo.while"(%iter, %limit) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      "stablehlo.return"(%cond) : (tensor<i1>) -> ()
  }, {
    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
      %v1 = stablehlo.constant dense<40> : tensor<i64>
      %v2 = stablehlo.add %v0, %v1 :  tensor<i64>  // 30 + 40 = 70
      %v3 = stablehlo.add %arg2, %v2 : tensor<i64> // 0 + 70, 210 + 70

      %results:2 = "stablehlo.while"(%v3, %limit) ({
        ^bb0(%arg4: tensor<i64>, %arg5: tensor<i64>):
          stablehlo.return %cond : tensor<i1>
      }, {
        ^bb0(%arg6: tensor<i64>, %arg7: tensor<i64>):
          %v4 = stablehlo.add %v2, %arg6 :  tensor<i64> // 70 + 70, 70 + 140
          %v5 = stablehlo.add %v4, %iter :  tensor<i64> // 70 + 70 + 0, 70 + 140 + 0
          stablehlo.return %v4, %arg7 : tensor<i64>, tensor<i64>
      }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)

      stablehlo.return %results#0, %arg3 : tensor<i64>, tensor<i64>
  }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)

  func.return %results0 : tensor<i64>
}
// CHECK-NEXT: tensor<i64>
// CHECK-NEXT: 420 : i64
