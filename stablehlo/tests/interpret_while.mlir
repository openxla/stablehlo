// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: region_test
func.func @region_test() -> tensor<i64> {
  %iter = stablehlo.constant dense<2> : tensor<i64>
  %step = stablehlo.constant dense<1> : tensor<i64>
  %v0 = stablehlo.constant dense<10> : tensor<i64>

  %results0, %results1 = "stablehlo.while"(%v0, %iter) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %cond = stablehlo.convert %arg1 :  (tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%cond) : (tensor<i1>) -> ()

  }, {
    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
      %v1 = stablehlo.constant dense<20> : tensor<i64>
      %v2 = stablehlo.add %v0, %v1 :  tensor<i64>  // I1: 10 + 20, I2: 10 + 20
      %v3 = stablehlo.add %v2, %arg2 :  tensor<i64>  // I1: 30 + 10, I2: 30 + 40

      %iter1 = stablehlo.subtract %arg3, %step :  tensor<i64>  // I1: 1, I2: 0
      stablehlo.return %v3, %iter1 : tensor<i64>, tensor<i64>
  }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)

  func.return %results0 : tensor<i64>
}
// CHECK-NEXT: tensor<i64>
// CHECK-NEXT: 70 : i64
