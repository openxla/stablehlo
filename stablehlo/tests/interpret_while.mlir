// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @while() {
  // int i = 0;
  // int sum = 0;
  // while (i < 10) {
  //   sum += 1;
  //   i += 1;
  // }
  %init_i = stablehlo.constant dense<0> : tensor<i64>
  %init_sum = stablehlo.constant dense<0> : tensor<i64>
  %one = stablehlo.constant dense<1> : tensor<i64>
  %ten = stablehlo.constant dense<10> : tensor<i64>
  %results0, %results1 = stablehlo.while(%arg0 = %init_i, %arg1 = %init_sum) : tensor<i64>, tensor<i64>
  cond {
    %cond = stablehlo.compare LT, %arg0, %ten : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cond : tensor<i1>
  } do {
    %new_sum = stablehlo.add %arg1, %one : tensor<i64>
    %new_i = stablehlo.add %arg0, %one : tensor<i64>
    stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
  }
  check.expect_eq_const %results0, dense<10> : tensor<i64>
  check.expect_eq_const %results1, dense<10> : tensor<i64>
  func.return
}
