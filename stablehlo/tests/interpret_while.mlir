// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @while() {
  // int i = 10;
  // int state = 0;
  // while (i > 0) {
  //   i -= 1;
  //   state += 1;
  // }
  %ten = stablehlo.constant dense<10> : tensor<i64>
  %zero = stablehlo.constant dense<0> : tensor<i64>
  %one = stablehlo.constant dense<1> : tensor<i64>
  %result_i, %result_state = "stablehlo.while"(%ten, %zero) ({
    ^bb0(%i: tensor<i64>, %state: tensor<i64>):
      %cond = stablehlo.convert %i : (tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
  }, {
    ^bb0(%i: tensor<i64>, %state: tensor<i64>):
      %new_i = stablehlo.subtract %i, %one :  tensor<i64>
      %new_state = stablehlo.add %state, %one :  tensor<i64>
      stablehlo.return %new_i, %new_state : tensor<i64>, tensor<i64>
  }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)

  check.expect_eq_const %result_state, dense<10> : tensor<i64>
  func.return
}
