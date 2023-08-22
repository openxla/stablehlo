// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @tuple() {
  %val0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
  %val00 = stablehlo.constant dense<3> : tensor<i64>
  %val1 = stablehlo.tuple %val00 : tuple<tensor<i64>>
  %operand = stablehlo.tuple %val0, %val1 : tuple<tensor<2xf64>, tuple<tensor<i64>>>
  func.return
}

// -----

func.func @get_tuple_element() {
  %val0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
  %val00 = stablehlo.constant dense<3> : tensor<i64>
  %val1 = stablehlo.tuple %val00 : tuple<tensor<i64>>
  %operand = stablehlo.tuple %val0, %val1 : tuple<tensor<2xf64>, tuple<tensor<i64>>>
  %result0 = stablehlo.get_tuple_element %operand[0] : (tuple<tensor<2xf64>, tuple<tensor<i64>>>) -> tensor<2xf64>
  %result00 = stablehlo.get_tuple_element %operand[1] : (tuple<tensor<2xf64>, tuple<tensor<i64>>>) -> tuple<tensor<i64>>
  %result1 = stablehlo.get_tuple_element %result00[0] : (tuple<tensor<i64>>) -> tensor<i64>
  check.expect_almost_eq_const %result0, dense<[1.0, 2.0]> : tensor<2xf64>
  check.expect_eq_const %result1, dense<3> : tensor<i64>
  func.return
}

