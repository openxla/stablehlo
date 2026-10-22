// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @fma_op_test_bf16() {
  %a = stablehlo.constant dense<[1.5, 2.0, -3.0, 0.0, 0x7F80]> : tensor<5xbf16>
  %b = stablehlo.constant dense<[2.0, 3.0, 4.0, 5.0, 2.0]> : tensor<5xbf16>
  %c = stablehlo.constant dense<[0.5, -1.0, 2.0, -0.0, 1.0]> : tensor<5xbf16>
  %result = stablehlo.fma %a, %b, %c : tensor<5xbf16>
  check.expect_almost_eq_const %result, dense<[3.5, 5.0, -10.0, 0.0, 0x7F80]> : tensor<5xbf16>
  func.return
}

// -----

func.func @fma_op_test_f16() {
  %a = stablehlo.constant dense<[1.5, 2.0, -3.0, 0.0, 0x7C00]> : tensor<5xf16>
  %b = stablehlo.constant dense<[2.0, 3.0, 4.0, 5.0, 2.0]> : tensor<5xf16>
  %c = stablehlo.constant dense<[0.5, -1.0, 2.0, -0.0, 1.0]> : tensor<5xf16>
  %result = stablehlo.fma %a, %b, %c : tensor<5xf16>
  check.expect_almost_eq_const %result, dense<[3.5, 5.0, -10.0, 0.0, 0x7C00]> : tensor<5xf16>
  func.return
}

// -----

func.func @fma_op_test_f32() {
  // Note index 3 tests single-rounding cancellation:
  // (1 + 2^-12) * (1 + 2^-12) - 1 = 2^-11 + 2^-24 = 0.00048828131
  %a = stablehlo.constant dense<[1.5, 2.0, -3.0, 0x3F800800]> : tensor<4xf32>
  %b = stablehlo.constant dense<[2.0, 3.0, 4.0, 0x3F800800]> : tensor<4xf32>
  %c = stablehlo.constant dense<[0.5, -1.0, 2.0, -1.0]> : tensor<4xf32>
  %result = stablehlo.fma %a, %b, %c : tensor<4xf32>
  check.expect_eq_const %result, dense<[3.5, 5.0, -10.0, 0x3A000400]> : tensor<4xf32>
  func.return
}

// -----

func.func @fma_op_test_f64() {
  %a = stablehlo.constant dense<[1.5, 2.0, -3.0]> : tensor<3xf64>
  %b = stablehlo.constant dense<[2.0, 3.0, 4.0]> : tensor<3xf64>
  %c = stablehlo.constant dense<[0.5, -1.0, 2.0]> : tensor<3xf64>
  %result = stablehlo.fma %a, %b, %c : tensor<3xf64>
  check.expect_almost_eq_const %result, dense<[3.5, 5.0, -10.0]> : tensor<3xf64>
  func.return
}
