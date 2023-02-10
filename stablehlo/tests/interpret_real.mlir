// RUN: stablehlo-interpreter --interpret -split-input-file %s

func.func @real_op_test_f64() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 0x0000000000000001, 0x8000000000000001]> : tensor<11xf64>
  %1 = stablehlo.real %0 : tensor<11xf64>
  check.almost_eq %1, dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 0x0000000000000001, 0x8000000000000001]> : tensor<11xf64>
  func.return
}

// -----

func.func @real_op_test_c128() {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f64>>
  %1 = stablehlo.real %0 : (tensor<2xcomplex<f64>>) -> tensor<2xf64>
  check.almost_eq %1, dense<[1.5, 3.5]> : tensor<2xf64>
  func.return
}
