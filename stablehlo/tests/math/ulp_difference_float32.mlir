// RUN: stablehlo-opt --chlo-legalize-to-stablehlo %s | stablehlo-translate --interpret
// This file is generated, see build_tools/math/README.md for more information.
module @ulp_difference_float32 {
  func.func public @main() {
    %0 = call @actual_0() : () -> tensor<12xf32>
    %1 = call @shifted_0() : () -> tensor<12xf32>
    check.expect_close %0, %1, max_ulp_difference = 0, min_ulp_difference = 0 : tensor<12xf32>, tensor<12xf32>
    %3 = call @actual_1() : () -> tensor<9xf32>
    %4 = call @shifted_1() : () -> tensor<9xf32>
    check.expect_close %3, %4, max_ulp_difference = 1, min_ulp_difference = 1 : tensor<9xf32>, tensor<9xf32>
    %6 = call @actual_5() : () -> tensor<9xf32>
    %7 = call @shifted_5() : () -> tensor<9xf32>
    check.expect_close %6, %7, max_ulp_difference = 5, min_ulp_difference = 5 : tensor<9xf32>, tensor<9xf32>
    %9 = call @actual_50() : () -> tensor<9xf32>
    %10 = call @shifted_50() : () -> tensor<9xf32>
    check.expect_close %9, %10, max_ulp_difference = 50, min_ulp_difference = 50 : tensor<9xf32>, tensor<9xf32>
    %12 = call @actual_nonfinite() : () -> tensor<5xf32>
    %13 = call @shifted_nonfinite() : () -> tensor<5xf32>
    check.expect_close %12, %13, max_ulp_difference = 18446744073709551615, min_ulp_difference = 18446744073709551615 : tensor<5xf32>, tensor<5xf32>
    func.return
  }
  func.func private @actual_0() -> tensor<12xf32> {
    // [-3.4028235e+38, -1000000000.0, -1.1754944e-38, -1e-45, 0.0, 1e-45, 1.1754944e-38, 1.2, 1000000000.0, inf, -inf, nan]
    %1 = stablehlo.constant dense<"0xFFFF7FFF286B6ECE00008080010000800000000001000000000080009A99993F286B6E4E0000807F000080FF0000C07F"> : tensor<12xf32>
    return %1 : tensor<12xf32>
  }
  func.func private @shifted_0() -> tensor<12xf32> {
    // [-3.4028235e+38, -1000000000.0, -1.1754944e-38, -1e-45, 0.0, 1e-45, 1.1754944e-38, 1.2, 1000000000.0, inf, -inf, nan]
    %1 = stablehlo.constant dense<"0xFFFF7FFF286B6ECE00008080010000800000000001000000000080009A99993F286B6E4E0000807F000080FF0000C07F"> : tensor<12xf32>
    return %1 : tensor<12xf32>
  }
  func.func private @actual_1() -> tensor<9xf32> {
    // [-3.4028235e+38, -1000000000.0, -1.1754944e-38, -1e-45, 0.0, 1e-45, 1.1754944e-38, 1.2, 1000000000.0]
    %1 = stablehlo.constant dense<"0xFFFF7FFF286B6ECE00008080010000800000000001000000000080009A99993F286B6E4E"> : tensor<9xf32>
    return %1 : tensor<9xf32>
  }
  func.func private @shifted_1() -> tensor<9xf32> {
    // [-3.4028233e+38, -999999940.0, -1.1754942e-38, -0.0, 1e-45, 3e-45, 1.1754945e-38, 1.2000002, 1000000060.0]
    %1 = stablehlo.constant dense<"0xFEFF7FFF276B6ECEFFFF7F80000000800100000002000000010080009B99993F296B6E4E"> : tensor<9xf32>
    return %1 : tensor<9xf32>
  }
  func.func private @actual_5() -> tensor<9xf32> {
    // [-3.4028235e+38, -1000000000.0, -1.1754944e-38, -1e-45, 0.0, 1e-45, 1.1754944e-38, 1.2, 1000000000.0]
    %1 = stablehlo.constant dense<"0xFFFF7FFF286B6ECE00008080010000800000000001000000000080009A99993F286B6E4E"> : tensor<9xf32>
    return %1 : tensor<9xf32>
  }
  func.func private @shifted_5() -> tensor<9xf32> {
    // [-3.4028225e+38, -999999700.0, -1.1754937e-38, 6e-45, 7e-45, 8e-45, 1.175495e-38, 1.2000006, 1000000300.0]
    %1 = stablehlo.constant dense<"0xFAFF7FFF236B6ECEFBFF7F80040000000500000006000000050080009F99993F2D6B6E4E"> : tensor<9xf32>
    return %1 : tensor<9xf32>
  }
  func.func private @actual_50() -> tensor<9xf32> {
    // [-3.4028235e+38, -1000000000.0, -1.1754944e-38, -1e-45, 0.0, 1e-45, 1.1754944e-38, 1.2, 1000000000.0]
    %1 = stablehlo.constant dense<"0xFFFF7FFF286B6ECE00008080010000800000000001000000000080009A99993F286B6E4E"> : tensor<9xf32>
    return %1 : tensor<9xf32>
  }
  func.func private @shifted_50() -> tensor<9xf32> {
    // [-3.4028133e+38, -999996800.0, -1.1754873e-38, 6.9e-44, 7e-44, 7.1e-44, 1.1755014e-38, 1.200006, 1000003200.0]
    %1 = stablehlo.constant dense<"0xCDFF7FFFF66A6ECECEFF7F8031000000320000003300000032008000CC99993F5A6B6E4E"> : tensor<9xf32>
    return %1 : tensor<9xf32>
  }
  func.func private @actual_nonfinite() -> tensor<5xf32> {
    // [inf, inf, inf, inf, inf]
    %1 = stablehlo.constant dense<"0x0000807F0000807F0000807F0000807F0000807F"> : tensor<5xf32>
    return %1 : tensor<5xf32>
  }
  func.func private @shifted_nonfinite() -> tensor<5xf32> {
    // [-inf, nan, 0.0, 1.2, 3.4028235e+38]
    %1 = stablehlo.constant dense<"0x000080FF0000C07F000000009A99993FFFFF7F7F"> : tensor<5xf32>
    return %1 : tensor<5xf32>
  }
}
