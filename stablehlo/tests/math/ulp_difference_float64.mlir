// RUN: stablehlo-opt --chlo-legalize-to-stablehlo %s | stablehlo-translate --interpret
// This file is generated, see build_tools/math/README.md for more information.
module @ulp_difference_float64 {
  func.func public @main() {
    %0 = call @actual_0() : () -> tensor<12xf64>
    %1 = call @shifted_0() : () -> tensor<12xf64>
    check.expect_close %0, %1, max_ulp_difference = 0, min_ulp_difference = 0 : tensor<12xf64>, tensor<12xf64>
    %3 = call @actual_1() : () -> tensor<9xf64>
    %4 = call @shifted_1() : () -> tensor<9xf64>
    check.expect_close %3, %4, max_ulp_difference = 1, min_ulp_difference = 1 : tensor<9xf64>, tensor<9xf64>
    %6 = call @actual_5() : () -> tensor<9xf64>
    %7 = call @shifted_5() : () -> tensor<9xf64>
    check.expect_close %6, %7, max_ulp_difference = 5, min_ulp_difference = 5 : tensor<9xf64>, tensor<9xf64>
    %9 = call @actual_50() : () -> tensor<9xf64>
    %10 = call @shifted_50() : () -> tensor<9xf64>
    check.expect_close %9, %10, max_ulp_difference = 50, min_ulp_difference = 50 : tensor<9xf64>, tensor<9xf64>
    %12 = call @actual_nonfinite() : () -> tensor<5xf64>
    %13 = call @shifted_nonfinite() : () -> tensor<5xf64>
    check.expect_close %12, %13, max_ulp_difference = 18446744073709551615, min_ulp_difference = 18446744073709551615 : tensor<5xf64>, tensor<5xf64>
    func.return
  }
  func.func private @actual_0() -> tensor<12xf64> {
    // [-1.7976931348623157e+308, -1000000001.2, -2.2250738585072014e-308, -5e-324, 0.0, 5e-324, 2.2250738585072014e-308, 1.2, 1000000000.0, inf, -inf, nan]
    %1 = stablehlo.constant dense<"0xFFFFFFFFFFFFEFFF9A99990065CDCDC100000000000010800100000000000080000000000000000001000000000000000000000000001000333333333333F33F0000000065CDCD41000000000000F07F000000000000F0FF000000000000F87F"> : tensor<12xf64>
    return %1 : tensor<12xf64>
  }
  func.func private @shifted_0() -> tensor<12xf64> {
    // [-1.7976931348623157e+308, -1000000001.2, -2.2250738585072014e-308, -5e-324, 0.0, 5e-324, 2.2250738585072014e-308, 1.2, 1000000000.0, inf, -inf, nan]
    %1 = stablehlo.constant dense<"0xFFFFFFFFFFFFEFFF9A99990065CDCDC100000000000010800100000000000080000000000000000001000000000000000000000000001000333333333333F33F0000000065CDCD41000000000000F07F000000000000F0FF000000000000F87F"> : tensor<12xf64>
    return %1 : tensor<12xf64>
  }
  func.func private @actual_1() -> tensor<9xf64> {
    // [-1.7976931348623157e+308, -1000000001.2, -2.2250738585072014e-308, -5e-324, 0.0, 5e-324, 2.2250738585072014e-308, 1.2, 1000000000.0]
    %1 = stablehlo.constant dense<"0xFFFFFFFFFFFFEFFF9A99990065CDCDC100000000000010800100000000000080000000000000000001000000000000000000000000001000333333333333F33F0000000065CDCD41"> : tensor<9xf64>
    return %1 : tensor<9xf64>
  }
  func.func private @shifted_1() -> tensor<9xf64> {
    // [-1.7976931348623155e+308, -1000000001.1999999, -2.225073858507201e-308, -0.0, 5e-324, 1e-323, 2.225073858507202e-308, 1.2000000000000002, 1000000000.0000001]
    %1 = stablehlo.constant dense<"0xFEFFFFFFFFFFEFFF9999990065CDCDC1FFFFFFFFFFFF0F800000000000000080010000000000000002000000000000000100000000001000343333333333F33F0100000065CDCD41"> : tensor<9xf64>
    return %1 : tensor<9xf64>
  }
  func.func private @actual_5() -> tensor<9xf64> {
    // [-1.7976931348623157e+308, -1000000001.2, -2.2250738585072014e-308, -5e-324, 0.0, 5e-324, 2.2250738585072014e-308, 1.2, 1000000000.0]
    %1 = stablehlo.constant dense<"0xFFFFFFFFFFFFEFFF9A99990065CDCDC100000000000010800100000000000080000000000000000001000000000000000000000000001000333333333333F33F0000000065CDCD41"> : tensor<9xf64>
    return %1 : tensor<9xf64>
  }
  func.func private @shifted_5() -> tensor<9xf64> {
    // [-1.7976931348623147e+308, -1000000001.1999995, -2.225073858507199e-308, 2e-323, 2.5e-323, 3e-323, 2.225073858507204e-308, 1.200000000000001, 1000000000.0000006]
    %1 = stablehlo.constant dense<"0xFAFFFFFFFFFFEFFF9599990065CDCDC1FBFFFFFFFFFF0F800400000000000000050000000000000006000000000000000500000000001000383333333333F33F0500000065CDCD41"> : tensor<9xf64>
    return %1 : tensor<9xf64>
  }
  func.func private @actual_50() -> tensor<9xf64> {
    // [-1.7976931348623157e+308, -1000000001.2, -2.2250738585072014e-308, -5e-324, 0.0, 5e-324, 2.2250738585072014e-308, 1.2, 1000000000.0]
    %1 = stablehlo.constant dense<"0xFFFFFFFFFFFFEFFF9A99990065CDCDC100000000000010800100000000000080000000000000000001000000000000000000000000001000333333333333F33F0000000065CDCD41"> : tensor<9xf64>
    return %1 : tensor<9xf64>
  }
  func.func private @shifted_50() -> tensor<9xf64> {
    // [-1.7976931348623057e+308, -1000000001.1999941, -2.2250738585071767e-308, 2.4e-322, 2.47e-322, 2.5e-322, 2.225073858507226e-308, 1.200000000000011, 1000000000.000006]
    %1 = stablehlo.constant dense<"0xCDFFFFFFFFFFEFFF6899990065CDCDC1CEFFFFFFFFFF0F803100000000000000320000000000000033000000000000003200000000001000653333333333F33F3200000065CDCD41"> : tensor<9xf64>
    return %1 : tensor<9xf64>
  }
  func.func private @actual_nonfinite() -> tensor<5xf64> {
    // [inf, inf, inf, inf, inf]
    %1 = stablehlo.constant dense<"0x000000000000F07F000000000000F07F000000000000F07F000000000000F07F000000000000F07F"> : tensor<5xf64>
    return %1 : tensor<5xf64>
  }
  func.func private @shifted_nonfinite() -> tensor<5xf64> {
    // [-inf, nan, 0.0, 1.2, 1.7976931348623157e+308]
    %1 = stablehlo.constant dense<"0x000000000000F0FF000000000000F87F0000000000000000333333333333F33FFFFFFFFFFFFFEF7F"> : tensor<5xf64>
    return %1 : tensor<5xf64>
  }
}
