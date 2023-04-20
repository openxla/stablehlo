// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<5x7xf32>
    %1 = call @expected() : () -> tensor<5x7xf32>
    %2 = stablehlo.reduce_precision %0, format = e5m10 : tensor<5x7xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<5x7xf32>, tensor<5x7xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<5x7xf32> {
    %0 = stablehlo.constant dense<[[-0.993861913, -0.716096282, -1.94070697, 0.474192441, 4.20493698, -3.38537836, -3.6731472], [3.16929746, -1.36569667, 4.73274279, -0.438586295, -0.517679751, 7.06460381, -2.35960102], [-0.523919463, 1.576170e+00, -1.32587886, -2.21371269, -0.403554231, 0.365458697, 0.996900379], [-2.89147782, -4.60086203, 1.01571786, 0.791271567, 5.432890e-01, -2.61252475, 2.32681966], [-3.13387656, -0.554039478, -3.36884379, -0.862945318, -0.893986999, 1.75586152, 6.04183912]]> : tensor<5x7xf32>
    return %0 : tensor<5x7xf32>
  }
  func.func private @expected() -> tensor<5x7xf32> {
    %0 = stablehlo.constant dense<[[-0.993652343, -0.716308593, -1.94042969, 0.474121094, 4.203125, -3.38476563, -3.67382813], [3.16992188, -1.36523438, 4.734375, -0.438476563, -0.517578125, 7.06640625, -2.359375], [-0.523925781, 1.57617188, -1.32617188, -2.21289063, -0.403564453, 0.365478516, 0.997070312], [-2.890625, -4.6015625, 1.015625, 0.791503906, 0.543457031, -2.61328125, 2.32617188], [-3.13476563, -0.554199219, -3.36914063, -0.862792968, -0.894042968, 1.75585938, 6.04296875]]> : tensor<5x7xf32>
    return %0 : tensor<5x7xf32>
  }
}
