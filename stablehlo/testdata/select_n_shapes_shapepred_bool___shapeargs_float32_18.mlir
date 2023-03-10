// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<i1>, tensor<18xf32>, tensor<18xf32>)
    %1 = call @expected() : () -> tensor<18xf32>
    %2 = stablehlo.select %0#0, %0#2, %0#1 : tensor<i1>, tensor<18xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<18xf32>, tensor<18xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<i1>, tensor<18xf32>, tensor<18xf32>) {
    %0 = stablehlo.constant dense<[-0.59687674, 0.444817394, -1.57262433, -0.391970962, -5.24718809, 0.475874126, -2.51704502, 1.11449134, 0.963905632, 2.5423243, 2.98997808, 1.36518657, 1.72895372, -3.13502455, -4.244880e+00, -2.06469131, -0.208799317, -5.24309206]> : tensor<18xf32>
    %1 = stablehlo.constant dense<[5.11399555, -1.22680259, -1.07809293, -0.865489065, -0.159358069, 1.40876615, 0.146112531, -2.07897186, 4.61126804, 2.6683445, -4.90580893, -2.06841159, 1.85906672, -2.39865279, -1.17596078, 9.531920e+00, -2.70751882, 7.10285807]> : tensor<18xf32>
    %2 = stablehlo.constant dense<true> : tensor<i1>
    return %2, %0, %1 : tensor<i1>, tensor<18xf32>, tensor<18xf32>
  }
  func.func private @expected() -> tensor<18xf32> {
    %0 = stablehlo.constant dense<[5.11399555, -1.22680259, -1.07809293, -0.865489065, -0.159358069, 1.40876615, 0.146112531, -2.07897186, 4.61126804, 2.6683445, -4.90580893, -2.06841159, 1.85906672, -2.39865279, -1.17596078, 9.531920e+00, -2.70751882, 7.10285807]> : tensor<18xf32>
    return %0 : tensor<18xf32>
  }
}
