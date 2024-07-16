// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi8>, tensor<3x6xf64>)
    %1 = call @expected() : () -> tensor<4x6xf64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi8>) -> tensor<4x3xf64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    return %4 : tensor<4x6xf64>
  }
  func.func private @inputs() -> (tensor<4x3xi8> {mhlo.layout_mode = "default"}, tensor<3x6xf64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[5, 3, -1], [0, 2, 1], [1, 2, 1], [-1, 4, 4]]> : tensor<4x3xi8>
    %cst = stablehlo.constant dense<[[0.1880662032510263, -2.023795304652626, 3.2473837819141886, 4.519621709716283, 4.1380737688287912, 5.7340414107154203], [1.0822822150164857, -1.3397161028972739, -4.1601253183831091, -4.1171799221794778, 1.6826484553523873, 1.481017467936709], [-1.6727293935913132, 0.37300986215698184, 2.1755047526098017, -1.7093730339521322, -0.91975375624215205, 2.2826186668177404]]> : tensor<3x6xf64>
    return %c, %cst : tensor<4x3xi8>, tensor<3x6xf64>
  }
  func.func private @expected() -> (tensor<4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[5.8599070548959009, -14.511134694111934, 1.5810382018118134, 11.955941815995114, 26.658067966443269, 30.83064079056949], [0.49183503644165816, -2.3064223436375659, -6.144745884156416, -9.9437328783110885, 2.4455431544626225, 5.2446536026911588], [0.67990123969268446, -4.3302176482901924, -2.8973621022422278, -5.4241111685948047, 6.5836169232914141, 10.978695013406579], [-2.549854917550336, -1.8430296583085424, -11.185866045007417, -27.825833534242722, -1.0864949723878503, 9.3205031283023771]]> : tensor<4x6xf64>
    return %cst : tensor<4x6xf64>
  }
}
