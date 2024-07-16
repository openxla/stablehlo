// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi16>, tensor<3x6xf64>)
    %1 = call @expected() : () -> tensor<4x6xf64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi16>) -> tensor<4x3xf64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    return %4 : tensor<4x6xf64>
  }
  func.func private @inputs() -> (tensor<4x3xi16> {mhlo.layout_mode = "default"}, tensor<3x6xf64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-2, -2, 0], [-1, -1, -2], [0, 0, 0], [-1, -1, 2]]> : tensor<4x3xi16>
    %cst = stablehlo.constant dense<[[1.7733081084817144, 2.6346815064499118, -0.74179466306578923, 2.6831100031818709, -2.1638504490293937, -1.1534140311890431], [-3.5500692259393842, 0.35657319387758213, -0.57431215270536828, -2.753694124019944, -1.3239960773300565, 1.8449571615017859], [1.2318585386686605, 0.39017048941749832, 2.9626323848973164, 0.50274335043940821, 2.2781374164626977, 1.4792723564806507]]> : tensor<3x6xf64>
    return %c, %cst : tensor<4x3xi16>, tensor<3x6xf64>
  }
  func.func private @expected() -> (tensor<4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[3.5535222349153397, -5.9825094006549877, 2.632213631542315, 0.14116824167614617, 6.9756930527189009, -1.3830862606254857], [-0.68695595987965108, -3.7715956791624903, -4.6091579540234751, -0.93490258004074334, -1.068428306565945, -3.6500878432740445], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [4.2404781947949903, -2.2109137214924974, 7.2413715855657905, 1.0760708217168895, 8.044121359284846, 2.2670015826485583]]> : tensor<4x6xf64>
    return %cst : tensor<4x6xf64>
  }
}
