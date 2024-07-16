// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x5xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x5xf64>
    %1 = call @expected() : () -> tensor<4x5xf64>
    %2 = stablehlo.reverse %0, dims = [0] : tensor<4x5xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x5xf64>, tensor<4x5xf64>) -> ()
    return %2 : tensor<4x5xf64>
  }
  func.func private @inputs() -> (tensor<4x5xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-0.068293202679957851, -2.6867475929167197, -2.2749228292440544, 3.5248234665276428, -0.11431189492327372], [1.6962983012984216, -7.0719194750213958, -0.32469278565706106, -3.2470842711756642, -10.343215461700757], [0.8597751311741596, -1.4606133085628474, 0.95919580967951789, 1.4313565361388514, 2.9807530678976066], [2.9634058562791141, 3.6780890335969096, -4.0214880095131598, 1.8323414945632726, 4.4693645223023495]]> : tensor<4x5xf64>
    return %cst : tensor<4x5xf64>
  }
  func.func private @expected() -> (tensor<4x5xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.9634058562791141, 3.6780890335969096, -4.0214880095131598, 1.8323414945632726, 4.4693645223023495], [0.8597751311741596, -1.4606133085628474, 0.95919580967951789, 1.4313565361388514, 2.9807530678976066], [1.6962983012984216, -7.0719194750213958, -0.32469278565706106, -3.2470842711756642, -10.343215461700757], [-0.068293202679957851, -2.6867475929167197, -2.2749228292440544, 3.5248234665276428, -0.11431189492327372]]> : tensor<4x5xf64>
    return %cst : tensor<4x5xf64>
  }
}
