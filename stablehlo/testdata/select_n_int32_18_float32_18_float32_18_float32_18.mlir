// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<18xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:4 = call @inputs() : () -> (tensor<18xi32>, tensor<18xf32>, tensor<18xf32>, tensor<18xf32>)
    %1 = call @expected() : () -> tensor<18xf32>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<18xi32>
    %3 = stablehlo.compare  LT, %0#0, %2,  SIGNED : (tensor<18xi32>, tensor<18xi32>) -> tensor<18xi1>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<18xi32>
    %5 = stablehlo.compare  LT, %0#0, %4,  SIGNED : (tensor<18xi32>, tensor<18xi32>) -> tensor<18xi1>
    %6 = stablehlo.select %5, %0#2, %0#3 : tensor<18xi1>, tensor<18xf32>
    %7 = stablehlo.select %3, %0#1, %6 : tensor<18xi1>, tensor<18xf32>
    stablehlo.custom_call @check.expect_close(%7, %1) {has_side_effect = true} : (tensor<18xf32>, tensor<18xf32>) -> ()
    return %7 : tensor<18xf32>
  }
  func.func private @inputs() -> (tensor<18xi32> {mhlo.layout_mode = "default"}, tensor<18xf32> {mhlo.layout_mode = "default"}, tensor<18xf32> {mhlo.layout_mode = "default"}, tensor<18xf32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 2, 0, 0, 1, 1, 2, 0, 1, 0, 0, 1, 2, 1, 2, 1, 1, 0]> : tensor<18xi32>
    %cst = stablehlo.constant dense<[2.56200647, 0.303029031, -2.33659768, -0.0842525735, -3.8680172, 2.72818685, -1.30297685, -4.97471142, -2.56493282, -5.70271301, -1.74544048, 3.00412154, -0.551256299, 3.3540206, 1.73112082, 2.13606191, -2.09324622, -0.971786737]> : tensor<18xf32>
    %cst_0 = stablehlo.constant dense<[-6.02375221, 2.72696185, 3.9226594, 1.65064454, -2.43719792, 0.226952314, 0.855229318, -1.46683788, 0.0280455742, -1.52327204, -0.0184807125, 0.266047239, -0.881479442, -1.93082952, 4.86475706, -1.67108071, -2.33280182, 5.68302345]> : tensor<18xf32>
    %cst_1 = stablehlo.constant dense<[-0.763047457, 3.95658016, 6.675350e-01, 0.90997231, -5.72888422, -1.85249972, -0.0301251821, -0.243915722, 0.0369620621, -0.119632974, -1.40885472, -3.06102085, -1.34765267, -2.44184375, 0.426090688, 0.480808973, 0.913791298, -0.0524138957]> : tensor<18xf32>
    return %c, %cst, %cst_0, %cst_1 : tensor<18xi32>, tensor<18xf32>, tensor<18xf32>, tensor<18xf32>
  }
  func.func private @expected() -> (tensor<18xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[2.56200647, 3.95658016, -2.33659768, -0.0842525735, -2.43719792, 0.226952314, -0.0301251821, -4.97471142, 0.0280455742, -5.70271301, -1.74544048, 0.266047239, -1.34765267, -1.93082952, 0.426090688, -1.67108071, -2.33280182, -0.971786737]> : tensor<18xf32>
    return %cst : tensor<18xf32>
  }
}
