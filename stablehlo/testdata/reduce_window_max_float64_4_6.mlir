// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xf64>
    %1 = call @expected() : () -> tensor<3x5xf64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %2 = "stablehlo.reduce_window"(%0, %cst) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<f64>
      stablehlo.return %3 : tensor<f64>
    }) : (tensor<4x6xf64>, tensor<f64>) -> tensor<3x5xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5xf64>, tensor<3x5xf64>) -> ()
    return %2 : tensor<3x5xf64>
  }
  func.func private @inputs() -> (tensor<4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.476679218597688, 2.0431316424698656, 3.8494337987637257, -1.08363294655146, -1.4434161453902417, -0.53526242799487644], [1.4392491346452876, -0.26102123543740829, -2.1599746841105572, 0.62490016576673846, -6.2721891031333659, 3.0606977917334515], [1.1824749984152432, 0.91804707732091594, -2.943773753481238, -3.6168826507005365, -1.2016922257354874, -3.1268082252224421], [0.33629952166437987, -5.4224933195938965, -4.4068957909920128, -0.18843731564639468, -3.4955070170733098, -0.54614499375548842]]> : tensor<4x6xf64>
    return %cst : tensor<4x6xf64>
  }
  func.func private @expected() -> (tensor<3x5xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.0431316424698656, 3.8494337987637257, 3.8494337987637257, 1.000000e+00, 3.0606977917334515], [1.4392491346452876, 1.000000e+00, 1.000000e+00, 1.000000e+00, 3.0606977917334515], [1.1824749984152432, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]]> : tensor<3x5xf64>
    return %cst : tensor<3x5xf64>
  }
}
