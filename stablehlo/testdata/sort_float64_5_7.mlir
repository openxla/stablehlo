// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xf64>
    %1 = call @expected() : () -> tensor<5x7xf64>
    %2 = "stablehlo.sort"(%0) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %3 = stablehlo.compare  EQ, %arg0, %cst,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %4 = stablehlo.select %3, %cst_0, %arg0 : tensor<i1>, tensor<f64>
      %5 = stablehlo.compare  NE, %arg0, %arg0,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_1 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
      %6 = stablehlo.select %5, %cst_1, %4 : tensor<i1>, tensor<f64>
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %7 = stablehlo.compare  EQ, %arg1, %cst_2,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %8 = stablehlo.select %7, %cst_3, %arg1 : tensor<i1>, tensor<f64>
      %9 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_4 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
      %10 = stablehlo.select %9, %cst_4, %8 : tensor<i1>, tensor<f64>
      %11 = stablehlo.compare  LT, %6, %10,  TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %11 : tensor<i1>
    }) : (tensor<5x7xf64>) -> tensor<5x7xf64>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x7xf64>, tensor<5x7xf64>) -> ()
    return %2 : tensor<5x7xf64>
  }
  func.func private @inputs() -> (tensor<5x7xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.0896635946524977, 2.7632953409939716, -0.50213168480988946, -2.0543716182315079, 4.7389017179183535, -5.0992561837764008, 3.2311120719177095], [-3.6800679015698829, -5.2035365268290956, -0.37137121268702389, 3.9131576334488489, 8.2306975220664498, 1.2187365390560596, 2.5896494222636242], [-0.92573122376613293, -0.71528440028190499, 2.6681087215860151, 0.97069439024483861, -2.1846940329152411, -2.2067052115096883, 0.92144406147571478], [-2.8321294558434205, 1.1302784794698757, 3.7176209758875078, -1.3679934880325813, -0.11983810579204865, -5.5580946032460581, -1.8742713896235315], [1.6630315746085409, 1.4959596210409398, -2.0998486044131641, 1.544143027486987, 0.033663053680914386, -1.625141303366769, -0.73063142936550929]]> : tensor<5x7xf64>
    return %cst : tensor<5x7xf64>
  }
  func.func private @expected() -> (tensor<5x7xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.0896635946524977, -5.2035365268290956, -2.0998486044131641, -2.0543716182315079, -2.1846940329152411, -5.5580946032460581, -1.8742713896235315], [-3.6800679015698829, -0.71528440028190499, -0.50213168480988946, -1.3679934880325813, -0.11983810579204865, -5.0992561837764008, -0.73063142936550929], [-2.8321294558434205, 1.1302784794698757, -0.37137121268702389, 0.97069439024483861, 0.033663053680914386, -2.2067052115096883, 0.92144406147571478], [-0.92573122376613293, 1.4959596210409398, 2.6681087215860151, 1.544143027486987, 4.7389017179183535, -1.625141303366769, 2.5896494222636242], [1.6630315746085409, 2.7632953409939716, 3.7176209758875078, 3.9131576334488489, 8.2306975220664498, 1.2187365390560596, 3.2311120719177095]]> : tensor<5x7xf64>
    return %cst : tensor<5x7xf64>
  }
}
