// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<18xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<i1>, tensor<18xf32>, tensor<18xf32>)
    %1 = call @expected() : () -> tensor<18xf32>
    %2 = stablehlo.select %0#0, %0#2, %0#1 : tensor<i1>, tensor<18xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<18xf32>, tensor<18xf32>) -> ()
    return %2 : tensor<18xf32>
  }
  func.func private @inputs() -> (tensor<i1> {mhlo.layout_mode = "default"}, tensor<18xf32> {mhlo.layout_mode = "default"}, tensor<18xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[-1.83992159, -5.53955698, -0.79379636, 1.26131332, 1.52134728, -2.55310225, -5.83623314, -2.70676374, -1.89610958, 5.02900076, 1.20761502, -1.07989526, -0.395469189, -2.707490e+00, 1.10248196, -0.0470473804, -1.68658864, -1.51551247]> : tensor<18xf32>
    %cst_0 = stablehlo.constant dense<[4.03491116, 4.54813766, 0.525275111, 6.68681145, -4.24286127, 7.29083967, -1.08588982, 1.0339278, 0.348950595, -3.03581524, -2.31413841, 1.69352174, -1.42940807, 5.65003777, 6.60452461, 2.04546762, 3.83973241, 0.686347365]> : tensor<18xf32>
    %c = stablehlo.constant dense<true> : tensor<i1>
    return %c, %cst, %cst_0 : tensor<i1>, tensor<18xf32>, tensor<18xf32>
  }
  func.func private @expected() -> (tensor<18xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[4.03491116, 4.54813766, 0.525275111, 6.68681145, -4.24286127, 7.29083967, -1.08588982, 1.0339278, 0.348950595, -3.03581524, -2.31413841, 1.69352174, -1.42940807, 5.65003777, 6.60452461, 2.04546762, 3.83973241, 0.686347365]> : tensor<18xf32>
    return %cst : tensor<18xf32>
  }
}
