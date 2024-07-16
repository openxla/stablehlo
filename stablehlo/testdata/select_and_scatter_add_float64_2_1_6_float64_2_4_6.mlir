// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x1x6xf64>, tensor<2x4x6xf64>)
    %1 = call @expected() : () -> tensor<2x4x6xf64>
    %cst = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %2 = stablehlo.pad %0#1, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xf64>, tensor<f64>) -> tensor<2x4x6xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %cst_0) <{window_dimensions = array<i64: 1, 3, 1>, window_strides = array<i64: 1, 2, 1>}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %5 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f64>
      stablehlo.return %5 : tensor<f64>
    }) : (tensor<2x4x6xf64>, tensor<2x1x6xf64>, tensor<f64>) -> tensor<2x4x6xf64>
    %4 = stablehlo.slice %3 [0:2, 0:4, 0:6] : (tensor<2x4x6xf64>) -> tensor<2x4x6xf64>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<2x4x6xf64>, tensor<2x4x6xf64>) -> ()
    return %4 : tensor<2x4x6xf64>
  }
  func.func private @inputs() -> (tensor<2x1x6xf64> {mhlo.layout_mode = "default"}, tensor<2x4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.65534158846581381, 2.0149102770534553, -3.9497793154591596, -0.84784950026977868, 0.20737941184148967, 1.8757323675827537]], [[5.7686491291416271, 4.0626762668865215, -1.0563197582057429, 0.4078860017863033, 1.1773181971925688, 7.7005518869319491]]]> : tensor<2x1x6xf64>
    %cst_0 = stablehlo.constant dense<[[[-0.91188340903326082, -2.4710970674799979, -0.92104401692248761, -2.3088683218007087, -3.0882282943931707, -0.19730095004274356], [5.0314787758858932, -1.0738800580670875, 0.38513681322200699, -2.1599522948671259, 2.0200116940160111, -2.2082048968279562], [-7.8978149272775831, -2.4943593800822326, -2.8310341747127472, 2.2036443618103254, -2.1779128165069794, 0.62039528728551752], [-1.2715074440429766, 1.0804422535410538, -1.9581270259674506, 1.6861035708929502, 2.686715461937089, 0.61568725222015164]], [[-3.3825742798177849, 3.3532386739088937, -3.3167885166654756, 5.005068848558377, -0.18242536091999154, 0.37390408014017096], [-1.404168952083892, 1.632259753969096, 1.4609210279428617, 1.8578002827028763, 0.14578291434060806, -0.22041549769900609], [-0.43491736596543723, 2.605963479901976, 1.5871054686508652, 0.6376353756926213, -2.0242079062777498, 1.999748887433044], [2.2884318173121345, -0.96852848645466061, -0.14375187992771277, -3.6057230073502269, -3.2691972473950717, -0.64468238524284938]]]> : tensor<2x4x6xf64>
    return %cst, %cst_0 : tensor<2x1x6xf64>, tensor<2x4x6xf64>
  }
  func.func private @expected() -> (tensor<2x4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.65534158846581381, 2.0149102770534553, -3.9497793154591596, 0.000000e+00, 0.20737941184148967, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, -0.84784950026977868, 0.000000e+00, 1.8757323675827537], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 4.0626762668865215, 0.000000e+00, 0.4078860017863033, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.1773181971925688, 0.000000e+00], [5.7686491291416271, 0.000000e+00, -1.0563197582057429, 0.000000e+00, 0.000000e+00, 7.7005518869319491], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf64>
    return %cst : tensor<2x4x6xf64>
  }
}
