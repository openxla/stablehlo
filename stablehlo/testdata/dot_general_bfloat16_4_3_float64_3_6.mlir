// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xbf16>, tensor<3x6xf64>)
    %1 = call @expected() : () -> tensor<4x6xf64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xbf16>) -> tensor<4x3xf64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    stablehlo.custom_call @check.expect_almost_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    return %4 : tensor<4x6xf64>
  }
  func.func private @inputs() -> (tensor<4x3xbf16> {mhlo.layout_mode = "default"}, tensor<3x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.828130e+00, -3.421880e+00, 1.781250e+00], [-3.796880e+00, -3.808590e-02, 3.710940e-01], [-6.132810e-01, 1.000000e+00, 1.806640e-01], [-6.562500e+00, 1.062500e+00, 3.796880e+00]]> : tensor<4x3xbf16>
    %cst_0 = stablehlo.constant dense<[[-0.050378914882338442, -3.7698173417195675, -3.4866203881372089, -2.8050004607341901, 2.4681323215395947, -0.39923405596434125], [5.0350009816030941, 2.7373967085451238, 0.90214104352151581, -0.46061932072390066, 4.5521023701500845, -1.3741391936675766], [3.7719597840306021, -0.35389950656538899, -2.7256187914652052, 1.9568224855047538, -1.0825140636812025, 0.45062046791383625]]> : tensor<3x6xf64>
    return %cst, %cst_0 : tensor<4x3xbf16>, tensor<3x6xf64>
  }
  func.func private @expected() -> (tensor<4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-10.367862749966964, 0.6641018114282069, 1.9185759296029601, 12.994663718421322, -24.485140195643627, 6.6338840763269129], [1.3992704108512108, 14.077939004372627, 12.192442800413428, 11.393943837240791, -9.9462751980279833, 1.7353996249508321], [5.7473550036653691, 4.9854180775737289, 2.5479985897413444, 1.6131623679083755, 2.8428717063728501, -1.0478855084042311], [20.001959976859826, 26.304198118873394, 13.490637307047594, 25.348217869949838, -15.470680177358691, 2.8709001881045366]]> : tensor<4x6xf64>
    return %cst : tensor<4x6xf64>
  }
}
