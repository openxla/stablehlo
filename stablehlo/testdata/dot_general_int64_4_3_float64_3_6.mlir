// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi64>, tensor<3x6xf64>)
    %1 = call @expected() : () -> tensor<4x6xf64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi64>) -> tensor<4x3xf64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    return %4 : tensor<4x6xf64>
  }
  func.func private @inputs() -> (tensor<4x3xi64> {mhlo.layout_mode = "default"}, tensor<3x6xf64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 5, 0], [0, 1, 1], [0, -3, -1], [-6, -2, 0]]> : tensor<4x3xi64>
    %cst = stablehlo.constant dense<[[0.46689794530021078, 2.4098564186994844, 0.35770504223105187, -0.3034896108819462, 1.0758237968131992, -4.0302353806309839], [0.72157394783660789, 3.5398915865393654, 4.517311938507123, 2.6702401338490911, -1.3477981427031955, -1.1355583044367723], [2.9085630897169379, -0.75708011429181343, 2.7875068980377034, -0.8049581818122109, -0.94209570310918933, 1.652101436272235]]> : tensor<3x6xf64>
    return %c, %cst : tensor<4x3xi64>, tensor<3x6xf64>
  }
  func.func private @expected() -> (tensor<4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[3.6078697391830392, 17.699457932696827, 22.586559692535616, 13.351200669245456, -6.7389907135159781, -5.6777915221838615], [3.6301370375535456, 2.7828114722475519, 7.3048188365448263, 1.8652819520368802, -2.2898938458123848, 0.51654313183546274], [-5.0732849332267618, -9.8625946453262827, -16.339442713559073, -7.2057622197350621, 4.9854901312187758, 1.7545734770380819], [-4.2445355674744807, -21.538921685275639, -11.180854130400558, -3.5195426024065051, -3.759346495472804, 26.452528892659448]]> : tensor<4x6xf64>
    return %cst : tensor<4x6xf64>
  }
}
