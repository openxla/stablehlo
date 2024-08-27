// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x6xbf16>, tensor<4x6xbf16>)
    %1 = call @expected() : () -> tensor<3x5xbf16>
    %cst = stablehlo.constant dense<0x7F80> : tensor<bf16>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2:2 = "stablehlo.reduce_window"(%0#1, %0#0, %cst, %cst_0) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<bf16>, %arg3: tensor<bf16>):
      %3 = stablehlo.compare  LE, %arg0, %arg2,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %4 = stablehlo.select %3, %arg0, %arg2 : tensor<i1>, tensor<bf16>
      %5 = stablehlo.select %3, %arg1, %arg3 : tensor<i1>, tensor<bf16>
      stablehlo.return %4, %5 : tensor<bf16>, tensor<bf16>
    }) : (tensor<4x6xbf16>, tensor<4x6xbf16>, tensor<bf16>, tensor<bf16>) -> (tensor<3x5xbf16>, tensor<3x5xbf16>)
    stablehlo.custom_call @check.expect_close(%2#1, %1) {has_side_effect = true} : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> ()
    return %2#1 : tensor<3x5xbf16>
  }
  func.func private @inputs() -> (tensor<4x6xbf16> {mhlo.layout_mode = "default"}, tensor<4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.296880e+00, 2.671880e+00, 2.046880e+00, -3.375000e+00, -1.515630e+00, 2.031250e+00], [-3.171880e+00, -3.656250e+00, 2.640630e+00, -3.156250e+00, -4.062500e+00, -1.166990e-01], [8.750000e+00, -2.953130e+00, 3.453130e+00, 1.492190e+00, -5.062500e+00, 1.617190e+00], [1.671880e+00, -8.164060e-01, 1.078130e+00, 1.632810e+00, 3.187500e+00, -1.156250e+00]]> : tensor<4x6xbf16>
    %cst_0 = stablehlo.constant dense<[[-3.265630e+00, -2.375000e+00, -1.804690e+00, -7.062500e+00, -2.640630e+00, -6.591800e-02], [4.062500e-01, 6.210940e-01, 4.062500e+00, 1.765630e+00, -3.015630e+00, -1.375000e+00], [3.484380e+00, -5.406250e+00, -2.250000e+00, 1.039060e+00, -9.687500e-01, -2.375000e+00], [-4.250000e+00, 8.554680e-01, -1.343750e+00, -6.031250e+00, -2.359380e+00, -1.187500e+00]]> : tensor<4x6xbf16>
    return %cst, %cst_0 : tensor<4x6xbf16>, tensor<4x6xbf16>
  }
  func.func private @expected() -> (tensor<3x5xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.296880e+00, 2.671880e+00, -3.375000e+00, -3.375000e+00, -4.062500e+00], [-2.953130e+00, -2.953130e+00, 3.453130e+00, -4.062500e+00, -4.062500e+00], [-2.953130e+00, -2.953130e+00, 1.632810e+00, 1.632810e+00, 1.617190e+00]]> : tensor<3x5xbf16>
    return %cst : tensor<3x5xbf16>
  }
}
