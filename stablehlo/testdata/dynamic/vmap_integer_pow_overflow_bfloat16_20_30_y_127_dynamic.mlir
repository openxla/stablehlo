// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x30xbf16> {mhlo.sharding = ""}) -> tensor<?x20x30xbf16> {
    %0 = call @integer_pow(%arg0, %arg1) : (tensor<i64>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    return %0 : tensor<?x20x30xbf16>
  }
  func.func private @integer_pow(%arg0: tensor<i64>, %arg1: tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16> {
    %0 = stablehlo.multiply %arg1, %arg1 : tensor<?x20x30xbf16>
    %1 = stablehlo.multiply %arg1, %0 : tensor<?x20x30xbf16>
    %2 = stablehlo.multiply %0, %0 : tensor<?x20x30xbf16>
    %3 = stablehlo.multiply %1, %2 : tensor<?x20x30xbf16>
    %4 = stablehlo.multiply %2, %2 : tensor<?x20x30xbf16>
    %5 = stablehlo.multiply %3, %4 : tensor<?x20x30xbf16>
    %6 = stablehlo.multiply %4, %4 : tensor<?x20x30xbf16>
    %7 = stablehlo.multiply %5, %6 : tensor<?x20x30xbf16>
    %8 = stablehlo.multiply %6, %6 : tensor<?x20x30xbf16>
    %9 = stablehlo.multiply %7, %8 : tensor<?x20x30xbf16>
    %10 = stablehlo.multiply %8, %8 : tensor<?x20x30xbf16>
    %11 = stablehlo.multiply %9, %10 : tensor<?x20x30xbf16>
    return %11 : tensor<?x20x30xbf16>
  }
}

