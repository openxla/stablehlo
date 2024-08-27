// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x1x2xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<7x5x3xf32>, tensor<3xi64>)
    %1 = call @expected() : () -> tensor<3x1x2xf32>
    %2 = stablehlo.slice %0#1 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %4 = stablehlo.slice %0#1 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %6 = stablehlo.slice %0#1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %8 = stablehlo.compare  LT, %3, %c,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<7> : tensor<i64>
    %9 = stablehlo.add %3, %c_0 : tensor<i64>
    %10 = stablehlo.select %8, %9, %3 : tensor<i1>, tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %11 = stablehlo.compare  LT, %5, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_2 = stablehlo.constant dense<5> : tensor<i64>
    %12 = stablehlo.add %5, %c_2 : tensor<i64>
    %13 = stablehlo.select %11, %12, %5 : tensor<i1>, tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %14 = stablehlo.compare  LT, %7, %c_3,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_4 = stablehlo.constant dense<3> : tensor<i64>
    %15 = stablehlo.add %7, %c_4 : tensor<i64>
    %16 = stablehlo.select %14, %15, %7 : tensor<i1>, tensor<i64>
    %17 = stablehlo.dynamic_slice %0#0, %10, %13, %16, sizes = [3, 1, 2] : (tensor<7x5x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<3x1x2xf32>
    stablehlo.custom_call @check.expect_close(%17, %1) {has_side_effect = true} : (tensor<3x1x2xf32>, tensor<3x1x2xf32>) -> ()
    return %17 : tensor<3x1x2xf32>
  }
  func.func private @inputs() -> (tensor<7x5x3xf32> {mhlo.layout_mode = "default"}, tensor<3xi64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xC4F2F1BFBAADEFC068289D3FBCEB153FBFDFE2BFE6D3C23FD9961C4004A2F03F1FC4FC3FDD00A7BFEC78E63F30C60D40CEC096C04786E63F7FCE0440D94D713ED0D601C1A9B4713FBA6A2840B56BDD3F49322BC054861EC031FD62C01C78AEC0BD6BDC40EC2EBFC016D10240AE7EC03F2BC91FC074B68A3EC54564400D008540ABDE43C074493FC088EB74C0B6E872BF0DCDC6C0EF1E4E4003B5563F3E3A773FC5CE004166FF943FFF57DB3F7650633F93283ABF1159EFBFD6DE533DCC7527C0B4FF80BF9FDAA9BF8972CCBEA8962A40AA441EC03848AA3FB0B50040732D8ABF55164E3FFEA6E0BF5DA0034094E036BF9499074051E793BDAA3034C028AB444083B89B40C9C011BFDD3FB7BEE8AE8740421A46C030D7BD3FC1F331400435C9405AB555C02F652840056B59BFC169B53FDEEFD73E9BECE23DDBDE144072B20240029BBEC08CBF1B409E4850BF479D07BFADDB02C08570E63F4795824033F99540C2355BBFB63978400019FEC0ADF9C3BF618BCD404EF9274038F0EB3F0DF6E23F2AE915C0EA0C21C02D931D40ADB90BC01C0BF0BF6F55EDBF5579B5BDD20EC33E920D01C0"> : tensor<7x5x3xf32>
    %c = stablehlo.constant dense<[4, 0, 1]> : tensor<3xi64>
    return %cst, %c : tensor<7x5x3xf32>, tensor<3xi64>
  }
  func.func private @expected() -> (tensor<3x1x2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-0.0722185448, -2.81547022]], [[0.421751916, 0.110802852]], [[-1.531057, 6.42326403]]]> : tensor<3x1x2xf32>
    return %cst : tensor<3x1x2xf32>
  }
}
