// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x2xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f64>>
    %1 = call @expected() : () -> tensor<3x2xcomplex<f64>>
    %2 = stablehlo.reshape %0 : (tensor<2x3xcomplex<f64>>) -> tensor<3x2xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x2xcomplex<f64>>, tensor<3x2xcomplex<f64>>) -> ()
    return %2 : tensor<3x2xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-1.722938445328158,-6.1887849433992859), (-0.073958819640521034,-2.545936194133934), (0.22891570948245527,0.72224451118857946)], [(0.43591571123138967,2.6209821608255051), (4.5898309260273464,-0.17641004400846497), (2.358777963541467,1.7903291239492709)]]> : tensor<2x3xcomplex<f64>>
    return %cst : tensor<2x3xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<3x2xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-1.722938445328158,-6.1887849433992859), (-0.073958819640521034,-2.545936194133934)], [(0.22891570948245527,0.72224451118857946), (0.43591571123138967,2.6209821608255051)], [(4.5898309260273464,-0.17641004400846497), (2.358777963541467,1.7903291239492709)]]> : tensor<3x2xcomplex<f64>>
    return %cst : tensor<3x2xcomplex<f64>>
  }
}
