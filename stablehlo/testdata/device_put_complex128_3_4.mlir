// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3x4xcomplex<f64>>
    %1 = call @expected() : () -> tensor<3x4xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%0, %1) {has_side_effect = true} : (tensor<3x4xcomplex<f64>>, tensor<3x4xcomplex<f64>>) -> ()
    return %0 : tensor<3x4xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<3x4xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-0.45521692254150775,-0.23587719752911818), (-0.39447183871733649,-4.2560388759332382), (-0.95398833183548393,-1.8260966232429259), (-1.4448256570362188,3.8849298889500621)], [(0.82203851579483245,-1.6857797245504615), (1.7814049744572444,6.7891611846845787), (4.9752563116102104,-2.6079749029367827), (5.0486585068590601,-3.4452677134332927)], [(-0.0086539344647242993,-1.0856169945983891), (-0.92600838629227899,-6.2065985016319543), (3.4331953212126152,-1.1479858385098698), (2.1759253353081385,-3.1236979514579031)]]> : tensor<3x4xcomplex<f64>>
    return %cst : tensor<3x4xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<3x4xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-0.45521692254150775,-0.23587719752911818), (-0.39447183871733649,-4.2560388759332382), (-0.95398833183548393,-1.8260966232429259), (-1.4448256570362188,3.8849298889500621)], [(0.82203851579483245,-1.6857797245504615), (1.7814049744572444,6.7891611846845787), (4.9752563116102104,-2.6079749029367827), (5.0486585068590601,-3.4452677134332927)], [(-0.0086539344647242993,-1.0856169945983891), (-0.92600838629227899,-6.2065985016319543), (3.4331953212126152,-1.1479858385098698), (2.1759253353081385,-3.1236979514579031)]]> : tensor<3x4xcomplex<f64>>
    return %cst : tensor<3x4xcomplex<f64>>
  }
}
