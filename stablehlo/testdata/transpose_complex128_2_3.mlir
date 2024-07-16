// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x2xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f64>>
    %1 = call @expected() : () -> tensor<3x2xcomplex<f64>>
    %2 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xcomplex<f64>>) -> tensor<3x2xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x2xcomplex<f64>>, tensor<3x2xcomplex<f64>>) -> ()
    return %2 : tensor<3x2xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-1.3119832852479361,-2.7942567775397702), (-5.2463848304345424,-0.96527108533955164), (0.81592895582045633,-3.094400866203606)], [(1.2780774567730127,-1.2231556133313144), (-2.2705888602008182,-0.24873578516395517), (-4.0748643883366746,-0.070090492007976304)]]> : tensor<2x3xcomplex<f64>>
    return %cst : tensor<2x3xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<3x2xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-1.3119832852479361,-2.7942567775397702), (1.2780774567730127,-1.2231556133313144)], [(-5.2463848304345424,-0.96527108533955164), (-2.2705888602008182,-0.24873578516395517)], [(0.81592895582045633,-3.094400866203606), (-4.0748643883366746,-0.070090492007976304)]]> : tensor<3x2xcomplex<f64>>
    return %cst : tensor<3x2xcomplex<f64>>
  }
}
