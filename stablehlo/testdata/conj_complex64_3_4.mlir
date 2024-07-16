// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3x4xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3x4xcomplex<f32>>
    %2 = stablehlo.real %0 : (tensor<3x4xcomplex<f32>>) -> tensor<3x4xf32>
    %3 = stablehlo.imag %0 : (tensor<3x4xcomplex<f32>>) -> tensor<3x4xf32>
    %4 = stablehlo.negate %3 : tensor<3x4xf32>
    %5 = stablehlo.complex %2, %4 : tensor<3x4xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%5, %1) {has_side_effect = true} : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> ()
    return %5 : tensor<3x4xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<3x4xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-3.92063093,0.947482943), (2.78115916,-2.56036592), (2.53200412,2.17843843), (0.609237313,-2.37602949)], [(-7.69268703,1.68608713), (5.67410326,-2.21368337), (1.93652391,-0.854346096), (-1.19848359,0.0837314874)], [(-0.950235664,-2.71151304), (-0.416245133,-1.51285613), (-1.48842835,0.668041825), (-5.31184292,-1.85015249)]]> : tensor<3x4xcomplex<f32>>
    return %cst : tensor<3x4xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<3x4xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-3.92063093,-0.947482943), (2.78115916,2.56036592), (2.53200412,-2.17843843), (0.609237313,2.37602949)], [(-7.69268703,-1.68608713), (5.67410326,2.21368337), (1.93652391,0.854346096), (-1.19848359,-0.0837314874)], [(-0.950235664,2.71151304), (-0.416245133,1.51285613), (-1.48842835,-0.668041825), (-5.31184292,1.85015249)]]> : tensor<3x4xcomplex<f32>>
    return %cst : tensor<3x4xcomplex<f32>>
  }
}
