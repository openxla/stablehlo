// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3x4xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3x4xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%0, %1) {has_side_effect = true} : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> ()
    return %0 : tensor<3x4xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<3x4xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-2.31154037,2.16501498), (-0.62014389,-3.54022598), (6.43273115,-0.0746307299), (-1.64691806,-2.16097689)], [(-4.05629301,-1.24208844), (1.9414624,0.0947651639), (-0.0559204295,3.14833903), (4.0608716,0.622570335)], [(0.526463509,8.46901607), (-0.513401508,-1.28572524), (-1.92279816,-2.3590281), (3.56461072,-6.23003673)]]> : tensor<3x4xcomplex<f32>>
    return %cst : tensor<3x4xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<3x4xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-2.31154037,2.16501498), (-0.62014389,-3.54022598), (6.43273115,-0.0746307299), (-1.64691806,-2.16097689)], [(-4.05629301,-1.24208844), (1.9414624,0.0947651639), (-0.0559204295,3.14833903), (4.0608716,0.622570335)], [(0.526463509,8.46901607), (-0.513401508,-1.28572524), (-1.92279816,-2.3590281), (3.56461072,-6.23003673)]]> : tensor<3x4xcomplex<f32>>
    return %cst : tensor<3x4xcomplex<f32>>
  }
}
