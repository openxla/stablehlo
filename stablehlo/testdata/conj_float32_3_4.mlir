// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3x4xf32>
    %1 = call @expected() : () -> tensor<3x4xcomplex<f32>>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x4xf32>
    %3 = stablehlo.complex %0, %2 : tensor<3x4xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%3, %1) {has_side_effect = true} : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> ()
    return %3 : tensor<3x4xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<3x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-0.320381522, -0.114326343, 2.78269958, -1.2944634], [-0.201011941, -0.804424703, 0.361741751, -3.98341298], [-1.02460182, -0.470660597, 0.961570084, -2.37234592]]> : tensor<3x4xf32>
    return %cst : tensor<3x4xf32>
  }
  func.func private @expected() -> (tensor<3x4xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-0.320381522,0.000000e+00), (-0.114326343,0.000000e+00), (2.78269958,0.000000e+00), (-1.2944634,0.000000e+00)], [(-0.201011941,0.000000e+00), (-0.804424703,0.000000e+00), (0.361741751,0.000000e+00), (-3.98341298,0.000000e+00)], [(-1.02460182,0.000000e+00), (-0.470660597,0.000000e+00), (0.961570084,0.000000e+00), (-2.37234592,0.000000e+00)]]> : tensor<3x4xcomplex<f32>>
    return %cst : tensor<3x4xcomplex<f32>>
  }
}
