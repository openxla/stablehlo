// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x2xf32>, tensor<3x1xf32>)
    %1 = call @expected() : () -> tensor<3x2xcomplex<f32>>
    %2 = stablehlo.broadcast_in_dim %0#1, dims = [0, 1] : (tensor<3x1xf32>) -> tensor<3x2xf32>
    %3 = stablehlo.complex %0#0, %2 : tensor<3x2xcomplex<f32>>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x2xcomplex<f32>>, tensor<3x2xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x2xf32>, tensor<3x1xf32>) {
    %0 = stablehlo.constant dense<[[0.38378045, -0.744436264], [0.604358494, -5.5137372], [5.48623514, 0.583725393]]> : tensor<3x2xf32>
    %1 = stablehlo.constant dense<[[-1.33247781], [-4.61012411], [2.31260014]]> : tensor<3x1xf32>
    return %0, %1 : tensor<3x2xf32>, tensor<3x1xf32>
  }
  func.func private @expected() -> tensor<3x2xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(0.38378045,-1.33247781), (-0.744436264,-1.33247781)], [(0.604358494,-4.61012411), (-5.5137372,-4.61012411)], [(5.48623514,2.31260014), (0.583725393,2.31260014)]]> : tensor<3x2xcomplex<f32>>
    return %0 : tensor<3x2xcomplex<f32>>
  }
}
