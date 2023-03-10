// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<2xcomplex<f32>>
    %2 = stablehlo.add %0#0, %0#1 : tensor<2xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[(-1.60350597,-2.17498136), (1.02513945,-1.20277262)]> : tensor<2xcomplex<f32>>
    %1 = stablehlo.constant dense<[(-1.37768435,-1.39068925), (-1.56822729,-2.69757295)]> : tensor<2xcomplex<f32>>
    return %0, %1 : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
  }
  func.func private @expected() -> tensor<2xcomplex<f32>> {
    %0 = stablehlo.constant dense<[(-2.9811902,-3.56567049), (-0.54308784,-3.90034556)]> : tensor<2xcomplex<f32>>
    return %0 : tensor<2xcomplex<f32>>
  }
}
