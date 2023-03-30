// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<3xcomplex<f32>>, tensor<1xcomplex<f32>>, tensor<1xi32>)
    %1 = call @expected() : () -> tensor<3xcomplex<f32>>
    %2 = "stablehlo.slice"(%0#2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
    %3 = stablehlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.compare  LT, %3, %4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %6 = stablehlo.constant dense<3> : tensor<i32>
    %7 = stablehlo.add %3, %6 : tensor<i32>
    %8 = stablehlo.select %5, %7, %3 : tensor<i1>, tensor<i32>
    %9 = stablehlo.dynamic_update_slice %0#0, %0#1, %8 : (tensor<3xcomplex<f32>>, tensor<1xcomplex<f32>>, tensor<i32>) -> tensor<3xcomplex<f32>>
    %10 = stablehlo.custom_call @check.eq(%9, %1) : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> tensor<i1>
    return %10 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xcomplex<f32>>, tensor<1xcomplex<f32>>, tensor<1xi32>) {
    %0 = stablehlo.constant dense<[(-0.872995615,1.36752987), (-1.24379516,-5.42408705), (-0.079742834,0.394861579)]> : tensor<3xcomplex<f32>>
    %1 = stablehlo.constant dense<(-3.46074414,-1.36701679)> : tensor<1xcomplex<f32>>
    %2 = stablehlo.constant dense<1> : tensor<1xi32>
    return %0, %1, %2 : tensor<3xcomplex<f32>>, tensor<1xcomplex<f32>>, tensor<1xi32>
  }
  func.func private @expected() -> tensor<3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[(-0.872995615,1.36752987), (-3.46074414,-1.36701679), (-0.079742834,0.394861579)]> : tensor<3xcomplex<f32>>
    return %0 : tensor<3xcomplex<f32>>
  }
}
