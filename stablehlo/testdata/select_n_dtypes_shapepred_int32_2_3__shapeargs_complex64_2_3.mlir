// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:4 = call @inputs() : () -> (tensor<2x3xi32>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<2x3xcomplex<f32>>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %4 = stablehlo.compare  LT, %0#0, %3,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %5 = stablehlo.constant dense<2> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %7 = stablehlo.compare  LT, %0#0, %6,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %8 = stablehlo.select %7, %0#2, %0#3 : tensor<2x3xi1>, tensor<2x3xcomplex<f32>>
    %9 = stablehlo.select %4, %0#1, %8 : tensor<2x3xi1>, tensor<2x3xcomplex<f32>>
    %10 = stablehlo.custom_call @check.eq(%9, %1) : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<i1>
    return %10 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xi32>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[0, 1, 0], [2, 1, 2]]> : tensor<2x3xi32>
    %1 = stablehlo.constant dense<[[(1.57032847,4.1053977), (-3.74037051,2.54515386), (2.77682853,-3.03926396)], [(2.92192292,1.67899239), (-2.6347487,2.77748942), (-2.2197423,3.77761197)]]> : tensor<2x3xcomplex<f32>>
    %2 = stablehlo.constant dense<[[(-5.50477648,-6.02844333), (1.75364339,0.262662739), (-1.96421671,-1.595613)], [(4.50061607,2.66209936), (-1.62665105,3.71576428), (-1.56580925,-1.74126494)]]> : tensor<2x3xcomplex<f32>>
    %3 = stablehlo.constant dense<[[(0.313747197,-0.271646649), (0.830076038,-4.83073282), (2.15453196,-3.55750418)], [(7.964890e-01,3.34007144), (2.32828379,1.79770422), (0.759251534,1.4883424)]]> : tensor<2x3xcomplex<f32>>
    return %0, %1, %2, %3 : tensor<2x3xi32>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> tensor<2x3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(1.57032847,4.1053977), (1.75364339,0.262662739), (2.77682853,-3.03926396)], [(7.964890e-01,3.34007144), (-1.62665105,3.71576428), (0.759251534,1.4883424)]]> : tensor<2x3xcomplex<f32>>
    return %0 : tensor<2x3xcomplex<f32>>
  }
}
