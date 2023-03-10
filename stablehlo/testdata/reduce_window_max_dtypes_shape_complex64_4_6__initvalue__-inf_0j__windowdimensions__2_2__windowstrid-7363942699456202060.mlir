// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3x5xcomplex<f32>>
    %2 = stablehlo.constant dense<(0xFF800000,0.000000e+00)> : tensor<complex<f32>>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<complex<f32>>) -> tensor<complex<f32>>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
      %6 = stablehlo.real %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %7 = stablehlo.real %arg1 : (tensor<complex<f32>>) -> tensor<f32>
      %8 = stablehlo.compare  EQ, %6, %7,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %9 = stablehlo.compare  GT, %6, %7,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %10 = stablehlo.imag %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %11 = stablehlo.imag %arg1 : (tensor<complex<f32>>) -> tensor<f32>
      %12 = stablehlo.compare  GT, %10, %11,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %13 = stablehlo.select %8, %12, %9 : tensor<i1>, tensor<i1>
      %14 = stablehlo.select %13, %arg0, %arg1 : tensor<i1>, tensor<complex<f32>>
      stablehlo.return %14 : tensor<complex<f32>>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3x5xcomplex<f32>>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xcomplex<f32>>, tensor<3x5xcomplex<f32>>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(3.19160271,1.93230247), (2.50072622,0.83774811), (0.203054637,2.64535379), (-3.36030579,-1.64477944), (-0.114631437,1.23255181), (-0.116937235,2.10134077)], [(1.37705767,-4.27031612), (0.854770839,-1.28244865), (-0.5449754,0.52257359), (1.71377635,1.11657381), (-1.06774914,-1.25968111), (-1.41271901,1.99156022)], [(7.21256113,3.94019437), (4.36284494,1.57882726), (3.3280437,1.45958543), (0.740917921,1.7568537), (-3.92812538,1.01615059), (1.54992008,2.86008215)], [(4.06971502,2.1043973), (7.16280841,-6.32898521), (-0.262441903,5.10018301), (2.14247918,-2.2662158), (0.827383518,-2.13803506), (2.02075124,-1.67654085)]]> : tensor<4x6xcomplex<f32>>
    return %0 : tensor<4x6xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(3.19160271,1.93230247), (2.50072622,0.83774811), (1.71377635,1.11657381), (1.71377635,1.11657381), (-0.114631437,1.23255181)], [(7.21256113,3.94019437), (4.36284494,1.57882726), (3.3280437,1.45958543), (1.71377635,1.11657381), (1.54992008,2.86008215)], [(7.21256113,3.94019437), (7.16280841,-6.32898521), (3.3280437,1.45958543), (2.14247918,-2.2662158), (2.02075124,-1.67654085)]]> : tensor<3x5xcomplex<f32>>
    return %0 : tensor<3x5xcomplex<f32>>
  }
}

