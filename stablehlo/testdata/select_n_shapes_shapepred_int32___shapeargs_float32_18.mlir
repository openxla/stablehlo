// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:4 = call @inputs() : () -> (tensor<18xi32>, tensor<18xf32>, tensor<18xf32>, tensor<18xf32>)
    %1 = call @expected() : () -> tensor<18xf32>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i32>) -> tensor<18xi32>
    %4 = stablehlo.compare  LT, %0#0, %3,  SIGNED : (tensor<18xi32>, tensor<18xi32>) -> tensor<18xi1>
    %5 = stablehlo.constant dense<2> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i32>) -> tensor<18xi32>
    %7 = stablehlo.compare  LT, %0#0, %6,  SIGNED : (tensor<18xi32>, tensor<18xi32>) -> tensor<18xi1>
    %8 = stablehlo.select %7, %0#2, %0#3 : tensor<18xi1>, tensor<18xf32>
    %9 = stablehlo.select %4, %0#1, %8 : tensor<18xi1>, tensor<18xf32>
    %10 = stablehlo.custom_call @check.eq(%9, %1) : (tensor<18xf32>, tensor<18xf32>) -> tensor<i1>
    return %10 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<18xi32>, tensor<18xf32>, tensor<18xf32>, tensor<18xf32>) {
    %0 = stablehlo.constant dense<[1, 0, 1, 1, 0, 1, 0, 0, 1, 2, 1, 2, 2, 2, 0, 0, 2, 0]> : tensor<18xi32>
    %1 = stablehlo.constant dense<[-2.56971025, -7.89731216, 0.0514946617, 4.50325537, 4.85297775, -1.72558701, -1.99758089, 0.172726586, -7.176060e-01, 1.34067202, 0.473809242, -3.26141739, -1.83130038, -1.43720102, -0.0554215796, -4.94478846, 1.58294404, 1.77953064]> : tensor<18xf32>
    %2 = stablehlo.constant dense<[-0.919511258, 2.8376255, 3.54919434, 4.88454437, -1.38952506, -2.16930485, -0.401708037, 0.481261939, -8.76869678, 0.688085675, 3.67170405, -3.37989855, -0.241823301, 3.570261, 0.611415386, 0.885005414, 0.0570253097, -0.893107176]> : tensor<18xf32>
    %3 = stablehlo.constant dense<[-0.444796294, -6.04211664, 2.37894702, 1.58824074, 3.19721127, -1.86271441, 0.137862161, 5.63979721, -1.51977253, -2.82285905, 0.047678344, 2.366310e+00, 5.4627285, -0.693504691, 4.94340086, -1.0458703, -2.03221393, 0.0244565886]> : tensor<18xf32>
    return %0, %1, %2, %3 : tensor<18xi32>, tensor<18xf32>, tensor<18xf32>, tensor<18xf32>
  }
  func.func private @expected() -> tensor<18xf32> {
    %0 = stablehlo.constant dense<[-0.919511258, -7.89731216, 3.54919434, 4.88454437, 4.85297775, -2.16930485, -1.99758089, 0.172726586, -8.76869678, -2.82285905, 3.67170405, 2.366310e+00, 5.4627285, -0.693504691, -0.0554215796, -4.94478846, -2.03221393, 1.77953064]> : tensor<18xf32>
    return %0 : tensor<18xf32>
  }
}
