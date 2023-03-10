// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x4x3xf32>, tensor<2x1x3xf32>)
    %1 = call @expected() : () -> tensor<2x4x3xf32>
    %2 = stablehlo.broadcast_in_dim %0#1, dims = [0, 1, 2] : (tensor<2x1x3xf32>) -> tensor<2x4x3xf32>
    %3 = stablehlo.remainder %0#0, %2 : tensor<2x4x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<2x4x3xf32>, tensor<2x4x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x4x3xf32>, tensor<2x1x3xf32>) {
    %0 = stablehlo.constant dense<[[[5.14350653, -0.670222342, 2.98462629], [-1.81911612, 1.21669233, 1.04855895], [-2.93446159, 7.40413523, 0.623148739], [0.862075746, 2.77537894, -1.98524344]], [[3.59573269, 3.58449674, -5.10428858], [-0.133532748, 3.45638442, 1.08678818], [2.1897552, 1.68665373, -2.78926778], [7.40427112, -1.27565074, -1.62288249]]]> : tensor<2x4x3xf32>
    %1 = stablehlo.constant dense<[[[3.04335117, -1.61257958, 1.48450732]], [[-1.64298856, 2.71912694, 4.39677715]]]> : tensor<2x1x3xf32>
    return %0, %1 : tensor<2x4x3xf32>, tensor<2x1x3xf32>
  }
  func.func private @expected() -> tensor<2x4x3xf32> {
    %0 = stablehlo.constant dense<[[[2.10015535, -0.670222342, 0.0156116486], [-1.81911612, 1.21669233, 1.04855895], [-2.93446159, 0.95381689, 0.623148739], [0.862075746, 1.16279936, -0.500736117]], [[0.309755564, 0.865369797, -0.707511425], [-0.133532748, 0.73725748, 1.08678818], [0.546766639, 1.68665373, -2.78926778], [0.832316875, -1.27565074, -1.62288249]]]> : tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
}
