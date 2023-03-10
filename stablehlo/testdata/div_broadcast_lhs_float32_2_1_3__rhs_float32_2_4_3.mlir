// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x1x3xf32>, tensor<2x4x3xf32>)
    %1 = call @expected() : () -> tensor<2x4x3xf32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [0, 1, 2] : (tensor<2x1x3xf32>) -> tensor<2x4x3xf32>
    %3 = stablehlo.divide %2, %0#1 : tensor<2x4x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<2x4x3xf32>, tensor<2x4x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x1x3xf32>, tensor<2x4x3xf32>) {
    %0 = stablehlo.constant dense<[[[1.70141661, -1.47774613, -1.90541553]], [[-1.22120333, -3.58539462, -1.52984381]]]> : tensor<2x1x3xf32>
    %1 = stablehlo.constant dense<[[[-3.35528398, 0.231270105, 0.135762051], [-5.97310591, -0.541369498, -4.03985214], [1.06742704, 1.40038991, -3.66399717], [1.2979691, 2.98837256, 1.21461976]], [[3.81585932, 0.0645933449, 2.97496819], [0.530538738, -6.87553119, 1.03047204], [6.61673164, 0.669437587, 2.12376714], [3.03284383, -0.471552938, -7.71977233]]]> : tensor<2x4x3xf32>
    return %0, %1 : tensor<2x1x3xf32>, tensor<2x4x3xf32>
  }
  func.func private @expected() -> tensor<2x4x3xf32> {
    %0 = stablehlo.constant dense<[[[-0.507085741, -6.38969803, -14.0349646], [-0.284846216, 2.7296443, 0.471654773], [1.59394181, -1.05523908, 0.520037413], [1.31082988, -0.49449864, -1.56873417]], [[-0.32003364, -55.5071831, -0.514238715], [-2.30181742, 0.521471679, -1.48460484], [-0.184562922, -5.35583115, -0.720344424], [-0.402659476, 7.60337687, 0.198172137]]]> : tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
}
