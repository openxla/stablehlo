// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xf32>
    %1 = call @expected() : () -> tensor<8x9xf32>
    %2 = call @cumprod(%0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xf32>, tensor<8x9xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-1.24729371, -1.34739697, 1.81460357, -3.141410e+00, -0.962018966, -1.329410e+00, -1.31378257, -5.86126614, 0.97861725], [-0.974442958, 3.52509952, 0.824250937, -1.940830e+00, -1.26002479, 1.58191442, 2.71779203, -2.15416145, 0.0722424685], [-0.0110738324, 2.75686669, -4.58146715, -2.28137732, 4.01758385, 0.149319991, 1.99673688, 2.11170411, 7.9814825], [-0.395987511, -1.93988502, -5.32596874, 5.79848909, -0.369597673, -0.792699873, 0.0412339084, -5.16402626, 1.77495289], [0.579719186, 2.12604904, -0.918148875, 1.03094983, 5.05238962, 1.17898691, -1.81588924, 2.74803877, -4.44567585], [2.05311418, -3.78843904, 3.94227195, -1.10080671, 1.95295024, -3.88314915, 4.257090e-01, 1.08508563, -4.17765141], [1.01031244, -4.79649925, 3.493200e+00, 0.619659483, -2.10138679, -1.62579262, -2.65190792, 0.679226934, -1.13532257], [0.934070289, -2.33822632, -2.34862804, 0.80499041, -1.03529727, 2.01134062, -2.77663732, 0.665591598, 3.28851891]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-1.24729371, 1.68059981, 3.0496223, -9.58011531, 9.21625232, -12.2521772, 16.0966969, -94.3470306, -92.3296279], [-0.974442958, -3.43500829, -2.83130884, 5.49508905, -6.92394829, -10.9530945, -29.7682323, 64.1255798, 4.63259029], [-0.0110738324, -0.03052908, 0.139867976, -0.319091618, -1.2819773, -0.191424847, -0.382225066, -0.807146191, -6.44222307], [-0.395987511, 0.768170237, -4.0912509, -23.7230721, 8.76799201, -6.95038604, -0.286591589, 1.47996652, 2.62687087], [0.579719186, 1.2325114, -1.13162899, -1.16665268, -5.89438391, -6.94940138, 12.6193428, 34.6784477, -154.169144], [2.05311418, -7.7780981, -30.6633778, 33.7544518, 65.9207611, -255.980164, -108.973061, -118.245102, 493.986816], [1.01031244, -4.845963, -16.9279175, -10.4895458, 22.042593, -35.8366852, 95.0355911, 64.5507278, -7.328590e+01], [0.934070289, -2.18406773, 5.12956285, 4.1292491, -4.27500057, -8.59848213, 23.8748665, 15.8909101, 52.2575569]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @cumprod(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<8> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x4xf32>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x4xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<8x4xf32>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 3]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x2xf32>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 4]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x2xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x2xf32>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x1xf32>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[8, 0]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x1xf32>) -> tensor<8x0xf32>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 2]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x0xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<8x0xf32>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %13 = stablehlo.concatenate %12, %11, dim = 1 : (tensor<8x1xf32>, tensor<8x0xf32>) -> tensor<8x1xf32>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<8x1xf32>, tensor<f32>) -> tensor<8x2xf32>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.pad %8, %16, low = [0, 1], high = [0, 0], interior = [0, 1] : (tensor<8x1xf32>, tensor<f32>) -> tensor<8x2xf32>
    %18 = stablehlo.add %15, %17 : tensor<8x2xf32>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 4]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x1xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<8x1xf32>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x1xf32>
    %23 = stablehlo.concatenate %22, %21, dim = 1 : (tensor<8x1xf32>, tensor<8x1xf32>) -> tensor<8x2xf32>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8x4xf32>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.pad %18, %26, low = [0, 1], high = [0, 0], interior = [0, 1] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8x4xf32>
    %28 = stablehlo.add %25, %27 : tensor<8x4xf32>
    %29 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x4xf32>
    %30 = stablehlo.multiply %28, %29 : tensor<8x4xf32>
    %31 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x1xf32>
    %32 = stablehlo.concatenate %31, %30, dim = 1 : (tensor<8x1xf32>, tensor<8x4xf32>) -> tensor<8x5xf32>
    %33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.pad %32, %33, low = [0, 0], high = [0, 0], interior = [0, 1] : (tensor<8x5xf32>, tensor<f32>) -> tensor<8x9xf32>
    %35 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %36 = stablehlo.pad %28, %35, low = [0, 1], high = [0, 1], interior = [0, 1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8x9xf32>
    %37 = stablehlo.add %34, %36 : tensor<8x9xf32>
    return %37 : tensor<8x9xf32>
  }
}
