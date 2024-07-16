// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xf32>
    %1 = call @expected() : () -> tensor<8x9xf32>
    %2 = call @cumlogsumexp(%0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<8x9xf32>, tensor<8x9xf32>) -> ()
    return %2 : tensor<8x9xf32>
  }
  func.func private @inputs() -> (tensor<8x9xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.917350947, -4.9364953, 5.78474236, 2.16007137, -5.87161112, 4.8677063, 2.89252067, 5.88916683, -3.99975801], [0.16120404, 2.30027127, -0.696499586, -1.8043313, -0.451449335, -2.34567666, -1.37980521, 3.12589049, -7.86885547], [0.168912634, 3.66638756, 2.70016122, 2.52134037, 1.62884533, -2.009800e+00, 3.8995502, -4.23454952, -0.558775961], [-3.0145781, -6.071230e-01, 3.07570076, 2.98369765, -2.89113092, 0.852195203, 4.07448292, -1.27202868, 0.5381338], [1.0474875, 1.46639109, -2.51472211, -0.783157349, 1.70092225, 0.817279934, -5.30891323, -3.18156838, -4.77801609], [6.301692, -4.54297352, -5.15177536, -1.8628242, -4.71607161, 1.01931059, -1.44714093, -1.18739903, -2.346100e+00], [2.58556485, -2.20311093, 2.46583056, -2.23851299, 0.406177461, 1.15396821, -4.52010155, 0.639283895, -2.14272141], [3.04544759, 1.34837973, -0.576765537, 2.59732938, -1.86460078, -5.48272514, -0.0322545208, 2.5423491, 0.93175137]]> : tensor<8x9xf32>
    return %cst : tensor<8x9xf32>
  }
  func.func private @expected() -> (tensor<8x9xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[6.37595367, 4.05949783, 5.926300e+00, 4.01076365, 2.56009102, 4.94739342, 4.84976196, 5.98928785, 1.62212908], [6.37168455, 4.05937386, 3.90130901, 3.83982301, 2.5598731, 2.37816644, 4.69748735, 3.63827181, 1.61850476], [6.36967516, 3.87039089, 3.89118385, 3.83627844, 2.50939369, 2.36924601, 4.69519043, 2.7243104, 1.61842895], [6.36764479, 2.18050385, 3.52890706, 3.523633, 1.974010e+00, 2.35662937, 4.09500122, 2.72335982, 1.49811506], [6.36756038, 2.116960e+00, 2.51944041, 2.64947724, 1.96626937, 2.10541701, 0.198307484, 2.70478821, 1.01539063], [6.36265612, 1.37934053, 2.51290751, 2.61664224, 0.509811461, 1.78275096, 0.194241866, 2.70200729, 1.0123384], [3.53486013, 1.37665796, 2.5124383, 2.60523796, 0.504421353, 1.15527868, -0.0210724436, 2.68133759, 0.976929306], [3.04544759, 1.34837973, -0.576765537, 2.59732938, -1.86460078, -5.48272514, -0.0322545208, 2.5423491, 0.93175137]]> : tensor<8x9xf32>
    return %cst : tensor<8x9xf32>
  }
  func.func private @cumlogsumexp(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[0, 7], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = func.call @logaddexp(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<8x9xf32>, tensor<f32>) -> tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @logaddexp(%arg0: tensor<f32> {mhlo.layout_mode = "default"}, %arg1: tensor<f32> {mhlo.layout_mode = "default"}) -> (tensor<f32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<f32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %4 = stablehlo.abs %1 : tensor<f32>
    %5 = stablehlo.negate %4 : tensor<f32>
    %6 = stablehlo.exponential %5 : tensor<f32>
    %7 = stablehlo.log_plus_one %6 : tensor<f32>
    %8 = stablehlo.add %0, %7 : tensor<f32>
    %9 = stablehlo.select %2, %3, %8 : tensor<i1>, tensor<f32>
    return %9 : tensor<f32>
  }
}
