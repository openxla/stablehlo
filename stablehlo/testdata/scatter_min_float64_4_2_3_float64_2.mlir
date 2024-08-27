// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3xf64>, tensor<2xf64>)
    %1 = call @expected() : () -> tensor<4x2x3xf64>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<f64>
      stablehlo.return %3 : tensor<f64>
    }) : (tensor<4x2x3xf64>, tensor<2xi64>, tensor<2xf64>) -> tensor<4x2x3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xf64>, tensor<4x2x3xf64>) -> ()
    return %2 : tensor<4x2x3xf64>
  }
  func.func private @inputs() -> (tensor<4x2x3xf64> {mhlo.layout_mode = "default"}, tensor<2xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-2.3526662379681222, -1.2324712404116762, 2.6422261581870661], [-4.3559773725479793, -0.26917589828851052, -0.93795906112829419]], [[-1.5396601208341738, -0.68141867900078923, -0.60178106819518318], [-3.3748304989591595, 1.0617148986765363, -4.0266360267899746]], [[-5.6525431861650768, 6.0369439616129092, -5.8650026954268206], [-1.3606761438164301, -2.9941353959656607, -0.87139350493244727]], [[0.67398007057735976, -2.3935979900223439, 3.5389976599829875], [0.58240383513087512, -3.8012835737577526, -5.5845643646665009]]]> : tensor<4x2x3xf64>
    %cst_0 = stablehlo.constant dense<[-3.0343853297848984, -2.9150274259447544]> : tensor<2xf64>
    return %cst, %cst_0 : tensor<4x2x3xf64>, tensor<2xf64>
  }
  func.func private @expected() -> (tensor<4x2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-2.3526662379681222, -1.2324712404116762, 2.6422261581870661], [-4.3559773725479793, -0.26917589828851052, -0.93795906112829419]], [[-1.5396601208341738, -0.68141867900078923, -0.60178106819518318], [-3.3748304989591595, 1.0617148986765363, -4.0266360267899746]], [[-5.6525431861650768, 6.0369439616129092, -5.8650026954268206], [-1.3606761438164301, -2.9941353959656607, -0.87139350493244727]], [[0.67398007057735976, -2.3935979900223439, -3.0343853297848984], [0.58240383513087512, -3.8012835737577526, -5.5845643646665009]]]> : tensor<4x2x3xf64>
    return %cst : tensor<4x2x3xf64>
  }
}
