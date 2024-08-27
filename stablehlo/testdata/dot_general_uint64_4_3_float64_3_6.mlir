// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui64>, tensor<3x6xf64>)
    %1 = call @expected() : () -> tensor<4x6xf64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui64>) -> tensor<4x3xf64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    return %4 : tensor<4x6xf64>
  }
  func.func private @inputs() -> (tensor<4x3xui64> {mhlo.layout_mode = "default"}, tensor<3x6xf64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 5, 2], [1, 0, 2], [2, 0, 0], [0, 5, 5]]> : tensor<4x3xui64>
    %cst = stablehlo.constant dense<[[-0.8843040214754414, 0.82296715994583824, -1.2351699753063525, -0.86249943731353984, 4.5050901229444422, 1.0298933086352715], [-0.50707948186462337, 6.0627238681772955, -3.0378494750898355, -1.1981674732128835, -2.9179088766888874, 6.6479246631177347], [-2.9505934533020959, -3.6157121940795598, -0.075095321841947255, -5.2368421362524042, 0.14531847235276535, 1.5175554543543561]]> : tensor<3x6xf64>
    return %c, %cst : tensor<4x3xui64>, tensor<3x6xf64>
  }
  func.func private @expected() -> (tensor<4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-9.3208883374027494, 23.905162112673196, -16.574607994439425, -17.327021075882765, -9.793817315794465, 37.304627532932656], [-6.7854909280796329, -6.4084572282132815, -1.385360618990247, -11.336183709818348, 4.795727067649973, 4.0650042173439838], [-1.7686080429508828, 1.6459343198916765, -2.4703399506127051, -1.7249988746270797, 9.0101802458888844, 2.059786617270543], [-17.288364675833595, 12.235058370488677, -15.564723984658913, -32.175048047326442, -13.862952021680611, 40.827400587360451]]> : tensor<4x6xf64>
    return %cst : tensor<4x6xf64>
  }
}
