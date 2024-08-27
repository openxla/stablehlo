// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xui64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xf64>
    %1 = call @expected() : () -> tensor<2x3xui64>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xf64>) -> tensor<2x3xui64>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<2x3xui64>, tensor<2x3xui64>) -> ()
    return %2 : tensor<2x3xui64>
  }
  func.func private @inputs() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.8308635969433493, 0.71900016032753111, -1.6188963883466747], [-2.954063914647203, 0.26588954076144977, -0.30983069279339981]]> : tensor<2x3xf64>
    return %cst : tensor<2x3xf64>
  }
  func.func private @expected() -> (tensor<2x3xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[4613556956920182793, 4604651397253537208, 13833341717198732246], [13837206416227410102, 4598461460064685830, 13822625070343130920]]> : tensor<2x3xui64>
    return %c : tensor<2x3xui64>
  }
}
