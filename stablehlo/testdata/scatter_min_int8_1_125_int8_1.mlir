// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x125xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x125xi8>, tensor<1xi8>)
    %1 = call @expected() : () -> tensor<1x125xi8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<1x125xi8>, tensor<1xi64>, tensor<1xi8>) -> tensor<1x125xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x125xi8>, tensor<1x125xi8>) -> ()
    return %2 : tensor<1x125xi8>
  }
  func.func private @inputs() -> (tensor<1x125xi8> {mhlo.layout_mode = "default"}, tensor<1xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFE040000010007FAFDFD01FF00F9FCFEFE0102000202FD030002FEFDFF060000FDFFFFFE020203FE000005010100070002FE00FE000302020400FD0002030200FDFB0100FE04FC0400FCFE00000200000003FD030003FD0000FD03FF01FD02FCFF0301FBFD00FF00FF040003FE00FB00FC04010000FFFEFD0001010003"> : tensor<1x125xi8>
    %c_0 = stablehlo.constant dense<-4> : tensor<1xi8>
    return %c, %c_0 : tensor<1x125xi8>, tensor<1xi8>
  }
  func.func private @expected() -> (tensor<1x125xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFC040000010007FAFDFD01FF00F9FCFEFE0102000202FD030002FEFDFF060000FDFFFFFE020203FE000005010100070002FE00FE000302020400FD0002030200FDFB0100FE04FC0400FCFE00000200000003FD030003FD0000FD03FF01FD02FCFF0301FBFD00FF00FF040003FE00FB00FC04010000FFFEFD0001010003"> : tensor<1x125xi8>
    return %c : tensor<1x125xi8>
  }
}
