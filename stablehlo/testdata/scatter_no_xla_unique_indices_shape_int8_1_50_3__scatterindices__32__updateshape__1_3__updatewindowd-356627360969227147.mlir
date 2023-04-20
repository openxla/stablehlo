// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xi8>, tensor<1x3xi8>)
    %2 = call @expected() : () -> tensor<1x50x3xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      stablehlo.return %arg1 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xi8>, tensor<1xi32>, tensor<1x3xi8>) -> tensor<1x50x3xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xi8>, tensor<1x50x3xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xi8>, tensor<1x3xi8>) {
    %0 = stablehlo.constant dense<"0x0403040004FE000100020100FD04FC0004FF0001000000030000000102FD02FBFEFF0104070000FFFDFDFC0200FE0003FEFCFE0001FC020008FFFF000003FE01FF03FD000100020502010400FF05020003FE030009FD04FC020002FC02FF0207FD0400FE020605FAFEFFFB0000010503FEF9FE01FD060300FB03FB02FFFDFFFFFBFFFDFC000000FF00FB000100020003FB0000020001"> : tensor<1x50x3xi8>
    %1 = stablehlo.constant dense<[[-3, -5, -5]]> : tensor<1x3xi8>
    return %0, %1 : tensor<1x50x3xi8>, tensor<1x3xi8>
  }
  func.func private @expected() -> tensor<1x50x3xi8> {
    %0 = stablehlo.constant dense<"0x0403040004FE000100020100FD04FC0004FF0001000000030000000102FD02FBFEFF0104070000FFFDFDFC0200FE0003FEFCFE0001FC020008FFFF000003FE01FF03FD000100020502010400FF05020003FE030009FD04FC020002FC02FF0207FDFBFBFE020605FAFEFFFB0000010503FEF9FE01FD060300FB03FB02FFFDFFFFFBFFFDFC000000FF00FB000100020003FB0000020001"> : tensor<1x50x3xi8>
    return %0 : tensor<1x50x3xi8>
  }
}

