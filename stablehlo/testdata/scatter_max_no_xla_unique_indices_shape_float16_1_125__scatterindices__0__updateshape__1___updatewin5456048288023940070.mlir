// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xf16>, tensor<1xf16>)
    %2 = call @expected() : () -> tensor<1x125xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xf16>, tensor<1xi32>, tensor<1xf16>) -> tensor<1x125xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xf16>, tensor<1x125xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xf16>, tensor<1xf16>) {
    %0 = stablehlo.constant dense<"0xCAB9FAC556C74AC3A9BD83B53DBF97BC9FC5FA42304344B27CC63D3F543AB044E04544BCA23E1347F73A45410F3C7DBB1DBA6E3E0CB95AC6884301455BB17EB252C1A1BD613CD5BF76441A47A7B2954277C0C9443FC429339734CE366341FFC4FD38B14305C0A0BD943CC041D93FEE3E40BA763C31C528BEF941543986C1AABB01B8C941BCBFA2B68DC694BC18A8C038063EBC42A1BF6B4423C495474ABD183F3F426C3014C38BC0B7C4D4BE23C17A43623A913916C4C5BC8CAF9FBE0746353F7342832B44C1F0C04D45C2BFD4BCEFC0F2C010BD104448440EC422381FC0044212C3DFC0F038203A12C4B0C07FBC2D4237C4B9B72B3DF0BB063B"> : tensor<1x125xf16>
    %1 = stablehlo.constant dense<5.792960e+00> : tensor<1xf16>
    return %0, %1 : tensor<1x125xf16>, tensor<1xf16>
  }
  func.func private @expected() -> tensor<1x125xf16> {
    %0 = stablehlo.constant dense<"0xCB45FAC556C74AC3A9BD83B53DBF97BC9FC5FA42304344B27CC63D3F543AB044E04544BCA23E1347F73A45410F3C7DBB1DBA6E3E0CB95AC6884301455BB17EB252C1A1BD613CD5BF76441A47A7B2954277C0C9443FC429339734CE366341FFC4FD38B14305C0A0BD943CC041D93FEE3E40BA763C31C528BEF941543986C1AABB01B8C941BCBFA2B68DC694BC18A8C038063EBC42A1BF6B4423C495474ABD183F3F426C3014C38BC0B7C4D4BE23C17A43623A913916C4C5BC8CAF9FBE0746353F7342832B44C1F0C04D45C2BFD4BCEFC0F2C010BD104448440EC422381FC0044212C3DFC0F038203A12C4B0C07FBC2D4237C4B9B72B3DF0BB063B"> : tensor<1x125xf16>
    return %0 : tensor<1x125xf16>
  }
}

