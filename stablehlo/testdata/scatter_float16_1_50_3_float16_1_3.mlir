// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x50x3xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<32> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x50x3xf16>, tensor<1x3xf16>)
    %1 = call @expected() : () -> tensor<1x50x3xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      stablehlo.return %arg1 : tensor<f16>
    }) : (tensor<1x50x3xf16>, tensor<1xi64>, tensor<1x3xf16>) -> tensor<1x50x3xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x50x3xf16>, tensor<1x50x3xf16>) -> ()
    return %2 : tensor<1x50x3xf16>
  }
  func.func private @inputs() -> (tensor<1x50x3xf16> {mhlo.layout_mode = "default"}, tensor<1x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x48BE67C4D6B22C46A6BF6ABA1FC19C44904721C3A4B823C18BC2204130C316C36AC0FA3987C0EC436BC42A3F293F5B441D428EBC3035A1B55DC2AAC0D5C0CB2D2CC4E743C9418C40633E5C301B408F406D354DC880C09A42FDC1FEC07EAE22B49FBB1BC4A2BF3B428144C3C089C0BBC1B1BBAFC68943BAB9874579405E42103A6FC53CC56EBC3CC2613EE13FDABC653FCEBB38477C44A2421DC312B4A4BDF4C54CBFA43B16B5533383C432416BC3F83A5BC161C38CBF6041DBBF083D65C19B3E8E38A9436E420943C0396B434DC3E54595BFAF3ABF41403CDDC0A4BDEEC4EC3D1F3F1FBC37C0C12A3740B43E3BC4DDBF2145D73F8C43C2BCD9427D3D73BD06423343A43863BE3FC44CB92FC603BE673D52B47B3B0CC0A23E98BBB4B8D3C2C13B0FC191C1AE395F40F93B6DB7"> : tensor<1x50x3xf16>
    %cst_0 = stablehlo.constant dense<[[5.285160e+00, -4.943850e-01, 1.293950e+00]]> : tensor<1x3xf16>
    return %cst, %cst_0 : tensor<1x50x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> (tensor<1x50x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x48BE67C4D6B22C46A6BF6ABA1FC19C44904721C3A4B823C18BC2204130C316C36AC0FA3987C0EC436BC42A3F293F5B441D428EBC3035A1B55DC2AAC0D5C0CB2D2CC4E743C9418C40633E5C301B408F406D354DC880C09A42FDC1FEC07EAE22B49FBB1BC4A2BF3B428144C3C089C0BBC1B1BBAFC68943BAB9874579405E42103A6FC53CC56EBC3CC2613EE13FDABC653FCEBB38477C44A2421DC312B4A4BDF4C54CBFA43B16B5533383C432416BC3F83A5BC161C38CBF6041DBBF083D65C19B3E4945E9B72D3D0943C0396B434DC3E54595BFAF3ABF41403CDDC0A4BDEEC4EC3D1F3F1FBC37C0C12A3740B43E3BC4DDBF2145D73F8C43C2BCD9427D3D73BD06423343A43863BE3FC44CB92FC603BE673D52B47B3B0CC0A23E98BBB4B8D3C2C13B0FC191C1AE395F40F93B6DB7"> : tensor<1x50x3xf16>
    return %cst : tensor<1x50x3xf16>
  }
}
