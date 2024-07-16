// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x50x3xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<32> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x50x3xbf16>, tensor<1x3xbf16>)
    %1 = call @expected() : () -> tensor<1x50x3xbf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<1x50x3xbf16>, tensor<1xi64>, tensor<1x3xbf16>) -> tensor<1x50x3xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x50x3xbf16>, tensor<1x50x3xbf16>) -> ()
    return %2 : tensor<1x50x3xbf16>
  }
  func.func private @inputs() -> (tensor<1x50x3xbf16> {mhlo.layout_mode = "default"}, tensor<1x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xA23F8940A6BEF13F713FA4C0B840E33E40C05F4077C05E3ED2400740894011C0DFBF1DC05CC088BF5E402CBFBDBF7B3FF83E02BF76BE96BF0B407DBFE740143EEA4066BF80BFFCBF954003408EC0763FCABFB5C0BC3EDF3F2FBF763F8340CABFFEBF03C0EF3E27405EBF3F405840773F25407CBF24C0BABF42C045C0C13F1B40974015BEBB402140C9BF45C0CD3FD0403340FEBF853DF9BFA53F44C029BFCDC06F4019BFE93D2140B93E7CC0733F014095408BC01AC0953FD34026C001BF25BFDE3E68408B3FA9BE6440923F6D40E1BD4240E33F6EC0B6BF623E03BFE23F8B40943F55BEBB3F2DBD15C099BEAF3F95406740EA40913E73C0714039C0EE40F3BFC73F23BF9040193E67C0DEC05CBF67BFDF3F0440303F86C033C0AFC05440B340CD4081C028C0433C513F0840"> : tensor<1x50x3xbf16>
    %cst_0 = stablehlo.constant dense<[[2.929690e-01, 3.031250e+00, -2.406250e+00]]> : tensor<1x3xbf16>
    return %cst, %cst_0 : tensor<1x50x3xbf16>, tensor<1x3xbf16>
  }
  func.func private @expected() -> (tensor<1x50x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xA23F8940A6BEF13F713FA4C0B840E33E40C05F4077C05E3ED2400740894011C0DFBF1DC05CC088BF5E402CBFBDBF7B3FF83E02BF76BE96BF0B407DBFE740143EEA4066BF80BFFCBF954003408EC0763FCABFB5C0BC3EDF3F2FBF763F8340CABFFEBF03C0EF3E27405EBF3F405840773F25407CBF24C0BABF42C045C0C13F1B40974015BEBB402140C9BF45C0CD3FD0403340FEBF853DF9BFA53F44C029BFCDC06F4019BFE93D2140B93E7CC0733F014095408BC01AC0953FD34026C001BF25BF3A3FD540A9BFA9BE6440923F6D40E1BD4240E33F6EC0B6BF623E03BFE23F8B40943F55BEBB3F2DBD15C099BEAF3F95406740EA40913E73C0714039C0EE40F3BFC73F23BF9040193E67C0DEC05CBF67BFDF3F0440303F86C033C0AFC05440B340CD4081C028C0433C513F0840"> : tensor<1x50x3xbf16>
    return %cst : tensor<1x50x3xbf16>
  }
}
