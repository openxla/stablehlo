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
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<1x50x3xbf16>, tensor<1xi64>, tensor<1x3xbf16>) -> tensor<1x50x3xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x50x3xbf16>, tensor<1x50x3xbf16>) -> ()
    return %2 : tensor<1x50x3xbf16>
  }
  func.func private @inputs() -> (tensor<1x50x3xbf16> {mhlo.layout_mode = "default"}, tensor<1x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xACBFCDBF1EC022C0ED3F22BF8540FF3EDD3B9AC003C034C06140CD3DEF3FD0BFC74014C09F3F4E40C83F79C048BF8DC046C0AB3E02C09640603F8AC02740A3405A401740F2BF594047C0243E86BF573F7FC0F83F00BFAF4051BF94BF8CC0724038C00540F93E03C036404BC0864080BFEFC0AABFEF3F17407DBF40C01EC0DFBFB1BE2B40453F4D3C174057C00B40C93E8B409B3FB9C08B408A3F2FC0A0400240074095C0CD3FCE3F9A4009C0DFC0AC3F2EBF983E23BEACBEDFBF9A3E273F96401040743F60BC45408EBE7240E0BE8940714099C0274055BF8ABE8BBF1C40AEC0D43E0740074023C015C068BE4DBF49405D4012C0A240A9BF80BF543EA240A13C44C085BF82C021C0A6BF1BC0203F91BFD33EB9BE37C0284064C03AC07AC0C2BF3140FBBE99BFD0C080BF7640"> : tensor<1x50x3xbf16>
    %cst_0 = stablehlo.constant dense<[[-9.875000e+00, -6.125000e+00, -8.828120e-01]]> : tensor<1x3xbf16>
    return %cst, %cst_0 : tensor<1x50x3xbf16>, tensor<1x3xbf16>
  }
  func.func private @expected() -> (tensor<1x50x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xACBFCDBF1EC022C0ED3F22BF8540FF3EDD3B9AC003C034C06140CD3DEF3FD0BFC74014C09F3F4E40C83F79C048BF8DC046C0AB3E02C09640603F8AC02740A3405A401740F2BF594047C0243E86BF573F7FC0F83F00BFAF4051BF94BF8CC0724038C00540F93E03C036404BC0864080BFEFC0AABFEF3F17407DBF40C01EC0DFBFB1BE2B40453F4D3C174057C00B40C93E8B409B3FB9C08B408A3F2FC0A0400240074095C0CD3FCE3F9A4009C0DFC0AC3F2EBF983E23BEACBEDFBF9A3E273F9640B2C1BBC0463C45408EBE7240E0BE8940714099C0274055BF8ABE8BBF1C40AEC0D43E0740074023C015C068BE4DBF49405D4012C0A240A9BF80BF543EA240A13C44C085BF82C021C0A6BF1BC0203F91BFD33EB9BE37C0284064C03AC07AC0C2BF3140FBBE99BFD0C080BF7640"> : tensor<1x50x3xbf16>
    return %cst : tensor<1x50x3xbf16>
  }
}
