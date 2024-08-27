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
      stablehlo.return %arg1 : tensor<bf16>
    }) : (tensor<1x50x3xbf16>, tensor<1xi64>, tensor<1x3xbf16>) -> tensor<1x50x3xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x50x3xbf16>, tensor<1x50x3xbf16>) -> ()
    return %2 : tensor<1x50x3xbf16>
  }
  func.func private @inputs() -> (tensor<1x50x3xbf16> {mhlo.layout_mode = "default"}, tensor<1x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x82BF043E8C40ABBF1C3FC33F92BF9B3F75C0A3400B3EB2BFAB3F0B40F9BEF6403BBFED3F784082BEC03F76C0C7BF46C0B23F59C094C0CB3F1AC0954009C047C0C1C073BE11C015C0DC3EEA3FC33EEF40883C27BFDCC0913FA63F443C95C0093EB93F15C0434014C09ABF0340FDBE0BC15CC0983DB9BFF03FD4BF7A40B63FB03D9FC001416CC099C049C071401DC0EAC00140BA409040E1BF59BE63409FC071405BC046C04D40143F28C00940C33FB8BEDEBEE33FAE3F974044409DBD094049409B3F98BF45C027406BC074C013C018C009BF99BE7CC093BF14C07BC008C0993F3ABE9EC007C02A4017BFDE3FBB3F24BF144031BE87BEEF3FAA3FAFC0F8C08DC040C057408740C93FBE3F69C0224028408DBF5BBFBB3FAE40AE3F604092C08EBF993F87BF26C038C04A406340"> : tensor<1x50x3xbf16>
    %cst_0 = stablehlo.constant dense<[[-4.707030e-01, 3.242190e-01, 1.382810e+00]]> : tensor<1x3xbf16>
    return %cst, %cst_0 : tensor<1x50x3xbf16>, tensor<1x3xbf16>
  }
  func.func private @expected() -> (tensor<1x50x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x82BF043E8C40ABBF1C3FC33F92BF9B3F75C0A3400B3EB2BFAB3F0B40F9BEF6403BBFED3F784082BEC03F76C0C7BF46C0B23F59C094C0CB3F1AC0954009C047C0C1C073BE11C015C0DC3EEA3FC33EEF40883C27BFDCC0913FA63F443C95C0093EB93F15C0434014C09ABF0340FDBE0BC15CC0983DB9BFF03FD4BF7A40B63FB03D9FC001416CC099C049C071401DC0EAC00140BA409040E1BF59BE63409FC071405BC046C04D40143F28C00940C33FB8BEDEBEE33FAE3F974044409DBD09404940F1BEA63EB13F27406BC074C013C018C009BF99BE7CC093BF14C07BC008C0993F3ABE9EC007C02A4017BFDE3FBB3F24BF144031BE87BEEF3FAA3FAFC0F8C08DC040C057408740C93FBE3F69C0224028408DBF5BBFBB3FAE40AE3F604092C08EBF993F87BF26C038C04A406340"> : tensor<1x50x3xbf16>
    return %cst : tensor<1x50x3xbf16>
  }
}
