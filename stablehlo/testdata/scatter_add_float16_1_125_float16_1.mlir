// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x125xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x125xf16>, tensor<1xf16>)
    %1 = call @expected() : () -> tensor<1x125xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<1x125xf16>, tensor<1xi64>, tensor<1xf16>) -> tensor<1x125xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x125xf16>, tensor<1x125xf16>) -> ()
    return %2 : tensor<1x125xf16>
  }
  func.func private @inputs() -> (tensor<1x125xf16> {mhlo.layout_mode = "default"}, tensor<1xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x5BC063C5D0BD6DAFEAB8DB3E23C406C5ADC0C9BDB73A3ABF74BAC3C55CBE42BC193F4D311CB95CBA8939D5C19A406FC566B4273BDFBD923E5C410F464F3FD93DEB3F99C1523C044556316328013A94B50744B743BD2FD442A4440FC31836C940B2BE6B2A303C0A417BBF03403644E9BC4B478143D8C5D1BEA4BFB7A926BE3ABEA03C7BC00835C037384570BB0E445FBB693951BB6CC10A464D414FC1D1C04FAFA2BD34B391B68441FDC4CFBA8E46FD3555BFA4C0533AD22A1BB17B43853C54B683461946304066BF8446E14423B457C0953C53437D435338304489B841384FB4184190BB1DC45842DA387330863FE93EEA39F4C486427E42D9C5"> : tensor<1x125xf16>
    %cst_0 = stablehlo.constant dense<-3.882810e+00> : tensor<1xf16>
    return %cst, %cst_0 : tensor<1x125xf16>, tensor<1xf16>
  }
  func.func private @expected() -> (tensor<1x125xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x10C663C5D0BD6DAFEAB8DB3E23C406C5ADC0C9BDB73A3ABF74BAC3C55CBE42BC193F4D311CB95CBA8939D5C19A406FC566B4273BDFBD923E5C410F464F3FD93DEB3F99C1523C044556316328013A94B50744B743BD2FD442A4440FC31836C940B2BE6B2A303C0A417BBF03403644E9BC4B478143D8C5D1BEA4BFB7A926BE3ABEA03C7BC00835C037384570BB0E445FBB693951BB6CC10A464D414FC1D1C04FAFA2BD34B391B68441FDC4CFBA8E46FD3555BFA4C0533AD22A1BB17B43853C54B683461946304066BF8446E14423B457C0953C53437D435338304489B841384FB4184190BB1DC45842DA387330863FE93EEA39F4C486427E42D9C5"> : tensor<1x125xf16>
    return %cst : tensor<1x125xf16>
  }
}
