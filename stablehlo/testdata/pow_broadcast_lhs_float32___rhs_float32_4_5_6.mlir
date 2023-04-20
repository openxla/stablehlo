// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<f32>, tensor<4x5x6xf32>)
    %1 = call @expected() : () -> tensor<4x5x6xf32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<f32>) -> tensor<4x5x6xf32>
    %3 = stablehlo.power %2, %0#1 : tensor<4x5x6xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<4x5x6xf32>, tensor<4x5x6xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<f32>, tensor<4x5x6xf32>) {
    %0 = stablehlo.constant dense<"0x28E018BF546537C04AC6C6BF0C1308C0DDBACDBFECC7A6C06214A5BF199481C00AF45EBF17D4A43E99BE08C05D6D70C0724C71C09289ED3E29744E3E789B6D3F324301C0D975EF3F4858DB3D6A00C23FB6B6173DBD87B6408E8D624026D43DC069D03040DDED1BC0A397804052497EC0D5E253C0B43931406539ACC0FFF4723FA784264019433940C2B80DC089BA8EC03E004FBFBC5627C0037AFBBE5E96933F838D83BFBD3F3E3E186A2040004485C020DC3EBF6E658140BC9535403550DCBE307D483F90ADC8C093272C3F42524A3F49058C40DCE821C085369DBFD4725C40773C34C0609700C0AC80E23F5CE80E40897E95BEBBB4843F9B7020C0030689408B35FA3F5EFEA5BF11F4B640678E0DC09CD9A4BF0225E43D19B2CBBF9C63703F7E7B75C0BEC82FBFF2E64440712E09BD18BA91BF3A1730BD9519AF40A9D586C0AA7C5140474235C08387B5BEA394A5404E5F2EBEFAAA37408AC09BC0CB1BB9C05F2832C0B761F53F5B7983BF3C8382C0BC05463D5F6C213F3D65B2BFDBB70DC0AED4083F5890C23D6C44CDBC2347CF40E71C4240206BBEBF1F4503C0250B5C4004077E40F92C8D40C22F1FC0E8F7DDBF7A1BB0BFEBC3F23F4D31BDBFC4460241F62D1CBE0089BB3E891A3E3F2FA765408879EBBF1551024022D252C0FF39263F"> : tensor<4x5x6xf32>
    %1 = stablehlo.constant dense<-1.90542066> : tensor<f32>
    return %1, %0 : tensor<f32>, tensor<4x5x6xf32>
  }
  func.func private @expected() -> tensor<4x5x6xf32> {
    %0 = stablehlo.constant dense<0xFFC00000> : tensor<4x5x6xf32>
    return %0 : tensor<4x5x6xf32>
  }
}
