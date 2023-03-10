// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xf16>, tensor<20x20xf16>)
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = call @igammac(%0#0, %0#1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<20x20xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xf16>, tensor<20x20xf16>) {
    %0 = stablehlo.constant dense<"0x1140E9BE56C4BFC3FF3E3F3FFE3B7A3BA43FBD462345253D24C2C62D8141B84126C449C2C23A39C647BA92C50F4462C1F63A1F3E7440543D03BD573D94454A448E41BAC35644C0BF1BC07B3D25C10BBF983E55B2A6C2633CADC064433E3D8435484109C5994692B9081E77C19DB913B808C222C123B89A3E60B125C4EEC60EBD0A44A6B66ABE1241B6BA27BCE6C34EBCB23D66BE14C00D417E403B3EFA41E0C14340ED3282C55FBA543D28C1D9C25F47E8314637E23DDC2E59C4E8387C405AB816BFB6BC15C3E4C34B400B3E35C15A3D12C5B54070BF9A41DF2EA8C0EDB670C79DBFB13FF13957414EC10DB9F0B793346E458D3C3743AAC1473339B8C13A1FBC92BD2D4033C120445AC0153F4BC2F136FA45193908BE93C3DFB9FEC43CC1D73FDC40793F54BA74C5E1AE4F44F1C10CC21C3E54C4EB41A7C30B3C9CC3AFC332C3DB400C40D4BC63C0F13EC03E4CC14844C0C298BC92C4B43A1D3C30B912B8DA4182C592C7254131BF9FB92746BA41073CA7C504A8B74141BD77B87DC11AAD37C0B24232BCED39A3458BBE06BCE6BAC2BBB037D6BADF4337B6F32E61BF7BC23B3EE93CC2B35CB804C4F5468EB7E745EEC0F5C11BC513C4F9445432494340C56DB92DBBD22D123AFC3B20BE813DF9308542A5BFF8B949338038573C54BC86408F412B4082BFB1206B3577C68AC2A24306C14EA4493D20B8ADC495416F420E42AE4379C33D3A324043B8B13AAEC5E4C284BA8EC149C2BC38CFC256426BA653421FB77BBBFB3DAB4387BF7C42773FFF408DC00A43583B933E453E64441F404BBC0DBF31379B42FFBE7043A23F27AD28BDC9B444C5F62E323EEEC53AC0F34364BC3AC6E0C155C521B1EC3EB23C353D7CBEAAC3543C6CB8EB451EBDAD49904135434C3BF9BCEFC3CB3CBD3E0EC3C938A6B500B52CC32D325DC07F3BAE3D5D36D3398BBD2243413D003F3D40553D2EC1003AF04263C5DFC398C439450AAA3C4752456E46CD3F25C44FC17745CBB84EC51041403F7739BC3FCC3E0BC2203FEDC076BE15C127C2C53F193CA1C3FE3146456D34A4C4183C2745A9BE95C03B3FD5BE36BF1CC2CDB839C31B40EABA3041BB45C5C1A4C4B740ACBFE73EBBC15EC6F138E33279C0"> : tensor<20x20xf16>
    %1 = stablehlo.constant dense<"0x89B0113F55C244BCE9374EC6D3BC71B406C4F041EF3CA83F2CC01FB9A331BBC158B456B7DCC1B7B8F134B9437CBCDA41C13F1DBE6945A344084423C2DC43C7AD663B97C86E413B3ACB3D92B77B478FBF5B399BC3F3AD8FB8513E9FC2A9C5A1B8F64726B9BEC48E42B6C5BB393EB9D1410836CEBEEAB50FBC5442714095C0F6447743ED43BB43FEC2FA3EE1BC10C5A7C1823E33B62EBA42C3F9BF8A25C7BAAEC199440C418C3EE9B9A6409BB8DF39603C823E02431D426C3D41B3DDC358443E4546346DBF823AB1BEC8445EB748BADFBEB7C14DC32BBE02BC31406E4629BD2643F3C0D743D838F8AD5AC2D1BF283E64C00FB6483D8B45B6C46C3D3FC658C6D23E0AA856BE21BD92C1DF432E3F10C2BBBD49BB4A4130C4DA48E59BC93686398538B13AB43EF1C29AC383401FBC2D3EB7BBCEBC98C4A0B713407A3CF942D44455C1C7C28D42A74403C4803F473D9F3F9242DBC548C6EDC0CD3D87401A46B3BB52BFA6C2DDBD0ABF6D3835C40540DF4524B5B8457F3BAABECCC397C1A3C0F1C50BC544C0CDC08B36AF2B81BFE04292C0BAAC7147EEC4BEBA26434AB698BD42BC9DBF6CB4A83E19C365C41144D0405ABEA9C086C4234439BDF5AF8DC5964068C569C0D2C0C3C18A4026B7564456B3A5BE10BE51383341B3447F3E7943A5C38DC2B3BF7347BFBD4542913E66C1ABC4EAB11E40CCBDF540DB35C24481C06CBCE240A93C683581C1ACB599424D48B1B881C099BDB73B843F2FC30334D0C4FC414339DBC41FBAA3B85DC66ABD41BB3DB9ADC15148E7C10240FDBCBC45E3B34A4314B812BBE5B808C36640DA408440CBBD1B3B76C504450EC06841B23DE33E13C17437A5BFE135222FFC421F42EDB401C4E4BBE2BF5FC0203FD9BFFCC17D4527C258B436BAC73B92B64CC40FC07CC0514121C249BB6F444840D6BDF6BD26C526BF4BC3123FC7B82C3C414471C666B911C04830BDC0EAC17E457E3D96C002437A4060BE5EB470B209C3CD3CE7BA0EC854C5D3B3E230CA3E2F409BC65A43B5413640C2BAB4C08DC4A73681B8963C372ACEC21FBD91C2A040C4C0393FFDC43540C9AB9E3B37C58B444B3F1245C93C79B8DAB2D4BDDE3541B96135123E3C439943C1C2F62120BE"> : tensor<20x20xf16>
    return %0, %1 : tensor<20x20xf16>, tensor<20x20xf16>
  }
  func.func private @expected() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0x003C003C003C003CEF3A003C003C003C003CAC3BF23B0E33003C003CFD3B003C003C003C003C003C003C003C003C003C5E2F003C0429EB24003C003CE739003C463B003C283A003C003C003C003C003C233A003C003C003C003C003C003C003C7320003C003C003C003C003C003C003C003C003C003C003C003C003C003C003CEE37003C003C003C003C003C003C003C4435003C003C003C003CFC3B003C003C442C901F003C003C2E31003C003C003CF424131F882EBE23003C003CF02D003C003C003C003C003C822B003C003C003C003C003C003C003C8C1D003C003C003C003CB12D9136003C003C003C003C003C003C1335B630003CA528003C003C003C003C003C003C003C003C4B36003C003C003C1828003C003C003C003C003C0F3B0E3B5937003C003C003C003C003C003C003C003C003C003C4B35003C003C003C003C4D31003C003CC4352338003C2039003C003C003CD831FB2E003C003C003C003C003C003C003C003CE33B862B003C003C003C003C003C003C003C003C003C003C003C3138003C003C003C003C003CEA06003C003C003C003C003C003C003C003C003C003C003C083B003C003C003C003C003C003C003C003CE839003C003C003C003C242C003C003C003C003C003C003C003CF110692D4327003C003C003CCE1D003CB9063F29003C003C003C003C003C7230003C003C003C003C8638B13B003C003C003C003C6607003C003C003C003C003C003C003C003C003CD33B003C003C003C003C003C003C003C003C003C003C812F003C7921003C2830003C003C003C003C003CC7390E35003C003C003C003C003CE930003C003C003C003C003C003C003C003C8A30003C003C003C003C003C003C003C003CDA3B003C003C003C003C003C003C003C003C4027003C003C003C4C21003C003C003C003C003C003C003CB737D92A003C003C003C2E3A003C003C003C003C003C003CF43B003C003C003C003C003C003C003C003C003CDC3B9A2ECE35003C003CAF31003C003C003C003C6A3B003C003C3936003C003C003C992E003C003C003C2F35003C003C003C003C003C3F29003C003C003C003C003C003C003C6737003C003C003CA738003C"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @igammac(%arg0: tensor<20x20xf16>, %arg1: tensor<20x20xf16>) -> tensor<20x20xf16> {
    %0 = call @xla_fallback_igammac(%arg0, %arg1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @igamma_body.171(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>> {
    %0 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xi1>
    %1 = stablehlo.get_tuple_element %arg0[2] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %2 = stablehlo.get_tuple_element %arg0[4] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %3 = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %6 = stablehlo.add %3, %5 : tensor<20x20xf32>
    %7 = stablehlo.divide %2, %6 : tensor<20x20xf32>
    %8 = stablehlo.multiply %1, %7 : tensor<20x20xf32>
    %9 = stablehlo.get_tuple_element %arg0[3] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %10 = stablehlo.add %9, %8 : tensor<20x20xf32>
    %11 = stablehlo.divide %8, %10 : tensor<20x20xf32>
    %12 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %13 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %14 = stablehlo.compare  GT, %11, %13 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %15 = stablehlo.and %0, %14 : tensor<20x20xi1>
    %16 = stablehlo.select %0, %6, %3 : tensor<20x20xi1>, tensor<20x20xf32>
    %17 = stablehlo.select %0, %8, %1 : tensor<20x20xi1>, tensor<20x20xf32>
    %18 = stablehlo.select %0, %10, %9 : tensor<20x20xi1>, tensor<20x20xf32>
    %19 = stablehlo.get_tuple_element %arg0[5] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %20 = stablehlo.divide %2, %6 : tensor<20x20xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<20x20xf32>
    %22 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %23 = stablehlo.constant dense<-1.000000e+00> : tensor<20x20xf32>
    %24 = stablehlo.multiply %23, %1 : tensor<20x20xf32>
    %25 = stablehlo.multiply %24, %2 : tensor<20x20xf32>
    %26 = stablehlo.multiply %6, %6 : tensor<20x20xf32>
    %27 = stablehlo.divide %25, %26 : tensor<20x20xf32>
    %28 = stablehlo.add %21, %27 : tensor<20x20xf32>
    %29 = stablehlo.select %0, %28, %19 : tensor<20x20xi1>, tensor<20x20xf32>
    %30 = stablehlo.get_tuple_element %arg0[6] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %31 = stablehlo.add %30, %28 : tensor<20x20xf32>
    %32 = stablehlo.select %0, %31, %30 : tensor<20x20xi1>, tensor<20x20xf32>
    %33 = stablehlo.tuple %15, %16, %17, %18, %2, %29, %32 {xla_shape = "(pred[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, /*index=5*/f32[20,20]{1,0}, f32[20,20]{1,0})"} : tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>
    return %33 : tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>
  }
  func.func private @or.208(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @igamma_condition.212(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<i1> {
    %0 = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %1 = stablehlo.get_tuple_element %arg0[2] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %2 = stablehlo.get_tuple_element %arg0[3] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %3 = stablehlo.get_tuple_element %arg0[4] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %4 = stablehlo.get_tuple_element %arg0[5] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %5 = stablehlo.get_tuple_element %arg0[6] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %6 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xi1>
    %7 = stablehlo.constant dense<false> : tensor<i1>
    %8 = stablehlo.reduce(%6 init: %7) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
     reducer(%arg1: tensor<i1>, %arg2: tensor<i1>)  {
      %9 = stablehlo.or %arg1, %arg2 : tensor<i1>
      stablehlo.return %9 : tensor<i1>
    }
    return %8 : tensor<i1>
  }
  func.func private @igammac_body.265(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>> {
    %0 = stablehlo.get_tuple_element %arg0[7] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %1 = stablehlo.get_tuple_element %arg0[4] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %2 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %3 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
    %4 = stablehlo.add %1, %3 : tensor<20x20xf32>
    %5 = stablehlo.multiply %0, %4 : tensor<20x20xf32>
    %6 = stablehlo.get_tuple_element %arg0[9] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %7 = stablehlo.get_tuple_element %arg0[3] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %9 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %10 = stablehlo.add %7, %9 : tensor<20x20xf32>
    %11 = stablehlo.get_tuple_element %arg0[5] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<f32>
    %12 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %13 = stablehlo.add %11, %12 : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %15 = stablehlo.multiply %10, %14 : tensor<20x20xf32>
    %16 = stablehlo.multiply %6, %15 : tensor<20x20xf32>
    %17 = stablehlo.subtract %5, %16 : tensor<20x20xf32>
    %18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %19 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %20 = stablehlo.compare  NE, %17, %19 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %21 = stablehlo.get_tuple_element %arg0[12] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %22 = stablehlo.multiply %21, %4 : tensor<20x20xf32>
    %23 = stablehlo.get_tuple_element %arg0[6] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %24 = stablehlo.subtract %22, %23 : tensor<20x20xf32>
    %25 = stablehlo.get_tuple_element %arg0[10] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %26 = stablehlo.multiply %25, %15 : tensor<20x20xf32>
    %27 = stablehlo.subtract %24, %26 : tensor<20x20xf32>
    %28 = stablehlo.get_tuple_element %arg0[8] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %29 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %30 = stablehlo.multiply %28, %29 : tensor<20x20xf32>
    %31 = stablehlo.add %27, %30 : tensor<20x20xf32>
    %32 = stablehlo.multiply %23, %4 : tensor<20x20xf32>
    %33 = stablehlo.multiply %28, %15 : tensor<20x20xf32>
    %34 = stablehlo.subtract %32, %33 : tensor<20x20xf32>
    %35 = stablehlo.divide %34, %17 : tensor<20x20xf32>
    %36 = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %37 = stablehlo.select %20, %35, %36 : tensor<20x20xi1>, tensor<20x20xf32>
    %38 = stablehlo.get_tuple_element %arg0[13] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %39 = stablehlo.multiply %38, %4 : tensor<20x20xf32>
    %40 = stablehlo.subtract %39, %0 : tensor<20x20xf32>
    %41 = stablehlo.get_tuple_element %arg0[11] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %42 = stablehlo.multiply %41, %15 : tensor<20x20xf32>
    %43 = stablehlo.subtract %40, %42 : tensor<20x20xf32>
    %44 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %45 = stablehlo.multiply %6, %44 : tensor<20x20xf32>
    %46 = stablehlo.add %43, %45 : tensor<20x20xf32>
    %47 = stablehlo.multiply %37, %46 : tensor<20x20xf32>
    %48 = stablehlo.subtract %31, %47 : tensor<20x20xf32>
    %49 = stablehlo.divide %48, %17 : tensor<20x20xf32>
    %50 = stablehlo.get_tuple_element %arg0[14] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %51 = stablehlo.select %20, %49, %50 : tensor<20x20xi1>, tensor<20x20xf32>
    %52 = stablehlo.subtract %51, %50 : tensor<20x20xf32>
    %53 = stablehlo.abs %52 : tensor<20x20xf32>
    %54 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %55 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %56 = stablehlo.select %20, %53, %55 : tensor<20x20xi1>, tensor<20x20xf32>
    %57 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xi1>
    %58 = stablehlo.subtract %36, %35 : tensor<20x20xf32>
    %59 = stablehlo.divide %58, %35 : tensor<20x20xf32>
    %60 = stablehlo.abs %59 : tensor<20x20xf32>
    %61 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %62 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %63 = stablehlo.select %20, %60, %62 : tensor<20x20xi1>, tensor<20x20xf32>
    %64 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %65 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %66 = stablehlo.compare  GT, %63, %65 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %67 = stablehlo.and %57, %66 : tensor<20x20xi1>
    %68 = stablehlo.select %57, %37, %36 : tensor<20x20xi1>, tensor<20x20xf32>
    %69 = stablehlo.get_tuple_element %arg0[2] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %70 = stablehlo.select %57, %63, %69 : tensor<20x20xi1>, tensor<20x20xf32>
    %71 = stablehlo.select %57, %10, %7 : tensor<20x20xi1>, tensor<20x20xf32>
    %72 = stablehlo.select %57, %4, %1 : tensor<20x20xi1>, tensor<20x20xf32>
    %73 = stablehlo.abs %34 : tensor<20x20xf32>
    %74 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %75 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %76 = stablehlo.constant dense<0x4B000000> : tensor<f32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %78 = stablehlo.compare  GT, %73, %77 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %79 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %80 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %81 = stablehlo.multiply %34, %80 : tensor<20x20xf32>
    %82 = stablehlo.select %78, %81, %34 : tensor<20x20xi1>, tensor<20x20xf32>
    %83 = stablehlo.select %57, %82, %23 : tensor<20x20xi1>, tensor<20x20xf32>
    %84 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %85 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %86 = stablehlo.multiply %17, %85 : tensor<20x20xf32>
    %87 = stablehlo.select %78, %86, %17 : tensor<20x20xi1>, tensor<20x20xf32>
    %88 = stablehlo.select %57, %87, %0 : tensor<20x20xi1>, tensor<20x20xf32>
    %89 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %90 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %91 = stablehlo.multiply %23, %90 : tensor<20x20xf32>
    %92 = stablehlo.select %78, %91, %23 : tensor<20x20xi1>, tensor<20x20xf32>
    %93 = stablehlo.select %57, %92, %28 : tensor<20x20xi1>, tensor<20x20xf32>
    %94 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %95 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %96 = stablehlo.multiply %0, %95 : tensor<20x20xf32>
    %97 = stablehlo.select %78, %96, %0 : tensor<20x20xi1>, tensor<20x20xf32>
    %98 = stablehlo.select %57, %97, %6 : tensor<20x20xi1>, tensor<20x20xf32>
    %99 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %100 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %101 = stablehlo.multiply %21, %100 : tensor<20x20xf32>
    %102 = stablehlo.select %78, %101, %21 : tensor<20x20xi1>, tensor<20x20xf32>
    %103 = stablehlo.select %57, %102, %25 : tensor<20x20xi1>, tensor<20x20xf32>
    %104 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %105 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %106 = stablehlo.multiply %38, %105 : tensor<20x20xf32>
    %107 = stablehlo.select %78, %106, %38 : tensor<20x20xi1>, tensor<20x20xf32>
    %108 = stablehlo.select %57, %107, %41 : tensor<20x20xi1>, tensor<20x20xf32>
    %109 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %110 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %111 = stablehlo.multiply %31, %110 : tensor<20x20xf32>
    %112 = stablehlo.select %78, %111, %31 : tensor<20x20xi1>, tensor<20x20xf32>
    %113 = stablehlo.select %57, %112, %21 : tensor<20x20xi1>, tensor<20x20xf32>
    %114 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %115 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %116 = stablehlo.multiply %46, %115 : tensor<20x20xf32>
    %117 = stablehlo.select %78, %116, %46 : tensor<20x20xi1>, tensor<20x20xf32>
    %118 = stablehlo.select %57, %117, %38 : tensor<20x20xi1>, tensor<20x20xf32>
    %119 = stablehlo.select %57, %51, %50 : tensor<20x20xi1>, tensor<20x20xf32>
    %120 = stablehlo.tuple %67, %68, %70, %71, %72, %13, %83, %88, %93, %98, %103, %108, %113, %118, %119 {xla_shape = "(pred[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, /*index=5*/f32[], f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, /*index=10*/f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0})"} : tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>
    return %120 : tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>
  }
  func.func private @or.388(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @igammac_condition.392(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<i1> {
    %0 = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %1 = stablehlo.get_tuple_element %arg0[2] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %2 = stablehlo.get_tuple_element %arg0[3] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %3 = stablehlo.get_tuple_element %arg0[4] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %4 = stablehlo.get_tuple_element %arg0[6] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %5 = stablehlo.get_tuple_element %arg0[7] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %6 = stablehlo.get_tuple_element %arg0[8] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %7 = stablehlo.get_tuple_element %arg0[9] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %8 = stablehlo.get_tuple_element %arg0[10] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %9 = stablehlo.get_tuple_element %arg0[11] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %10 = stablehlo.get_tuple_element %arg0[12] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %11 = stablehlo.get_tuple_element %arg0[13] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %12 = stablehlo.get_tuple_element %arg0[14] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %13 = stablehlo.get_tuple_element %arg0[5] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<f32>
    %14 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
    %15 = stablehlo.compare  LT, %13, %14 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %16 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xi1>
    %17 = stablehlo.constant dense<false> : tensor<i1>
    %18 = stablehlo.reduce(%16 init: %17) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
     reducer(%arg1: tensor<i1>, %arg2: tensor<i1>)  {
      %20 = stablehlo.or %arg1, %arg2 : tensor<i1>
      stablehlo.return %20 : tensor<i1>
    }
    %19 = stablehlo.and %15, %18 : tensor<i1>
    return %19 : tensor<i1>
  }
  func.func private @xla_fallback_igammac(%arg0: tensor<20x20xf16>, %arg1: tensor<20x20xf16>) -> tensor<20x20xf16> {
    %0 = stablehlo.convert %arg1 : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %3 = stablehlo.compare  LE, %0, %2 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %4 = stablehlo.convert %arg0 : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %7 = stablehlo.compare  LE, %4, %6 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %8 = stablehlo.or %3, %7 : tensor<20x20xi1>
    %9 = stablehlo.log %0 : tensor<20x20xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<20x20xf32>
    %11 = stablehlo.subtract %10, %0 : tensor<20x20xf32>
    %12 = stablehlo.abs %4 : tensor<20x20xf32>
    %13 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %14 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %15 = stablehlo.compare  EQ, %12, %14 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %16 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %17 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %18 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %19 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %20 = stablehlo.compare  LT, %4, %19 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %21 = stablehlo.constant dense<3.14159274> : tensor<f32>
    %22 = stablehlo.constant dense<3.14159274> : tensor<20x20xf32>
    %23 = stablehlo.abs %4 : tensor<20x20xf32>
    %24 = stablehlo.floor %23 : tensor<20x20xf32>
    %25 = stablehlo.subtract %23, %24 : tensor<20x20xf32>
    %26 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %27 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %28 = stablehlo.compare  GT, %25, %27 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %29 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %30 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %31 = stablehlo.subtract %30, %25 : tensor<20x20xf32>
    %32 = stablehlo.select %28, %31, %25 : tensor<20x20xi1>, tensor<20x20xf32>
    %33 = stablehlo.multiply %22, %32 : tensor<20x20xf32>
    %34 = stablehlo.sine %33 : tensor<20x20xf32>
    %35 = stablehlo.log %34 : tensor<20x20xf32>
    %36 = stablehlo.is_finite %35 : (tensor<20x20xf32>) -> tensor<20x20xi1>
    %37 = stablehlo.constant dense<1.14472985> : tensor<f32>
    %38 = stablehlo.constant dense<1.14472985> : tensor<20x20xf32>
    %39 = stablehlo.subtract %38, %35 : tensor<20x20xf32>
    %40 = stablehlo.constant dense<0.918938517> : tensor<f32>
    %41 = stablehlo.constant dense<0.918938517> : tensor<20x20xf32>
    %42 = stablehlo.negate %4 : tensor<20x20xf32>
    %43 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %44 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %45 = stablehlo.subtract %4, %44 : tensor<20x20xf32>
    %46 = stablehlo.select %20, %42, %45 : tensor<20x20xi1>, tensor<20x20xf32>
    %47 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %48 = stablehlo.add %46, %47 : tensor<20x20xf32>
    %49 = stablehlo.constant dense<7.500000e+00> : tensor<f32>
    %50 = stablehlo.constant dense<7.500000e+00> : tensor<20x20xf32>
    %51 = stablehlo.add %50, %46 : tensor<20x20xf32>
    %52 = stablehlo.constant dense<2.01490307> : tensor<f32>
    %53 = stablehlo.constant dense<2.01490307> : tensor<20x20xf32>
    %54 = stablehlo.constant dense<7.500000e+00> : tensor<20x20xf32>
    %55 = stablehlo.divide %46, %54 : tensor<20x20xf32>
    %56 = stablehlo.log_plus_one %55 : tensor<20x20xf32>
    %57 = stablehlo.add %53, %56 : tensor<20x20xf32>
    %58 = stablehlo.divide %51, %57 : tensor<20x20xf32>
    %59 = stablehlo.subtract %48, %58 : tensor<20x20xf32>
    %60 = stablehlo.multiply %59, %57 : tensor<20x20xf32>
    %61 = stablehlo.add %41, %60 : tensor<20x20xf32>
    %62 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %63 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %64 = stablehlo.constant dense<676.520386> : tensor<f32>
    %65 = stablehlo.constant dense<676.520386> : tensor<20x20xf32>
    %66 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %67 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %68 = stablehlo.add %46, %67 : tensor<20x20xf32>
    %69 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %70 = stablehlo.add %68, %69 : tensor<20x20xf32>
    %71 = stablehlo.divide %65, %70 : tensor<20x20xf32>
    %72 = stablehlo.add %63, %71 : tensor<20x20xf32>
    %73 = stablehlo.constant dense<-1259.13916> : tensor<f32>
    %74 = stablehlo.constant dense<-1259.13916> : tensor<20x20xf32>
    %75 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %76 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %77 = stablehlo.add %46, %76 : tensor<20x20xf32>
    %78 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %79 = stablehlo.add %77, %78 : tensor<20x20xf32>
    %80 = stablehlo.divide %74, %79 : tensor<20x20xf32>
    %81 = stablehlo.add %72, %80 : tensor<20x20xf32>
    %82 = stablehlo.constant dense<771.323425> : tensor<f32>
    %83 = stablehlo.constant dense<771.323425> : tensor<20x20xf32>
    %84 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %85 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
    %86 = stablehlo.add %46, %85 : tensor<20x20xf32>
    %87 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %88 = stablehlo.add %86, %87 : tensor<20x20xf32>
    %89 = stablehlo.divide %83, %88 : tensor<20x20xf32>
    %90 = stablehlo.add %81, %89 : tensor<20x20xf32>
    %91 = stablehlo.constant dense<-176.615036> : tensor<f32>
    %92 = stablehlo.constant dense<-176.615036> : tensor<20x20xf32>
    %93 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %94 = stablehlo.constant dense<3.000000e+00> : tensor<20x20xf32>
    %95 = stablehlo.add %46, %94 : tensor<20x20xf32>
    %96 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %97 = stablehlo.add %95, %96 : tensor<20x20xf32>
    %98 = stablehlo.divide %92, %97 : tensor<20x20xf32>
    %99 = stablehlo.add %90, %98 : tensor<20x20xf32>
    %100 = stablehlo.constant dense<12.5073433> : tensor<f32>
    %101 = stablehlo.constant dense<12.5073433> : tensor<20x20xf32>
    %102 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %103 = stablehlo.constant dense<4.000000e+00> : tensor<20x20xf32>
    %104 = stablehlo.add %46, %103 : tensor<20x20xf32>
    %105 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %106 = stablehlo.add %104, %105 : tensor<20x20xf32>
    %107 = stablehlo.divide %101, %106 : tensor<20x20xf32>
    %108 = stablehlo.add %99, %107 : tensor<20x20xf32>
    %109 = stablehlo.constant dense<-0.138571098> : tensor<f32>
    %110 = stablehlo.constant dense<-0.138571098> : tensor<20x20xf32>
    %111 = stablehlo.constant dense<5.000000e+00> : tensor<f32>
    %112 = stablehlo.constant dense<5.000000e+00> : tensor<20x20xf32>
    %113 = stablehlo.add %46, %112 : tensor<20x20xf32>
    %114 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %115 = stablehlo.add %113, %114 : tensor<20x20xf32>
    %116 = stablehlo.divide %110, %115 : tensor<20x20xf32>
    %117 = stablehlo.add %108, %116 : tensor<20x20xf32>
    %118 = stablehlo.constant dense<9.98436917E-6> : tensor<f32>
    %119 = stablehlo.constant dense<9.98436917E-6> : tensor<20x20xf32>
    %120 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %121 = stablehlo.constant dense<6.000000e+00> : tensor<20x20xf32>
    %122 = stablehlo.add %46, %121 : tensor<20x20xf32>
    %123 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %124 = stablehlo.add %122, %123 : tensor<20x20xf32>
    %125 = stablehlo.divide %119, %124 : tensor<20x20xf32>
    %126 = stablehlo.add %117, %125 : tensor<20x20xf32>
    %127 = stablehlo.constant dense<1.50563267E-7> : tensor<f32>
    %128 = stablehlo.constant dense<1.50563267E-7> : tensor<20x20xf32>
    %129 = stablehlo.constant dense<7.000000e+00> : tensor<f32>
    %130 = stablehlo.constant dense<7.000000e+00> : tensor<20x20xf32>
    %131 = stablehlo.add %46, %130 : tensor<20x20xf32>
    %132 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %133 = stablehlo.add %131, %132 : tensor<20x20xf32>
    %134 = stablehlo.divide %128, %133 : tensor<20x20xf32>
    %135 = stablehlo.add %126, %134 : tensor<20x20xf32>
    %136 = stablehlo.log %135 : tensor<20x20xf32>
    %137 = stablehlo.add %61, %136 : tensor<20x20xf32>
    %138 = stablehlo.subtract %39, %137 : tensor<20x20xf32>
    %139 = stablehlo.negate %35 : tensor<20x20xf32>
    %140 = stablehlo.select %36, %138, %139 : tensor<20x20xi1>, tensor<20x20xf32>
    %141 = stablehlo.select %20, %140, %137 : tensor<20x20xi1>, tensor<20x20xf32>
    %142 = stablehlo.select %15, %17, %141 : tensor<20x20xi1>, tensor<20x20xf32>
    %143 = stablehlo.subtract %11, %142 : tensor<20x20xf32>
    %144 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
    %145 = stablehlo.constant dense<88.7228394> : tensor<f32>
    %146 = stablehlo.negate %145 : tensor<f32>
    %147 = stablehlo.broadcast_in_dim %146, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %148 = stablehlo.compare  LT, %143, %147 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %149 = stablehlo.or %8, %148 : tensor<20x20xi1>
    %150 = stablehlo.not %149 : tensor<20x20xi1>
    %151 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %152 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %153 = stablehlo.compare  LT, %0, %152 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %154 = stablehlo.compare  LT, %0, %4 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %155 = stablehlo.or %153, %154 : tensor<20x20xi1>
    %156 = stablehlo.and %150, %155 : tensor<20x20xi1>
    %157 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %158 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %159 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %160 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %161 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %162 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %163 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %164 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %165:7 = stablehlo.while(%iterArg = %156, %iterArg_0 = %4, %iterArg_1 = %158, %iterArg_2 = %160, %iterArg_3 = %0, %iterArg_4 = %162, %iterArg_5 = %164) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %214 = stablehlo.constant dense<false> : tensor<i1>
      %215 = stablehlo.reduce(%iterArg init: %214) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %216 = stablehlo.or %arg2, %arg3 : tensor<i1>
        stablehlo.return %216 : tensor<i1>
      }
      stablehlo.return %215 : tensor<i1>
    } do {
      %214 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %215 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %216 = stablehlo.add %iterArg_0, %215 : tensor<20x20xf32>
      %217 = stablehlo.divide %iterArg_3, %216 : tensor<20x20xf32>
      %218 = stablehlo.multiply %iterArg_1, %217 : tensor<20x20xf32>
      %219 = stablehlo.add %iterArg_2, %218 : tensor<20x20xf32>
      %220 = stablehlo.divide %218, %219 : tensor<20x20xf32>
      %221 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %222 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %223 = stablehlo.compare  GT, %220, %222 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %224 = stablehlo.and %iterArg, %223 : tensor<20x20xi1>
      %225 = stablehlo.select %iterArg, %216, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %226 = stablehlo.select %iterArg, %218, %iterArg_1 : tensor<20x20xi1>, tensor<20x20xf32>
      %227 = stablehlo.select %iterArg, %219, %iterArg_2 : tensor<20x20xi1>, tensor<20x20xf32>
      %228 = stablehlo.divide %iterArg_3, %216 : tensor<20x20xf32>
      %229 = stablehlo.multiply %iterArg_4, %228 : tensor<20x20xf32>
      %230 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
      %231 = stablehlo.constant dense<-1.000000e+00> : tensor<20x20xf32>
      %232 = stablehlo.multiply %231, %iterArg_1 : tensor<20x20xf32>
      %233 = stablehlo.multiply %232, %iterArg_3 : tensor<20x20xf32>
      %234 = stablehlo.multiply %216, %216 : tensor<20x20xf32>
      %235 = stablehlo.divide %233, %234 : tensor<20x20xf32>
      %236 = stablehlo.add %229, %235 : tensor<20x20xf32>
      %237 = stablehlo.select %iterArg, %236, %iterArg_4 : tensor<20x20xi1>, tensor<20x20xf32>
      %238 = stablehlo.add %iterArg_5, %236 : tensor<20x20xf32>
      %239 = stablehlo.select %iterArg, %238, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %224, %225, %226, %227, %iterArg_3, %237, %239 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %166 = stablehlo.not %155 : tensor<20x20xi1>
    %167 = stablehlo.and %150, %166 : tensor<20x20xi1>
    %168 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %169 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %170 = stablehlo.add %0, %169 : tensor<20x20xf32>
    %171 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %172 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %173 = stablehlo.subtract %172, %4 : tensor<20x20xf32>
    %174 = stablehlo.add %0, %173 : tensor<20x20xf32>
    %175 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %176 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %177 = stablehlo.add %174, %176 : tensor<20x20xf32>
    %178 = stablehlo.multiply %177, %0 : tensor<20x20xf32>
    %179 = stablehlo.divide %170, %178 : tensor<20x20xf32>
    %180 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %181 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %182 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %183 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %184 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %185 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %186 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %187 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %188 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %189 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %190 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %191 = stablehlo.negate %0 : tensor<20x20xf32>
    %192 = stablehlo.multiply %179, %191 : tensor<20x20xf32>
    %193 = stablehlo.subtract %190, %192 : tensor<20x20xf32>
    %194 = stablehlo.divide %193, %178 : tensor<20x20xf32>
    %195:15 = stablehlo.while(%iterArg = %167, %iterArg_0 = %179, %iterArg_1 = %181, %iterArg_2 = %173, %iterArg_3 = %177, %iterArg_4 = %182, %iterArg_5 = %170, %iterArg_6 = %178, %iterArg_7 = %184, %iterArg_8 = %0, %iterArg_9 = %186, %iterArg_10 = %188, %iterArg_11 = %190, %iterArg_12 = %191, %iterArg_13 = %194) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %214 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
      %215 = stablehlo.compare  LT, %iterArg_4, %214 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %216 = stablehlo.constant dense<false> : tensor<i1>
      %217 = stablehlo.reduce(%iterArg init: %216) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %219 = stablehlo.or %arg2, %arg3 : tensor<i1>
        stablehlo.return %219 : tensor<i1>
      }
      %218 = stablehlo.and %215, %217 : tensor<i1>
      stablehlo.return %218 : tensor<i1>
    } do {
      %214 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %215 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
      %216 = stablehlo.add %iterArg_3, %215 : tensor<20x20xf32>
      %217 = stablehlo.multiply %iterArg_6, %216 : tensor<20x20xf32>
      %218 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %219 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %220 = stablehlo.add %iterArg_2, %219 : tensor<20x20xf32>
      %221 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %222 = stablehlo.add %iterArg_4, %221 : tensor<f32>
      %223 = stablehlo.broadcast_in_dim %222, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %224 = stablehlo.multiply %220, %223 : tensor<20x20xf32>
      %225 = stablehlo.multiply %iterArg_8, %224 : tensor<20x20xf32>
      %226 = stablehlo.subtract %217, %225 : tensor<20x20xf32>
      %227 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %228 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
      %229 = stablehlo.compare  NE, %226, %228 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %230 = stablehlo.multiply %iterArg_11, %216 : tensor<20x20xf32>
      %231 = stablehlo.subtract %230, %iterArg_5 : tensor<20x20xf32>
      %232 = stablehlo.multiply %iterArg_9, %224 : tensor<20x20xf32>
      %233 = stablehlo.subtract %231, %232 : tensor<20x20xf32>
      %234 = stablehlo.broadcast_in_dim %222, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %235 = stablehlo.multiply %iterArg_7, %234 : tensor<20x20xf32>
      %236 = stablehlo.add %233, %235 : tensor<20x20xf32>
      %237 = stablehlo.multiply %iterArg_5, %216 : tensor<20x20xf32>
      %238 = stablehlo.multiply %iterArg_7, %224 : tensor<20x20xf32>
      %239 = stablehlo.subtract %237, %238 : tensor<20x20xf32>
      %240 = stablehlo.divide %239, %226 : tensor<20x20xf32>
      %241 = stablehlo.select %229, %240, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %242 = stablehlo.multiply %iterArg_12, %216 : tensor<20x20xf32>
      %243 = stablehlo.subtract %242, %iterArg_6 : tensor<20x20xf32>
      %244 = stablehlo.multiply %iterArg_10, %224 : tensor<20x20xf32>
      %245 = stablehlo.subtract %243, %244 : tensor<20x20xf32>
      %246 = stablehlo.broadcast_in_dim %222, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %247 = stablehlo.multiply %iterArg_8, %246 : tensor<20x20xf32>
      %248 = stablehlo.add %245, %247 : tensor<20x20xf32>
      %249 = stablehlo.multiply %241, %248 : tensor<20x20xf32>
      %250 = stablehlo.subtract %236, %249 : tensor<20x20xf32>
      %251 = stablehlo.divide %250, %226 : tensor<20x20xf32>
      %252 = stablehlo.select %229, %251, %iterArg_13 : tensor<20x20xi1>, tensor<20x20xf32>
      %253 = stablehlo.subtract %252, %iterArg_13 : tensor<20x20xf32>
      %254 = stablehlo.abs %253 : tensor<20x20xf32>
      %255 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %256 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %257 = stablehlo.select %229, %254, %256 : tensor<20x20xi1>, tensor<20x20xf32>
      %258 = stablehlo.subtract %iterArg_0, %240 : tensor<20x20xf32>
      %259 = stablehlo.divide %258, %240 : tensor<20x20xf32>
      %260 = stablehlo.abs %259 : tensor<20x20xf32>
      %261 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %262 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %263 = stablehlo.select %229, %260, %262 : tensor<20x20xi1>, tensor<20x20xf32>
      %264 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %265 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %266 = stablehlo.compare  GT, %263, %265 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %267 = stablehlo.and %iterArg, %266 : tensor<20x20xi1>
      %268 = stablehlo.select %iterArg, %241, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %269 = stablehlo.select %iterArg, %263, %iterArg_1 : tensor<20x20xi1>, tensor<20x20xf32>
      %270 = stablehlo.select %iterArg, %220, %iterArg_2 : tensor<20x20xi1>, tensor<20x20xf32>
      %271 = stablehlo.select %iterArg, %216, %iterArg_3 : tensor<20x20xi1>, tensor<20x20xf32>
      %272 = stablehlo.abs %239 : tensor<20x20xf32>
      %273 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %274 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %275 = stablehlo.constant dense<0x4B000000> : tensor<f32>
      %276 = stablehlo.broadcast_in_dim %275, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %277 = stablehlo.compare  GT, %272, %276 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %278 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %279 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %280 = stablehlo.multiply %239, %279 : tensor<20x20xf32>
      %281 = stablehlo.select %277, %280, %239 : tensor<20x20xi1>, tensor<20x20xf32>
      %282 = stablehlo.select %iterArg, %281, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      %283 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %284 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %285 = stablehlo.multiply %226, %284 : tensor<20x20xf32>
      %286 = stablehlo.select %277, %285, %226 : tensor<20x20xi1>, tensor<20x20xf32>
      %287 = stablehlo.select %iterArg, %286, %iterArg_6 : tensor<20x20xi1>, tensor<20x20xf32>
      %288 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %289 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %290 = stablehlo.multiply %iterArg_5, %289 : tensor<20x20xf32>
      %291 = stablehlo.select %277, %290, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      %292 = stablehlo.select %iterArg, %291, %iterArg_7 : tensor<20x20xi1>, tensor<20x20xf32>
      %293 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %294 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %295 = stablehlo.multiply %iterArg_6, %294 : tensor<20x20xf32>
      %296 = stablehlo.select %277, %295, %iterArg_6 : tensor<20x20xi1>, tensor<20x20xf32>
      %297 = stablehlo.select %iterArg, %296, %iterArg_8 : tensor<20x20xi1>, tensor<20x20xf32>
      %298 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %299 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %300 = stablehlo.multiply %iterArg_11, %299 : tensor<20x20xf32>
      %301 = stablehlo.select %277, %300, %iterArg_11 : tensor<20x20xi1>, tensor<20x20xf32>
      %302 = stablehlo.select %iterArg, %301, %iterArg_9 : tensor<20x20xi1>, tensor<20x20xf32>
      %303 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %304 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %305 = stablehlo.multiply %iterArg_12, %304 : tensor<20x20xf32>
      %306 = stablehlo.select %277, %305, %iterArg_12 : tensor<20x20xi1>, tensor<20x20xf32>
      %307 = stablehlo.select %iterArg, %306, %iterArg_10 : tensor<20x20xi1>, tensor<20x20xf32>
      %308 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %309 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %310 = stablehlo.multiply %236, %309 : tensor<20x20xf32>
      %311 = stablehlo.select %277, %310, %236 : tensor<20x20xi1>, tensor<20x20xf32>
      %312 = stablehlo.select %iterArg, %311, %iterArg_11 : tensor<20x20xi1>, tensor<20x20xf32>
      %313 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %314 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %315 = stablehlo.multiply %248, %314 : tensor<20x20xf32>
      %316 = stablehlo.select %277, %315, %248 : tensor<20x20xi1>, tensor<20x20xf32>
      %317 = stablehlo.select %iterArg, %316, %iterArg_12 : tensor<20x20xi1>, tensor<20x20xf32>
      %318 = stablehlo.select %iterArg, %252, %iterArg_13 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %267, %268, %269, %270, %271, %222, %282, %287, %292, %297, %302, %307, %312, %317, %318 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %196 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %197 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %198 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %199 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %200 = stablehlo.compare  EQ, %0, %199 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %201 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %202 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %203 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %204 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %205 = stablehlo.exponential %143 : tensor<20x20xf32>
    %206 = stablehlo.multiply %165#3, %205 : tensor<20x20xf32>
    %207 = stablehlo.divide %206, %4 : tensor<20x20xf32>
    %208 = stablehlo.subtract %204, %207 : tensor<20x20xf32>
    %209 = stablehlo.multiply %195#1, %205 : tensor<20x20xf32>
    %210 = stablehlo.select %155, %208, %209 : tensor<20x20xi1>, tensor<20x20xf32>
    %211 = stablehlo.select %200, %202, %210 : tensor<20x20xi1>, tensor<20x20xf32>
    %212 = stablehlo.select %8, %197, %211 : tensor<20x20xi1>, tensor<20x20xf32>
    %213 = stablehlo.convert %212 : (tensor<20x20xf32>) -> tensor<20x20xf16>
    return %213 : tensor<20x20xf16>
  }
}
