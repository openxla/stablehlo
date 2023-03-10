// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xbf16>, tensor<20x20xbf16>)
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = call @igammac(%0#0, %0#1) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> tensor<20x20xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xbf16>, tensor<20x20xbf16>) {
    %0 = stablehlo.constant dense<"0xF24092401C404CBFC03F38C0433F1BC060C04DBF13C0A5C0B240E1BF18BFCA40A3BF85C0D8BF22C02F4017C0873F1FBF86BFAA4053BFA83E13BE824005BEAABF36C0853F053E1C3F854050C0A5C095BF0ABF2D4021C0B44025C03CC005BF0DBD8B4008409E3F02BF074002BF293F4EBF9D3F143F26C0D4BE69404FC00DC049C039402740763F604048C00FBFC9BEBABF1FC08D3F393F00BF6B400F3FA84057BF0C3F4BC0AC4010C0E340F93F813ED13F1FC001BE0BBF633F24C0953E03C135BF983FB24051C086BFAF406540314047C0DD408C40ED3E494091BE54406F40B6BFA6BF3CC088BD03BFF7BE3BC032C0354000418D4049C099BFC640A3C084C08D406DC0294008C0E8BF22C082C00A40953FA5BF3DBFF3C042C0F53E9ABF0F402DC03EC043BF2FC0D9C01440924009BFD040E6BFFEBE5FC0AEBE673E5540C4C02CC0CC3DE0BF72C0F53F0A40C740D5BF913EBBBF463D84C0DC4097BEAA408940443F4540CABECFBE863E19BF2F409CBF7640C9BF0EC1DC3FB74098C02B4092BFD33FAEBF863F9AC09CBF533FB3BE9BBFF6BD7A40693E79BF8DBE9BBE024071BF22C02C40EA3F1240B5C0C63FD54026BF833F26BFA84092C0E440BABFA1C06AC01DBF9CC0A7BF08C0A9BFCF3FCFBFEF3E60C0564039BFAC40554025BF17C00EC02DC0344054404FBFFDBED9BF92C02B41043E04408DBF3E3F9B3EF03D0EC0CDC0664054C03FC025BF2C40A7C0E33FAB408F4009C1A1C0184084C01DC08CC07CC0E0BF9CBFAAC0E73F08C06EBF25C0653F8ABF04BFD4BDEA3FF7BF154089C073C02140F13F9B401A4076407040D2BF60408EBFC23FF03F1E4051C0953F9FBF3F40B8BF83BF1EBE59C01AC09740EEBECDBE48BF66BFF540FF3FFB3F5C4087BF4A409C408DC051C013C069BF63C095BFC33F63BE25C0CCC0BB3EA2C07440BD40E33F2D4047403EC0B53FABBF4740193EACBF8ABF81405D3F0E400BC02E40A5405BC0F3BFBE3F8E401E3EA3C018C090C002C075BFE13FEC3E05BE993F573FAABCAA3D57C05E40D3BF36C046C08BBF54C0D63F743FDFBFCEBF5ABF413FB74085BF8CC0FC3F59BF6EC01A3F83C0233EDABFA0C0FDBFA9C0943F3D3F6BBFAAC041406740F43E"> : tensor<20x20xbf16>
    %1 = stablehlo.constant dense<"0xDB3FE13F943F4BC07DC088BEBDC0DB3DCCBFA94084BF9BBFC83F5BBE674030BF2A3F40C0364023C0EF40A13E89C085BF0CBE3DBF5BC0ECC0CB40703F66C075BF00BFAD3FC93D0DC0C13F3740D7402E40BA40CABFDCBFE73F29C0D53F8A3F90C03F3F32C0C13FA93F4DC015C05DC085C08D3D774060C0CFC0AE4064C001407EC08B4097BF5E401A409E40004049BD9DBFAF40C0BF4DBF65C01940504039BF32C076409C405CBF9FC0D73F0E403ABE9F40DFC019BFA73D0BC041C028C089C01DC0ADC00DC001BFA9BF99C0C63F4E40373F4EC0D9BD0B4146C0E93E0CC000401240F4BF973F19401A3F9E3F0CC0434088C008C058BE68BFA4405640C1BF2FBFA3C046C004C09F403E4095BF9D3F3CC03B3FECBF36C0C3C0F2C0D7BF56C0104047BD8E407E3F0F40983FDABFE140913E5F40A540C84096BF3B401A4064BFE3BF423FB43EB43F86BFD1407ABFAEBE89C0983F853F4EBF9D401B409EC0EEBE4FC06C3ECABF06C004C06D3F07C042C0343FA03F1240803F1FBF06BFDABE0940C9BF93403340D03F3C40C0BFEEBDC8BE7CBF64BF883F51BF5CBFAAC08B3F7C40BCBB0D40A6BF9E3FD4C086409D3F37C043C08CC0EFBFFFBFBB3E04C001405E3FAE3F34C0124088406640DA4069402FC0004097BF46C075400A4080404AC094C093C0493F2EC02B4081C0A93FDDBFAC40CFBF39C0C9BFB9405BC04F40EA3F1FC086C0DA3F0D3DEA3FE4BFC73FA13E2CC0F83F15C062BBBE40C0C050C030BFE440F23E85BFA1403ABE06C087BF9FC017BF7140A640F63D9CBF0E40BEBF1C408B4018C033BFD83FDD3F9CC082C0AC409B3F73408A40303FE93FE13EA8BF034058C086BFA43FD8BFDC40DFBF71C0ADBEBDBE5A4091BF4ABF51401DBFB5C004C0DD406E4031BF784055C01FC08340B0BE46C079C0B640E93FB7BFDC3FB8406140A93F723F90C0F93FBF3F25405BBFF9BF78C015BE9CBF5140B33F4FC0ECC07640F03FA5BD204081C01EC0BB40B5BF0FBE8A406BC0B2C085C02740393FC0BF7ABFD3BE7EBD3B40154049BFDC4013417540D4BD604063C042C07EC0A5C0823FB8BF37BF8F3F15405BBF1CC0EB3F5640DBBDB4408D3D79BFD8BFB53FFEC051BE6BC096BDAABF63C0"> : tensor<20x20xbf16>
    return %0, %1 : tensor<20x20xbf16>, tensor<20x20xbf16>
  }
  func.func private @expected() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0x803F723F4B3F803F803F803F803F803F803F803F803F803F7D3F803F803F803F803F803F803F803F713C803F803F803F803F803F803F803F803F7C3F803F803F803F8C3E633E803F723F803F803F803F803F803F803F7C3F803F803F803F803F7F3F803F993E803F803F803F803F803F783FE83B803F803F253E803F803F803F343E803FEB3C2F3F803F803F803F803F803F803F803F803F393F563C803F803FD63B803F803F803F803FAC3E803FC43C803F803F803F803F803F803F803F803F803F803F803F803F803F633FA63E803F803F803FDB37803F803F803F523F803F803F803F803F803F803F803F803F803F803F803F803F803F653F803F803F803F803F803F803F803F803F803F803F0F3F803F803F803F803F803F803FD03E803F803F803F803F803F803F013E803F673F803F803F803F803F143C803F803F803FA53D803F803F1C3C803F803F803F763D803F803F803F7C3F803F803F803F2E3F803F803F803FA43D803F803F803F743F803F803F803F803F803F0E3F803F0C3D803F583E803F803F803F803F803F803F793F803F803F803F803FCD3D803F803F803F193F803F803F003F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F9A3D803F283D803F803F803F743F993E803F803F803F803F803F133F803F803F803F803F803F803F803F803F803FA33B0D3C803F803F5C3F803F803F803F3B3F803F803F783F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F963C803F803F803F9B3E803FC63E803F803F803FEC3E763F803F803F3A3E803FBD3E803F373FD63E783F803F2A3E803F803F803F803F803F803F803F803F803F803F803F803F793F803F803F803F803FA13E803F803F803F803F803F803F803F803F803F803F803F283D803FFD3E7F3F313F803F373F803F133E803F803F803F803F803F183F4E3E803F803F583E773F803F803F803F803FC938803F803F803F803F803F803FA13C803F803F803F803F803F803F323F803F803F803F803F803FBB3D803F803F803F803F843E803F803F803FA23E803F803F953D803F803F803F803F803F803F983E803F803F803F803F803F803F"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @igammac(%arg0: tensor<20x20xbf16>, %arg1: tensor<20x20xbf16>) -> tensor<20x20xbf16> {
    %0 = call @xla_fallback_igammac(%arg0, %arg1) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
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
  func.func private @xla_fallback_igammac(%arg0: tensor<20x20xbf16>, %arg1: tensor<20x20xbf16>) -> tensor<20x20xbf16> {
    %0 = stablehlo.convert %arg1 : (tensor<20x20xbf16>) -> tensor<20x20xf32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %3 = stablehlo.compare  LE, %0, %2 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %4 = stablehlo.convert %arg0 : (tensor<20x20xbf16>) -> tensor<20x20xf32>
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
    %213 = stablehlo.convert %212 : (tensor<20x20xf32>) -> tensor<20x20xbf16>
    return %213 : tensor<20x20xbf16>
  }
}
