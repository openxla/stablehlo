// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x20xf32>, tensor<20x20xf32>)
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = call @igammac(%0#0, %0#1) : (tensor<1x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x20xf32>, tensor<20x20xf32>) {
    %0 = stablehlo.constant dense<[[-2.95355916, 0.790388584, -2.96554661, -5.026270e+00, -2.61630774, 0.975202798, -0.409306079, 2.53560352, 1.00893152, 0.434700698, -0.938411056, -4.01808834, -0.494305551, 3.76039696, -3.28323269, 2.60344315, 0.439905107, -2.17292476, -2.129350e+00, -0.895822942]]> : tensor<1x20xf32>
    %1 = stablehlo.constant dense<"0x583A89C0CE9378C02F9E0CC0377BA940142FF03F74A652BFE5126B403EEF83C0E19D3240C4DA714061BE70406AE3DE3EECB101C0EE933F3FFFAAAA40C36258C0B90CC83F0CF4803EE9CA69BF48A14B4083C85DC0352F1BC097B98E3F8DEDF1BF789C954037AC2C409C2411C0012DB1BE3BADF6BFA1B4C5C0D72E86404D3DC3BE554E17BFFE68883E3D2C80BFBDA8763DED0F2E3EFE68F93FD2172D40B561A63F5DF73440414966408AB169C0C6882340A362BE3FBB172540D5BE2D40B022913F3640C9BF7A21DA3F481081C08E6258C0FC7925C013E2B33F31E1BA40FD8546407CFF81BFC6EF05BF0A569EBE1BBAC63FC06614C0842F8440BFE12ABEAD8E74C00F65DF3F0E379C3F024E87C0C53A3FBF1BC1BE3FDF35703F5C384A40220EC7BF80925DC037C18D40A0774D3FD4A1783E953887BF9D410EC06126B53F734016C0DC18A9407E3FBD3E941EAEBF8EFE2A3F96421F401CCE8040F16178BF59D9EC3F85C580C09476A53EC55CB03F06EA483F66E703C06B5900407F36BB4028B88C40DBE3873FBA5922C09E17084154BDC9BEDB649EBF914830C0474FEEBFBE260440D67D36406FA78EC0D4FD31C0C4790EC0B6F752BF7E0D06406CBFB040E569513F428697C01820DF3FA8DA19405ABE2940BE2995BF8EDD90BF2513823F68EB813F0FE08A3F27B3C5BFFA15B5BF8F5107402D8294C0F50099BF405DA33F780E313F3854D33F90AF6CC036FD60C01E3BC0C049485F3E3F83314065ACA7401447DDBF5E3AE1400CE796409DB04C40F1C39A3FE0018740B10200C0AE4C53C0BA778BBF811192C0670E50405F8FD83F3372C040CD6F38BFD8D9F0C028056C3F82D1CF3F5FC25BC042F39FBE3389334015CF75BF67F49E3FFDBD57C0E1D0F3BF33436240464F8ABF4F09723F1F700D3EC709AA3F05BADEBE88AA3440540E8E40118BBDC0AA7458BD520DBC40486EBB3FE630813FD2F587C0D5549E407E02ACC032789BC00B5837C0C36A2CC02AFC2F40550B17C070C6A03FDFD48BC00FB101C03C68BAC0B942633F0AEAC7BE9D4F8DC0DF0002C03E900740F813B1BFDD0F783FB9DCC93F2F05DDBFA7C26FBF9DC080C0B65394C047D9F9BE09BE653F1180D7BE507B2E40D25D733F1A3A444097AE49C0CCC77EC087CA7740A5C49A3DC900A2C04BBB13C0AFDC09C0DDAC313E356576C0ED743E40CA4A4A40CF8C11401E3F8D405E5F2040E175A7BF3EE8A6C0660C04409CC0A5BF4725B93ED88960409BF8C03C996FE73F9CE90ABFA19E5CC06357D640872BB0C0B0D117BF3A80E1BD1A50CF3FC053F53F410ED13FC3DB98BFB2204DC0A490FDBCB53CB4408A464840E0611DBF14DDDCC0FF8916BFB873A74027C25EBE14E919BF06E73340010591BF36D7D84089C31CC0A1C607C1EAC63640D4E5D6C0EBCDD7BFF52A733F30D932C077A5A63FFD5742405E046BC0ACCD7A3E2E4D9E40C4ADAA4004A1ADBFB00C31C0CD7BC63F7D969A40C05D13C070FC87C0C580EA40B03951C0BBF062BF2704F8BC47A48EC0F9F3663F4368E53FFCCAA5BED8838C40442D163FC3F125BF5880B53FFDD17B3DC6563740672F28C0B72F3DC02FD2C140CF6B18BFB16F92C0D053EE3CC1925EC027700E40B5DD833FC32FC0BF6A988CC099D666401A60DBC0062ED1BF57795440ECD725BF7680A9C06ED9A040C7E836BF0580A6BEE21764C03A41FBBF1457AABF2C188E40E0C92D3EA37D71C0D8758F40F347D73ED36C1E3FCF624A40249C3C3FF7B11840E61BCC3FE6CCFBBF99E83AC0B6B2E4C05F8B2A3F07894C3FABF2BBBF6892B3BFAFF410BED8A44E401B9508C1D06F4840BD3B3DBE8E57913FCB665540983A88BF3BDCD8C0087EDF409DB7F33F9EDA48C023EC88401927D8BE46865EC03E2687BE9FE807409F9B4EBE1EFED1408D60654069A1B14041533140185D90C0E290F73FCC80AD3F7FF1C3C0DEF5EFBFE47C05C0EB96F4C0A6438240F99B9E40D9800840F005394026B30B3D8ECD93BF25D23640DF8A5BC0CD65A240C36489C07A8D873E9201B1BF779CECBF16013ABFE460D1C01EE6FD3F9573A2C06FD8D3C0AA841B3F034F84C0E20B2540BD246B3FC9F81DBEC4AA1F407F40903ECF0E83C0D7F96540C3F8D7BF517694BF74E443C0D54DBE3F6CADA8BEE636CCC0D811CABFEED8B73FFB944E3FFEDAEA3EFCF59940D3B979406CF8F03F34BD32BF0B082C40F5F6A540CB1EA8BDB65DB7BF02A7F73F5C3838406A2712408E4C8E3F9B99B5BF1BD08ABF"> : tensor<20x20xf32>
    return %0, %1 : tensor<1x20xf32>, tensor<20x20xf32>
  }
  func.func private @expected() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0x0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F169A7F3D77169A3B0000803F0000803F0000803F61257D3F0000803F0000803FB02C843D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FFFA4833D0000803F0000803F0000803F0000803F0000803F0000803F0000803F2AEA7F3F0000803FCFF47F3F7C02023F0000803F0000803F0000803F0000803F39438B3C0000803F0000803F0000803FE752943D0000803F4873513F0000803F45FF593D0000803F0000803F0000803F026D6D3F0000803F20A29E3E0000803F0000803F0000803F0000803F0000803FD6A81F3C0000803F0000803F0000803FC93E923E0000803F0000803F02AA693E3479133E0000803F0000803F0000803F077D9E3E0000803FBB8D7E3F0000803F0000803F0000803F0000803F0000803FFAC0143F0000803F0000803F0000803FCB8F8A3C0000803F2F571A3F0000803F00E1BD3E0000803F0000803F0000803F3172523F0000803FAE5E053EDE00FD3D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F7251073D0000803F0000803F0000803FC3755F3F0000803F42DDCF3E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F33196E3F8C17473E0000803F0000803F0000803F0000803FF65C263F0000803F0000803F73070F390000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FBEEB163D0000803FA561133D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F5F4AC73D0000803F0000803F0000803F0000803F2A04973E0000803F0000803F0000803F782C683D0000803F0000803F0000803F1778F4390000803F0000803F0000803F74606F3E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F44F7F93D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FD39DF33C0000803F0000803F0000803F6CEB6B3F0000803F0000803F0000803F5A96FF3E0000803F0000803F0000803F79B4433F0000803F3EFCE13E0000803F0000803F0000803F0000803F0000803F0102993C0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F5FDD223A0000803F0000803F0000803F0000803F676A4B3B0000803F0000803F0000803F0000803F0000803F0000803F0000803F6D9E5C3C0000803F0000803F0000803F0000803F0000803F742DA53E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F87B6743F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FFDE6773F0000803F0200FE3ED1B1B83E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F14E3793F66FD0A3F7C0C1B3C0000803F0000803F0000803F0000803F0000803F0000803F54A15C3E0000803F0000803F0000803F0000803F883CCD3C0000803F0000803F0000803FF3639F3E0000803F0000803F0000803FA02C14390000803F0000803F0000803F0000803F0000803F0000803FD7FD043D0000803F0000803F0000803F0000803F7D6C263D0000803F0000803F0000803F0000803F0000803F0000803F0000803FB63D5E3B0000803F0000803F0000803FFDFF7F3F0000803F6848B83E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F7959723F0000803F7350983C0000803F0000803F0000803F61E57F3F0000803F5076673E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FC9B31E3F0000803F4318313EF9101E3E0000803F0000803F0000803F0000803F0000803F0000803F0FE1B53EADB5DB3C0000803F0000803F0000803F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @igammac(%arg0: tensor<1x20xf32>, %arg1: tensor<20x20xf32>) -> tensor<20x20xf32> {
    %0 = call @xla_fallback_igammac(%arg0, %arg1) : (tensor<1x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @igamma_body.172(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>> {
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
  func.func private @or.209(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @igamma_condition.213(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<i1> {
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
  func.func private @igammac_body.266(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>> {
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
  func.func private @or.389(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @igammac_condition.393(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<i1> {
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
  func.func private @xla_fallback_igammac(%arg0: tensor<1x20xf32>, %arg1: tensor<20x20xf32>) -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %2 = stablehlo.compare  LE, %arg1, %1 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %3 = stablehlo.reshape %arg0 : (tensor<1x20xf32>) -> tensor<20xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<20xf32>) -> tensor<20x20xf32>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %7 = stablehlo.compare  LE, %4, %6 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %8 = stablehlo.or %2, %7 : tensor<20x20xi1>
    %9 = stablehlo.log %arg1 : tensor<20x20xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<20x20xf32>
    %11 = stablehlo.subtract %10, %arg1 : tensor<20x20xf32>
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
    %153 = stablehlo.compare  LT, %arg1, %152 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %154 = stablehlo.compare  LT, %arg1, %4 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
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
    %165:7 = stablehlo.while(%iterArg = %156, %iterArg_0 = %4, %iterArg_1 = %158, %iterArg_2 = %160, %iterArg_3 = %arg1, %iterArg_4 = %162, %iterArg_5 = %164) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %213 = stablehlo.constant dense<false> : tensor<i1>
      %214 = stablehlo.reduce(%iterArg init: %213) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %215 = stablehlo.or %arg2, %arg3 : tensor<i1>
        stablehlo.return %215 : tensor<i1>
      }
      stablehlo.return %214 : tensor<i1>
    } do {
      %213 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %214 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %215 = stablehlo.add %iterArg_0, %214 : tensor<20x20xf32>
      %216 = stablehlo.divide %iterArg_3, %215 : tensor<20x20xf32>
      %217 = stablehlo.multiply %iterArg_1, %216 : tensor<20x20xf32>
      %218 = stablehlo.add %iterArg_2, %217 : tensor<20x20xf32>
      %219 = stablehlo.divide %217, %218 : tensor<20x20xf32>
      %220 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %221 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %222 = stablehlo.compare  GT, %219, %221 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %223 = stablehlo.and %iterArg, %222 : tensor<20x20xi1>
      %224 = stablehlo.select %iterArg, %215, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %225 = stablehlo.select %iterArg, %217, %iterArg_1 : tensor<20x20xi1>, tensor<20x20xf32>
      %226 = stablehlo.select %iterArg, %218, %iterArg_2 : tensor<20x20xi1>, tensor<20x20xf32>
      %227 = stablehlo.divide %iterArg_3, %215 : tensor<20x20xf32>
      %228 = stablehlo.multiply %iterArg_4, %227 : tensor<20x20xf32>
      %229 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
      %230 = stablehlo.constant dense<-1.000000e+00> : tensor<20x20xf32>
      %231 = stablehlo.multiply %230, %iterArg_1 : tensor<20x20xf32>
      %232 = stablehlo.multiply %231, %iterArg_3 : tensor<20x20xf32>
      %233 = stablehlo.multiply %215, %215 : tensor<20x20xf32>
      %234 = stablehlo.divide %232, %233 : tensor<20x20xf32>
      %235 = stablehlo.add %228, %234 : tensor<20x20xf32>
      %236 = stablehlo.select %iterArg, %235, %iterArg_4 : tensor<20x20xi1>, tensor<20x20xf32>
      %237 = stablehlo.add %iterArg_5, %235 : tensor<20x20xf32>
      %238 = stablehlo.select %iterArg, %237, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %223, %224, %225, %226, %iterArg_3, %236, %238 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %166 = stablehlo.not %155 : tensor<20x20xi1>
    %167 = stablehlo.and %150, %166 : tensor<20x20xi1>
    %168 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %169 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %170 = stablehlo.add %arg1, %169 : tensor<20x20xf32>
    %171 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %172 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %173 = stablehlo.subtract %172, %4 : tensor<20x20xf32>
    %174 = stablehlo.add %arg1, %173 : tensor<20x20xf32>
    %175 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %176 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %177 = stablehlo.add %174, %176 : tensor<20x20xf32>
    %178 = stablehlo.multiply %177, %arg1 : tensor<20x20xf32>
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
    %191 = stablehlo.negate %arg1 : tensor<20x20xf32>
    %192 = stablehlo.multiply %179, %191 : tensor<20x20xf32>
    %193 = stablehlo.subtract %190, %192 : tensor<20x20xf32>
    %194 = stablehlo.divide %193, %178 : tensor<20x20xf32>
    %195:15 = stablehlo.while(%iterArg = %167, %iterArg_0 = %179, %iterArg_1 = %181, %iterArg_2 = %173, %iterArg_3 = %177, %iterArg_4 = %182, %iterArg_5 = %170, %iterArg_6 = %178, %iterArg_7 = %184, %iterArg_8 = %arg1, %iterArg_9 = %186, %iterArg_10 = %188, %iterArg_11 = %190, %iterArg_12 = %191, %iterArg_13 = %194) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %213 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
      %214 = stablehlo.compare  LT, %iterArg_4, %213 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %215 = stablehlo.constant dense<false> : tensor<i1>
      %216 = stablehlo.reduce(%iterArg init: %215) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %218 = stablehlo.or %arg2, %arg3 : tensor<i1>
        stablehlo.return %218 : tensor<i1>
      }
      %217 = stablehlo.and %214, %216 : tensor<i1>
      stablehlo.return %217 : tensor<i1>
    } do {
      %213 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %214 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
      %215 = stablehlo.add %iterArg_3, %214 : tensor<20x20xf32>
      %216 = stablehlo.multiply %iterArg_6, %215 : tensor<20x20xf32>
      %217 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %218 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %219 = stablehlo.add %iterArg_2, %218 : tensor<20x20xf32>
      %220 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %221 = stablehlo.add %iterArg_4, %220 : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %221, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %223 = stablehlo.multiply %219, %222 : tensor<20x20xf32>
      %224 = stablehlo.multiply %iterArg_8, %223 : tensor<20x20xf32>
      %225 = stablehlo.subtract %216, %224 : tensor<20x20xf32>
      %226 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %227 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
      %228 = stablehlo.compare  NE, %225, %227 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %229 = stablehlo.multiply %iterArg_11, %215 : tensor<20x20xf32>
      %230 = stablehlo.subtract %229, %iterArg_5 : tensor<20x20xf32>
      %231 = stablehlo.multiply %iterArg_9, %223 : tensor<20x20xf32>
      %232 = stablehlo.subtract %230, %231 : tensor<20x20xf32>
      %233 = stablehlo.broadcast_in_dim %221, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %234 = stablehlo.multiply %iterArg_7, %233 : tensor<20x20xf32>
      %235 = stablehlo.add %232, %234 : tensor<20x20xf32>
      %236 = stablehlo.multiply %iterArg_5, %215 : tensor<20x20xf32>
      %237 = stablehlo.multiply %iterArg_7, %223 : tensor<20x20xf32>
      %238 = stablehlo.subtract %236, %237 : tensor<20x20xf32>
      %239 = stablehlo.divide %238, %225 : tensor<20x20xf32>
      %240 = stablehlo.select %228, %239, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %241 = stablehlo.multiply %iterArg_12, %215 : tensor<20x20xf32>
      %242 = stablehlo.subtract %241, %iterArg_6 : tensor<20x20xf32>
      %243 = stablehlo.multiply %iterArg_10, %223 : tensor<20x20xf32>
      %244 = stablehlo.subtract %242, %243 : tensor<20x20xf32>
      %245 = stablehlo.broadcast_in_dim %221, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %246 = stablehlo.multiply %iterArg_8, %245 : tensor<20x20xf32>
      %247 = stablehlo.add %244, %246 : tensor<20x20xf32>
      %248 = stablehlo.multiply %240, %247 : tensor<20x20xf32>
      %249 = stablehlo.subtract %235, %248 : tensor<20x20xf32>
      %250 = stablehlo.divide %249, %225 : tensor<20x20xf32>
      %251 = stablehlo.select %228, %250, %iterArg_13 : tensor<20x20xi1>, tensor<20x20xf32>
      %252 = stablehlo.subtract %251, %iterArg_13 : tensor<20x20xf32>
      %253 = stablehlo.abs %252 : tensor<20x20xf32>
      %254 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %255 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %256 = stablehlo.select %228, %253, %255 : tensor<20x20xi1>, tensor<20x20xf32>
      %257 = stablehlo.subtract %iterArg_0, %239 : tensor<20x20xf32>
      %258 = stablehlo.divide %257, %239 : tensor<20x20xf32>
      %259 = stablehlo.abs %258 : tensor<20x20xf32>
      %260 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %261 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %262 = stablehlo.select %228, %259, %261 : tensor<20x20xi1>, tensor<20x20xf32>
      %263 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %264 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %265 = stablehlo.compare  GT, %262, %264 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %266 = stablehlo.and %iterArg, %265 : tensor<20x20xi1>
      %267 = stablehlo.select %iterArg, %240, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %268 = stablehlo.select %iterArg, %262, %iterArg_1 : tensor<20x20xi1>, tensor<20x20xf32>
      %269 = stablehlo.select %iterArg, %219, %iterArg_2 : tensor<20x20xi1>, tensor<20x20xf32>
      %270 = stablehlo.select %iterArg, %215, %iterArg_3 : tensor<20x20xi1>, tensor<20x20xf32>
      %271 = stablehlo.abs %238 : tensor<20x20xf32>
      %272 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %273 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %274 = stablehlo.constant dense<0x4B000000> : tensor<f32>
      %275 = stablehlo.broadcast_in_dim %274, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %276 = stablehlo.compare  GT, %271, %275 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %277 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %278 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %279 = stablehlo.multiply %238, %278 : tensor<20x20xf32>
      %280 = stablehlo.select %276, %279, %238 : tensor<20x20xi1>, tensor<20x20xf32>
      %281 = stablehlo.select %iterArg, %280, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      %282 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %283 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %284 = stablehlo.multiply %225, %283 : tensor<20x20xf32>
      %285 = stablehlo.select %276, %284, %225 : tensor<20x20xi1>, tensor<20x20xf32>
      %286 = stablehlo.select %iterArg, %285, %iterArg_6 : tensor<20x20xi1>, tensor<20x20xf32>
      %287 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %288 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %289 = stablehlo.multiply %iterArg_5, %288 : tensor<20x20xf32>
      %290 = stablehlo.select %276, %289, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      %291 = stablehlo.select %iterArg, %290, %iterArg_7 : tensor<20x20xi1>, tensor<20x20xf32>
      %292 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %293 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %294 = stablehlo.multiply %iterArg_6, %293 : tensor<20x20xf32>
      %295 = stablehlo.select %276, %294, %iterArg_6 : tensor<20x20xi1>, tensor<20x20xf32>
      %296 = stablehlo.select %iterArg, %295, %iterArg_8 : tensor<20x20xi1>, tensor<20x20xf32>
      %297 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %298 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %299 = stablehlo.multiply %iterArg_11, %298 : tensor<20x20xf32>
      %300 = stablehlo.select %276, %299, %iterArg_11 : tensor<20x20xi1>, tensor<20x20xf32>
      %301 = stablehlo.select %iterArg, %300, %iterArg_9 : tensor<20x20xi1>, tensor<20x20xf32>
      %302 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %303 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %304 = stablehlo.multiply %iterArg_12, %303 : tensor<20x20xf32>
      %305 = stablehlo.select %276, %304, %iterArg_12 : tensor<20x20xi1>, tensor<20x20xf32>
      %306 = stablehlo.select %iterArg, %305, %iterArg_10 : tensor<20x20xi1>, tensor<20x20xf32>
      %307 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %308 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %309 = stablehlo.multiply %235, %308 : tensor<20x20xf32>
      %310 = stablehlo.select %276, %309, %235 : tensor<20x20xi1>, tensor<20x20xf32>
      %311 = stablehlo.select %iterArg, %310, %iterArg_11 : tensor<20x20xi1>, tensor<20x20xf32>
      %312 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %313 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %314 = stablehlo.multiply %247, %313 : tensor<20x20xf32>
      %315 = stablehlo.select %276, %314, %247 : tensor<20x20xi1>, tensor<20x20xf32>
      %316 = stablehlo.select %iterArg, %315, %iterArg_12 : tensor<20x20xi1>, tensor<20x20xf32>
      %317 = stablehlo.select %iterArg, %251, %iterArg_13 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %266, %267, %268, %269, %270, %221, %281, %286, %291, %296, %301, %306, %311, %316, %317 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %196 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %197 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %198 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %199 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %200 = stablehlo.compare  EQ, %arg1, %199 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
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
    return %212 : tensor<20x20xf32>
  }
}
