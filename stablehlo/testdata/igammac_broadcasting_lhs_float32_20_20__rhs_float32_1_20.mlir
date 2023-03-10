// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xf32>, tensor<1x20xf32>)
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = call @igammac(%0#0, %0#1) : (tensor<20x20xf32>, tensor<1x20xf32>) -> tensor<20x20xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xf32>, tensor<1x20xf32>) {
    %0 = stablehlo.constant dense<"0x3020913EB8B2D83E46101640A38FC5BF890401402FE2A4402F3697C0E18CABBF4D6DDD40DCAC24C0A6B2C33FB166F43FFC9772BFEC701940EEFB07BF72F4103FA5FFAE40442514403B00FCBFF8D011C04037BE400CF57A40AA04143FD5C627C01A7350BFEC4AB8BF9110813FF4F6BF406491E03F2A7DB0C04BD4A1408E415D3E8DE2FFBE221E59C022306B40E948A140D8980AC0DA609F3FAB8E2FC07EA16FBF23187F40BA3E183F2D1476C075B1D93F121044C0F0DB5D3F7CD8C73F8744F3BF430F1B405F42E73F8B1A4940097D7EBEAD36603FB9911340389B2440870AB43FD73CD2C08D465BBEF7FF78C07A784B3F447C24BF00381FC0B16BFDBF98E47FC0F086C33FA7449B409ED600BFA0927DC0123CDBBF7BCECC3F492A10402CC378BFC4FEBDC03C67A94085606840FD389DC01FCD284014C4B23E0DB291C0372AA2C07935BD3FB2BAAEBFFB53343F451349409D3C2CC096C8BF3FD3A87C40F3FC92408CD5043F9BDB19BF66F263C0E91FCB3FE1AF5E3F8ABE50C0AFC4F73FB653544061A95BBFF30C023F03BF01C06D386DC0C590C03F5CBFB73FCB0078BED7578140BEEE5F3F1EABA5BF75FDDD40678C84C01AB9A8402CB514C008875C3FD21A92C0690E2AC00611793E7EED0DC040C79440A2774B40BD8C44BF554761C06292D3BF8690B6BFE35161404AF6B7C0B11E544029506040BA401B4132B955BE3555674086196AC02BFE15C05A4BD93F04A8B8C007FC5940401E26C07CFA06C06C218D40A272DB40E547793F9B95E1407DAFA1C0592C1E40082470BFD729253FFED13A3F8E19BA3F9C1D3140FD6B7DC0D0EEC8405AC67540F0FF07C03F1790C0B57984C0FD21C9BF3B7776BFD91404BFF6BD1540E2577BBF59B591BF9ED5E5C0F486FCBDDBD8B9BDDD9690C0EB710FBF5363BF3F2A19793F45D3B340D00105C09388D03F00D1BCBF8E5D833F1526BB3E34F8D53F97CB6AC07AC38EC049D404C096379A3FF070AD40526794BE3C626B40DFCDA6403D7669C0DF4052BE997042C0502C693E7BA94E3F491AFF3FAD62363E92CBF43F99BC30BFCC2907C0604BB6C009103DC0948949C0DD2DDCBF83031E406B29B6408151BABF711066BF719322C0F10BF33F21263F4027788F4083FB76C0DC0C8BBFBD1BA440B90B3DC0F3136F3F43F6923FE6371140317B3840B2256940DF695840C291CABF1A41E33D2C89483F3074563E42FB80BDD27CC6BE224A9340CD0631402C85EBBF341704BF7C500941ACEAF8C067810D40A8FD96C0654941402845C7BC22216FBEE548B4BD4BC30BC0A247873FCCE41DC0BA88C73F694D08C07491223F4A75E23F6F3200C071A993C0004117C073474140E4908040A1F55ABF7C8D15BEF4EC6A3F80060740D8E97B40E5BF64401060A53C8507DE3FA9FE1BC085D87640BD2EF240691E57BE458B7FC01DD9014059128840BFC5E73F77CCA2C03065D33F06B115BFDA64B440C8F48EBFFAA0A8C0E641D43FB356644057F6B1C0D817483E058191C03D5BC5BF9B2519BE4A4625404F32703FC0DE80C026FE07C1E190D540082F41400957E4C0B39957C0D9AFFA3ECFE0FEBF0815173FBD2B41BFEA97A93E972352C08A54573FFE58BB3F27CA883FF1A1F53F2610BAC0256D86C07104844088D54CC04FE0A73FC058A44008D315C17BB188BF58A6E53D783B0140749DEABF96C03440E3B9A3BF691EF83FCA51054038B04D40B602354035452CC079DE743F2A9E16C0676D53BF9AEF7EBFCF17044090712FC0D2290D3F2F279C4088C8913F8E388AC07F741FC05C5317C0D39B2D3F2CFFD43FD78E823E2F0504C03C614EC047562140975FCD4024C825BE4D7E3EC0CA223140089F1BC09BB58FBFF14E68C0F919A23FE133DD40E69BDEBF9626D9BFC99004404EE0D83F49E36A4099E74EBE7FC45EC027C384C00A90DCC0E73637C0682F0F40834B8AC0450021BFA5079340023D9A40621C60BFA8CE1C4070440AC08C48114082EB9CBF182B334008A2BDBE1CB5943E706623BEAB81303F9BA443C0328F01C04D8CCEC0123517C07DA98BC09349ED3ED87848BFF8DD1BC0BC765E4074783EC0437FADBF0AC9DABFFCC031C089C800C190EF86C0175CF6BFC50ADD3FE7C9C3C0DC1D19C0F5FC83C0DCC6EABFF2E68F4083F9C4409C8B584026009DBF3C692EC037DB3A40CC92C74050D593C076617B3F93ED74C0A531433F7E8EB0C0A3FD0B407F5580C0DE69A9BFB4B6A940794023C0525FBE3FA1CB4AC0F87FFFBF"> : tensor<20x20xf32>
    %1 = stablehlo.constant dense<[[3.87998796, 1.99467063, -4.02589035, 0.275933355, 3.16329694, -4.52458382, 0.70135945, 3.37606835, 2.81681681, -2.13368535, 3.31388211, -5.57836485, -0.569259942, -5.06082821, -3.05091333, -1.24116778, -0.161324903, -4.411600e+00, -0.566983759, -4.6893115]]> : tensor<1x20xf32>
    return %0, %1 : tensor<20x20xf32>, tensor<1x20xf32>
  }
  func.func private @expected() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0x8FB40B3B90E2123D0000803F0000803F693C373E0000803F0000803F0000803FA104793F0000803FD4CFB43D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F55ED4B3F7AFA583F0000803F0000803F0000803F0000803FB6F5FF3E119A5F3FDACD353E0000803F54C8443F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F289FE83EDFD1733D0000803F2561703F0000803F0000803F68C3393F0000803FDD64A63E0000803F98D8C63E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F25D3CD3D0000803F0000803F0000803F0000803F0000803FC2744E3E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FDFDF4A3D0000803F0000803F6A7E7F3F0000803F0000803F265D7E3F07172E3FB857993C0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F1F53533DDAB77A3E0000803FB5F47F3FFA5B053D0000803F51FF7F3F0000803F870D603F0000803FDC9FDD3C0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F31E1483F0000803F95AD7F3FD6D6003F0000803F0000803FBA00F63E0000803F0000803FF04DE23D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F5A7E293E0000803F0000803FBC501E3FC0B0B93D0000803F0000803FFEE0653F7A06283F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F370D683FD11A243D0000803F0000803F347AC43D0000803F0000803F6EFBC53B0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F34B2613EA6D5E43C0000803FE083983D9D110B3E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F12EB813EE21F693F0000803F0000803FF1154E3F0000803FAE50ED3E83183A3DE9D3933E0000803F6418003F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F203A5D3E0000803F7A81773F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FCE18853ECE405C3F0000803F0000803F0553123D0000803FDF537E3F7E71F13EC022B5390000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F6682793F0000803F0000803FD84BF43D0000803F0000803F2683153B0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FC6F7703D0000803F129CA13E0000803F0000803F5E2C313FB761203D1E08583E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FFEA0653E0000803F0000803FB243793F94A9DF3E0000803F0000803F3AF1FF3C0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F11BD853DEF51923C0000803F0000803FEF788F3E0000803F0000803F0000803FFC43D13E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F518E623E0000803F0000803F85262E3F3381523F0000803FBE8D753E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0DDB2B3C0000803F0000803F484DE63E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FC0D20E3F768B7C3F0000803F0000803F0000803F0000803F82FC7F3F0000803F01A86C3D0000803F1C43AF3C0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @igammac(%arg0: tensor<20x20xf32>, %arg1: tensor<1x20xf32>) -> tensor<20x20xf32> {
    %0 = call @xla_fallback_igammac(%arg0, %arg1) : (tensor<20x20xf32>, tensor<1x20xf32>) -> tensor<20x20xf32>
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
  func.func private @xla_fallback_igammac(%arg0: tensor<20x20xf32>, %arg1: tensor<1x20xf32>) -> tensor<20x20xf32> {
    %0 = stablehlo.reshape %arg1 : (tensor<1x20xf32>) -> tensor<20xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<20xf32>) -> tensor<20x20xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %4 = stablehlo.compare  LE, %1, %3 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %7 = stablehlo.compare  LE, %arg0, %6 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %8 = stablehlo.or %4, %7 : tensor<20x20xi1>
    %9 = stablehlo.log %1 : tensor<20x20xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<20x20xf32>
    %11 = stablehlo.subtract %10, %1 : tensor<20x20xf32>
    %12 = stablehlo.abs %arg0 : tensor<20x20xf32>
    %13 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %14 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %15 = stablehlo.compare  EQ, %12, %14 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %16 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %17 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %18 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %19 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %20 = stablehlo.compare  LT, %arg0, %19 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %21 = stablehlo.constant dense<3.14159274> : tensor<f32>
    %22 = stablehlo.constant dense<3.14159274> : tensor<20x20xf32>
    %23 = stablehlo.abs %arg0 : tensor<20x20xf32>
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
    %42 = stablehlo.negate %arg0 : tensor<20x20xf32>
    %43 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %44 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %45 = stablehlo.subtract %arg0, %44 : tensor<20x20xf32>
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
    %153 = stablehlo.compare  LT, %1, %152 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %154 = stablehlo.compare  LT, %1, %arg0 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
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
    %165:7 = stablehlo.while(%iterArg = %156, %iterArg_0 = %arg0, %iterArg_1 = %158, %iterArg_2 = %160, %iterArg_3 = %1, %iterArg_4 = %162, %iterArg_5 = %164) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
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
    %170 = stablehlo.add %1, %169 : tensor<20x20xf32>
    %171 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %172 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %173 = stablehlo.subtract %172, %arg0 : tensor<20x20xf32>
    %174 = stablehlo.add %1, %173 : tensor<20x20xf32>
    %175 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %176 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %177 = stablehlo.add %174, %176 : tensor<20x20xf32>
    %178 = stablehlo.multiply %177, %1 : tensor<20x20xf32>
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
    %191 = stablehlo.negate %1 : tensor<20x20xf32>
    %192 = stablehlo.multiply %179, %191 : tensor<20x20xf32>
    %193 = stablehlo.subtract %190, %192 : tensor<20x20xf32>
    %194 = stablehlo.divide %193, %178 : tensor<20x20xf32>
    %195:15 = stablehlo.while(%iterArg = %167, %iterArg_0 = %179, %iterArg_1 = %181, %iterArg_2 = %173, %iterArg_3 = %177, %iterArg_4 = %182, %iterArg_5 = %170, %iterArg_6 = %178, %iterArg_7 = %184, %iterArg_8 = %1, %iterArg_9 = %186, %iterArg_10 = %188, %iterArg_11 = %190, %iterArg_12 = %191, %iterArg_13 = %194) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
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
    %200 = stablehlo.compare  EQ, %1, %199 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %201 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %202 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %203 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %204 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %205 = stablehlo.exponential %143 : tensor<20x20xf32>
    %206 = stablehlo.multiply %165#3, %205 : tensor<20x20xf32>
    %207 = stablehlo.divide %206, %arg0 : tensor<20x20xf32>
    %208 = stablehlo.subtract %204, %207 : tensor<20x20xf32>
    %209 = stablehlo.multiply %195#1, %205 : tensor<20x20xf32>
    %210 = stablehlo.select %155, %208, %209 : tensor<20x20xi1>, tensor<20x20xf32>
    %211 = stablehlo.select %200, %202, %210 : tensor<20x20xi1>, tensor<20x20xf32>
    %212 = stablehlo.select %8, %197, %211 : tensor<20x20xi1>, tensor<20x20xf32>
    return %212 : tensor<20x20xf32>
  }
}
