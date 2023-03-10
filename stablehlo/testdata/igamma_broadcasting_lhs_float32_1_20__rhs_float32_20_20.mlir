// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x20xf32>, tensor<20x20xf32>)
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = call @igamma(%0#0, %0#1) : (tensor<1x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x20xf32>, tensor<20x20xf32>) {
    %0 = stablehlo.constant dense<[[1.20794058, 6.0737462, -2.49186802, -3.00700927, -5.10266972, -3.79685378, -6.76943779, -2.56331301, -4.67658138, -4.37845135, -0.540752769, 1.26569831, -5.04493952, 3.2862289, 2.1054132, 0.39229244, -3.86122465, -2.30690217, 2.17852426, -1.44895661]]> : tensor<1x20xf32>
    %1 = stablehlo.constant dense<"0x21948CC051E6C23E7B0AAB4036AA393F6ADB3DC04C45ADC007E2F6C0317B9640FA838ABF2216C4BFFF60FDBF75CDBC3F905930BD0214D3BFE19C594027CBB4BF9BE8FA3FAC30C53FBAA21CC06D18C2BFD0010B3F34EBCB3FE269FD3FC3E659BF41E4223F7FFD9ABF819CB040B7878FC067A6154064E42F403A0E7BBE3BDCA73E49E18340E94A0FC0376CCF3EA2CD303FB20644BF2677E9BFF6A9AAC0524FCCBFCAC3BD40A71011C0C6A61FC04A723740A86DBF4098F6E83F2D55DEBF91468F401FAADE3F91B9A13FB42D6CBFD6405240333C5540D8272740F88390C0234C5540F4F9A9BF5A861EC061F73FC0CD0ECB40628C2F402DA8F540D78C9BC046E40AC0E9B989C0AD14163FCFD3D53FA45601C006E75FBF0ADF6EC080AF1FBEB2AEAFC0B9C9AABFA55A31C0CD99014030BA764005A99FC082CCEE3E80EA13BF64C0954016F4BF403B7CB6BF84245A3E96E563C0ED4CB63E179040400AD7863F9D8FAFBF9CA2CB3FE7766FC0D821D3BEF4CF973F84BE73C0A0454B4012E6A93E13F91BC0D39E1FC0249616C0E163E43E3CF1E940D680D0BE7EB26BBF1C7B3BC01396824040C2763FAA89153E423F77409F4A5F3FADDE30408EE18FBF040E6A3FDDB805C0FDA430BE50D59C40FD94BE3FA088823F93594240BBFF45C063F9553DE55D9CC0E8655D3E21D5AA40E47BFFBD37F63DBF98F518C06DEA2ABFE0CD9CBEB0D158402004FDBF636372BFF7F90BC0388B01C0AD9B8BC04C3D32C0FF7A6B40975C08C09E6D18BF9756F93FFE376BC09F31F33FC2022E4087AA993F0A500FC0FAED8E4094DC17C067F5D83F6A1FBFBECB20994079FFBCC0C1B31F3F9938993DDF2A24BF5F244540BF152F40B5C37FC0BD7A6EC0EA4FD63FCC05273D221326402053D73FC828C9BF6F7AA2BF62C3B13FB4050EC0A58EB74057249AC0FE8851BF83A488C05EB93E40AB901CBF2A8A63C05C0444BFDD76364092339DBFEB890BC00672EABF4ABDC23F9BE8E5BF0C7FBBBFED9518C0DF5F9CC0743C81BF3C7E433FC393E83FBF135EBF0F106CC0025A6BBD066451C03A740E40FF66AEC0381A073ED1B60BC140D9B7407C3FDFBDDD2AB4BFF36E99C05DD214C0321D5D407493AE3FB74F42BDED5043BF7BC5E5BF81177FBEF80707C04DBDE83F10AD9EBEA30A58C047FAEE40D51F584036C241C08ACF5FBED1BBB6BF3A9F1FBF94969A3EEB463DC0BF8225408A0BECBFA0BA52C033639340FB46463FCE5755C0FFCD1240BE13C9BFD68F503F1657CCBE31007A40AAEBBE40E18DD33E921C65BF1679043F4C1D893F2F421CC18D023640F67DAF3FDC46063E42A3C5BF7F418440DCE0A5C04F9A4E4034DFA6BFB6178E3EBE3A15C02F2F86BEE8F07ABF51974C4060C0D2BFC153A6BFCEC4D03E658BD43FC2DE26BFAB9E0340C0EF8BBF4732B1BFAAEA67407FD61A40BBBCDDBEE623424061C6A2C09478D8BF2E74E5BF8ABC0040EEAD93BF060E683ECFDF2D3F8E1722C03F4B99BFA68D92C0A78D533F464C8CC06C834BC0F3E2B3BFB60E85C0BA806A40023E5CC023C6B2BE6F3AA3C011535E40234C963F2624EE3F5B1A67404AEF924080965940E97E1D408CB3593F8A072DC05198DEBE9A1D37C0EEAE50C09358A140F898D9BF83044F3F651A753F04DFA43E6DA4BCC0D1D38C401526D340523401BF7EBF17BF604A3340D0DA00C0A70DFABC939112BFF42C1F3EF55A38C0B308A83E1D205C4053ABFDC0083C163F39A257C0DC30D5BEE709FABFE200FBBF0C7728C0CCDE27C069C129C052458ABFFC094A3FB1B6D43E0F9526C0774B58400537034033B3EF3F93A3E8BF29432B40FA0D38C09D4908C022ACBD40D8FB03C0766DF93FA78F5EC0572081BF2BC4253E510C6840C723AE40431543BEA9C110C0C57D1F40149FB13F0E5BDBBF679BC73FF47EBBBF24B5AFC078270C3D2F38523F9DB5B5BF9C4B56BE01EBB640CAB2D9BFD846BA3C417D45BFCB690E3F5B1DC240DB34A840001F6EBF38911DC040B5724089B552C03D6E51BE56B0C63F34CB30404DDF46BFC9795EC0D262BF3FEDCC55C014AD0FC04730B03FDF2D9FBF4C3C823FFD5EA1C000BBC33FB03E83405E42F7BF691AE5BFC3383D3EBC98F93F06973FC0B471E9BFCA05B23E32E10540448D4D4030D42C402BF9FCBFACEEA83FDC3BE9BF46C3F7BF7ABF594005D556C0384032C086B45B3F8D8B8A3F9CB5ABBFED719ABFA8D8DE3ED2998FC073FDC1BE5BC517BFBDC638C0D92FA0C02076A24014C8A1C0"> : tensor<20x20xf32>
    return %0, %1 : tensor<1x20xf32>, tensor<20x20xf32>
  }
  func.func private @expected() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0x0000C07F4E0226360000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FF68E2E3F0000C07F0000C07F3602563F0000C07F0000C07F0000C07F0000C07F0000C07FBF4CA63EC94DAE3B0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F848A363E0000C07F0000C07F0A27533DB53B513F0000C07F0000C07F0000C07F0000C07F07E47E3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F2774703F0000C07F09DCD43E0000C07F234F7E3F0000C07F0000C07F0000C07F0000C07F3971683FACB8443F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F6AE9113F5A117F3F0000C07F0000C07F0000C07F0000C07F41F67E3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FAB29173F0000C07F11A40C3F9891113D0000C07F0000C07F0000C07F90355C3D0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0987543F7DE1CF3EF9F1623F0000C07F0000C07F2BC12B3A0000C07FD881013EE99ADC3E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F53035E3F0000C07F0000C07F0000C07F0000C07F0000C07F37E7673F908CAD3A0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F2887E43E0000C07F0000C07F0000C07F0000C07F1627303F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F334AAF3E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FF50AEA3A0000C07F7EF67B3F0000C07F0000C07F40086E3F0000C07F0000C07FEDD4DF3C0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F5F5CF03DDB75BC3B0000C07F0000C07F0000C07FCBC74C3F0000C07F0384293E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F5257243FD41E2B3F0000C07F0000C07F0000C07F0000C07F0000C07F7553513F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F6E5B003F0000C07F3DF57B3F4FD2FA3D0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FF68C003F0000C07F0000C07FB9A76C3F69F47F3F0000C07F0000C07FAEAE3A3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FB4F8523F068D383C0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FF374A33D0000C07F1D47613F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F7AE57E3F0000C07F0000C07F0000C07FC3FF7E3F0000C07F0000C07F0667CF3E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F70CBCB390A8D0C3F0000C07F0000C07F0000C07FBA2C113F0000C07F087B673F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F20AB733F0000C07F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @igamma(%arg0: tensor<1x20xf32>, %arg1: tensor<20x20xf32>) -> tensor<20x20xf32> {
    %0 = call @xla_fallback_igamma(%arg0, %arg1) : (tensor<1x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @igammac_body.202(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>> {
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
  func.func private @or.325(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @igammac_condition.329(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<i1> {
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
  func.func private @igamma_body.381(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>> {
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
  func.func private @or.418(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @igamma_condition.422(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<i1> {
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
  func.func private @xla_fallback_igamma(%arg0: tensor<1x20xf32>, %arg1: tensor<20x20xf32>) -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %2 = stablehlo.compare  EQ, %arg1, %1 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %5 = stablehlo.compare  LT, %arg1, %4 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %6 = stablehlo.reshape %arg0 : (tensor<1x20xf32>) -> tensor<20xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [1] : (tensor<20xf32>) -> tensor<20x20xf32>
    %8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %10 = stablehlo.compare  LE, %7, %9 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %11 = stablehlo.or %5, %10 : tensor<20x20xi1>
    %12 = stablehlo.or %2, %11 : tensor<20x20xi1>
    %13 = stablehlo.log %arg1 : tensor<20x20xf32>
    %14 = stablehlo.multiply %7, %13 : tensor<20x20xf32>
    %15 = stablehlo.subtract %14, %arg1 : tensor<20x20xf32>
    %16 = stablehlo.abs %7 : tensor<20x20xf32>
    %17 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %18 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %19 = stablehlo.compare  EQ, %16, %18 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %20 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %21 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %22 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %23 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %24 = stablehlo.compare  LT, %7, %23 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %25 = stablehlo.constant dense<3.14159274> : tensor<f32>
    %26 = stablehlo.constant dense<3.14159274> : tensor<20x20xf32>
    %27 = stablehlo.abs %7 : tensor<20x20xf32>
    %28 = stablehlo.floor %27 : tensor<20x20xf32>
    %29 = stablehlo.subtract %27, %28 : tensor<20x20xf32>
    %30 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %31 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %32 = stablehlo.compare  GT, %29, %31 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %33 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %34 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %35 = stablehlo.subtract %34, %29 : tensor<20x20xf32>
    %36 = stablehlo.select %32, %35, %29 : tensor<20x20xi1>, tensor<20x20xf32>
    %37 = stablehlo.multiply %26, %36 : tensor<20x20xf32>
    %38 = stablehlo.sine %37 : tensor<20x20xf32>
    %39 = stablehlo.log %38 : tensor<20x20xf32>
    %40 = stablehlo.is_finite %39 : (tensor<20x20xf32>) -> tensor<20x20xi1>
    %41 = stablehlo.constant dense<1.14472985> : tensor<f32>
    %42 = stablehlo.constant dense<1.14472985> : tensor<20x20xf32>
    %43 = stablehlo.subtract %42, %39 : tensor<20x20xf32>
    %44 = stablehlo.constant dense<0.918938517> : tensor<f32>
    %45 = stablehlo.constant dense<0.918938517> : tensor<20x20xf32>
    %46 = stablehlo.negate %7 : tensor<20x20xf32>
    %47 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %48 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %49 = stablehlo.subtract %7, %48 : tensor<20x20xf32>
    %50 = stablehlo.select %24, %46, %49 : tensor<20x20xi1>, tensor<20x20xf32>
    %51 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %52 = stablehlo.add %50, %51 : tensor<20x20xf32>
    %53 = stablehlo.constant dense<7.500000e+00> : tensor<f32>
    %54 = stablehlo.constant dense<7.500000e+00> : tensor<20x20xf32>
    %55 = stablehlo.add %54, %50 : tensor<20x20xf32>
    %56 = stablehlo.constant dense<2.01490307> : tensor<f32>
    %57 = stablehlo.constant dense<2.01490307> : tensor<20x20xf32>
    %58 = stablehlo.constant dense<7.500000e+00> : tensor<20x20xf32>
    %59 = stablehlo.divide %50, %58 : tensor<20x20xf32>
    %60 = stablehlo.log_plus_one %59 : tensor<20x20xf32>
    %61 = stablehlo.add %57, %60 : tensor<20x20xf32>
    %62 = stablehlo.divide %55, %61 : tensor<20x20xf32>
    %63 = stablehlo.subtract %52, %62 : tensor<20x20xf32>
    %64 = stablehlo.multiply %63, %61 : tensor<20x20xf32>
    %65 = stablehlo.add %45, %64 : tensor<20x20xf32>
    %66 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %67 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %68 = stablehlo.constant dense<676.520386> : tensor<f32>
    %69 = stablehlo.constant dense<676.520386> : tensor<20x20xf32>
    %70 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %71 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %72 = stablehlo.add %50, %71 : tensor<20x20xf32>
    %73 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %74 = stablehlo.add %72, %73 : tensor<20x20xf32>
    %75 = stablehlo.divide %69, %74 : tensor<20x20xf32>
    %76 = stablehlo.add %67, %75 : tensor<20x20xf32>
    %77 = stablehlo.constant dense<-1259.13916> : tensor<f32>
    %78 = stablehlo.constant dense<-1259.13916> : tensor<20x20xf32>
    %79 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %80 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %81 = stablehlo.add %50, %80 : tensor<20x20xf32>
    %82 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %83 = stablehlo.add %81, %82 : tensor<20x20xf32>
    %84 = stablehlo.divide %78, %83 : tensor<20x20xf32>
    %85 = stablehlo.add %76, %84 : tensor<20x20xf32>
    %86 = stablehlo.constant dense<771.323425> : tensor<f32>
    %87 = stablehlo.constant dense<771.323425> : tensor<20x20xf32>
    %88 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %89 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
    %90 = stablehlo.add %50, %89 : tensor<20x20xf32>
    %91 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %92 = stablehlo.add %90, %91 : tensor<20x20xf32>
    %93 = stablehlo.divide %87, %92 : tensor<20x20xf32>
    %94 = stablehlo.add %85, %93 : tensor<20x20xf32>
    %95 = stablehlo.constant dense<-176.615036> : tensor<f32>
    %96 = stablehlo.constant dense<-176.615036> : tensor<20x20xf32>
    %97 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %98 = stablehlo.constant dense<3.000000e+00> : tensor<20x20xf32>
    %99 = stablehlo.add %50, %98 : tensor<20x20xf32>
    %100 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %101 = stablehlo.add %99, %100 : tensor<20x20xf32>
    %102 = stablehlo.divide %96, %101 : tensor<20x20xf32>
    %103 = stablehlo.add %94, %102 : tensor<20x20xf32>
    %104 = stablehlo.constant dense<12.5073433> : tensor<f32>
    %105 = stablehlo.constant dense<12.5073433> : tensor<20x20xf32>
    %106 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %107 = stablehlo.constant dense<4.000000e+00> : tensor<20x20xf32>
    %108 = stablehlo.add %50, %107 : tensor<20x20xf32>
    %109 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %110 = stablehlo.add %108, %109 : tensor<20x20xf32>
    %111 = stablehlo.divide %105, %110 : tensor<20x20xf32>
    %112 = stablehlo.add %103, %111 : tensor<20x20xf32>
    %113 = stablehlo.constant dense<-0.138571098> : tensor<f32>
    %114 = stablehlo.constant dense<-0.138571098> : tensor<20x20xf32>
    %115 = stablehlo.constant dense<5.000000e+00> : tensor<f32>
    %116 = stablehlo.constant dense<5.000000e+00> : tensor<20x20xf32>
    %117 = stablehlo.add %50, %116 : tensor<20x20xf32>
    %118 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %119 = stablehlo.add %117, %118 : tensor<20x20xf32>
    %120 = stablehlo.divide %114, %119 : tensor<20x20xf32>
    %121 = stablehlo.add %112, %120 : tensor<20x20xf32>
    %122 = stablehlo.constant dense<9.98436917E-6> : tensor<f32>
    %123 = stablehlo.constant dense<9.98436917E-6> : tensor<20x20xf32>
    %124 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %125 = stablehlo.constant dense<6.000000e+00> : tensor<20x20xf32>
    %126 = stablehlo.add %50, %125 : tensor<20x20xf32>
    %127 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %128 = stablehlo.add %126, %127 : tensor<20x20xf32>
    %129 = stablehlo.divide %123, %128 : tensor<20x20xf32>
    %130 = stablehlo.add %121, %129 : tensor<20x20xf32>
    %131 = stablehlo.constant dense<1.50563267E-7> : tensor<f32>
    %132 = stablehlo.constant dense<1.50563267E-7> : tensor<20x20xf32>
    %133 = stablehlo.constant dense<7.000000e+00> : tensor<f32>
    %134 = stablehlo.constant dense<7.000000e+00> : tensor<20x20xf32>
    %135 = stablehlo.add %50, %134 : tensor<20x20xf32>
    %136 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %137 = stablehlo.add %135, %136 : tensor<20x20xf32>
    %138 = stablehlo.divide %132, %137 : tensor<20x20xf32>
    %139 = stablehlo.add %130, %138 : tensor<20x20xf32>
    %140 = stablehlo.log %139 : tensor<20x20xf32>
    %141 = stablehlo.add %65, %140 : tensor<20x20xf32>
    %142 = stablehlo.subtract %43, %141 : tensor<20x20xf32>
    %143 = stablehlo.negate %39 : tensor<20x20xf32>
    %144 = stablehlo.select %40, %142, %143 : tensor<20x20xi1>, tensor<20x20xf32>
    %145 = stablehlo.select %24, %144, %141 : tensor<20x20xi1>, tensor<20x20xf32>
    %146 = stablehlo.select %19, %21, %145 : tensor<20x20xi1>, tensor<20x20xf32>
    %147 = stablehlo.subtract %15, %146 : tensor<20x20xf32>
    %148 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
    %149 = stablehlo.constant dense<88.7228394> : tensor<f32>
    %150 = stablehlo.negate %149 : tensor<f32>
    %151 = stablehlo.broadcast_in_dim %150, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %152 = stablehlo.compare  LT, %147, %151 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %153 = stablehlo.or %12, %152 : tensor<20x20xi1>
    %154 = stablehlo.compare  NE, %7, %7 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %155 = stablehlo.compare  NE, %arg1, %arg1 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %156 = stablehlo.or %154, %155 : tensor<20x20xi1>
    %157 = stablehlo.or %153, %156 : tensor<20x20xi1>
    %158 = stablehlo.not %157 : tensor<20x20xi1>
    %159 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %160 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %161 = stablehlo.compare  GT, %arg1, %160 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %162 = stablehlo.compare  GT, %arg1, %7 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %163 = stablehlo.and %161, %162 : tensor<20x20xi1>
    %164 = stablehlo.and %158, %163 : tensor<20x20xi1>
    %165 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %166 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %167 = stablehlo.add %arg1, %166 : tensor<20x20xf32>
    %168 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %169 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %170 = stablehlo.subtract %169, %7 : tensor<20x20xf32>
    %171 = stablehlo.add %arg1, %170 : tensor<20x20xf32>
    %172 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %173 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %174 = stablehlo.add %171, %173 : tensor<20x20xf32>
    %175 = stablehlo.multiply %174, %arg1 : tensor<20x20xf32>
    %176 = stablehlo.divide %167, %175 : tensor<20x20xf32>
    %177 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %178 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %179 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %180 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %181 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %182 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %183 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %184 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %185 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %186 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %187 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %188 = stablehlo.negate %arg1 : tensor<20x20xf32>
    %189 = stablehlo.multiply %176, %188 : tensor<20x20xf32>
    %190 = stablehlo.subtract %187, %189 : tensor<20x20xf32>
    %191 = stablehlo.divide %190, %175 : tensor<20x20xf32>
    %192:15 = stablehlo.while(%iterArg = %164, %iterArg_0 = %176, %iterArg_1 = %178, %iterArg_2 = %170, %iterArg_3 = %174, %iterArg_4 = %179, %iterArg_5 = %167, %iterArg_6 = %175, %iterArg_7 = %181, %iterArg_8 = %arg1, %iterArg_9 = %183, %iterArg_10 = %185, %iterArg_11 = %187, %iterArg_12 = %188, %iterArg_13 = %191) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %225 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
      %226 = stablehlo.compare  LT, %iterArg_4, %225 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %227 = stablehlo.constant dense<false> : tensor<i1>
      %228 = stablehlo.reduce(%iterArg init: %227) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %230 = stablehlo.or %arg2, %arg3 : tensor<i1>
        stablehlo.return %230 : tensor<i1>
      }
      %229 = stablehlo.and %226, %228 : tensor<i1>
      stablehlo.return %229 : tensor<i1>
    } do {
      %225 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %226 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
      %227 = stablehlo.add %iterArg_3, %226 : tensor<20x20xf32>
      %228 = stablehlo.multiply %iterArg_6, %227 : tensor<20x20xf32>
      %229 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %230 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %231 = stablehlo.add %iterArg_2, %230 : tensor<20x20xf32>
      %232 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %233 = stablehlo.add %iterArg_4, %232 : tensor<f32>
      %234 = stablehlo.broadcast_in_dim %233, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %235 = stablehlo.multiply %231, %234 : tensor<20x20xf32>
      %236 = stablehlo.multiply %iterArg_8, %235 : tensor<20x20xf32>
      %237 = stablehlo.subtract %228, %236 : tensor<20x20xf32>
      %238 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %239 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
      %240 = stablehlo.compare  NE, %237, %239 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %241 = stablehlo.multiply %iterArg_11, %227 : tensor<20x20xf32>
      %242 = stablehlo.subtract %241, %iterArg_5 : tensor<20x20xf32>
      %243 = stablehlo.multiply %iterArg_9, %235 : tensor<20x20xf32>
      %244 = stablehlo.subtract %242, %243 : tensor<20x20xf32>
      %245 = stablehlo.broadcast_in_dim %233, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %246 = stablehlo.multiply %iterArg_7, %245 : tensor<20x20xf32>
      %247 = stablehlo.add %244, %246 : tensor<20x20xf32>
      %248 = stablehlo.multiply %iterArg_5, %227 : tensor<20x20xf32>
      %249 = stablehlo.multiply %iterArg_7, %235 : tensor<20x20xf32>
      %250 = stablehlo.subtract %248, %249 : tensor<20x20xf32>
      %251 = stablehlo.divide %250, %237 : tensor<20x20xf32>
      %252 = stablehlo.select %240, %251, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %253 = stablehlo.multiply %iterArg_12, %227 : tensor<20x20xf32>
      %254 = stablehlo.subtract %253, %iterArg_6 : tensor<20x20xf32>
      %255 = stablehlo.multiply %iterArg_10, %235 : tensor<20x20xf32>
      %256 = stablehlo.subtract %254, %255 : tensor<20x20xf32>
      %257 = stablehlo.broadcast_in_dim %233, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %258 = stablehlo.multiply %iterArg_8, %257 : tensor<20x20xf32>
      %259 = stablehlo.add %256, %258 : tensor<20x20xf32>
      %260 = stablehlo.multiply %252, %259 : tensor<20x20xf32>
      %261 = stablehlo.subtract %247, %260 : tensor<20x20xf32>
      %262 = stablehlo.divide %261, %237 : tensor<20x20xf32>
      %263 = stablehlo.select %240, %262, %iterArg_13 : tensor<20x20xi1>, tensor<20x20xf32>
      %264 = stablehlo.subtract %263, %iterArg_13 : tensor<20x20xf32>
      %265 = stablehlo.abs %264 : tensor<20x20xf32>
      %266 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %267 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %268 = stablehlo.select %240, %265, %267 : tensor<20x20xi1>, tensor<20x20xf32>
      %269 = stablehlo.subtract %iterArg_0, %251 : tensor<20x20xf32>
      %270 = stablehlo.divide %269, %251 : tensor<20x20xf32>
      %271 = stablehlo.abs %270 : tensor<20x20xf32>
      %272 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %273 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %274 = stablehlo.select %240, %271, %273 : tensor<20x20xi1>, tensor<20x20xf32>
      %275 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %276 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %277 = stablehlo.compare  GT, %274, %276 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %278 = stablehlo.and %iterArg, %277 : tensor<20x20xi1>
      %279 = stablehlo.select %iterArg, %252, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %280 = stablehlo.select %iterArg, %274, %iterArg_1 : tensor<20x20xi1>, tensor<20x20xf32>
      %281 = stablehlo.select %iterArg, %231, %iterArg_2 : tensor<20x20xi1>, tensor<20x20xf32>
      %282 = stablehlo.select %iterArg, %227, %iterArg_3 : tensor<20x20xi1>, tensor<20x20xf32>
      %283 = stablehlo.abs %250 : tensor<20x20xf32>
      %284 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %285 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %286 = stablehlo.constant dense<0x4B000000> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %286, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %288 = stablehlo.compare  GT, %283, %287 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %289 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %290 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %291 = stablehlo.multiply %250, %290 : tensor<20x20xf32>
      %292 = stablehlo.select %288, %291, %250 : tensor<20x20xi1>, tensor<20x20xf32>
      %293 = stablehlo.select %iterArg, %292, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      %294 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %295 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %296 = stablehlo.multiply %237, %295 : tensor<20x20xf32>
      %297 = stablehlo.select %288, %296, %237 : tensor<20x20xi1>, tensor<20x20xf32>
      %298 = stablehlo.select %iterArg, %297, %iterArg_6 : tensor<20x20xi1>, tensor<20x20xf32>
      %299 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %300 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %301 = stablehlo.multiply %iterArg_5, %300 : tensor<20x20xf32>
      %302 = stablehlo.select %288, %301, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      %303 = stablehlo.select %iterArg, %302, %iterArg_7 : tensor<20x20xi1>, tensor<20x20xf32>
      %304 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %305 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %306 = stablehlo.multiply %iterArg_6, %305 : tensor<20x20xf32>
      %307 = stablehlo.select %288, %306, %iterArg_6 : tensor<20x20xi1>, tensor<20x20xf32>
      %308 = stablehlo.select %iterArg, %307, %iterArg_8 : tensor<20x20xi1>, tensor<20x20xf32>
      %309 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %310 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %311 = stablehlo.multiply %iterArg_11, %310 : tensor<20x20xf32>
      %312 = stablehlo.select %288, %311, %iterArg_11 : tensor<20x20xi1>, tensor<20x20xf32>
      %313 = stablehlo.select %iterArg, %312, %iterArg_9 : tensor<20x20xi1>, tensor<20x20xf32>
      %314 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %315 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %316 = stablehlo.multiply %iterArg_12, %315 : tensor<20x20xf32>
      %317 = stablehlo.select %288, %316, %iterArg_12 : tensor<20x20xi1>, tensor<20x20xf32>
      %318 = stablehlo.select %iterArg, %317, %iterArg_10 : tensor<20x20xi1>, tensor<20x20xf32>
      %319 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %320 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %321 = stablehlo.multiply %247, %320 : tensor<20x20xf32>
      %322 = stablehlo.select %288, %321, %247 : tensor<20x20xi1>, tensor<20x20xf32>
      %323 = stablehlo.select %iterArg, %322, %iterArg_11 : tensor<20x20xi1>, tensor<20x20xf32>
      %324 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %325 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %326 = stablehlo.multiply %259, %325 : tensor<20x20xf32>
      %327 = stablehlo.select %288, %326, %259 : tensor<20x20xi1>, tensor<20x20xf32>
      %328 = stablehlo.select %iterArg, %327, %iterArg_12 : tensor<20x20xi1>, tensor<20x20xf32>
      %329 = stablehlo.select %iterArg, %263, %iterArg_13 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %278, %279, %280, %281, %282, %233, %293, %298, %303, %308, %313, %318, %323, %328, %329 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %193 = stablehlo.not %163 : tensor<20x20xi1>
    %194 = stablehlo.and %158, %193 : tensor<20x20xi1>
    %195 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %196 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %197 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %198 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %199 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %200 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %201 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %202 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %203:7 = stablehlo.while(%iterArg = %194, %iterArg_0 = %7, %iterArg_1 = %196, %iterArg_2 = %198, %iterArg_3 = %arg1, %iterArg_4 = %200, %iterArg_5 = %202) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %225 = stablehlo.constant dense<false> : tensor<i1>
      %226 = stablehlo.reduce(%iterArg init: %225) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %227 = stablehlo.or %arg2, %arg3 : tensor<i1>
        stablehlo.return %227 : tensor<i1>
      }
      stablehlo.return %226 : tensor<i1>
    } do {
      %225 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %226 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %227 = stablehlo.add %iterArg_0, %226 : tensor<20x20xf32>
      %228 = stablehlo.divide %iterArg_3, %227 : tensor<20x20xf32>
      %229 = stablehlo.multiply %iterArg_1, %228 : tensor<20x20xf32>
      %230 = stablehlo.add %iterArg_2, %229 : tensor<20x20xf32>
      %231 = stablehlo.divide %229, %230 : tensor<20x20xf32>
      %232 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %233 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %234 = stablehlo.compare  GT, %231, %233 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %235 = stablehlo.and %iterArg, %234 : tensor<20x20xi1>
      %236 = stablehlo.select %iterArg, %227, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %237 = stablehlo.select %iterArg, %229, %iterArg_1 : tensor<20x20xi1>, tensor<20x20xf32>
      %238 = stablehlo.select %iterArg, %230, %iterArg_2 : tensor<20x20xi1>, tensor<20x20xf32>
      %239 = stablehlo.divide %iterArg_3, %227 : tensor<20x20xf32>
      %240 = stablehlo.multiply %iterArg_4, %239 : tensor<20x20xf32>
      %241 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
      %242 = stablehlo.constant dense<-1.000000e+00> : tensor<20x20xf32>
      %243 = stablehlo.multiply %242, %iterArg_1 : tensor<20x20xf32>
      %244 = stablehlo.multiply %243, %iterArg_3 : tensor<20x20xf32>
      %245 = stablehlo.multiply %227, %227 : tensor<20x20xf32>
      %246 = stablehlo.divide %244, %245 : tensor<20x20xf32>
      %247 = stablehlo.add %240, %246 : tensor<20x20xf32>
      %248 = stablehlo.select %iterArg, %247, %iterArg_4 : tensor<20x20xi1>, tensor<20x20xf32>
      %249 = stablehlo.add %iterArg_5, %247 : tensor<20x20xf32>
      %250 = stablehlo.select %iterArg, %249, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %235, %236, %237, %238, %iterArg_3, %248, %250 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %204 = stablehlo.or %11, %156 : tensor<20x20xi1>
    %205 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %206 = stablehlo.constant dense<0x7FC00000> : tensor<20x20xf32>
    %207 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %208 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %209 = stablehlo.compare  EQ, %arg1, %208 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %210 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %211 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %212 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %213 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %214 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %215 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %216 = stablehlo.exponential %147 : tensor<20x20xf32>
    %217 = stablehlo.multiply %192#1, %216 : tensor<20x20xf32>
    %218 = stablehlo.subtract %215, %217 : tensor<20x20xf32>
    %219 = stablehlo.multiply %203#3, %216 : tensor<20x20xf32>
    %220 = stablehlo.divide %219, %7 : tensor<20x20xf32>
    %221 = stablehlo.select %163, %218, %220 : tensor<20x20xi1>, tensor<20x20xf32>
    %222 = stablehlo.select %2, %213, %221 : tensor<20x20xi1>, tensor<20x20xf32>
    %223 = stablehlo.select %209, %211, %222 : tensor<20x20xi1>, tensor<20x20xf32>
    %224 = stablehlo.select %204, %206, %223 : tensor<20x20xi1>, tensor<20x20xf32>
    return %224 : tensor<20x20xf32>
  }
}
