// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xf16>, tensor<20x20xf16>)
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = call @igamma(%0#0, %0#1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<20x20xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xf16>, tensor<20x20xf16>) {
    %0 = stablehlo.constant dense<"0x02382EBEF24017BB93402BB9B6C2283527BFD7BBB6B84E43143A4E38A6C4943EA24263461D4341391D395546FAC35BC42EBC9E3A03C5B24025300A417E464BC5693E403C60C1D4C2072B11386B3F2FBD48C182BB75448EC13DAA0340FE444DB9E1417F4072427CB5C8C449C1F6B8D14358C6DD412B3E9AB9D2C03E3D063C27BDD7C4C3C37C417FB82EB934418EC3F4423FC4B4B639C51EBDF3A83236C1C1EE44433B22B5CF4056308FA63A439741363931C5B3442EC433445F420739DFC4C63583A900BEE9BABDC03FAC8F4268C3FFB913403BC48E3845C55B432D3C95C0503614BCFE45B7BE0246973467BEE1C64FC0CBC0783BECC01BC3533DC7BDC838374513C5EC3FF6401AC310C35FBC27437B37F2401EBD1E3FB5C42C3E90C12B3EF9C006C0DFC1D23D2D41D43D93C0C544123A6E3CB33704BB913C1040FF4441B4EA38DFBA5C31F53DA5BAF4448DC434427FBD7D43D743E7BC07BD76BCC8BC773EA0B45041B54200C12E41EDB7E646494134C1B0A9F5B4E73B94400D3C4F3D5AB184C1FE35E735ABBE6A3ECF423B364947893DD0BFEF3AADB918405C348FC46AC2843FB1BFBCBFFF424CC15FB6A8B2D84345C85041A83B22BA3F44BAC3803974C5EDBBA540994202C4C742ABAD4C35143F4D3346C51744843938A923BDAA396C43D6C0AF42AABC0244834037B62F3975BD59AF96B9073523AD0645F1C545B1EF391DBC7340BBBFF5C229C57E43793FCEC259C52AC1ED41F03E2EC466C0BBBDD844CD3F6740CE416FC4E3C15FC18A3F38C46642B9BEF03F8B4124C0A0BEBAB3854951C28AC4524253B4E8BC05B933B6E6BAFA3C703951BC62B6ACC52AB2DE4233BA053D25415FC0E4313AC639BF06BB903C9CC491BE65C802B9C13EE6C2E3C07041ED35C8BD6B445F413D3E07B40140E03372B131B404C0BD375242F1BEFE3FAD3F8A3E38C807B7C0C0B03AFBBC6143104479B9BABC2027EDBDB2BD2EB9313D914456410038D4BEF7C04D3CD23F4142993ACAA83EBFAB41D8C0463AB1BA9A45CA3D92BA6B4008439ABECCB600C52BC137B62EBA12BDC935C141A5325442684545AD4A3011C53E42EBBC40C09BC48F3ED0B83F37C83E82B128B90143BDBE074123BE0443"> : tensor<20x20xf16>
    %1 = stablehlo.constant dense<"0x1D444BBC7AC278467744F23A193A1144C343C8AEE2B83B41B5C0FB4242C555C58FC462C2EA3EC3BC41C166C56341054428BBC1415CBC5B40C04535C2DF427CB458406640BE316DBF7C391144D13FFA36F3C076C60EBCDBBB63BC5238F2C6F345B338F9BCCB39403A3C3D0ABD48C30643FE3A99BF4FC72DC8B9B76F41C3B7314535BF6741E23930B6E4BEC6BA27BC8341AA4368365547EEC5C6BC284102C143C2E0C0823C93B118B8ED39D63EABC4E73D6B445CBFF1C175C68AC020C6E2364238223EDB3C75417A35F83E1B46EA42A2BD4840EC41B1BF29C3904491B86CBDB2BF15C4C237C140D6BFC8406C383542B1C5203963BECC419C343C3AB1C03D3B9D36D2C26D3D513E00BF2AC4F4389BC1BFC546BC9BC540439ABD3C347CC56B45943748C1AAC3FAC0633824BD044378C5C1BC11B252C3C9BDE2C2A13E443FD045C4C612A1513562C02B4366C663C00645F6C0214000B695C635C129C4B64290C00344B23FA544F4C10EC459C69443C1C34244FD43D5BB64C4FEC1C3BAB341E8B6E839304035C1B44016C021C57A388D4008C0FD3C24C361C307C52EC119C4C1BA393B9DC4FFBFFDC46D4754BF52B924C5A140B03DB4BB78B0A42816C26EC1953C03C4863E1CC007C557BD28BF3BC03AC501C18541453C9BB4E23DCA42C535D9BE094312C4274257BDDE438E43984103412639BA478A4404BE633FAB3818B9124076BEB3C203AAC241D33F4C40DFC0FEBCC8BD4AC022BBECC47D452BBF46B9873D134073C474BB58412F3E11BCDE3827C591BF4CBC713C00C5C4C502B10FC14D44DF442CC4463E6AC12AC1B0C2A84038BE9C3C49B9353D0FC154BDDDC44442623C1EC026B4C7442F30E5BF5F2812C0AF3C4AC075B62840ABC4C4BED9C16941BD4476B5C443C83EB13901BEE242F7BC6345063315C7A0B020AE2F4512B597405F40864040BD9342B0B9323C5F4494C2A74115B9ABC03F449E42CCC477C3823EE631D9BB6AC2D643C5309ABBC1B373B3C6C46AC290BFE440E23FB2C26044BB43A044CA4071C3974221BD5D43ADBE5BC1B7302B46AE4068C34A3C1C411DB9F3C1B5C09FB80D43853D87C19D383DC39E39EEB38D4084C07FC13B3ED240A6C435B9F53579BF"> : tensor<20x20xf16>
    return %0, %1 : tensor<20x20xf16>, tensor<20x20xf16>
  }
  func.func private @expected() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0xF83B007E007E007E4A3B007E007EFC3B007E007E007E5E35007EED3B007E007E007E007EDC30007E007E007E007E007E007EAF3B007E5738003C007EE22D007EFA39043B007E007ED33BF73BFF38007E007E007E007E007E007E812E007E007E4226007E9326007E007E007E007EC437007E007E007E007E007E1E3B007E007E007E007E6C2B007E007E007E007E8536007E007E007E007E007EE03B007E007E007E007E007E007E007E7B30007E023B007E007E007E007E007E007E007E493A007E007E007E007E007E643B007E007EF638007E007E007ECF39007E007E007E007ECC00007E007EE63B007E007E007E007E007E007E007E5436007E3B3AAA02007E63364635007E007E007E007E007E007E007E3E3B007E202D007EE43B007E007E007E007E0029007E007E007E007E007E007E007E007EBD37D628007E007E007EDC3A007E007E007E007EED3A007E2B32007E007E007E007E007E007E007E3036533A007E007E007E372E007E007E007E007E007E007E007E333B007E007EC93B007E007E007E007E393A591E007E007E007E007E007E007E007E007E2F34007E007E007E007E007E007E007E007EAA33007E007E0000007E007E007E007E3836007E007E007E007E007E007E007E007E3D25007E007E007E6537007E007E007E007E007EF93A007EC73B007E007E007EFE3B007EF828007E007E5B3B007E007E007E007E007EBC32007E007E007E007E007E007E007E007E007E5324F338007E007E007E007E007EBA30007E007E007EF334007E007E007E007EC11C007E007EBD31007E007E007E007E007EB238007E007E007E007E007E0C38007E007E007E007ED839007E007E007E1339007E007E007E007E007E007E007E153B007E007E7B2B882B007E007E007EFF3B007E007E007E007EFB3A007E593951390D3A007E007E007EC039007E007ED534007E007E003C007E007E007EB2396E00007E007E007E007E007E007E007E007E007E007EBD37007E007E007EED33CF3B007E007E3138007E007E007E007E007E007E007E007E052FF23B007E007E007E007E007E8E30007E007E007EFE33007EC63B007E007E007E1535007E007E007E007E"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @igamma(%arg0: tensor<20x20xf16>, %arg1: tensor<20x20xf16>) -> tensor<20x20xf16> {
    %0 = call @xla_fallback_igamma(%arg0, %arg1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @igammac_body.201(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>> {
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
  func.func private @or.324(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @igammac_condition.328(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<i1> {
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
  func.func private @igamma_body.380(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>> {
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
  func.func private @or.417(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @igamma_condition.421(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<i1> {
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
  func.func private @xla_fallback_igamma(%arg0: tensor<20x20xf16>, %arg1: tensor<20x20xf16>) -> tensor<20x20xf16> {
    %0 = stablehlo.convert %arg1 : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %3 = stablehlo.compare  EQ, %0, %2 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %6 = stablehlo.compare  LT, %0, %5 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %7 = stablehlo.convert %arg0 : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %10 = stablehlo.compare  LE, %7, %9 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %11 = stablehlo.or %6, %10 : tensor<20x20xi1>
    %12 = stablehlo.or %3, %11 : tensor<20x20xi1>
    %13 = stablehlo.log %0 : tensor<20x20xf32>
    %14 = stablehlo.multiply %7, %13 : tensor<20x20xf32>
    %15 = stablehlo.subtract %14, %0 : tensor<20x20xf32>
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
    %155 = stablehlo.compare  NE, %0, %0 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %156 = stablehlo.or %154, %155 : tensor<20x20xi1>
    %157 = stablehlo.or %153, %156 : tensor<20x20xi1>
    %158 = stablehlo.not %157 : tensor<20x20xi1>
    %159 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %160 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %161 = stablehlo.compare  GT, %0, %160 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %162 = stablehlo.compare  GT, %0, %7 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %163 = stablehlo.and %161, %162 : tensor<20x20xi1>
    %164 = stablehlo.and %158, %163 : tensor<20x20xi1>
    %165 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %166 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %167 = stablehlo.add %0, %166 : tensor<20x20xf32>
    %168 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %169 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %170 = stablehlo.subtract %169, %7 : tensor<20x20xf32>
    %171 = stablehlo.add %0, %170 : tensor<20x20xf32>
    %172 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %173 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %174 = stablehlo.add %171, %173 : tensor<20x20xf32>
    %175 = stablehlo.multiply %174, %0 : tensor<20x20xf32>
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
    %188 = stablehlo.negate %0 : tensor<20x20xf32>
    %189 = stablehlo.multiply %176, %188 : tensor<20x20xf32>
    %190 = stablehlo.subtract %187, %189 : tensor<20x20xf32>
    %191 = stablehlo.divide %190, %175 : tensor<20x20xf32>
    %192:15 = stablehlo.while(%iterArg = %164, %iterArg_0 = %176, %iterArg_1 = %178, %iterArg_2 = %170, %iterArg_3 = %174, %iterArg_4 = %179, %iterArg_5 = %167, %iterArg_6 = %175, %iterArg_7 = %181, %iterArg_8 = %0, %iterArg_9 = %183, %iterArg_10 = %185, %iterArg_11 = %187, %iterArg_12 = %188, %iterArg_13 = %191) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %226 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
      %227 = stablehlo.compare  LT, %iterArg_4, %226 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %228 = stablehlo.constant dense<false> : tensor<i1>
      %229 = stablehlo.reduce(%iterArg init: %228) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %231 = stablehlo.or %arg2, %arg3 : tensor<i1>
        stablehlo.return %231 : tensor<i1>
      }
      %230 = stablehlo.and %227, %229 : tensor<i1>
      stablehlo.return %230 : tensor<i1>
    } do {
      %226 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %227 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
      %228 = stablehlo.add %iterArg_3, %227 : tensor<20x20xf32>
      %229 = stablehlo.multiply %iterArg_6, %228 : tensor<20x20xf32>
      %230 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %231 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %232 = stablehlo.add %iterArg_2, %231 : tensor<20x20xf32>
      %233 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %234 = stablehlo.add %iterArg_4, %233 : tensor<f32>
      %235 = stablehlo.broadcast_in_dim %234, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %236 = stablehlo.multiply %232, %235 : tensor<20x20xf32>
      %237 = stablehlo.multiply %iterArg_8, %236 : tensor<20x20xf32>
      %238 = stablehlo.subtract %229, %237 : tensor<20x20xf32>
      %239 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %240 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
      %241 = stablehlo.compare  NE, %238, %240 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %242 = stablehlo.multiply %iterArg_11, %228 : tensor<20x20xf32>
      %243 = stablehlo.subtract %242, %iterArg_5 : tensor<20x20xf32>
      %244 = stablehlo.multiply %iterArg_9, %236 : tensor<20x20xf32>
      %245 = stablehlo.subtract %243, %244 : tensor<20x20xf32>
      %246 = stablehlo.broadcast_in_dim %234, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %247 = stablehlo.multiply %iterArg_7, %246 : tensor<20x20xf32>
      %248 = stablehlo.add %245, %247 : tensor<20x20xf32>
      %249 = stablehlo.multiply %iterArg_5, %228 : tensor<20x20xf32>
      %250 = stablehlo.multiply %iterArg_7, %236 : tensor<20x20xf32>
      %251 = stablehlo.subtract %249, %250 : tensor<20x20xf32>
      %252 = stablehlo.divide %251, %238 : tensor<20x20xf32>
      %253 = stablehlo.select %241, %252, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %254 = stablehlo.multiply %iterArg_12, %228 : tensor<20x20xf32>
      %255 = stablehlo.subtract %254, %iterArg_6 : tensor<20x20xf32>
      %256 = stablehlo.multiply %iterArg_10, %236 : tensor<20x20xf32>
      %257 = stablehlo.subtract %255, %256 : tensor<20x20xf32>
      %258 = stablehlo.broadcast_in_dim %234, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %259 = stablehlo.multiply %iterArg_8, %258 : tensor<20x20xf32>
      %260 = stablehlo.add %257, %259 : tensor<20x20xf32>
      %261 = stablehlo.multiply %253, %260 : tensor<20x20xf32>
      %262 = stablehlo.subtract %248, %261 : tensor<20x20xf32>
      %263 = stablehlo.divide %262, %238 : tensor<20x20xf32>
      %264 = stablehlo.select %241, %263, %iterArg_13 : tensor<20x20xi1>, tensor<20x20xf32>
      %265 = stablehlo.subtract %264, %iterArg_13 : tensor<20x20xf32>
      %266 = stablehlo.abs %265 : tensor<20x20xf32>
      %267 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %268 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %269 = stablehlo.select %241, %266, %268 : tensor<20x20xi1>, tensor<20x20xf32>
      %270 = stablehlo.subtract %iterArg_0, %252 : tensor<20x20xf32>
      %271 = stablehlo.divide %270, %252 : tensor<20x20xf32>
      %272 = stablehlo.abs %271 : tensor<20x20xf32>
      %273 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %274 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %275 = stablehlo.select %241, %272, %274 : tensor<20x20xi1>, tensor<20x20xf32>
      %276 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %277 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %278 = stablehlo.compare  GT, %275, %277 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %279 = stablehlo.and %iterArg, %278 : tensor<20x20xi1>
      %280 = stablehlo.select %iterArg, %253, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %281 = stablehlo.select %iterArg, %275, %iterArg_1 : tensor<20x20xi1>, tensor<20x20xf32>
      %282 = stablehlo.select %iterArg, %232, %iterArg_2 : tensor<20x20xi1>, tensor<20x20xf32>
      %283 = stablehlo.select %iterArg, %228, %iterArg_3 : tensor<20x20xi1>, tensor<20x20xf32>
      %284 = stablehlo.abs %251 : tensor<20x20xf32>
      %285 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %286 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %287 = stablehlo.constant dense<0x4B000000> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %287, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %289 = stablehlo.compare  GT, %284, %288 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %290 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %291 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %292 = stablehlo.multiply %251, %291 : tensor<20x20xf32>
      %293 = stablehlo.select %289, %292, %251 : tensor<20x20xi1>, tensor<20x20xf32>
      %294 = stablehlo.select %iterArg, %293, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      %295 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %296 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %297 = stablehlo.multiply %238, %296 : tensor<20x20xf32>
      %298 = stablehlo.select %289, %297, %238 : tensor<20x20xi1>, tensor<20x20xf32>
      %299 = stablehlo.select %iterArg, %298, %iterArg_6 : tensor<20x20xi1>, tensor<20x20xf32>
      %300 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %301 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %302 = stablehlo.multiply %iterArg_5, %301 : tensor<20x20xf32>
      %303 = stablehlo.select %289, %302, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      %304 = stablehlo.select %iterArg, %303, %iterArg_7 : tensor<20x20xi1>, tensor<20x20xf32>
      %305 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %306 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %307 = stablehlo.multiply %iterArg_6, %306 : tensor<20x20xf32>
      %308 = stablehlo.select %289, %307, %iterArg_6 : tensor<20x20xi1>, tensor<20x20xf32>
      %309 = stablehlo.select %iterArg, %308, %iterArg_8 : tensor<20x20xi1>, tensor<20x20xf32>
      %310 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %311 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %312 = stablehlo.multiply %iterArg_11, %311 : tensor<20x20xf32>
      %313 = stablehlo.select %289, %312, %iterArg_11 : tensor<20x20xi1>, tensor<20x20xf32>
      %314 = stablehlo.select %iterArg, %313, %iterArg_9 : tensor<20x20xi1>, tensor<20x20xf32>
      %315 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %316 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %317 = stablehlo.multiply %iterArg_12, %316 : tensor<20x20xf32>
      %318 = stablehlo.select %289, %317, %iterArg_12 : tensor<20x20xi1>, tensor<20x20xf32>
      %319 = stablehlo.select %iterArg, %318, %iterArg_10 : tensor<20x20xi1>, tensor<20x20xf32>
      %320 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %321 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %322 = stablehlo.multiply %248, %321 : tensor<20x20xf32>
      %323 = stablehlo.select %289, %322, %248 : tensor<20x20xi1>, tensor<20x20xf32>
      %324 = stablehlo.select %iterArg, %323, %iterArg_11 : tensor<20x20xi1>, tensor<20x20xf32>
      %325 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %326 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %327 = stablehlo.multiply %260, %326 : tensor<20x20xf32>
      %328 = stablehlo.select %289, %327, %260 : tensor<20x20xi1>, tensor<20x20xf32>
      %329 = stablehlo.select %iterArg, %328, %iterArg_12 : tensor<20x20xi1>, tensor<20x20xf32>
      %330 = stablehlo.select %iterArg, %264, %iterArg_13 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %279, %280, %281, %282, %283, %234, %294, %299, %304, %309, %314, %319, %324, %329, %330 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
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
    %203:7 = stablehlo.while(%iterArg = %194, %iterArg_0 = %7, %iterArg_1 = %196, %iterArg_2 = %198, %iterArg_3 = %0, %iterArg_4 = %200, %iterArg_5 = %202) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %226 = stablehlo.constant dense<false> : tensor<i1>
      %227 = stablehlo.reduce(%iterArg init: %226) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %228 = stablehlo.or %arg2, %arg3 : tensor<i1>
        stablehlo.return %228 : tensor<i1>
      }
      stablehlo.return %227 : tensor<i1>
    } do {
      %226 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %227 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %228 = stablehlo.add %iterArg_0, %227 : tensor<20x20xf32>
      %229 = stablehlo.divide %iterArg_3, %228 : tensor<20x20xf32>
      %230 = stablehlo.multiply %iterArg_1, %229 : tensor<20x20xf32>
      %231 = stablehlo.add %iterArg_2, %230 : tensor<20x20xf32>
      %232 = stablehlo.divide %230, %231 : tensor<20x20xf32>
      %233 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %234 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %235 = stablehlo.compare  GT, %232, %234 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %236 = stablehlo.and %iterArg, %235 : tensor<20x20xi1>
      %237 = stablehlo.select %iterArg, %228, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %238 = stablehlo.select %iterArg, %230, %iterArg_1 : tensor<20x20xi1>, tensor<20x20xf32>
      %239 = stablehlo.select %iterArg, %231, %iterArg_2 : tensor<20x20xi1>, tensor<20x20xf32>
      %240 = stablehlo.divide %iterArg_3, %228 : tensor<20x20xf32>
      %241 = stablehlo.multiply %iterArg_4, %240 : tensor<20x20xf32>
      %242 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
      %243 = stablehlo.constant dense<-1.000000e+00> : tensor<20x20xf32>
      %244 = stablehlo.multiply %243, %iterArg_1 : tensor<20x20xf32>
      %245 = stablehlo.multiply %244, %iterArg_3 : tensor<20x20xf32>
      %246 = stablehlo.multiply %228, %228 : tensor<20x20xf32>
      %247 = stablehlo.divide %245, %246 : tensor<20x20xf32>
      %248 = stablehlo.add %241, %247 : tensor<20x20xf32>
      %249 = stablehlo.select %iterArg, %248, %iterArg_4 : tensor<20x20xi1>, tensor<20x20xf32>
      %250 = stablehlo.add %iterArg_5, %248 : tensor<20x20xf32>
      %251 = stablehlo.select %iterArg, %250, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %236, %237, %238, %239, %iterArg_3, %249, %251 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %204 = stablehlo.or %11, %156 : tensor<20x20xi1>
    %205 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %206 = stablehlo.constant dense<0x7FC00000> : tensor<20x20xf32>
    %207 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %208 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %209 = stablehlo.compare  EQ, %0, %208 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
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
    %222 = stablehlo.select %3, %213, %221 : tensor<20x20xi1>, tensor<20x20xf32>
    %223 = stablehlo.select %209, %211, %222 : tensor<20x20xi1>, tensor<20x20xf32>
    %224 = stablehlo.select %204, %206, %223 : tensor<20x20xi1>, tensor<20x20xf32>
    %225 = stablehlo.convert %224 : (tensor<20x20xf32>) -> tensor<20x20xf16>
    return %225 : tensor<20x20xf16>
  }
}
