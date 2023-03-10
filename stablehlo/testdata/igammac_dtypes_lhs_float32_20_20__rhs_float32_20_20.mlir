// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xf32>, tensor<20x20xf32>)
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = call @igammac(%0#0, %0#1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xf32>, tensor<20x20xf32>) {
    %0 = stablehlo.constant dense<"0x5EEF0B4090BF873BCBE8CC401DA1ED3FB8B82440514398402FCDA2C0C17584C07389DC3F739E583FC71AB940E254D53FD4828B40186E203F0DC199BF1C24B140A714DFBFB1B7553D6E01634070F45A4090F495BF5739FE3DD6A695BFC5D09640B0AF1F3E66438640F2708EBEC8F9A140D8FA0DC000741EC0939FE9C050DFD03E50D99DBE65294D40B8761D3EC13F0E40E0C2EABFBC20FCBDAAF68640EE341A40A767513E30FDD03F1BC302C0556A4B4020A845C032DA91BEBE9AC33F37B33CC001C78EC0CF0174C0AFA532BF4622A6BE77F7D0C025174DC06FEC6C3FC85F0A40373B31404A7B29C071552B40BA3E973FE25AD2BFAF51243F06F91E406CF65C4047E332BF037F3FC060928D3F9F01FFBF861462BF7F63A2BFC47F4840D81756C0EF7C20BF6DC21FC02D1C434080FC6F4010207D404605FEBF439F8C40FECA613F72E3B2C0A34D8A403BA1F8C0E2AFC4BFBE8677BD9C4D253F9DC740C040A74DBF2568954023A8DE40B11CEBBFED8D3D404F91CF3F63D5F1BF139EF2C059C8C2BF740FA2BF89E6A4BFB1D190408F5FD8BF7DA0EBBEFB7639C0ABDD90C0399B8B3F384EC0BE89DF63C02E7013C027A287406FB8123F60E605C03EE6FBBF50134C3EEFC08240BDCF3AC0F44429C0F34A38407FF4103F5ACDA44046D06FC094CF6E3F5049B33FE7D995405A1EC03F599B7EBF215BA0BEFB6F3B3FA4AB44C0E07BA5BFF75F3F4002C94D40069AFC4016DD96BF82702DC0D484284083694EC04439AA3FC1F14540199C2BC035F616C0A8BC42C0E5E0BE3F53B98B3FEFD9903F23B8DD3EE02D67BE56CB12C06C7B0BC003B5EF3E442937BFBC91874073577ABFAD78BD3FA1F0B63EF76684BF76025FBF92EEDE3FD8E95CC055DDD8C0A7DC98BE054ADFBF1876DABC7737403F54E43BBEAF9B2040A9F759C0A50E19C0BC2CF3BE7C2901C0C914C5BFDAC5BBBD97DC543FC736133F960A853CD685B9BE693ACABE15D63EBF7E4302C0B5C041406380023FF42DBCBE9B92D74094F7EA3F1A122B3FD8B8C23F7F11CE3F481C903DA1EC893FCA275040FC3CD6BF736CBEBE7D853140D16C92C01F642C40E08EA4BF33FF8340D4A539BF5E336B40B65A543F3BE30EC0C6A71EC0537B3240AFB95DBFFCA41AC0440A2FC0A6E18C40AC8B14C07B544DC01A3C2AC073208BBF37C0EB3F7676A540DF9FE13FC6A0B7C066FC9E3FC149623FE64E33C02DA4803F3E8F50BE78DD1A40BB75B0403F8A98C034E81CC09BA06BC045A54440170BCD403C9E9E3F0D7E1440F2DD7CBF76ECE43FA4F6634021E7A63F7A6C263FAB6C034091E42740A646A8C0622A54408A049D3EAD16413FC7061CBEF08F9CBE38352D40DC333440494C9B3FF188E7C0068730402C0D2A404F2795C06B46B4BEAF42083F180DC5BF2AC845BF9EBF533F4F3282BF73A52540205E413D4B4442C0A85EDAC02AEC0AC0392E68407E51BABE321DEFBF443535C01DCF97C01D5A8F40CB51A13F10E0D03E6ECF4FBF1C3EA13F86422AC0947040C0D447AFC07363C6402657A040F4AA9840A672BFBF3AAF7C4038F699406FC118403ABD68C0119E91BFC28176BE5868F33FB4357040D0338ABFC9B1BFC05EC1DDBF3B194A400B301F40D0EAF1BD845A50C0B2890FC027E4FE3DE2226FBF81CDCDBDB5D30540062D70407EC283C0176F2BC094E8A4BED096E43DC12725403281BAC0E3BE414076A8BDBF9D49373F34171C4067930CBFD1AD873F454671BE8F0E07C0B253EABE380C76BE03DECEBF82B89940146184C0AF08C1BF33B3B7C04B67353F8FBA0740E1EDEBBFEA530E40A6618DBF8B3C85C08F776AC0FE3CA640F00993BE399DAD3F88E98C400D3BCFBF0A7AD43DB2231BC03DCCD1BFFD62A73F9E70B340A24B33BF119549C0B3EF844063123FBF2584C63F9848B5C0326326BFB3E1983D5BC104C0D7A1EB3D8DE1A13E424EA14079B2FCBE15B235C03051A83FADEE1540ECFCB9C0094DC5BF400645C0C5B823C002474FBE5C12AFBFF7A4BF3F224C10C058F0A6BFB590EB40BFE46FC0988E89405B269B407170B33E6A12BEBF877268BF46E3163DA6A47A3F00AF0C40F220C63F4D57B13F9BF756C039B447BFF8F98D406454A43F359798C037886240DF06C3C0486877C0A4C9A9BEB68601C04D0E9A3E68209C3F8E54F0C042B70E408C0E7840E0EEBF406AE31A40AACFC5401216BE3FE8B8024073A43540AF9382C093A6C3BFA22FB8C012F88D3E184119C067B6754031738C408CE0E1C0"> : tensor<20x20xf32>
    %1 = stablehlo.constant dense<"0x035498C028EB1AC008D9803F04C04CBFCD1F3DBE92583FC0C0673EBF168CA03FD77155BFBF4C8B3E9871423F6E9A2A40015F11C024744EC0C761DA3FF8EADB3FBA2C01C028A479BD67167BBF6D5CB5BFF7B02E4040CCC4BFA4E4D2BC1D861BC099A7A33EA2F83340DAAB8240BB3B1AC0AFBD69C030A9DC3FFEB83FBE871E2A401F8C96BF3C5939C0588189C0CC4B38C04859313E39710FC0BEF509408ED6133FA0FF3DBFD64C4AC05B6281C09EF791BFFDA154405AA01AC06FD6E63F995DF9C0EC6A2B405CB11EC068B19AC04F3714402F27ED3F42037F3EABE5B4BF721E8C40378A8C3FE6CFCEBFEEC552C0D98064C05E5B7340BFDA9A40810D5F40C756034079ADA140A086E43F079B34C0930DBD3E63254DC0181576BFAEF2853FE90DC2C094AE54C058D533C082621340BAA10540FFDB54C0FA3E65BE06764DC04150284048C7443FCB7F95BE778364406FCB4ABE40F9123FF35808400BFF144198719840960EB63F5E4AC53F2D82C9C08CC41041A4D77D40E9DAADBE0DC2463FF786C3BFB2A73BBF254D30406472953FE38F7DC09048DAC0E4235440994533C0856022C02D06A3407319FCBF8E48A8BF21187E40FF4816BF6AAC67BF0A2A6240B9806BC06E7766C05102CB4065451F4099B643BFD5B7AFC0EB5F7640ABEE0FC1190FB2BFD1655AC034AEB83FD4A56440FA487E40C3179EC0F3435EC0A4404CC0C8E26BBF378BC5BF785EC03FF4CA184055577A409AA5493F5BFB9240B97D0B40310218C09A4B2040AAF0F2BF58F70A40179D4A40406E0D40AAB5ABC0AE0785C0AC40B73F56B5F9BD52139C3FEFD57CBF4AB5AD4066220A401EAC283EDCAB3B3F758B044018648B3F685A4A40ABDC3F4096A9C5BEECBD4D40200371BF4F45A33F824DCEBEF0364C402A641DC082B1BD40462985C0418E98C0DCFA22BF6D6A16C01633923F83A778BF822F6B3FEA3728405AC0DE3F80A69CC0D9EA0340FC690BC0713E92BE72A3FFBE2C11154067826840DB3A07BF81BA6E405805CC3E1BE077BF581980C0B8908840712B6EC063E895C046618FC0F10AEBBF0A898DBE443A90C0BE6143BF179CABBFBC85243F417C66C0947627C07490C14066FD993F3F2C37407EF48F40CBE79B3FC7450740173C62BFF00148BE86029AC0817A20C04AD31FC0242135C01C5E8FBFB9DD08C0E5750EC03129CC3F8DB32A40488716BFD2823640DB665540438343BFA178CC3FC018EEBFFE7B524024A968C09A22823F4CD18840816490BF42C685402834F6BF57C25FBFB25BAC3D9DBB5540FD798D40EAE58140C7F7A2BF924D413FD99BD0C0802B77C0C2A586408B31B2BF5DCC80BFA75B63C02C7A133FFE2D87C0B6A49A3F733B0DC085E210BF7C092C40E8F63640EA300040CD95AC40BB168B4062FE16404C7CF83FC80F8040CFA780BF5938BA3EEBFBE93F317617BED2FB32C02A61A84088108EBFEB7366C075F44B409BCFE8BFBA1A1AC06E7A01C0E513FBBF48DB51C0DD7F92C0D5FF1D4042F38FC0EB3404C03DA1C7C0C5AFA5C022DDADC0AAF6BE40A738C640AF9501C09B9DAEBE5EFC533F2A3A28C0FF68FDBF1474DEBFB1CB5ABED33002C075C21B40A8BD58C0687CB1BFC64F564086C835C029379EBF7123874001E2CABFF47A74C0F3DFCD3F71B50B3FD99F64C026FD90C0307844C0753E17C041ABE9BF1D50B2BF22888DC0ABCEBABFB49F2EC009D0C93F02AF194055D7723F3D2DAC3F359CDABFEF8AB8C0ACF45D3F8A5993C0E5540840884E99BFC2D633401556A34007412A3F20EEDABE4EE72B41E6A58BC0136B983E2D8A1040845105C0BA82B9C094D034BF860419BF2B1B633F7230123F2BCB51C06B260DC066FD2EC0C839C44047E8A63FBCEF5FBF948CE33ECA0CF2BDB669C43FF97A1140FB2F36406107AABFC02DDDC0955315C01D7FACC09529C340B9D3333F55BAB8C0D44DDEBF8C12CBBF0549B6BFC9D3CBBF873A1040346B8EC0AF6758C04B27EFBFCA4F76C0588F803E9E551341AD9943BF917E9AC0E54931C030E69BBFA566B53D3A9328C0ACB37E408E1D3940879AC1BE1FEF38C044A57C3D484F7AC0F3345240D1D751402F4B95C08798EABF8F556340533C35BFA1782D3F1CEB9640488FF63ED5FFF43E8AA985BF9E4FB440FF9CEBBE6A206ABF5296AC3CBBC671BF8EC4A5408DEE92BFCA6FB8C0FEB54C40F3270C40309916BF2CD637C0ABC0A3C0EAAC803E26C905C09FCF53C057A0C9C0F19D12407542BCBF76FBD33E740ED53F"> : tensor<20x20xf32>
    return %0, %1 : tensor<20x20xf32>, tensor<20x20xf32>
  }
  func.func private @expected() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0x0000803F0000803F6AED7F3F0000803F0000803F0000803F0000803F0000803F0000803F0036303F74F17F3F9B3B3A3E0000803F0000803F0000803FFEFE7B3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F4CF40B3EA2CB393F0000803F0000803F0000803F0000803F0000803FAAD27B3C0000803F0000803F0000803F0000803F0000803F0000803FDA865B3F05D3703F0000803F0000803F0000803F0000803F0000803F0000803F73ADA13E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F8591A83DC76C5E3F0000803F0000803F0000803F0000803F3A04463B411C613E4940423F0000803F0000803F0000803F0000803F0000803F0000803FCDD16C3F0000803F0000803F0000803FAD691B3F909A4D3F0000803F0000803F0000803F03FA6A3D0000803F0000803F0000803F0000803F0000803F7C946D3D0000803F0000803F54E4793F32B47F3F0000803F0137BA3B8B81693D0000803F0000803F0000803F0000803F0000803FAD4D7C3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FB4CBF93E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F40202F3F0000803F0000803F0000803F39B1793FA03E8A3D0000803F0000803F0000803F0000803F0000803F0000803F6E41583F890C7F3F0000803F0000803FCEEFEE3D0000803F0000803FD2BB103F0000803F0000803F0000803F489D5E3E0000803F0000803FBBC5983D0000803F0000803F0000803FA907643A0000803F33FF7F3F0000803F320E773E282ABC3D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FB7BD533DE2D29B3D0000803F0000803F0000803F0000803F0000803F8F52183F8A24EE3B0000803F1289653FB8C16A3F0000803F0000803FA018303D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FDE27E83D0507713E0000803F0000803F32A4573F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F5E8FE63E0000803F0000803FAC983B3D0000803F0000803F0000803F0000803FD279553F0000803F0000803F0000803F0000803F8A034F3F0000803F0000803F0000803F39D5F73D97D78C3E9F91003D0000803FA32D563F0000803F0000803FB6E7863E0000803F0000803F0000803F0000803F0000803F1A71593F0000803F0000803F2278DF3ED6A7BE3E0000803F0000803F48A36A3B0000803F0000803F1F71483C0000803F11127C3FD2184E3B0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F8F93033E0000803F0000803F0000803F0000803F0000803FCCA9823E0000803F0000803F0000803FED7F603F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FCBF5C43E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F7F62503DFA3B5A3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F4640523F0000803F0000803F0000803F859F09370000803F0000803F34D2CD3E0000803F0000803F0000803F0000803F0000803F86C8373F0000803F0000803F0000803F0000803F0000803F0000803FA3FE7F3F0000803F0000803FC8EB533F0000803F0000803F0000803F0000803F0000803F0000803FD0B23D3D0000803F0000803F0000803F0000803F0000803F2836DF3E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F7EE61A3F50E0173C0000803F0000803F7829A63D0000803F4D64483E1E68BF3D0000803F0000803F0000803F0000803FEC8B223F0000803F36EB7E3F0000803F0000803F0000803F0000803F0000803F52F37D3F0000803F0BF5413D0000803F0000803F8663813EFAED7A3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FB3E97F3F0000803F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @igammac(%arg0: tensor<20x20xf32>, %arg1: tensor<20x20xf32>) -> tensor<20x20xf32> {
    %0 = call @xla_fallback_igammac(%arg0, %arg1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @igamma_body.169(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>> {
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
  func.func private @or.206(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @igamma_condition.210(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<i1> {
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
  func.func private @igammac_body.263(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>> {
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
  func.func private @or.386(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @igammac_condition.390(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<i1> {
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
  func.func private @xla_fallback_igammac(%arg0: tensor<20x20xf32>, %arg1: tensor<20x20xf32>) -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %2 = stablehlo.compare  LE, %arg1, %1 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %5 = stablehlo.compare  LE, %arg0, %4 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %6 = stablehlo.or %2, %5 : tensor<20x20xi1>
    %7 = stablehlo.log %arg1 : tensor<20x20xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<20x20xf32>
    %9 = stablehlo.subtract %8, %arg1 : tensor<20x20xf32>
    %10 = stablehlo.abs %arg0 : tensor<20x20xf32>
    %11 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %12 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %13 = stablehlo.compare  EQ, %10, %12 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %14 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %15 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %16 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %17 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %18 = stablehlo.compare  LT, %arg0, %17 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %19 = stablehlo.constant dense<3.14159274> : tensor<f32>
    %20 = stablehlo.constant dense<3.14159274> : tensor<20x20xf32>
    %21 = stablehlo.abs %arg0 : tensor<20x20xf32>
    %22 = stablehlo.floor %21 : tensor<20x20xf32>
    %23 = stablehlo.subtract %21, %22 : tensor<20x20xf32>
    %24 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %25 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %26 = stablehlo.compare  GT, %23, %25 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %27 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %28 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %29 = stablehlo.subtract %28, %23 : tensor<20x20xf32>
    %30 = stablehlo.select %26, %29, %23 : tensor<20x20xi1>, tensor<20x20xf32>
    %31 = stablehlo.multiply %20, %30 : tensor<20x20xf32>
    %32 = stablehlo.sine %31 : tensor<20x20xf32>
    %33 = stablehlo.log %32 : tensor<20x20xf32>
    %34 = stablehlo.is_finite %33 : (tensor<20x20xf32>) -> tensor<20x20xi1>
    %35 = stablehlo.constant dense<1.14472985> : tensor<f32>
    %36 = stablehlo.constant dense<1.14472985> : tensor<20x20xf32>
    %37 = stablehlo.subtract %36, %33 : tensor<20x20xf32>
    %38 = stablehlo.constant dense<0.918938517> : tensor<f32>
    %39 = stablehlo.constant dense<0.918938517> : tensor<20x20xf32>
    %40 = stablehlo.negate %arg0 : tensor<20x20xf32>
    %41 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %42 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %43 = stablehlo.subtract %arg0, %42 : tensor<20x20xf32>
    %44 = stablehlo.select %18, %40, %43 : tensor<20x20xi1>, tensor<20x20xf32>
    %45 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %46 = stablehlo.add %44, %45 : tensor<20x20xf32>
    %47 = stablehlo.constant dense<7.500000e+00> : tensor<f32>
    %48 = stablehlo.constant dense<7.500000e+00> : tensor<20x20xf32>
    %49 = stablehlo.add %48, %44 : tensor<20x20xf32>
    %50 = stablehlo.constant dense<2.01490307> : tensor<f32>
    %51 = stablehlo.constant dense<2.01490307> : tensor<20x20xf32>
    %52 = stablehlo.constant dense<7.500000e+00> : tensor<20x20xf32>
    %53 = stablehlo.divide %44, %52 : tensor<20x20xf32>
    %54 = stablehlo.log_plus_one %53 : tensor<20x20xf32>
    %55 = stablehlo.add %51, %54 : tensor<20x20xf32>
    %56 = stablehlo.divide %49, %55 : tensor<20x20xf32>
    %57 = stablehlo.subtract %46, %56 : tensor<20x20xf32>
    %58 = stablehlo.multiply %57, %55 : tensor<20x20xf32>
    %59 = stablehlo.add %39, %58 : tensor<20x20xf32>
    %60 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %61 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %62 = stablehlo.constant dense<676.520386> : tensor<f32>
    %63 = stablehlo.constant dense<676.520386> : tensor<20x20xf32>
    %64 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %65 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %66 = stablehlo.add %44, %65 : tensor<20x20xf32>
    %67 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %68 = stablehlo.add %66, %67 : tensor<20x20xf32>
    %69 = stablehlo.divide %63, %68 : tensor<20x20xf32>
    %70 = stablehlo.add %61, %69 : tensor<20x20xf32>
    %71 = stablehlo.constant dense<-1259.13916> : tensor<f32>
    %72 = stablehlo.constant dense<-1259.13916> : tensor<20x20xf32>
    %73 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %74 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %75 = stablehlo.add %44, %74 : tensor<20x20xf32>
    %76 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %77 = stablehlo.add %75, %76 : tensor<20x20xf32>
    %78 = stablehlo.divide %72, %77 : tensor<20x20xf32>
    %79 = stablehlo.add %70, %78 : tensor<20x20xf32>
    %80 = stablehlo.constant dense<771.323425> : tensor<f32>
    %81 = stablehlo.constant dense<771.323425> : tensor<20x20xf32>
    %82 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %83 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
    %84 = stablehlo.add %44, %83 : tensor<20x20xf32>
    %85 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %86 = stablehlo.add %84, %85 : tensor<20x20xf32>
    %87 = stablehlo.divide %81, %86 : tensor<20x20xf32>
    %88 = stablehlo.add %79, %87 : tensor<20x20xf32>
    %89 = stablehlo.constant dense<-176.615036> : tensor<f32>
    %90 = stablehlo.constant dense<-176.615036> : tensor<20x20xf32>
    %91 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %92 = stablehlo.constant dense<3.000000e+00> : tensor<20x20xf32>
    %93 = stablehlo.add %44, %92 : tensor<20x20xf32>
    %94 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %95 = stablehlo.add %93, %94 : tensor<20x20xf32>
    %96 = stablehlo.divide %90, %95 : tensor<20x20xf32>
    %97 = stablehlo.add %88, %96 : tensor<20x20xf32>
    %98 = stablehlo.constant dense<12.5073433> : tensor<f32>
    %99 = stablehlo.constant dense<12.5073433> : tensor<20x20xf32>
    %100 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %101 = stablehlo.constant dense<4.000000e+00> : tensor<20x20xf32>
    %102 = stablehlo.add %44, %101 : tensor<20x20xf32>
    %103 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %104 = stablehlo.add %102, %103 : tensor<20x20xf32>
    %105 = stablehlo.divide %99, %104 : tensor<20x20xf32>
    %106 = stablehlo.add %97, %105 : tensor<20x20xf32>
    %107 = stablehlo.constant dense<-0.138571098> : tensor<f32>
    %108 = stablehlo.constant dense<-0.138571098> : tensor<20x20xf32>
    %109 = stablehlo.constant dense<5.000000e+00> : tensor<f32>
    %110 = stablehlo.constant dense<5.000000e+00> : tensor<20x20xf32>
    %111 = stablehlo.add %44, %110 : tensor<20x20xf32>
    %112 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %113 = stablehlo.add %111, %112 : tensor<20x20xf32>
    %114 = stablehlo.divide %108, %113 : tensor<20x20xf32>
    %115 = stablehlo.add %106, %114 : tensor<20x20xf32>
    %116 = stablehlo.constant dense<9.98436917E-6> : tensor<f32>
    %117 = stablehlo.constant dense<9.98436917E-6> : tensor<20x20xf32>
    %118 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %119 = stablehlo.constant dense<6.000000e+00> : tensor<20x20xf32>
    %120 = stablehlo.add %44, %119 : tensor<20x20xf32>
    %121 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %122 = stablehlo.add %120, %121 : tensor<20x20xf32>
    %123 = stablehlo.divide %117, %122 : tensor<20x20xf32>
    %124 = stablehlo.add %115, %123 : tensor<20x20xf32>
    %125 = stablehlo.constant dense<1.50563267E-7> : tensor<f32>
    %126 = stablehlo.constant dense<1.50563267E-7> : tensor<20x20xf32>
    %127 = stablehlo.constant dense<7.000000e+00> : tensor<f32>
    %128 = stablehlo.constant dense<7.000000e+00> : tensor<20x20xf32>
    %129 = stablehlo.add %44, %128 : tensor<20x20xf32>
    %130 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %131 = stablehlo.add %129, %130 : tensor<20x20xf32>
    %132 = stablehlo.divide %126, %131 : tensor<20x20xf32>
    %133 = stablehlo.add %124, %132 : tensor<20x20xf32>
    %134 = stablehlo.log %133 : tensor<20x20xf32>
    %135 = stablehlo.add %59, %134 : tensor<20x20xf32>
    %136 = stablehlo.subtract %37, %135 : tensor<20x20xf32>
    %137 = stablehlo.negate %33 : tensor<20x20xf32>
    %138 = stablehlo.select %34, %136, %137 : tensor<20x20xi1>, tensor<20x20xf32>
    %139 = stablehlo.select %18, %138, %135 : tensor<20x20xi1>, tensor<20x20xf32>
    %140 = stablehlo.select %13, %15, %139 : tensor<20x20xi1>, tensor<20x20xf32>
    %141 = stablehlo.subtract %9, %140 : tensor<20x20xf32>
    %142 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
    %143 = stablehlo.constant dense<88.7228394> : tensor<f32>
    %144 = stablehlo.negate %143 : tensor<f32>
    %145 = stablehlo.broadcast_in_dim %144, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %146 = stablehlo.compare  LT, %141, %145 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %147 = stablehlo.or %6, %146 : tensor<20x20xi1>
    %148 = stablehlo.not %147 : tensor<20x20xi1>
    %149 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %150 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %151 = stablehlo.compare  LT, %arg1, %150 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %152 = stablehlo.compare  LT, %arg1, %arg0 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %153 = stablehlo.or %151, %152 : tensor<20x20xi1>
    %154 = stablehlo.and %148, %153 : tensor<20x20xi1>
    %155 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %156 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %157 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %158 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %159 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %160 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %161 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %162 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %163:7 = stablehlo.while(%iterArg = %154, %iterArg_0 = %arg0, %iterArg_1 = %156, %iterArg_2 = %158, %iterArg_3 = %arg1, %iterArg_4 = %160, %iterArg_5 = %162) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %211 = stablehlo.constant dense<false> : tensor<i1>
      %212 = stablehlo.reduce(%iterArg init: %211) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %213 = stablehlo.or %arg2, %arg3 : tensor<i1>
        stablehlo.return %213 : tensor<i1>
      }
      stablehlo.return %212 : tensor<i1>
    } do {
      %211 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %212 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %213 = stablehlo.add %iterArg_0, %212 : tensor<20x20xf32>
      %214 = stablehlo.divide %iterArg_3, %213 : tensor<20x20xf32>
      %215 = stablehlo.multiply %iterArg_1, %214 : tensor<20x20xf32>
      %216 = stablehlo.add %iterArg_2, %215 : tensor<20x20xf32>
      %217 = stablehlo.divide %215, %216 : tensor<20x20xf32>
      %218 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %219 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %220 = stablehlo.compare  GT, %217, %219 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %221 = stablehlo.and %iterArg, %220 : tensor<20x20xi1>
      %222 = stablehlo.select %iterArg, %213, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %223 = stablehlo.select %iterArg, %215, %iterArg_1 : tensor<20x20xi1>, tensor<20x20xf32>
      %224 = stablehlo.select %iterArg, %216, %iterArg_2 : tensor<20x20xi1>, tensor<20x20xf32>
      %225 = stablehlo.divide %iterArg_3, %213 : tensor<20x20xf32>
      %226 = stablehlo.multiply %iterArg_4, %225 : tensor<20x20xf32>
      %227 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
      %228 = stablehlo.constant dense<-1.000000e+00> : tensor<20x20xf32>
      %229 = stablehlo.multiply %228, %iterArg_1 : tensor<20x20xf32>
      %230 = stablehlo.multiply %229, %iterArg_3 : tensor<20x20xf32>
      %231 = stablehlo.multiply %213, %213 : tensor<20x20xf32>
      %232 = stablehlo.divide %230, %231 : tensor<20x20xf32>
      %233 = stablehlo.add %226, %232 : tensor<20x20xf32>
      %234 = stablehlo.select %iterArg, %233, %iterArg_4 : tensor<20x20xi1>, tensor<20x20xf32>
      %235 = stablehlo.add %iterArg_5, %233 : tensor<20x20xf32>
      %236 = stablehlo.select %iterArg, %235, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %221, %222, %223, %224, %iterArg_3, %234, %236 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %164 = stablehlo.not %153 : tensor<20x20xi1>
    %165 = stablehlo.and %148, %164 : tensor<20x20xi1>
    %166 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %167 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %168 = stablehlo.add %arg1, %167 : tensor<20x20xf32>
    %169 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %170 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %171 = stablehlo.subtract %170, %arg0 : tensor<20x20xf32>
    %172 = stablehlo.add %arg1, %171 : tensor<20x20xf32>
    %173 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %174 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %175 = stablehlo.add %172, %174 : tensor<20x20xf32>
    %176 = stablehlo.multiply %175, %arg1 : tensor<20x20xf32>
    %177 = stablehlo.divide %168, %176 : tensor<20x20xf32>
    %178 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %179 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %180 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %181 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %182 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %183 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %184 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %185 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %186 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %187 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %188 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %189 = stablehlo.negate %arg1 : tensor<20x20xf32>
    %190 = stablehlo.multiply %177, %189 : tensor<20x20xf32>
    %191 = stablehlo.subtract %188, %190 : tensor<20x20xf32>
    %192 = stablehlo.divide %191, %176 : tensor<20x20xf32>
    %193:15 = stablehlo.while(%iterArg = %165, %iterArg_0 = %177, %iterArg_1 = %179, %iterArg_2 = %171, %iterArg_3 = %175, %iterArg_4 = %180, %iterArg_5 = %168, %iterArg_6 = %176, %iterArg_7 = %182, %iterArg_8 = %arg1, %iterArg_9 = %184, %iterArg_10 = %186, %iterArg_11 = %188, %iterArg_12 = %189, %iterArg_13 = %192) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %211 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
      %212 = stablehlo.compare  LT, %iterArg_4, %211 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %213 = stablehlo.constant dense<false> : tensor<i1>
      %214 = stablehlo.reduce(%iterArg init: %213) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %216 = stablehlo.or %arg2, %arg3 : tensor<i1>
        stablehlo.return %216 : tensor<i1>
      }
      %215 = stablehlo.and %212, %214 : tensor<i1>
      stablehlo.return %215 : tensor<i1>
    } do {
      %211 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %212 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
      %213 = stablehlo.add %iterArg_3, %212 : tensor<20x20xf32>
      %214 = stablehlo.multiply %iterArg_6, %213 : tensor<20x20xf32>
      %215 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %216 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %217 = stablehlo.add %iterArg_2, %216 : tensor<20x20xf32>
      %218 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %219 = stablehlo.add %iterArg_4, %218 : tensor<f32>
      %220 = stablehlo.broadcast_in_dim %219, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %221 = stablehlo.multiply %217, %220 : tensor<20x20xf32>
      %222 = stablehlo.multiply %iterArg_8, %221 : tensor<20x20xf32>
      %223 = stablehlo.subtract %214, %222 : tensor<20x20xf32>
      %224 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %225 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
      %226 = stablehlo.compare  NE, %223, %225 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %227 = stablehlo.multiply %iterArg_11, %213 : tensor<20x20xf32>
      %228 = stablehlo.subtract %227, %iterArg_5 : tensor<20x20xf32>
      %229 = stablehlo.multiply %iterArg_9, %221 : tensor<20x20xf32>
      %230 = stablehlo.subtract %228, %229 : tensor<20x20xf32>
      %231 = stablehlo.broadcast_in_dim %219, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %232 = stablehlo.multiply %iterArg_7, %231 : tensor<20x20xf32>
      %233 = stablehlo.add %230, %232 : tensor<20x20xf32>
      %234 = stablehlo.multiply %iterArg_5, %213 : tensor<20x20xf32>
      %235 = stablehlo.multiply %iterArg_7, %221 : tensor<20x20xf32>
      %236 = stablehlo.subtract %234, %235 : tensor<20x20xf32>
      %237 = stablehlo.divide %236, %223 : tensor<20x20xf32>
      %238 = stablehlo.select %226, %237, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %239 = stablehlo.multiply %iterArg_12, %213 : tensor<20x20xf32>
      %240 = stablehlo.subtract %239, %iterArg_6 : tensor<20x20xf32>
      %241 = stablehlo.multiply %iterArg_10, %221 : tensor<20x20xf32>
      %242 = stablehlo.subtract %240, %241 : tensor<20x20xf32>
      %243 = stablehlo.broadcast_in_dim %219, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %244 = stablehlo.multiply %iterArg_8, %243 : tensor<20x20xf32>
      %245 = stablehlo.add %242, %244 : tensor<20x20xf32>
      %246 = stablehlo.multiply %238, %245 : tensor<20x20xf32>
      %247 = stablehlo.subtract %233, %246 : tensor<20x20xf32>
      %248 = stablehlo.divide %247, %223 : tensor<20x20xf32>
      %249 = stablehlo.select %226, %248, %iterArg_13 : tensor<20x20xi1>, tensor<20x20xf32>
      %250 = stablehlo.subtract %249, %iterArg_13 : tensor<20x20xf32>
      %251 = stablehlo.abs %250 : tensor<20x20xf32>
      %252 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %253 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %254 = stablehlo.select %226, %251, %253 : tensor<20x20xi1>, tensor<20x20xf32>
      %255 = stablehlo.subtract %iterArg_0, %237 : tensor<20x20xf32>
      %256 = stablehlo.divide %255, %237 : tensor<20x20xf32>
      %257 = stablehlo.abs %256 : tensor<20x20xf32>
      %258 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %259 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %260 = stablehlo.select %226, %257, %259 : tensor<20x20xi1>, tensor<20x20xf32>
      %261 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %262 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %263 = stablehlo.compare  GT, %260, %262 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %264 = stablehlo.and %iterArg, %263 : tensor<20x20xi1>
      %265 = stablehlo.select %iterArg, %238, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %266 = stablehlo.select %iterArg, %260, %iterArg_1 : tensor<20x20xi1>, tensor<20x20xf32>
      %267 = stablehlo.select %iterArg, %217, %iterArg_2 : tensor<20x20xi1>, tensor<20x20xf32>
      %268 = stablehlo.select %iterArg, %213, %iterArg_3 : tensor<20x20xi1>, tensor<20x20xf32>
      %269 = stablehlo.abs %236 : tensor<20x20xf32>
      %270 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %271 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %272 = stablehlo.constant dense<0x4B000000> : tensor<f32>
      %273 = stablehlo.broadcast_in_dim %272, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %274 = stablehlo.compare  GT, %269, %273 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %275 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %276 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %277 = stablehlo.multiply %236, %276 : tensor<20x20xf32>
      %278 = stablehlo.select %274, %277, %236 : tensor<20x20xi1>, tensor<20x20xf32>
      %279 = stablehlo.select %iterArg, %278, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      %280 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %281 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %282 = stablehlo.multiply %223, %281 : tensor<20x20xf32>
      %283 = stablehlo.select %274, %282, %223 : tensor<20x20xi1>, tensor<20x20xf32>
      %284 = stablehlo.select %iterArg, %283, %iterArg_6 : tensor<20x20xi1>, tensor<20x20xf32>
      %285 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %286 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %287 = stablehlo.multiply %iterArg_5, %286 : tensor<20x20xf32>
      %288 = stablehlo.select %274, %287, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      %289 = stablehlo.select %iterArg, %288, %iterArg_7 : tensor<20x20xi1>, tensor<20x20xf32>
      %290 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %291 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %292 = stablehlo.multiply %iterArg_6, %291 : tensor<20x20xf32>
      %293 = stablehlo.select %274, %292, %iterArg_6 : tensor<20x20xi1>, tensor<20x20xf32>
      %294 = stablehlo.select %iterArg, %293, %iterArg_8 : tensor<20x20xi1>, tensor<20x20xf32>
      %295 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %296 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %297 = stablehlo.multiply %iterArg_11, %296 : tensor<20x20xf32>
      %298 = stablehlo.select %274, %297, %iterArg_11 : tensor<20x20xi1>, tensor<20x20xf32>
      %299 = stablehlo.select %iterArg, %298, %iterArg_9 : tensor<20x20xi1>, tensor<20x20xf32>
      %300 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %301 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %302 = stablehlo.multiply %iterArg_12, %301 : tensor<20x20xf32>
      %303 = stablehlo.select %274, %302, %iterArg_12 : tensor<20x20xi1>, tensor<20x20xf32>
      %304 = stablehlo.select %iterArg, %303, %iterArg_10 : tensor<20x20xi1>, tensor<20x20xf32>
      %305 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %306 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %307 = stablehlo.multiply %233, %306 : tensor<20x20xf32>
      %308 = stablehlo.select %274, %307, %233 : tensor<20x20xi1>, tensor<20x20xf32>
      %309 = stablehlo.select %iterArg, %308, %iterArg_11 : tensor<20x20xi1>, tensor<20x20xf32>
      %310 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %311 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %312 = stablehlo.multiply %245, %311 : tensor<20x20xf32>
      %313 = stablehlo.select %274, %312, %245 : tensor<20x20xi1>, tensor<20x20xf32>
      %314 = stablehlo.select %iterArg, %313, %iterArg_12 : tensor<20x20xi1>, tensor<20x20xf32>
      %315 = stablehlo.select %iterArg, %249, %iterArg_13 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %264, %265, %266, %267, %268, %219, %279, %284, %289, %294, %299, %304, %309, %314, %315 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %194 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %195 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %196 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %197 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %198 = stablehlo.compare  EQ, %arg1, %197 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %199 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %200 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %201 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %202 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %203 = stablehlo.exponential %141 : tensor<20x20xf32>
    %204 = stablehlo.multiply %163#3, %203 : tensor<20x20xf32>
    %205 = stablehlo.divide %204, %arg0 : tensor<20x20xf32>
    %206 = stablehlo.subtract %202, %205 : tensor<20x20xf32>
    %207 = stablehlo.multiply %193#1, %203 : tensor<20x20xf32>
    %208 = stablehlo.select %153, %206, %207 : tensor<20x20xi1>, tensor<20x20xf32>
    %209 = stablehlo.select %198, %200, %208 : tensor<20x20xi1>, tensor<20x20xf32>
    %210 = stablehlo.select %6, %195, %209 : tensor<20x20xi1>, tensor<20x20xf32>
    return %210 : tensor<20x20xf32>
  }
}
