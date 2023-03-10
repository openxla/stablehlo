// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xbf16>, tensor<20x20xbf16>)
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = call @igamma(%0#0, %0#1) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> tensor<20x20xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xbf16>, tensor<20x20xbf16>) {
    %0 = stablehlo.constant dense<"0x8B40233E4E401640ADC0933FC440C93FF9BF0CC018BFAC3F2AC0843F8F3FD7BE26BFC53D9C4039BF2E40C93F7040333ECB3EBDC032BF123E1DBFBAC029402EC054BF3AC09CBF00C084C0DABE9EBFD3BF5B3F14BEC63F9EC04A3F8340AA3DB43E6BC0BD3E2B400840DABE223F14C0043F92BF1240CABF1340414058BF21C01340C33F5BC0EF3EC93E20406F40E1C0CDBF2B3F693F9FBF67C09A3F58407CBF8A40D7BFA0C055C01C4085BFE6BEFF3E6FC099C003C0D4BEE43FB23F86C07DBE80408F401C40CDBF653E903E59C018404FC073C00AC0EBBE804044BD284062BF3E3F8C4020407540C8C0ACBEA7C0D2C051BF55C09D3FD9BFDEBFFCBECE4082C06DBF714089406FC081C0ABC0C3BEF93F52C0C1BF593F3240DDC0A3BF9E3F37BFEBBDAC40DDBF26C09E3F1AC081404F4009C034407E3F693FC6BFC93FD7C0EE3F853E9EC056BF6FC062C0F6BF2AC06A40F4BE1DC092C01DC073C0BE3FC83F2CC0E13F6B40B3C017C05440FA3F903E38C07FC0543FCDBE033F03BFA5BEBD3E5340933F8640C83F0340A1BFB23FADBCC6C050400BC08740EB3F59BF734024C018401F3F2E40FC3D193EB53FD8BE39C0D93F1C40FDBF5BBFA0C093408BC0C03FD03E1F3F4540DD3F61BF14BFD3BFC2BE84C000BF3ABF95BD90C06EBFBB4013C0FA3E7B3F3EBE9C3F9540AB3F64BFA3C0F6BF9BBF74BF1C404E40443FD83F853F40C0C13FEFBF20C04A40E73F82BF03C065C095C092C012BE89402E3F83C09E40F33FA03F86C0ADBEE4BE15C06BBF483E483F6C404F401A3F3DC01C402740CEBE983F07C048C0733EDA3E754090BF13C01F3F54BE44BFD13FD2C0B3C08DC001401E40DEBF25BF9BBFB5403BC064C007409AC066C01FBF9040A9BF0B3DD04099BF4040C8BEAFBE8F4034C0A7BFB7BEFEBF55402240303FABBE3E40B3C02E3F3E40A240903F26C018C094C049C09BC0A2C0D13FA53EDB3FC64020C0114046401D3E8FC0FCBDE13E4D40833FC2BF00C01CC1A4BF0B4020BF13BF423E744093BF63403D40BF401AC06B40C43F534018408DC05A3F8D3F04C0944016C0DBBF094011C069C079407B40F83E9AC0F03F5ABFBB40C3BF4C3E1B40B13F2C40D63DB03E3D40A9BF6C40"> : tensor<20x20xbf16>
    %1 = stablehlo.constant dense<"0x05BE6C409D3F784021C04CC020C01DC05E3EA5C076BF43C06C3C6ABF2640D2C07ABF4CC07EC029C050BE9A40DC3F473FCA3F14C02EC0D33FD23E9D3F02C157BF1DC07F3EEC3E3640073F26BF1840D5BE9A40AA3F36BF7CBF053FEE3FF340CF3F2BC0E7C0F13FC2BF453F19407E3F30406C3F0A3F2EC0E9BF90402DBF9CBD92C00B408ABF30BFDB402DBF2DC0383FBDBFDA3E07C0A2BBC23FEE3F31C0CC3FE23E6A3F92C00240DB3EA740BABF4E3F6CBFF2BFD0BF76401B4038BFC2BE55C057C06AC085C09B3DC9404DC042C03C4086BF793FFA40CE3FD3BF3FC0F13FCD40323F203C4B407640613F804099BE3CBF35400DBFACBF0FC07340F33EB23FA34097C00CC0B8BF8A3EC63F353F5B40CCC0323FBE3F15C041C078BF2ABFF6C061C0603F9F409A40054099C0A8BFC640DDBE10400540A04087C0AB3F99BFB2BF71C08E3F1EC0FBBF503E31BF544008C1BFC08BBF5BBF8140B54099C06A40A7C0784000BF963FB94098BFFF4000BB92BFC33F383C3E40FE3F4B40E0402A4012C034BF36C08840B63F5DC00ABEFB3F404089C019C0E5BF6F40E2C0F93F9140084006C05AC0B33F4540B5C0E9BE98C00DC087C0D1BF7A4095C054BE0340E63F3B3FD4BE1A3F2440C03F62BFFEBF0BC08D4084BF83C0B4BF28C081C013401B40E1BF2A40B53FA5BE8A4027C086C040C09E3EE0BF9F409A40E03F0C40EF3EC9C05EBFB53FEBBF0D4163C0C23EB94078C044BF2E3F883EDBBF5B3F0EC0D2BFB3BEDABF3F406B404A3F9540A7BF8FBEE73FE83F0F3F00BFC73E413F9B408B3EE63F53BF4DC055C0E8BE2AC0BABF934073BFE1C0B8BF903FA53FAB3F8C40864003C0F7BE7040FCBF7F402A40283F2BC00CC0603F154038C00241704060401DC025400A4000400FC02340963FD5BBB73FEF3EFEBE10C0043F993FC5BF36C0BABE1BC03FBF49C0D7BE854071C092BFBF40D4BFE3408F3EA8BF86C040C0B43FBBBF90C07A40314011C02CC0AAC067BF2DC09EC00C40033FC940B94037C081BFEA404540DF3F6040BF3F064001BFEC3F033F3E402540C63F3640CF3E44C03D40F33FC23F904076BFDC3DCD3ECE3F4CBD04C0AABF4D40C5BE02C04CBFC5BFC2C09EBF06C09FBF9CBF0B40"> : tensor<20x20xbf16>
    return %0, %1 : tensor<20x20xbf16>, tensor<20x20xbf16>
  }
  func.func private @expected() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0xC07F803FC83D5A3FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F693FC07FC07FC07FC07FC07FC07F7A3FFE3D703F723FC07FC07F7C3FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F7F3FC07FC07FC07F043FDD3D803F753FC07FC07FBE3EC07FC07F753FC07F7B3FC07F843DC07FC07F533FC07FC07FC07F443FC07FC07F803FC07FC07FC07FC07F083FC07FC07FC07F4B3FC07FC07F0B3AC07FC07FC07FF73CC07FC07F4C3FC07FC07FC07FC07F403FC07FC07FC07FC07FC07FC07FC07F803FC07FC07F363FC07FC07FC07FC07FC07FC07FC43EC07F223F232E3A3F123FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC93AC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FF93EC07FC07FC07FC07F5D3FC07FC07FC83E7E3FC07FC07FC07FC07FC07F713FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F703FC07FC07FC07F493DC07FC07F7B3FC07FC07FC07FC07F773FC07F7D3FC07FC07FC07FC07FC07F143F103FC07FC07F433FC07FC07FC07FC07FF03EC07FC07F333FC07FC07FC07F613E803FC07FC07FC07FC07FC07FC07FC07FC07FC07FB13DC07F9E3EC07F283FE83E083FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F323DC07F7B3F433FC07F7B3FC07FC07FC07FC07FC07FC07FC07FCA3EA53EFF3EC07FC07FC07FC07FC07FC07F9D3B7C3FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F513F753FC07FC07FC07FC07FC07F7C3F0B3FC07F8E3B3B3FC07F363CB93EC07FC07FC07FC07FC07FC07F343FC07FC07FC07FC07FC07F023FC07FC07FC07FC07F523FC07FC07FC07F1239C07FC07F453EC07FC07FC07FD43EC07FC07FE93CC07FA63EC07FC07F833CC07FC07FC07FC07FC07F1B3D513FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F933DC07FC07FC07FC07FC07FC07F803FC07FC07FC07FC07FC07FC07FC07FC07FC07F7B3FC07FC07FC07F723FC07F203E303F953BC07FC07F323F0A3C373FC07F553F6E3FC07FC07FC07FC07FCF3EC07FC07F0037773A6E3FC07FC07FC07FF53DC07FC07FC07FC07FC07FC07FC07FC07FC07F673E"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @igamma(%arg0: tensor<20x20xbf16>, %arg1: tensor<20x20xbf16>) -> tensor<20x20xbf16> {
    %0 = call @xla_fallback_igamma(%arg0, %arg1) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
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
  func.func private @xla_fallback_igamma(%arg0: tensor<20x20xbf16>, %arg1: tensor<20x20xbf16>) -> tensor<20x20xbf16> {
    %0 = stablehlo.convert %arg1 : (tensor<20x20xbf16>) -> tensor<20x20xf32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %3 = stablehlo.compare  EQ, %0, %2 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %6 = stablehlo.compare  LT, %0, %5 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %7 = stablehlo.convert %arg0 : (tensor<20x20xbf16>) -> tensor<20x20xf32>
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
    %225 = stablehlo.convert %224 : (tensor<20x20xf32>) -> tensor<20x20xbf16>
    return %225 : tensor<20x20xbf16>
  }
}
