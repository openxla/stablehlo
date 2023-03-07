// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf16>
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = stablehlo.convert %0 : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %3 = stablehlo.multiply %2, %2 : tensor<20x20xf32>
    %4 = stablehlo.negate %3 : tensor<20x20xf32>
    %5 = stablehlo.abs %2 : tensor<20x20xf32>
    %6 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %7 = stablehlo.divide %6, %3 : tensor<20x20xf32>
    %8 = stablehlo.exponential %4 : tensor<20x20xf32>
    %9 = stablehlo.divide %6, %5 : tensor<20x20xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<20x20xf32>
    %11 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
    %12 = stablehlo.compare  LT, %5, %11 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %13 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %14 = stablehlo.multiply %13, %7 : tensor<20x20xf32>
    %15 = stablehlo.constant dense<2.326820e-02> : tensor<20x20xf32>
    %16 = stablehlo.add %14, %15 : tensor<20x20xf32>
    %17 = stablehlo.multiply %16, %7 : tensor<20x20xf32>
    %18 = stablehlo.constant dense<-0.138703942> : tensor<20x20xf32>
    %19 = stablehlo.add %17, %18 : tensor<20x20xf32>
    %20 = stablehlo.multiply %19, %7 : tensor<20x20xf32>
    %21 = stablehlo.constant dense<0.368742466> : tensor<20x20xf32>
    %22 = stablehlo.add %20, %21 : tensor<20x20xf32>
    %23 = stablehlo.multiply %22, %7 : tensor<20x20xf32>
    %24 = stablehlo.constant dense<-0.582473278> : tensor<20x20xf32>
    %25 = stablehlo.add %23, %24 : tensor<20x20xf32>
    %26 = stablehlo.multiply %25, %7 : tensor<20x20xf32>
    %27 = stablehlo.constant dense<0.621000468> : tensor<20x20xf32>
    %28 = stablehlo.add %26, %27 : tensor<20x20xf32>
    %29 = stablehlo.multiply %28, %7 : tensor<20x20xf32>
    %30 = stablehlo.constant dense<-0.494451523> : tensor<20x20xf32>
    %31 = stablehlo.add %29, %30 : tensor<20x20xf32>
    %32 = stablehlo.multiply %31, %7 : tensor<20x20xf32>
    %33 = stablehlo.constant dense<3.404880e-01> : tensor<20x20xf32>
    %34 = stablehlo.add %32, %33 : tensor<20x20xf32>
    %35 = stablehlo.multiply %34, %7 : tensor<20x20xf32>
    %36 = stablehlo.constant dense<-0.274112701> : tensor<20x20xf32>
    %37 = stablehlo.add %35, %36 : tensor<20x20xf32>
    %38 = stablehlo.multiply %37, %7 : tensor<20x20xf32>
    %39 = stablehlo.constant dense<0.563825965> : tensor<20x20xf32>
    %40 = stablehlo.add %38, %39 : tensor<20x20xf32>
    %41 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %42 = stablehlo.multiply %41, %7 : tensor<20x20xf32>
    %43 = stablehlo.constant dense<-10.477664> : tensor<20x20xf32>
    %44 = stablehlo.add %42, %43 : tensor<20x20xf32>
    %45 = stablehlo.multiply %44, %7 : tensor<20x20xf32>
    %46 = stablehlo.constant dense<1.297720e+01> : tensor<20x20xf32>
    %47 = stablehlo.add %45, %46 : tensor<20x20xf32>
    %48 = stablehlo.multiply %47, %7 : tensor<20x20xf32>
    %49 = stablehlo.constant dense<-7.49551868> : tensor<20x20xf32>
    %50 = stablehlo.add %48, %49 : tensor<20x20xf32>
    %51 = stablehlo.multiply %50, %7 : tensor<20x20xf32>
    %52 = stablehlo.constant dense<2.92101908> : tensor<20x20xf32>
    %53 = stablehlo.add %51, %52 : tensor<20x20xf32>
    %54 = stablehlo.multiply %53, %7 : tensor<20x20xf32>
    %55 = stablehlo.constant dense<-1.01526523> : tensor<20x20xf32>
    %56 = stablehlo.add %54, %55 : tensor<20x20xf32>
    %57 = stablehlo.multiply %56, %7 : tensor<20x20xf32>
    %58 = stablehlo.constant dense<0.42184633> : tensor<20x20xf32>
    %59 = stablehlo.add %57, %58 : tensor<20x20xf32>
    %60 = stablehlo.multiply %59, %7 : tensor<20x20xf32>
    %61 = stablehlo.constant dense<-0.282076746> : tensor<20x20xf32>
    %62 = stablehlo.add %60, %61 : tensor<20x20xf32>
    %63 = stablehlo.multiply %62, %7 : tensor<20x20xf32>
    %64 = stablehlo.constant dense<0.564189494> : tensor<20x20xf32>
    %65 = stablehlo.add %63, %64 : tensor<20x20xf32>
    %66 = stablehlo.select %12, %40, %65 : tensor<20x20xi1>, tensor<20x20xf32>
    %67 = stablehlo.multiply %10, %66 : tensor<20x20xf32>
    %68 = stablehlo.constant dense<-88.7228394> : tensor<20x20xf32>
    %69 = stablehlo.compare  LT, %4, %68 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %70 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %71 = stablehlo.select %69, %70, %67 : tensor<20x20xi1>, tensor<20x20xf32>
    %72 = stablehlo.compare  LT, %2, %70 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %73 = stablehlo.subtract %11, %71 : tensor<20x20xf32>
    %74 = stablehlo.select %72, %73, %71 : tensor<20x20xi1>, tensor<20x20xf32>
    %75 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %76 = stablehlo.multiply %2, %2 : tensor<20x20xf32>
    %77 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %78 = stablehlo.multiply %77, %76 : tensor<20x20xf32>
    %79 = stablehlo.constant dense<7.85386146E-5> : tensor<20x20xf32>
    %80 = stablehlo.add %78, %79 : tensor<20x20xf32>
    %81 = stablehlo.multiply %80, %76 : tensor<20x20xf32>
    %82 = stablehlo.constant dense<-8.0101937E-4> : tensor<20x20xf32>
    %83 = stablehlo.add %81, %82 : tensor<20x20xf32>
    %84 = stablehlo.multiply %83, %76 : tensor<20x20xf32>
    %85 = stablehlo.constant dense<0.00518832775> : tensor<20x20xf32>
    %86 = stablehlo.add %84, %85 : tensor<20x20xf32>
    %87 = stablehlo.multiply %86, %76 : tensor<20x20xf32>
    %88 = stablehlo.constant dense<-0.0268538129> : tensor<20x20xf32>
    %89 = stablehlo.add %87, %88 : tensor<20x20xf32>
    %90 = stablehlo.multiply %89, %76 : tensor<20x20xf32>
    %91 = stablehlo.constant dense<0.112835854> : tensor<20x20xf32>
    %92 = stablehlo.add %90, %91 : tensor<20x20xf32>
    %93 = stablehlo.multiply %92, %76 : tensor<20x20xf32>
    %94 = stablehlo.constant dense<-0.37612626> : tensor<20x20xf32>
    %95 = stablehlo.add %93, %94 : tensor<20x20xf32>
    %96 = stablehlo.multiply %95, %76 : tensor<20x20xf32>
    %97 = stablehlo.constant dense<1.12837911> : tensor<20x20xf32>
    %98 = stablehlo.add %96, %97 : tensor<20x20xf32>
    %99 = stablehlo.multiply %2, %98 : tensor<20x20xf32>
    %100 = stablehlo.subtract %75, %99 : tensor<20x20xf32>
    %101 = stablehlo.abs %2 : tensor<20x20xf32>
    %102 = stablehlo.compare  LT, %101, %75 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %103 = stablehlo.select %102, %100, %74 : tensor<20x20xi1>, tensor<20x20xf32>
    %104 = stablehlo.convert %103 : (tensor<20x20xf32>) -> tensor<20x20xf16>
    %105 = stablehlo.custom_call @check.eq(%104, %1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<i1>
    return %105 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0xCE3C7245704341BEF93BEEB6563CC1C0D7AFD0307C3A43B1BE443AC17DC30BC5A146184050C16BC248BC5F3DDFBE5DC2C2BCB9C15131004047C23844C343B5B292B73D32164760BC35C8B33C3BBD793C1544F43D1BC13F4298C02C42AE4132C5034633C27EC0254212BE3BC57EBF0B471AC27F397EC604B99E40D54005BCBDC326B384362941AC248EB640C675C4F13F15406E43F3BD30416FC0DDC145403CA9153C884139C52C4230BB76C412BEBBC044BE42C554C607C609C1FBB99A3A4FBC7D424AC6F430A541643F88BDD7BEF7C10F469DC2E740FCC203B9553664442A38A9C5AABBBF4003B78343A0A331BCFEC52BC067C221BCF4B4273F0CBD68425FB655C2BF3ADE44E6413FC24CBD4EC08345DDC1E73E34BFB5C25A2811C4133D56C3F3C0EC3E9A40E4B665C2C3319F447DB5F8C191B883C6373DF5C245C596B50DC6AD414B40DDC0CDBFA3B8BB3DACC11D46E5BAFD3C4FAFC4B7B0BC25404C42CEC22D44A1BDD332D44016412842F344694445C0DAB4EF420340E93BF7401BC37BC2D3C2C5C0EABC92C2AC3E1A3DA3C25AC19940FDBC0343C541BE361FC4B9B0C4C3D0BF3AB62538A0C0E04425356D345BC4B73EEC3E4046CA410AC20EC30F41D2C58E375FC1DE3F6841C2424EBFFDC04C31263F2B367642614674BB23C2AEC0634220C2D2C0F9C3E841B5C420B3F24531C0033FF3BE6F418038AD3E0E30AB4290B6673A52BEA0AAFEBEA8B73D26492EBCC5D84139442F42F4C17D350A46FBC49CC4A0B52B3CC43A57C5093ADC3F473ACFC526C058447BBC303F1AC3A74182C26C428ABEFD3451AD4842F2386745F7310C37EB3EC938BCB55F3D6742A636D73B4D30EFC2B1B4B3C458C0403C32BE91C4BC41F9B015C863BD19C17D3B09BD1A405D40B030C3C66F420939AE3B46448B3F4FB0E7B013C74938ACC0B1426D419EBC034138432CC07240D4C027388444B932C93DFB3B85B45EC4103C6742223A3EBDC6BD663820BA4A4346412F42E83628C283439AC077C1FA3D0E41CA3C51433EABA5321CB81FBE01BEB4BC25B14E3DA340EEBF4C2FF3B8673E0BBE68C1CC45EDBFE641FD3BB6BEFDBD9D3FA5B20E45D83F2EBDB5C14640CC3D72C35AC4DBB4A5415AC1"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @expected() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0xB82D00000200E43F1431D73D0230FF3F8D3CA73A0734BC3C00000040004000400000C61B004000407B3F5E2BF13F0040A13F0040843ACA1C004000000100EF3CFD3D433A0000833F00402F2EBE3F492F000085280040A800FF3FD600E103004000000040FE3FEA00DF3F0040F83F000000404D350040803E7C143111613F0040FE3C8438510CD63BC03D004000401C1DFB1B0300DC3F010CFE3F004032192F3CC43005060040D6002F3F0040DF3FFF3FE53F0040004000400040D73EC8337D3F4B0000409D3A4F049820CC3FF03F004000000040521000407F3E9B380000633700404C3F7A12DC3D0200113C723F0040FD3F00406C3F5B3DDC21B43F6300B53D004075330000FD010040C13FFE3F000000408323F53F0040B13B0040A92C004000405F23AA14D43D0040643A00007D3D0040523E00402D2C00400040843D0040EC03EB18FF3FFA3F5A3E7929004000001C3FFB2C833C083E9C3FEE1A8E0000400000D03F1B3A3F114A0DE10000000000FD3F543D1000AB1C2F31530F004000400040FF3FAC3F0040B124902C00400040B614B03F0C00F50269380040A93C0040FA3FAC3D6C37FF3F000032399139004081245F230000C90200400040B20D0040093800408B1D53081E00F63F0040853AE221AF3852000000403F0040FF3F6A000040FF3F0040F1010040FD3C0000FD3FBF22F23FFE07D236AD24DD3A2900C13D2034E63F3C3CF23F013EC83B1E3B00405B020000CE0000400539000000400040863D80306A3300409434971D46340040FD3F00008C3FA8210040350400405E00EB3F4639603C96001D360000563A443866235D368D3D5E2B640074384E31CB3A0040493D0040FE3F4130E33F00404A03B23C0040C63F0040F031B33FA31B2A18B03A00405A00FA3596310000D71F9B3CB03C00402E37FF3F25001608973F760E0600FD3FD816FF3F68370000223A3A2911313E3D0040D43064007434BF3FD63FFD36E23E04004D0ACE00553800400200FF3F00406E28C10DC92D0400413C273A213EE13FDD3F9D3FB83CC62B4614FB3FFA3A793E0B26DF3F00400000FB3FFD010E31EE3FDD3F481FED3C0000AF1DBB3F004026192C2900400040543D4F040040"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
}

