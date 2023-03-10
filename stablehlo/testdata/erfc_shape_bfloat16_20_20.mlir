// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xbf16>
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = stablehlo.convert %0 : (tensor<20x20xbf16>) -> tensor<20x20xf32>
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
    %104 = stablehlo.convert %103 : (tensor<20x20xf32>) -> tensor<20x20xbf16>
    %105 = stablehlo.custom_call @check.eq(%104, %1) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> tensor<i1>
    return %105 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0x123FA6C091BEA1401AC0FC3FCF3FB5BFCD3E21C020C099C0DDC004413CC0304062C0353F963F4F40ADC086402B3FCFBFC33FADBF324050BF893C6D403140D83EB0BF64BF44BFCDBF05C05B404BC0CE3F03C0BB3F354092BF7540803F22C00DC063404AC0B6BE9740B4400940E33F2C408B40543F24BFCF3FFFBF35405240264042BF14408AC02D40BA3F33C0E53FB9C0AB3EDBBF9D40BEBFD54006404CC0D33FBCBF1240C9BF53BD68C0ACBFA6BF3C405CC0673FAC3F7CBF603A1AC011407EC06FC06EBF9D4017C09EBF3F40D7BDA0C0AF409ABFC03F3DC087BFDD3FD3BE1BBB4EC0BF3F4BBFD0C03FC00E3F6DC003BF84405940363F823F03C064C0B1BFAFC00040783F3C40A9C0163E20BD23BF0E3FA13F074003BD803F48BFDA3E16C080C01E3F9ABE15BE83C093BF0FBF61C084C0C33F074053BF97404A3FFEBF0B3F6EC0A540FFBF84BF91C09BC02140E6C07BC067C023C0BFC0984092BF1EBE563F28C0E0BF6FBEEA3F0440414032402C3E3EBE66C09BBE2A4023C05CC003403340B4BF14BF1AC05FC08EBE23BFA540E64085C054C0AC403FBFE0C099C093BEC83FCF3F8C3F74BFEABF0AC0C63EE5C0E8BEC240FEBD354068BFACBFD6BE3440E73F3C4056BF10405A40D7BF7B407FC084C0E5BEA94084BF7440BBBF85C06FBF2B40E740E73FCDBF55C008C03E40E1BE67C060C0A53F01C0E33DF23EE0C0A6BF86C0D4BDF2BE863F4EC0473F6240BE3F41C0C6C0D53EABC0923FCCC0A63E5DBF06BCADBF82BF6C405F3FD5BEBBBF344049C0C240893F25C0C1BF953FD73C93C092BE82406AC0E73E22C0643E59405C403CBF89C0DEBF31403A405A40CA3F96BF99BE0C40A1C061C0ACBFD13F5340DBBF35C0A7C0AB40043FCDC0B23FA3C0A4BE233F7CC098C078C02FC04CBFACC085C074408A3FE8BF3DC02C408EC083C05B3C24BF22408C402F3FB6BF0F3F03C0A5C071C076BF883F62BF34BE40C04B40314038C011C042BF6AC0434019BF8DC0BD4064403A3F063F0540D6BFB7BF823DA53E2A3F23BF8A4029BF3640ABBE24C01D40B4404740BE3E63BF1C40EB3D6F40FA3ED6C0AC40E93FA3BD99BD88C0CC3D8CC0E43F5FBF4EC096C0DBBF3D3F9D3FE1BFDDC0683E"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @expected() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0xD73E0040A83F9D2B0040B03BB63CFA3F123F0040004000400040730C0040D3380040A23EC83DA03600405B31B13EFD3F003DF93FB038E03F7B3F2F34C1380D3FF93FE53FDC3FFD3F0040AF350040BB3C00401F3D8538F23F8533213E004000400E350040B13FDC2D0127223B473C17395F30773ED13FB63CFF3F853869368039DC3F8D3A00400A39233D00403B3C0040233FFE3F8B2CFB3FB61D493B0040A23CFB3FA43AFD3F873F0040F93FF73F093800404F3E6B3DEB3F803F0040B23A00400040E83F8B2C0040F63FCC378F3F00403C28F53F0B3D0040EF3F6F3CB83F803F00400F3DDE3F00400040DE3E0040C43FBA31DA35A13E1B3E00400040FA3F0040993B2F3E09380040563F863FD13FDE3E9A3D3B3B853F213EDE3F0C3F00400040C43EAA3F953F0040F33FC93F00400040003D3B3BE13FDC2D873EFF3FE33E0040AC2AFF3FED3F00400040C43900400040004000400040A32DF23F963F733E0040FE3FA13F1F3C683BA837B038503F9A3F0040AA3F353900400040793BA038FA3FCB3F00400040A73FD13FAC2A5D18004000400429DB3F00400040A83FDE3CB63CFA3DE93FFF3F0040163F0040BD3F3923923F8538E63FF93FB93F92382F3C0938E23FC03AC335FE3FFB3200400040BD3FB629ED3F9633FB3F0040E83F25390C182F3CFD3F00400040E137BC3F004000408C3DFF3F603F013F0040F73F00408F3FC03F0E3E00408B3E1F35133D004000400E3F0040DB3D0040263FE43F813FF93FED3F45345F3EB93FFB3F923800403923053E0040FC3FCC3D783F0040A83F1E320040063F0040413FDA359C35DA3F0040FE3FC1382638C335D23CF43FAA3F023B00400040F93FAC3C5236FE3F004000403929EF3E00404A3D0040AD3FBC3E0040004000400040DF3F004000409633023EFF3F00401739004000407C3FD13FB4392830AB3EFA3FDC3E004000400040EA3F083EE53F993F0040F436C13800400040DC3F00408A37CD3F00409A24FC349C3EEB3E583BFE3FFA3F6E3F263FB23ED13F9330D33F7238AF3F0040093A012738371A3FE53F153A5F3F0A34FB3E00400429253C8B3F8B3F0040633F0040413CE43F00400040FE3F983EAA3DFE3F0040403F"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
}

