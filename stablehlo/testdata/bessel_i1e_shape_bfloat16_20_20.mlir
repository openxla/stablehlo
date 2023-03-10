// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xbf16>
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = stablehlo.convert %0 : (tensor<20x20xbf16>) -> tensor<20x20xf32>
    %3 = stablehlo.abs %2 : tensor<20x20xf32>
    %4 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %5 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
    %6 = stablehlo.constant dense<3.200000e+01> : tensor<20x20xf32>
    %7 = stablehlo.constant dense<8.000000e+00> : tensor<20x20xf32>
    %8 = stablehlo.multiply %4, %3 : tensor<20x20xf32>
    %9 = stablehlo.subtract %8, %5 : tensor<20x20xf32>
    %10 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %11 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %12 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %13 = stablehlo.multiply %9, %10 : tensor<20x20xf32>
    %14 = stablehlo.subtract %13, %11 : tensor<20x20xf32>
    %15 = stablehlo.constant dense<9.38153732E-9> : tensor<20x20xf32>
    %16 = stablehlo.add %14, %15 : tensor<20x20xf32>
    %17 = stablehlo.multiply %9, %16 : tensor<20x20xf32>
    %18 = stablehlo.subtract %17, %10 : tensor<20x20xf32>
    %19 = stablehlo.constant dense<-4.44505908E-8> : tensor<20x20xf32>
    %20 = stablehlo.add %18, %19 : tensor<20x20xf32>
    %21 = stablehlo.multiply %9, %20 : tensor<20x20xf32>
    %22 = stablehlo.subtract %21, %16 : tensor<20x20xf32>
    %23 = stablehlo.constant dense<2.00329481E-7> : tensor<20x20xf32>
    %24 = stablehlo.add %22, %23 : tensor<20x20xf32>
    %25 = stablehlo.multiply %9, %24 : tensor<20x20xf32>
    %26 = stablehlo.subtract %25, %20 : tensor<20x20xf32>
    %27 = stablehlo.constant dense<-8.568720e-07> : tensor<20x20xf32>
    %28 = stablehlo.add %26, %27 : tensor<20x20xf32>
    %29 = stablehlo.multiply %9, %28 : tensor<20x20xf32>
    %30 = stablehlo.subtract %29, %24 : tensor<20x20xf32>
    %31 = stablehlo.constant dense<3.47025139E-6> : tensor<20x20xf32>
    %32 = stablehlo.add %30, %31 : tensor<20x20xf32>
    %33 = stablehlo.multiply %9, %32 : tensor<20x20xf32>
    %34 = stablehlo.subtract %33, %28 : tensor<20x20xf32>
    %35 = stablehlo.constant dense<-1.32731639E-5> : tensor<20x20xf32>
    %36 = stablehlo.add %34, %35 : tensor<20x20xf32>
    %37 = stablehlo.multiply %9, %36 : tensor<20x20xf32>
    %38 = stablehlo.subtract %37, %32 : tensor<20x20xf32>
    %39 = stablehlo.constant dense<4.78156508E-5> : tensor<20x20xf32>
    %40 = stablehlo.add %38, %39 : tensor<20x20xf32>
    %41 = stablehlo.multiply %9, %40 : tensor<20x20xf32>
    %42 = stablehlo.subtract %41, %36 : tensor<20x20xf32>
    %43 = stablehlo.constant dense<-1.61760821E-4> : tensor<20x20xf32>
    %44 = stablehlo.add %42, %43 : tensor<20x20xf32>
    %45 = stablehlo.multiply %9, %44 : tensor<20x20xf32>
    %46 = stablehlo.subtract %45, %40 : tensor<20x20xf32>
    %47 = stablehlo.constant dense<5.122860e-04> : tensor<20x20xf32>
    %48 = stablehlo.add %46, %47 : tensor<20x20xf32>
    %49 = stablehlo.multiply %9, %48 : tensor<20x20xf32>
    %50 = stablehlo.subtract %49, %44 : tensor<20x20xf32>
    %51 = stablehlo.constant dense<-0.00151357241> : tensor<20x20xf32>
    %52 = stablehlo.add %50, %51 : tensor<20x20xf32>
    %53 = stablehlo.multiply %9, %52 : tensor<20x20xf32>
    %54 = stablehlo.subtract %53, %48 : tensor<20x20xf32>
    %55 = stablehlo.constant dense<0.0041564228> : tensor<20x20xf32>
    %56 = stablehlo.add %54, %55 : tensor<20x20xf32>
    %57 = stablehlo.multiply %9, %56 : tensor<20x20xf32>
    %58 = stablehlo.subtract %57, %52 : tensor<20x20xf32>
    %59 = stablehlo.constant dense<-0.0105640851> : tensor<20x20xf32>
    %60 = stablehlo.add %58, %59 : tensor<20x20xf32>
    %61 = stablehlo.multiply %9, %60 : tensor<20x20xf32>
    %62 = stablehlo.subtract %61, %56 : tensor<20x20xf32>
    %63 = stablehlo.constant dense<0.0247264486> : tensor<20x20xf32>
    %64 = stablehlo.add %62, %63 : tensor<20x20xf32>
    %65 = stablehlo.multiply %9, %64 : tensor<20x20xf32>
    %66 = stablehlo.subtract %65, %60 : tensor<20x20xf32>
    %67 = stablehlo.constant dense<-0.0529459827> : tensor<20x20xf32>
    %68 = stablehlo.add %66, %67 : tensor<20x20xf32>
    %69 = stablehlo.multiply %9, %68 : tensor<20x20xf32>
    %70 = stablehlo.subtract %69, %64 : tensor<20x20xf32>
    %71 = stablehlo.constant dense<0.102643661> : tensor<20x20xf32>
    %72 = stablehlo.add %70, %71 : tensor<20x20xf32>
    %73 = stablehlo.multiply %9, %72 : tensor<20x20xf32>
    %74 = stablehlo.subtract %73, %68 : tensor<20x20xf32>
    %75 = stablehlo.constant dense<-0.176416516> : tensor<20x20xf32>
    %76 = stablehlo.add %74, %75 : tensor<20x20xf32>
    %77 = stablehlo.multiply %9, %76 : tensor<20x20xf32>
    %78 = stablehlo.subtract %77, %72 : tensor<20x20xf32>
    %79 = stablehlo.constant dense<0.252587199> : tensor<20x20xf32>
    %80 = stablehlo.add %78, %79 : tensor<20x20xf32>
    %81 = stablehlo.subtract %80, %72 : tensor<20x20xf32>
    %82 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %83 = stablehlo.multiply %81, %82 : tensor<20x20xf32>
    %84 = stablehlo.multiply %3, %83 : tensor<20x20xf32>
    %85 = stablehlo.divide %6, %3 : tensor<20x20xf32>
    %86 = stablehlo.subtract %85, %5 : tensor<20x20xf32>
    %87 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %88 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %89 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %90 = stablehlo.multiply %86, %87 : tensor<20x20xf32>
    %91 = stablehlo.subtract %90, %88 : tensor<20x20xf32>
    %92 = stablehlo.constant dense<-3.83538046E-9> : tensor<20x20xf32>
    %93 = stablehlo.add %91, %92 : tensor<20x20xf32>
    %94 = stablehlo.multiply %86, %93 : tensor<20x20xf32>
    %95 = stablehlo.subtract %94, %87 : tensor<20x20xf32>
    %96 = stablehlo.constant dense<-2.63146891E-8> : tensor<20x20xf32>
    %97 = stablehlo.add %95, %96 : tensor<20x20xf32>
    %98 = stablehlo.multiply %86, %97 : tensor<20x20xf32>
    %99 = stablehlo.subtract %98, %93 : tensor<20x20xf32>
    %100 = stablehlo.constant dense<-2.51223611E-7> : tensor<20x20xf32>
    %101 = stablehlo.add %99, %100 : tensor<20x20xf32>
    %102 = stablehlo.multiply %86, %101 : tensor<20x20xf32>
    %103 = stablehlo.subtract %102, %97 : tensor<20x20xf32>
    %104 = stablehlo.constant dense<-3.88256467E-6> : tensor<20x20xf32>
    %105 = stablehlo.add %103, %104 : tensor<20x20xf32>
    %106 = stablehlo.multiply %86, %105 : tensor<20x20xf32>
    %107 = stablehlo.subtract %106, %101 : tensor<20x20xf32>
    %108 = stablehlo.constant dense<-1.10588939E-4> : tensor<20x20xf32>
    %109 = stablehlo.add %107, %108 : tensor<20x20xf32>
    %110 = stablehlo.multiply %86, %109 : tensor<20x20xf32>
    %111 = stablehlo.subtract %110, %105 : tensor<20x20xf32>
    %112 = stablehlo.constant dense<-0.00976109784> : tensor<20x20xf32>
    %113 = stablehlo.add %111, %112 : tensor<20x20xf32>
    %114 = stablehlo.multiply %86, %113 : tensor<20x20xf32>
    %115 = stablehlo.subtract %114, %109 : tensor<20x20xf32>
    %116 = stablehlo.constant dense<0.778576254> : tensor<20x20xf32>
    %117 = stablehlo.add %115, %116 : tensor<20x20xf32>
    %118 = stablehlo.subtract %117, %109 : tensor<20x20xf32>
    %119 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %120 = stablehlo.multiply %118, %119 : tensor<20x20xf32>
    %121 = stablehlo.sqrt %3 : tensor<20x20xf32>
    %122 = stablehlo.divide %120, %121 : tensor<20x20xf32>
    %123 = stablehlo.compare  LE, %3, %7 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %124 = stablehlo.select %123, %84, %122 : tensor<20x20xi1>, tensor<20x20xf32>
    %125 = stablehlo.sign %2 : tensor<20x20xf32>
    %126 = stablehlo.multiply %125, %124 : tensor<20x20xf32>
    %127 = stablehlo.convert %126 : (tensor<20x20xf32>) -> tensor<20x20xbf16>
    %128 = stablehlo.custom_call @check.eq(%127, %1) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> tensor<i1>
    return %128 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0x99C08A3F29C00D3C253F903F1BC0263E463FE23EC2BE65BFD3BF82406AC08640AEC023C0BABF873F86BFC03F733F3AC076C091BF2C3FFC3F0EC0534036C0344050403940CB3E67403CC07ABF0AC023408540B23F37BFAE4098C06FBFCFBF2340323FDDBF34404CC0173F3AC0373F1840AEBF9DC055C07AC08F40FFBF0940E43C1E40E5BEFCBF9D40143F594055C0134048406CBDA8BFEFBE734029BE17C02540593F69C033BF3CC01FC0BCBF7FBF85BF9EBF7940E03F354085BF7C3F86402F405AC0A1BF8C40A73F27C0A93FF2BE8D3F4EBFCA3FA7C006C0C640BBBF47C09540BC3F3DC0A740C9405E3F9340D4C083404F3FD63F45407DBF15C1283FBDBFF43F55C0DD3F9AC085C0A03FBFBF42C0BC400940893F45401B3E42BFFCBF864032C07F3F1640D1BFC0BFA5C04340CD3F3DBE7F402B3F5F3F58C0AEBFA6BFF03F08BFD23E203F334052BE8B3E5CC015C0ADC04CBF433F47C01D402A4056BF01C0953FDABE8DC0B13FD4408E40754017C07540A640C3BF6AC03C40DF3F2BC09340C43F27C0F1C0ADC0CB3EA93F30408140C03D9A4090BF943F64C0073D4D3E523FE3BF2CBFA8C0F1BF77BE1A4008C072C033408640A8401AC091BF11C034C051C00CC1E73E34C03B408ABF6F3FC2BF3640053F1EBEFB3FA7BF82C0EA3EFBBFF13F55C0623F77C0993FB1BD18C00D4005404A400D4016C091BF8840DF3FC83F36C02EBF7DC06A4051408FC01DC00EBF27BF5AC022C04ABF2C3F0840AEC028BE3540FE3EFABE393FB84038C0623D07BF84C05BC05E3E19C0073F58C01CC04A40D63E923FB43E72C08F40044035C0D73F37C02A407C4036BF963FB63F0C40A74019BF40C030BF693D44C027BE61408D4035C0B0BED33FDAC09BC029BF333FE4BF06BFE73F93C091BF47BF033F50BDF33F0F4033C0A9406A4092BF2E400A3FDE3F90BE124033C0063FB4BF9EBE83BFFB3FC3BF7BBF59C081403EC0C3BF54C09540233F0840264060C056400E407DC0D83F3E40653F48BFDCBBA94016C06540CF4017C08D3E28BF95BFC83E8FBE9D3D87402540DF3FBAC06D400240A83F3C4070C014409DC001409240FF3FF0BF1840913FC9409E3F17C0683F993F463F29408C3F17C01140"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @expected() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0x2BBE583E51BE8C3B363E5A3E55BE8E3D453E153E07BE4FBE60BE363E3DBE343E22BE53BE60BE573E57BE603E523E4BBE3ABE5ABE3A3E5D3E59BE443E4DBE4D3E453E4C3E0B3E3E3E4BBE54BE5ABE533E343E603E3FBE223E2BBE51BE60BE533E3D3E60BE4D3E46BE2F3E4BBE3F3E563E5FBE29BE43BE39BE303E5DBE5A3E5E3C543E16BE5DBE293E2D3E423E43BE573E473EDFBC5FBE1ABE3B3E90BD56BE523E4B3E3DBE3DBE4BBE54BE60BE55BE57BE5DBE393E5F3E4D3E57BE543E343E4F3E42BE5EBE313E5F3E51BE5F3E1BBE593E48BE603E25BE5BBE1A3E60BE47BE2D3E603E4ABE253E193E4D3E2E3E15BE353E483E603E483E54BE00BE383E60BE5E3E43BE603E2ABE34BE5E3E60BE49BE1D3E5A3E583E483E863D43BE5DBE343E4EBE553E573E60BE60BE26BE493E603E9EBD373E393E4D3E42BE5FBE5EBE5E3E26BE0E3E343E4E3EACBDD63D41BE57BE23BE47BE443E47BE543E503E4ABE5CBE5B3E12BE30BE603E153E303E3A3E56BE3A3E253E60BE3DBE4B3E5F3E50BE2E3E603E51BE0DBE23BE0B3E5F3E4F3E373E2F3D2A3E5ABE5B3E3FBE833CA93D493E5FBE3ABE25BE5EBEC3BD553E5ABE3BBE4E3E343E253E55BE5ABE58BE4DBE44BE04BE173E4DBE4B3E58BE513E60BE4D3E243E88BD5D3E5FBE36BE183E5DBE5E3E43BE4E3E39BE5C3E22BD56BE593E5B3E463E593E57BE5ABE333E5F3E603E4DBE3BBE38BE3D3E443E30BE54BE29BE37BE42BE53BE46BE3A3E5A3E22BE8FBD4D3E1F3E1EBE403E1F3E4CBED63C25BE35BE41BEB43D56BE253E42BE55BE463E103E5B3E013E3BBE303E5B3E4DBE603E4CBE503E383E3EBE5C3E603E593E253E30BE4ABE3CBEDC3C48BE8EBD403E303E4DBEFDBD603E13BE2ABE38BE3D3E5FBE24BE5F3E2EBE5ABE45BE223EC6BC5E3E593E4EBE243E3D3E5BBE4F3E273E603EDCBD583E4EBE243E60BEEBBD56BE5D3E60BE54BE42BE373E4ABE60BE43BE2D3E353E5A3E523E40BE433E593E38BE603E4A3E4F3E45BE5BBB243E57BE3E3E173E56BED83D38BE5BBE0A3EDABD123D333E523E5F3E1EBE3C3E5C3E5F3E4B3E3BBE573E29BE5C3E2E3E5D3E5EBE563E5A3E193E5D3E56BE4F3E5C3E453E513E593E56BE583E"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
}

