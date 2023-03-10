// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xbf16>
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = call @bessel_i0e(%0) : (tensor<20x20xbf16>) -> tensor<20x20xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0xDF401FBFA8C02D4060C06E40B5BF1BBF5CBFB4C0673F5E40D33E5DBF7FC0BB3FB8BF01BF6D4035C092C0C73FB4C007401240D1C08CBFB43F04C040C0A9C094C0B23FA2C0893C7F3FA5401E409440F13F20C0D83E37C0833FEF3F4440C33E07BF7540B1406EC09DBF56BE604072BF9640C9C0F9BF2BC006BF0A400D40D43F2F40274017405DC04740A1409ABFBB4020C00640BFBFEB4047BF64BC8E3FF23E9640FF3FACC088BE6DBF6AC0953FA540C23F8EC02A3F25BF1E3ECBBEC8BF7E400A408C4059408A404940E1BF453FAD3F243EACBE033E32408740F63FC04070C0E43F5940F7BFC8BFC840D8BE354094C01D4074405EC07FC015C058BF6140F5C051C0E0C0FB3F67C09BBFEE3FB03F68401E40AE3FB63E2F40C1BF13409BC0553F27BEAA3E83BF0FBF5B40AFBF02C12B4047C0AABFA7BF553FA13F5D3EBFC01A3EBC407C40A4C0F0BE923F8840DA402940A1BFAEBD8CC0E5BFE7BFB5C06040E240BCC04EC010C08EC0034046C05ABF94BF54C00E4043C0E2C0543F6D3FFA3EA03D89C0B8C065BF9040E73F1140E93F874031C0DB3F6A402540B0401EBFEFBE12BE0F40D0405D4060C02B3F5340034006403740DC40E2BFA0BF833ED43F8C40F6BFECBF58C01540D3BF623ECBC034403940B63F0DBF3540C9C0DBBEB53F664012C08BBF44C00B40A93F883D10402F3F8DBF6A3F65BF8ABE04C11AC01D3FAAC053BFDCBF213F0C40093FC5BF5CBE54BF29BF91BF4B3FCFC04FBF104007C08FBF2DC0AD3EBB40BDBB34400CC02A400FC0D23FD3BF92C0DC3FBF3F58C0C4BF9B40B1BF4FBFD1BFB9400FBF6CBFE1BF73BF20C0D23F26C027404D40BC3F2AC0D13F944051C04140B5BF8B40E5BE0B3FCCBF95C0943D02C02EC08140B73E4B3F6A3FBFBE96C063C09C403AC0B1BFEE3F1940564087C076C09340E2BF51C0C6C0A73EC0BF6CBFCFC037BD2F3E23C0CCBE13BFF73F71BD38C08B3D81C0BBBF3740A2BFE33FCD40B0BF1AC0F13EBA3E2640384080BF29C09E405A4062C0E1BFCCBF78400FC0F13FAF4094BFB3BFF0BF3A400C40AF3FB4C0864017C0EDC0213F55C021BE9DBE8D3FBE4061C0CB404C3F0CC028C003402A3F1A40E23F2D40F73F854043C07BC06CBF"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @expected() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0x1E3E173F373E843E643E5D3EC33E193F013F313EFC3E653E313F013F543EBF3EC13E253F5D3E813E453EB83E313E993E923E233EE33EC43E9B3E793E373E443EC53E3B3E7C3FEF3E393E8B3E443EA43E8A3E2F3F803EEB3EA53E763E353F223F593E323E5D3ED43E523F643EF63E433E273EA13E853E223F973E953EB13E833E873E8F3E663E743E3B3ED73E2D3E8A3E9A3EBD3E1A3E083F7C3FE13E293F433E9E3E353E483FF93E5F3EDB3E393EBB3E483E133F153F5D3F333FB83E553E973E4A3E683E4B3E723EAB3E093FC83E5C3F3C3F623F823E4E3EA23E2B3E5C3EA93E683EA13EB83E273E2F3F813E443E8C3E5A3E653E543E903E033F643E163E6D3E1D3EA03E603ED63EA53EC63E603E8B3EC83E393F833EBC3E913E3F3E043F5B3F3D3FEB3E1E3F673EC73E123E853E743ECA3ECD3E043FD13E513F2B3E5E3F2D3E563E393E293FDD3E4D3E203E863ED13E6C3F4A3EA93EA83E303E643E1D3E2D3E6F3E933E483E9C3E753E023FDC3E6B3E943E773E1D3E043FF93E273F6D3F4C3E2F3EFD3E473EA83E933EA73E4E3E823EAE3E5F3E883E333E183F293F5F3F943E243E663E643E123F6C3E9C3E9A3E803E1F3EAA3ED23E493FB13E4A3EA23EA63E693E903EB23E503F263E813E7E3EC23E1F3F813E273E2F3FC33E613E923EE43E763E963ECB3E703F933E113FE23EFA3EFD3E473F113E8D3E183F363E043FAD3E163F963E213FB93E513F043F133FDE3E073F243E053F933E993EE03E843E3C3F2D3E7F3F813E963E853E943EB23EB23E453EAD3EBD3E693EBA3E3F3EC63E053FB33E2E3E1E3FF93EAB3EF53E8A3EB23E873E873E703EBF3E853EB33E443E6D3E783EC33E4B3E2C3F203FB53E433E6E3F9C3E843E533E393F073FFA3E363F433E623E3F3E7D3EC63EA53E8E3E6A3E4E3E593E453EAA3E6D3E283E3E3FBC3EF93E243E753F593F893E333F1C3FA13E723F7F3E6F3F533EBF3E803ED03EAA3E253EC63E8D3E293F383F873E7F3EEE3E863E3D3E683E633EAB3EB53E583E943EA43E333EDC3EC43EA43E7D3E963EC73E313E4F3E8F3E193E163F6B3E5C3F413FE23E2C3E643E263E063F963E863E9C3E133F8D3EAA3E843EA13E503E773E563EF93E"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @bessel_i0e(%arg0: tensor<20x20xbf16>) -> tensor<20x20xbf16> {
    %0 = call @xla_fallback_bessel_i0e(%arg0) : (tensor<20x20xbf16>) -> tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @xla_fallback_bessel_i0e(%arg0: tensor<20x20xbf16>) -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.convert %arg0 : (tensor<20x20xbf16>) -> tensor<20x20xf32>
    %3 = stablehlo.abs %2 : tensor<20x20xf32>
    %4 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %5 = stablehlo.constant dense<8.000000e+00> : tensor<20x20xf32>
    %6 = stablehlo.compare  LE, %3, %5 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %7 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %8 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %9 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %10 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %11 = stablehlo.multiply %10, %3 : tensor<20x20xf32>
    %12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %13 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
    %14 = stablehlo.subtract %11, %13 : tensor<20x20xf32>
    %15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %17 = stablehlo.multiply %14, %16 : tensor<20x20xf32>
    %18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %19 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %20 = stablehlo.subtract %17, %19 : tensor<20x20xf32>
    %21 = stablehlo.constant dense<-1.30002498E-8> : tensor<f32>
    %22 = stablehlo.constant dense<-1.30002498E-8> : tensor<20x20xf32>
    %23 = stablehlo.add %20, %22 : tensor<20x20xf32>
    %24 = stablehlo.multiply %14, %23 : tensor<20x20xf32>
    %25 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %26 = stablehlo.subtract %24, %25 : tensor<20x20xf32>
    %27 = stablehlo.constant dense<6.04699508E-8> : tensor<f32>
    %28 = stablehlo.constant dense<6.04699508E-8> : tensor<20x20xf32>
    %29 = stablehlo.add %26, %28 : tensor<20x20xf32>
    %30 = stablehlo.multiply %14, %29 : tensor<20x20xf32>
    %31 = stablehlo.subtract %30, %23 : tensor<20x20xf32>
    %32 = stablehlo.constant dense<-2.67079372E-7> : tensor<f32>
    %33 = stablehlo.constant dense<-2.67079372E-7> : tensor<20x20xf32>
    %34 = stablehlo.add %31, %33 : tensor<20x20xf32>
    %35 = stablehlo.multiply %14, %34 : tensor<20x20xf32>
    %36 = stablehlo.subtract %35, %29 : tensor<20x20xf32>
    %37 = stablehlo.constant dense<1.11738757E-6> : tensor<f32>
    %38 = stablehlo.constant dense<1.11738757E-6> : tensor<20x20xf32>
    %39 = stablehlo.add %36, %38 : tensor<20x20xf32>
    %40 = stablehlo.multiply %14, %39 : tensor<20x20xf32>
    %41 = stablehlo.subtract %40, %34 : tensor<20x20xf32>
    %42 = stablehlo.constant dense<-4.41673819E-6> : tensor<f32>
    %43 = stablehlo.constant dense<-4.41673819E-6> : tensor<20x20xf32>
    %44 = stablehlo.add %41, %43 : tensor<20x20xf32>
    %45 = stablehlo.multiply %14, %44 : tensor<20x20xf32>
    %46 = stablehlo.subtract %45, %39 : tensor<20x20xf32>
    %47 = stablehlo.constant dense<1.64484482E-5> : tensor<f32>
    %48 = stablehlo.constant dense<1.64484482E-5> : tensor<20x20xf32>
    %49 = stablehlo.add %46, %48 : tensor<20x20xf32>
    %50 = stablehlo.multiply %14, %49 : tensor<20x20xf32>
    %51 = stablehlo.subtract %50, %44 : tensor<20x20xf32>
    %52 = stablehlo.constant dense<-5.75419508E-5> : tensor<f32>
    %53 = stablehlo.constant dense<-5.75419508E-5> : tensor<20x20xf32>
    %54 = stablehlo.add %51, %53 : tensor<20x20xf32>
    %55 = stablehlo.multiply %14, %54 : tensor<20x20xf32>
    %56 = stablehlo.subtract %55, %49 : tensor<20x20xf32>
    %57 = stablehlo.constant dense<1.88502891E-4> : tensor<f32>
    %58 = stablehlo.constant dense<1.88502891E-4> : tensor<20x20xf32>
    %59 = stablehlo.add %56, %58 : tensor<20x20xf32>
    %60 = stablehlo.multiply %14, %59 : tensor<20x20xf32>
    %61 = stablehlo.subtract %60, %54 : tensor<20x20xf32>
    %62 = stablehlo.constant dense<-5.76375576E-4> : tensor<f32>
    %63 = stablehlo.constant dense<-5.76375576E-4> : tensor<20x20xf32>
    %64 = stablehlo.add %61, %63 : tensor<20x20xf32>
    %65 = stablehlo.multiply %14, %64 : tensor<20x20xf32>
    %66 = stablehlo.subtract %65, %59 : tensor<20x20xf32>
    %67 = stablehlo.constant dense<0.00163947558> : tensor<f32>
    %68 = stablehlo.constant dense<0.00163947558> : tensor<20x20xf32>
    %69 = stablehlo.add %66, %68 : tensor<20x20xf32>
    %70 = stablehlo.multiply %14, %69 : tensor<20x20xf32>
    %71 = stablehlo.subtract %70, %64 : tensor<20x20xf32>
    %72 = stablehlo.constant dense<-4.324310e-03> : tensor<f32>
    %73 = stablehlo.constant dense<-4.324310e-03> : tensor<20x20xf32>
    %74 = stablehlo.add %71, %73 : tensor<20x20xf32>
    %75 = stablehlo.multiply %14, %74 : tensor<20x20xf32>
    %76 = stablehlo.subtract %75, %69 : tensor<20x20xf32>
    %77 = stablehlo.constant dense<0.0105464607> : tensor<f32>
    %78 = stablehlo.constant dense<0.0105464607> : tensor<20x20xf32>
    %79 = stablehlo.add %76, %78 : tensor<20x20xf32>
    %80 = stablehlo.multiply %14, %79 : tensor<20x20xf32>
    %81 = stablehlo.subtract %80, %74 : tensor<20x20xf32>
    %82 = stablehlo.constant dense<-0.0237374157> : tensor<f32>
    %83 = stablehlo.constant dense<-0.0237374157> : tensor<20x20xf32>
    %84 = stablehlo.add %81, %83 : tensor<20x20xf32>
    %85 = stablehlo.multiply %14, %84 : tensor<20x20xf32>
    %86 = stablehlo.subtract %85, %79 : tensor<20x20xf32>
    %87 = stablehlo.constant dense<0.0493052825> : tensor<f32>
    %88 = stablehlo.constant dense<0.0493052825> : tensor<20x20xf32>
    %89 = stablehlo.add %86, %88 : tensor<20x20xf32>
    %90 = stablehlo.multiply %14, %89 : tensor<20x20xf32>
    %91 = stablehlo.subtract %90, %84 : tensor<20x20xf32>
    %92 = stablehlo.constant dense<-9.490110e-02> : tensor<f32>
    %93 = stablehlo.constant dense<-9.490110e-02> : tensor<20x20xf32>
    %94 = stablehlo.add %91, %93 : tensor<20x20xf32>
    %95 = stablehlo.multiply %14, %94 : tensor<20x20xf32>
    %96 = stablehlo.subtract %95, %89 : tensor<20x20xf32>
    %97 = stablehlo.constant dense<0.171620905> : tensor<f32>
    %98 = stablehlo.constant dense<0.171620905> : tensor<20x20xf32>
    %99 = stablehlo.add %96, %98 : tensor<20x20xf32>
    %100 = stablehlo.multiply %14, %99 : tensor<20x20xf32>
    %101 = stablehlo.subtract %100, %94 : tensor<20x20xf32>
    %102 = stablehlo.constant dense<-0.304682672> : tensor<f32>
    %103 = stablehlo.constant dense<-0.304682672> : tensor<20x20xf32>
    %104 = stablehlo.add %101, %103 : tensor<20x20xf32>
    %105 = stablehlo.multiply %14, %104 : tensor<20x20xf32>
    %106 = stablehlo.subtract %105, %99 : tensor<20x20xf32>
    %107 = stablehlo.constant dense<0.676795303> : tensor<f32>
    %108 = stablehlo.constant dense<0.676795303> : tensor<20x20xf32>
    %109 = stablehlo.add %106, %108 : tensor<20x20xf32>
    %110 = stablehlo.subtract %109, %99 : tensor<20x20xf32>
    %111 = stablehlo.multiply %8, %110 : tensor<20x20xf32>
    %112 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %113 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %114 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %115 = stablehlo.constant dense<3.200000e+01> : tensor<20x20xf32>
    %116 = stablehlo.divide %115, %3 : tensor<20x20xf32>
    %117 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
    %118 = stablehlo.subtract %116, %117 : tensor<20x20xf32>
    %119 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %120 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %121 = stablehlo.multiply %118, %120 : tensor<20x20xf32>
    %122 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %123 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %124 = stablehlo.subtract %121, %123 : tensor<20x20xf32>
    %125 = stablehlo.constant dense<3.39623196E-9> : tensor<f32>
    %126 = stablehlo.constant dense<3.39623196E-9> : tensor<20x20xf32>
    %127 = stablehlo.add %124, %126 : tensor<20x20xf32>
    %128 = stablehlo.multiply %118, %127 : tensor<20x20xf32>
    %129 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %130 = stablehlo.subtract %128, %129 : tensor<20x20xf32>
    %131 = stablehlo.constant dense<2.26666899E-8> : tensor<f32>
    %132 = stablehlo.constant dense<2.26666899E-8> : tensor<20x20xf32>
    %133 = stablehlo.add %130, %132 : tensor<20x20xf32>
    %134 = stablehlo.multiply %118, %133 : tensor<20x20xf32>
    %135 = stablehlo.subtract %134, %127 : tensor<20x20xf32>
    %136 = stablehlo.constant dense<2.04891862E-7> : tensor<f32>
    %137 = stablehlo.constant dense<2.04891862E-7> : tensor<20x20xf32>
    %138 = stablehlo.add %135, %137 : tensor<20x20xf32>
    %139 = stablehlo.multiply %118, %138 : tensor<20x20xf32>
    %140 = stablehlo.subtract %139, %133 : tensor<20x20xf32>
    %141 = stablehlo.constant dense<2.89137051E-6> : tensor<f32>
    %142 = stablehlo.constant dense<2.89137051E-6> : tensor<20x20xf32>
    %143 = stablehlo.add %140, %142 : tensor<20x20xf32>
    %144 = stablehlo.multiply %118, %143 : tensor<20x20xf32>
    %145 = stablehlo.subtract %144, %138 : tensor<20x20xf32>
    %146 = stablehlo.constant dense<6.88975852E-5> : tensor<f32>
    %147 = stablehlo.constant dense<6.88975852E-5> : tensor<20x20xf32>
    %148 = stablehlo.add %145, %147 : tensor<20x20xf32>
    %149 = stablehlo.multiply %118, %148 : tensor<20x20xf32>
    %150 = stablehlo.subtract %149, %143 : tensor<20x20xf32>
    %151 = stablehlo.constant dense<0.00336911646> : tensor<f32>
    %152 = stablehlo.constant dense<0.00336911646> : tensor<20x20xf32>
    %153 = stablehlo.add %150, %152 : tensor<20x20xf32>
    %154 = stablehlo.multiply %118, %153 : tensor<20x20xf32>
    %155 = stablehlo.subtract %154, %148 : tensor<20x20xf32>
    %156 = stablehlo.constant dense<0.804490387> : tensor<f32>
    %157 = stablehlo.constant dense<0.804490387> : tensor<20x20xf32>
    %158 = stablehlo.add %155, %157 : tensor<20x20xf32>
    %159 = stablehlo.subtract %158, %148 : tensor<20x20xf32>
    %160 = stablehlo.multiply %113, %159 : tensor<20x20xf32>
    %161 = stablehlo.sqrt %3 : tensor<20x20xf32>
    %162 = stablehlo.divide %160, %161 : tensor<20x20xf32>
    %163 = stablehlo.select %6, %111, %162 : tensor<20x20xi1>, tensor<20x20xf32>
    %164 = stablehlo.convert %163 : (tensor<20x20xf32>) -> tensor<20x20xbf16>
    return %164 : tensor<20x20xbf16>
  }
}
