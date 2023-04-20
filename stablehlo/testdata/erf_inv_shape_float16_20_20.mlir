// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf16>
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = call @erf_inv(%0) : (tensor<20x20xf16>) -> tensor<20x20xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0x203CB5BAF041003D79420F3BEAC4C0A08C46C4C3BA4374BC024406C1F1C0F5B6C1C4993A173FF8BF61C4854217BE59BDAAAE65C4E73C42BDFA3D4B470844EFBEC8447437AE3D0F4014C414C0543180C498C43D439345F544CEBD9E2515402DC41FB384424AC46F4290C2B73A8CC5654429C2713F40BE06C34844262817C310449C3E20BEC9C01B3A47BEE1A98736443F5EC5ECC49A4351C560C03FBDA8B825429E425A36283CEC4119326FBB46433F3D23C51ABF21BD2FC4A130B5B8F442CE41B3BD8EC5E23D7442A33F4D4146440DC00B4019B7B33EB746E539A0BFCC41C94439B925C02B41F3355D39FAC0A49AE33BA1C2413CD844A14373463AC02D43BAC5A4C0C436A0B9FEBB66B4EF3ACD445AC09EB8BDAF2232B3BD403CAD3EC9C3EAC6B6BDBF44E0C07F41E531733E38C5393A02449FBD444147C0F9403E4305C32D43C6456DB7424370BC86BE48A507AC8B437EBD41C6CF38C847193CC83CE9C7CABD1E42B43E7CBCC13BB0C5AF39D23F633DE4404BC891C131BAEAB406B0B646FA40A93E4DBE5ABB11BB1C4096398B90AF42D0C33C4065434BC0FF40FA3295BC7E3419398EC5FF3D6CBAD63FDC3120441B424E423FBBF7BB4344CFB99A4538422EBE383DEDBACBBD6DC58C2030BC1F3CD4C1DAB7B6B95C434B9D36C13627EAC274C2DC413CBC79BFE7BFC2415F3AF63E52C036AD11C1694509B8E542B4C267C1B62DA13F95C25EC2B1378EB56D409DB1DC2DFCB62CBB0B44B2BE9C477F44B63DD345183988B5E8C377412CAFD13D5F356940AFBAFDC40041DE34BE3FDBB805429FC1AF43A838B539A7BC25B1484504AB153AFA3D964329BCD03C32B5BAC10145AE40FD40B64058B58B40FFB152BC0242A4B947C6D33C9CC51440BD3406C472460834C1A7D2AFC8C47FBEA439A3464FC06BC4DCBA83C215BD2F3CE9459CC28D3A48BEE844C840EF3C6CC8B13CABC500C04140A240CB3A09C2D146EEB6F0B878C006434A3EB5BE504148432AC236BF55C276C4224228C22EBCF1C542C55AC2B533F0AD74C232C1ECC0B34178BCD44133C0812AF74279B79540F23264BC2BB589C494BE7ABBC5BADF380245C341B2C0CFB684C4F1C4A0C1C69D0BAF03376A3F1A42B936"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @expected() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0xFF7FEBBBFF7FFF7FFF7F6D3CFF7F36A0FF7FFF7FFF7FFF7FFF7FFF7FFF7F82B6FF7FAB3BFF7FFF7FFF7FFF7FFF7FFF7FECADFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F0837FF7FFF7FFF7FFF7FC230FF7FFF7FFF7FFF7FFF7FFF7FFB24FF7FFF7F65B2FF7FFF7FFF7FFF7FF03BFF7FFF7FFF7FFF7FFF7FFF7FFF7F5B27FF7FFF7FFF7FFF7FFF7FB13AFF7F37A91036FF7FFF7FFF7FFF7FFF7FFF7FFF7F95B8FF7FFF7FE235FF7FFF7F75311CBDFF7FFF7FFF7FFF7FFF7FFF7F2030A5B8FF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FA8B6FF7FFF7F543AFF7FFF7FFF7F51B9FF7FFF7F7B358439FF7FE399F03EFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F4F36E7B9AAC0F5B33F3CFF7FFF7F89B8E2AE7D31FF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F4531FF7FFF7FE83AFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F01B7FF7FFF7FFF7FAEA425ABFF7FFF7FFF7FC638FF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F1C3EFF7FFE39FF7FFF7FFF7FFF7FFF7FD9BA78B429AFFF7FFF7FFF7FFF7FEFBC70BCFF7FD7390790FF7FFF7FFF7FFF7FFF7FFF7F4332FF7F11342639FF7FFF7F4BBBFF7F3D31FF7FFF7FFF7FBCBC07C0FF7F30BAFF7FFF7FFF7FFF7F3CBCFF7FFF7F0820FF7FFF7FFF7F77B709BAFF7FB19CFF7F6426FF7FFF7FFF7FFF7FFF7FFF7FFF7F313BFF7FFF7FA0ACFF7FFF7FB6B7FF7FFF7FFF7F122DFF7FFF7FFF7F4B3717B5FF7F04B1342D89B69BBCFF7FFF7FFF7FFF7FFF7FFF7F253911B5FF7FFF7F60AEFF7FE934FF7FDDBBFF7FFF7F6C34FF7FD5B8FF7FFF7FFF7F9538073AFF7F97B0FF7F39AAA63AFF7FFF7FFF7FFF7FBDB4FF7FFF7FFF7FFF7FFF7FE2B4FF7F5DB1FF7FFF7FEDB9FF7FFF7FFF7FFF7F4D34FF7FFF7F4533E0A6F5AEFF7FFF7FED39FF7FFF7FFF7F26BCFF7FFF7FFF7FFF7FFF7F903BFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F103CFF7FFF7F7BB6F0B8FF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FF03246ADFF7FFF7FFF7FFF7FFF7FFF7FFF7FC529FF7F0EB7FF7F3C32FF7FB6B4FF7FFF7F36BD09BCDA38FF7FFF7FFF7F5AB6FF7FFF7FFF7F1E9D43AE9036FF7FFF7F4336"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @erf_inv(%arg0: tensor<20x20xf16>) -> tensor<20x20xf16> {
    %0 = call @xla_fallback_erf_inv(%arg0) : (tensor<20x20xf16>) -> tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @xla_fallback_erf_inv(%arg0: tensor<20x20xf16>) -> tensor<20x20xf16> {
    %0 = stablehlo.convert %arg0 : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %1 = stablehlo.abs %0 : tensor<20x20xf32>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %4 = stablehlo.compare  EQ, %1, %3 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %5 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %6 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %7 = stablehlo.multiply %0, %6 : tensor<20x20xf32>
    %8 = stablehlo.negate %0 : tensor<20x20xf32>
    %9 = stablehlo.multiply %8, %0 : tensor<20x20xf32>
    %10 = stablehlo.log_plus_one %9 : tensor<20x20xf32>
    %11 = stablehlo.negate %10 : tensor<20x20xf32>
    %12 = stablehlo.constant dense<5.000000e+00> : tensor<f32>
    %13 = stablehlo.constant dense<5.000000e+00> : tensor<20x20xf32>
    %14 = stablehlo.compare  LT, %11, %13 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %15 = stablehlo.constant dense<1.50140941> : tensor<f32>
    %16 = stablehlo.constant dense<1.50140941> : tensor<20x20xf32>
    %17 = stablehlo.constant dense<2.83297682> : tensor<f32>
    %18 = stablehlo.constant dense<2.83297682> : tensor<20x20xf32>
    %19 = stablehlo.select %14, %16, %18 : tensor<20x20xi1>, tensor<20x20xf32>
    %20 = stablehlo.constant dense<0.246640727> : tensor<f32>
    %21 = stablehlo.constant dense<0.246640727> : tensor<20x20xf32>
    %22 = stablehlo.constant dense<1.00167406> : tensor<f32>
    %23 = stablehlo.constant dense<1.00167406> : tensor<20x20xf32>
    %24 = stablehlo.select %14, %21, %23 : tensor<20x20xi1>, tensor<20x20xf32>
    %25 = stablehlo.constant dense<-0.00417768164> : tensor<f32>
    %26 = stablehlo.constant dense<-0.00417768164> : tensor<20x20xf32>
    %27 = stablehlo.constant dense<0.00943887047> : tensor<f32>
    %28 = stablehlo.constant dense<0.00943887047> : tensor<20x20xf32>
    %29 = stablehlo.select %14, %26, %28 : tensor<20x20xi1>, tensor<20x20xf32>
    %30 = stablehlo.constant dense<-0.00125372503> : tensor<f32>
    %31 = stablehlo.constant dense<-0.00125372503> : tensor<20x20xf32>
    %32 = stablehlo.constant dense<-0.0076224613> : tensor<f32>
    %33 = stablehlo.constant dense<-0.0076224613> : tensor<20x20xf32>
    %34 = stablehlo.select %14, %31, %33 : tensor<20x20xi1>, tensor<20x20xf32>
    %35 = stablehlo.constant dense<2.1858087E-4> : tensor<f32>
    %36 = stablehlo.constant dense<2.1858087E-4> : tensor<20x20xf32>
    %37 = stablehlo.constant dense<0.00573950773> : tensor<f32>
    %38 = stablehlo.constant dense<0.00573950773> : tensor<20x20xf32>
    %39 = stablehlo.select %14, %36, %38 : tensor<20x20xi1>, tensor<20x20xf32>
    %40 = stablehlo.constant dense<-4.39150654E-6> : tensor<f32>
    %41 = stablehlo.constant dense<-4.39150654E-6> : tensor<20x20xf32>
    %42 = stablehlo.constant dense<-0.00367342844> : tensor<f32>
    %43 = stablehlo.constant dense<-0.00367342844> : tensor<20x20xf32>
    %44 = stablehlo.select %14, %41, %43 : tensor<20x20xi1>, tensor<20x20xf32>
    %45 = stablehlo.constant dense<-3.5233877E-6> : tensor<f32>
    %46 = stablehlo.constant dense<-3.5233877E-6> : tensor<20x20xf32>
    %47 = stablehlo.constant dense<0.00134934322> : tensor<f32>
    %48 = stablehlo.constant dense<0.00134934322> : tensor<20x20xf32>
    %49 = stablehlo.select %14, %46, %48 : tensor<20x20xi1>, tensor<20x20xf32>
    %50 = stablehlo.constant dense<3.43273939E-7> : tensor<f32>
    %51 = stablehlo.constant dense<3.43273939E-7> : tensor<20x20xf32>
    %52 = stablehlo.constant dense<1.00950558E-4> : tensor<f32>
    %53 = stablehlo.constant dense<1.00950558E-4> : tensor<20x20xf32>
    %54 = stablehlo.select %14, %51, %53 : tensor<20x20xi1>, tensor<20x20xf32>
    %55 = stablehlo.constant dense<2.81022636E-8> : tensor<f32>
    %56 = stablehlo.constant dense<2.81022636E-8> : tensor<20x20xf32>
    %57 = stablehlo.constant dense<-2.00214257E-4> : tensor<f32>
    %58 = stablehlo.constant dense<-2.00214257E-4> : tensor<20x20xf32>
    %59 = stablehlo.select %14, %56, %58 : tensor<20x20xi1>, tensor<20x20xf32>
    %60 = stablehlo.constant dense<2.500000e+00> : tensor<f32>
    %61 = stablehlo.constant dense<2.500000e+00> : tensor<20x20xf32>
    %62 = stablehlo.subtract %11, %61 : tensor<20x20xf32>
    %63 = stablehlo.sqrt %11 : tensor<20x20xf32>
    %64 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %65 = stablehlo.constant dense<3.000000e+00> : tensor<20x20xf32>
    %66 = stablehlo.subtract %63, %65 : tensor<20x20xf32>
    %67 = stablehlo.select %14, %62, %66 : tensor<20x20xi1>, tensor<20x20xf32>
    %68 = stablehlo.multiply %59, %67 : tensor<20x20xf32>
    %69 = stablehlo.add %54, %68 : tensor<20x20xf32>
    %70 = stablehlo.multiply %69, %67 : tensor<20x20xf32>
    %71 = stablehlo.add %49, %70 : tensor<20x20xf32>
    %72 = stablehlo.multiply %71, %67 : tensor<20x20xf32>
    %73 = stablehlo.add %44, %72 : tensor<20x20xf32>
    %74 = stablehlo.multiply %73, %67 : tensor<20x20xf32>
    %75 = stablehlo.add %39, %74 : tensor<20x20xf32>
    %76 = stablehlo.multiply %75, %67 : tensor<20x20xf32>
    %77 = stablehlo.add %34, %76 : tensor<20x20xf32>
    %78 = stablehlo.multiply %77, %67 : tensor<20x20xf32>
    %79 = stablehlo.add %29, %78 : tensor<20x20xf32>
    %80 = stablehlo.multiply %79, %67 : tensor<20x20xf32>
    %81 = stablehlo.add %24, %80 : tensor<20x20xf32>
    %82 = stablehlo.multiply %81, %67 : tensor<20x20xf32>
    %83 = stablehlo.add %19, %82 : tensor<20x20xf32>
    %84 = stablehlo.multiply %83, %0 : tensor<20x20xf32>
    %85 = stablehlo.select %4, %7, %84 : tensor<20x20xi1>, tensor<20x20xf32>
    %86 = stablehlo.convert %85 : (tensor<20x20xf32>) -> tensor<20x20xf16>
    return %86 : tensor<20x20xf16>
  }
}
