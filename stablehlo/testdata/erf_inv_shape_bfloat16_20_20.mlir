// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xbf16>
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = call @erf_inv(%0) : (tensor<20x20xbf16>) -> tensor<20x20xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0xECBFEEBFAA40D43F14C0A5400EC0D6BDADBF2E400AC105C00EC03040593FE2408F4015C0263E87C0384000C07E409A40ED3E09BF1CC0C4C072BF9A3F51C092BF6E405DBFB140E4BFFFBE0840AEBED5BF703F1840AFBF3CC0293F2ABF4CC00B3FD44013C070408C3FA4BF63C02440ABC06CC02C404140B4C00F407EC02240CFBF59BFC2C0CDBFF8BF604016C05FC0B240A73FE8BF7AC0D6BF8D3F82BF4840E33FA8BF74C0FEBE2540FD40F0BDD1BFC1C07B406C403D40EBBECD3E72C00A40193FC53E883F4B40CDBF99405CBF59C0A33E53BF57C015C05CC0AAC00FC0A2C0F7BE5A4019C00F3FD83F23C014C02A3F0B3F014099C0DF3F89BFB8C0C640763F53401E408BC083C0653FCE40CA4056BFE33F943F18C081C0DA400FBF01BC973FDDBF953F0C408AC00AC075C042BF2CBFA840134015C0AFC0B83F3A3DB040E1BFFAC0BBC0A8406B3F92C021C08CBE733F693F42C08CC015C0043ECE3F294169BF74C09ABE2B404D401D401AC019BF5F40C63E9DBF95BFBABF4DC0B53F8FC0913FB1BE263F7A401C40A1C0A7BE56C010406EC0AEC0CCBFDA3F71BF91BF994083C08FBF99402ABF38BD504051406140EE3F0C3FAB40A33E4FBF434007405C40FAC0C93FB5BFDDC02D3F9340D24005C096C02440BBBE03C0E53FCD3E463FC6C053C01D3E66BF9B3FD1C03840A5BFB14032BF933DE5BF90403B3ED1BE86BFA1BF91C09F3F4BBEB44052C003C03540B5C0733F37BEDA3E1E404C3F743F84C015BFF4BFDC4027C0303F593F9C3D7AC0204084BF8C3FCABCCF3FA53ECE3F84C0A5BE9640DDBFCD4093C0EF3E47C0CF3F3ABE01C05D3F75C08D407AC0D0BE55C086BF0040A9C0F23F993FBF3F813F6CBF853E3940274098C0C24008BF863E74406E4094402C40D43E49408F4005C17D4021401240B13F904026BF9CBF4A3CAB3C75BF8CC01B3F3C409A3F0DC0023FFE3F1BC0C9BFCE3FED3FCBBFC5C09FBF5CBF7EC09EBF6CC03BC0E13F39BF8B3FA23E823F89BFBFBE38C00CC0A5BF3D3F283E164093BFD840A73F6D3FACBF164069BEFFBF95C00840AEC0EDBDA6C018C13B4091405CC0C540AB402A406540C4BFA33E9ABF8A4043C046C0404059403EC00C403B402E402540"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @expected() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0xFF7FFF7FFF7FFF7FFF7FFF7FFF7FBEBDFF7FFF7FFF7FFF7FFF7FFF7F823FFF7FFF7FFF7F143EFF7FFF7FFF7FFF7FFF7FDF3E04BFFF7FFF7FAEBFFF7FFF7FFF7FFF7F87BFFF7FFF7FF3BEFF7F9FBEFF7FA93FFF7FFF7FFF7F2D3F2EBFFF7F073FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F82BFFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FF2BEFF7FFF7FD5BDFF7FFF7FFF7FFF7FFF7FDDBEBE3EFF7FFF7F183FB63EFF7FFF7FFF7FFF7F85BFFF7F953E75BFFF7FFF7FFF7FFF7FFF7FFF7FEABEFF7FFF7F0B3FFF7FFF7FFF7F2E3F073FFF7FFF7FFF7FFF7FFF7FFF7FBB3FFF7FFF7FFF7FFF7F933FFF7FFF7F7CBFFF7FFF7FFF7FFF7FFF7F0BBFE5BBFF7FFF7FFF7FFF7FFF7FFF7FFF7F54BF31BFFF7FFF7FFF7FFF7FFF7F253DFF7FFF7FFF7FFF7FFF7F9D3FFF7FFF7F7DBEB13F9A3FFF7FFF7FFF7FEB3DFF7FFF7F9ABFFF7F8CBEFF7FFF7FFF7FFF7F18BFFF7FB73EFF7FFF7FFF7FFF7FFF7FFF7FFF7FA2BE293FFF7FFF7FFF7F98BEFF7FFF7FFF7FFF7FFF7FFF7FABBFFF7FFF7FFF7FFF7FFF7F2EBF23BDFF7FFF7FFF7FFF7F083FFF7F953E6CBFFF7FFF7FFF7FFF7FFF7FFF7FFF7F323FFF7FFF7FFF7FFF7FFF7FACBEFF7FFF7FBE3E5B3FFF7FFF7F0C3E94BFFF7FFF7FFF7FFF7FFF7F3ABF823DFF7FFF7F273EC2BEFF7FFF7FFF7FFF7F36BEFF7FFF7FFF7FFF7FFF7FB13F24BECB3EFF7F663FB43FFF7F13BFFF7FFF7FFF7F373F823F8A3DFF7FFF7FFF7FFF7FB3BCFF7F963EFF7FFF7F96BEFF7FFF7FFF7FFF7FE23EFF7FFF7F26BEFF7F873FFF7FFF7FFF7FC1BEFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F9FBF703EFF7FFF7FFF7FFF7F03BF723EFF7FFF7FFF7FFF7FC53EFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F29BFFF7F333C983CB7BFFF7F1A3FFF7FFF7FFF7FF93EFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F85BFFF7FFF7FFF7FFF7FFF7F45BFFF7F943EFF7FFF7FB0BEFF7FFF7FFF7F4B3F163EFF7FFF7FFF7FFF7FA23FFF7FFF7F51BEFF7FFF7FFF7FFF7FD3BDFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F953EFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @erf_inv(%arg0: tensor<20x20xbf16>) -> tensor<20x20xbf16> {
    %0 = call @xla_fallback_erf_inv(%arg0) : (tensor<20x20xbf16>) -> tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @xla_fallback_erf_inv(%arg0: tensor<20x20xbf16>) -> tensor<20x20xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<20x20xbf16>) -> tensor<20x20xf32>
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
    %86 = stablehlo.convert %85 : (tensor<20x20xf32>) -> tensor<20x20xbf16>
    return %86 : tensor<20x20xbf16>
  }
}
