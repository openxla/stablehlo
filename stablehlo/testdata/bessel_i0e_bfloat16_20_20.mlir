// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xbf16>
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = stablehlo.convert %0 : (tensor<20x20xbf16>) -> tensor<20x20xf32>
    %3 = stablehlo.abs %2 : tensor<20x20xf32>
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_1 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %7 = stablehlo.multiply %4, %3 : tensor<20x20xf32>
    %8 = stablehlo.subtract %7, %5 : tensor<20x20xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %12 = stablehlo.multiply %8, %9 : tensor<20x20xf32>
    %13 = stablehlo.subtract %12, %10 : tensor<20x20xf32>
    %cst_5 = stablehlo.constant dense<-1.300025009986248E-8> : tensor<f64>
    %14 = stablehlo.convert %cst_5 : (tensor<f64>) -> tensor<f32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %16 = stablehlo.add %13, %15 : tensor<20x20xf32>
    %17 = stablehlo.multiply %8, %16 : tensor<20x20xf32>
    %18 = stablehlo.subtract %17, %9 : tensor<20x20xf32>
    %cst_6 = stablehlo.constant dense<6.0469950225419186E-8> : tensor<f64>
    %19 = stablehlo.convert %cst_6 : (tensor<f64>) -> tensor<f32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %21 = stablehlo.add %18, %20 : tensor<20x20xf32>
    %22 = stablehlo.multiply %8, %21 : tensor<20x20xf32>
    %23 = stablehlo.subtract %22, %16 : tensor<20x20xf32>
    %cst_7 = stablehlo.constant dense<-2.6707938539406119E-7> : tensor<f64>
    %24 = stablehlo.convert %cst_7 : (tensor<f64>) -> tensor<f32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %26 = stablehlo.add %23, %25 : tensor<20x20xf32>
    %27 = stablehlo.multiply %8, %26 : tensor<20x20xf32>
    %28 = stablehlo.subtract %27, %21 : tensor<20x20xf32>
    %cst_8 = stablehlo.constant dense<1.1173875391201037E-6> : tensor<f64>
    %29 = stablehlo.convert %cst_8 : (tensor<f64>) -> tensor<f32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %31 = stablehlo.add %28, %30 : tensor<20x20xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<20x20xf32>
    %33 = stablehlo.subtract %32, %26 : tensor<20x20xf32>
    %cst_9 = stablehlo.constant dense<-4.4167383584587505E-6> : tensor<f64>
    %34 = stablehlo.convert %cst_9 : (tensor<f64>) -> tensor<f32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %36 = stablehlo.add %33, %35 : tensor<20x20xf32>
    %37 = stablehlo.multiply %8, %36 : tensor<20x20xf32>
    %38 = stablehlo.subtract %37, %31 : tensor<20x20xf32>
    %cst_10 = stablehlo.constant dense<1.6448448070728896E-5> : tensor<f64>
    %39 = stablehlo.convert %cst_10 : (tensor<f64>) -> tensor<f32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %41 = stablehlo.add %38, %40 : tensor<20x20xf32>
    %42 = stablehlo.multiply %8, %41 : tensor<20x20xf32>
    %43 = stablehlo.subtract %42, %36 : tensor<20x20xf32>
    %cst_11 = stablehlo.constant dense<-5.754195010082104E-5> : tensor<f64>
    %44 = stablehlo.convert %cst_11 : (tensor<f64>) -> tensor<f32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %46 = stablehlo.add %43, %45 : tensor<20x20xf32>
    %47 = stablehlo.multiply %8, %46 : tensor<20x20xf32>
    %48 = stablehlo.subtract %47, %41 : tensor<20x20xf32>
    %cst_12 = stablehlo.constant dense<1.8850288509584165E-4> : tensor<f64>
    %49 = stablehlo.convert %cst_12 : (tensor<f64>) -> tensor<f32>
    %50 = stablehlo.broadcast_in_dim %49, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %51 = stablehlo.add %48, %50 : tensor<20x20xf32>
    %52 = stablehlo.multiply %8, %51 : tensor<20x20xf32>
    %53 = stablehlo.subtract %52, %46 : tensor<20x20xf32>
    %cst_13 = stablehlo.constant dense<-5.7637557453858236E-4> : tensor<f64>
    %54 = stablehlo.convert %cst_13 : (tensor<f64>) -> tensor<f32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %56 = stablehlo.add %53, %55 : tensor<20x20xf32>
    %57 = stablehlo.multiply %8, %56 : tensor<20x20xf32>
    %58 = stablehlo.subtract %57, %51 : tensor<20x20xf32>
    %cst_14 = stablehlo.constant dense<0.0016394756169413357> : tensor<f64>
    %59 = stablehlo.convert %cst_14 : (tensor<f64>) -> tensor<f32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %61 = stablehlo.add %58, %60 : tensor<20x20xf32>
    %62 = stablehlo.multiply %8, %61 : tensor<20x20xf32>
    %63 = stablehlo.subtract %62, %56 : tensor<20x20xf32>
    %cst_15 = stablehlo.constant dense<-0.0043243099950505759> : tensor<f64>
    %64 = stablehlo.convert %cst_15 : (tensor<f64>) -> tensor<f32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %66 = stablehlo.add %63, %65 : tensor<20x20xf32>
    %67 = stablehlo.multiply %8, %66 : tensor<20x20xf32>
    %68 = stablehlo.subtract %67, %61 : tensor<20x20xf32>
    %cst_16 = stablehlo.constant dense<0.010546460394594998> : tensor<f64>
    %69 = stablehlo.convert %cst_16 : (tensor<f64>) -> tensor<f32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %71 = stablehlo.add %68, %70 : tensor<20x20xf32>
    %72 = stablehlo.multiply %8, %71 : tensor<20x20xf32>
    %73 = stablehlo.subtract %72, %66 : tensor<20x20xf32>
    %cst_17 = stablehlo.constant dense<-0.023737414805899471> : tensor<f64>
    %74 = stablehlo.convert %cst_17 : (tensor<f64>) -> tensor<f32>
    %75 = stablehlo.broadcast_in_dim %74, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %76 = stablehlo.add %73, %75 : tensor<20x20xf32>
    %77 = stablehlo.multiply %8, %76 : tensor<20x20xf32>
    %78 = stablehlo.subtract %77, %71 : tensor<20x20xf32>
    %cst_18 = stablehlo.constant dense<0.049305284239670712> : tensor<f64>
    %79 = stablehlo.convert %cst_18 : (tensor<f64>) -> tensor<f32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %81 = stablehlo.add %78, %80 : tensor<20x20xf32>
    %82 = stablehlo.multiply %8, %81 : tensor<20x20xf32>
    %83 = stablehlo.subtract %82, %76 : tensor<20x20xf32>
    %cst_19 = stablehlo.constant dense<-0.094901097048047639> : tensor<f64>
    %84 = stablehlo.convert %cst_19 : (tensor<f64>) -> tensor<f32>
    %85 = stablehlo.broadcast_in_dim %84, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %86 = stablehlo.add %83, %85 : tensor<20x20xf32>
    %87 = stablehlo.multiply %8, %86 : tensor<20x20xf32>
    %88 = stablehlo.subtract %87, %81 : tensor<20x20xf32>
    %cst_20 = stablehlo.constant dense<0.17162090152220877> : tensor<f64>
    %89 = stablehlo.convert %cst_20 : (tensor<f64>) -> tensor<f32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %91 = stablehlo.add %88, %90 : tensor<20x20xf32>
    %92 = stablehlo.multiply %8, %91 : tensor<20x20xf32>
    %93 = stablehlo.subtract %92, %86 : tensor<20x20xf32>
    %cst_21 = stablehlo.constant dense<-0.3046826723431984> : tensor<f64>
    %94 = stablehlo.convert %cst_21 : (tensor<f64>) -> tensor<f32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %96 = stablehlo.add %93, %95 : tensor<20x20xf32>
    %97 = stablehlo.multiply %8, %96 : tensor<20x20xf32>
    %98 = stablehlo.subtract %97, %91 : tensor<20x20xf32>
    %cst_22 = stablehlo.constant dense<0.67679527440947607> : tensor<f64>
    %99 = stablehlo.convert %cst_22 : (tensor<f64>) -> tensor<f32>
    %100 = stablehlo.broadcast_in_dim %99, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %101 = stablehlo.add %98, %100 : tensor<20x20xf32>
    %102 = stablehlo.subtract %101, %91 : tensor<20x20xf32>
    %cst_23 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %103 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %104 = stablehlo.multiply %103, %102 : tensor<20x20xf32>
    %105 = stablehlo.divide %6, %3 : tensor<20x20xf32>
    %106 = stablehlo.subtract %105, %5 : tensor<20x20xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %107 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %108 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %109 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %110 = stablehlo.multiply %106, %107 : tensor<20x20xf32>
    %111 = stablehlo.subtract %110, %108 : tensor<20x20xf32>
    %cst_27 = stablehlo.constant dense<3.3962320257083865E-9> : tensor<f64>
    %112 = stablehlo.convert %cst_27 : (tensor<f64>) -> tensor<f32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %114 = stablehlo.add %111, %113 : tensor<20x20xf32>
    %115 = stablehlo.multiply %106, %114 : tensor<20x20xf32>
    %116 = stablehlo.subtract %115, %107 : tensor<20x20xf32>
    %cst_28 = stablehlo.constant dense<2.266668990498178E-8> : tensor<f64>
    %117 = stablehlo.convert %cst_28 : (tensor<f64>) -> tensor<f32>
    %118 = stablehlo.broadcast_in_dim %117, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %119 = stablehlo.add %116, %118 : tensor<20x20xf32>
    %120 = stablehlo.multiply %106, %119 : tensor<20x20xf32>
    %121 = stablehlo.subtract %120, %114 : tensor<20x20xf32>
    %cst_29 = stablehlo.constant dense<2.0489185894690638E-7> : tensor<f64>
    %122 = stablehlo.convert %cst_29 : (tensor<f64>) -> tensor<f32>
    %123 = stablehlo.broadcast_in_dim %122, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %124 = stablehlo.add %121, %123 : tensor<20x20xf32>
    %125 = stablehlo.multiply %106, %124 : tensor<20x20xf32>
    %126 = stablehlo.subtract %125, %119 : tensor<20x20xf32>
    %cst_30 = stablehlo.constant dense<2.8913705208347567E-6> : tensor<f64>
    %127 = stablehlo.convert %cst_30 : (tensor<f64>) -> tensor<f32>
    %128 = stablehlo.broadcast_in_dim %127, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %129 = stablehlo.add %126, %128 : tensor<20x20xf32>
    %130 = stablehlo.multiply %106, %129 : tensor<20x20xf32>
    %131 = stablehlo.subtract %130, %124 : tensor<20x20xf32>
    %cst_31 = stablehlo.constant dense<6.8897583469168245E-5> : tensor<f64>
    %132 = stablehlo.convert %cst_31 : (tensor<f64>) -> tensor<f32>
    %133 = stablehlo.broadcast_in_dim %132, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %134 = stablehlo.add %131, %133 : tensor<20x20xf32>
    %135 = stablehlo.multiply %106, %134 : tensor<20x20xf32>
    %136 = stablehlo.subtract %135, %129 : tensor<20x20xf32>
    %cst_32 = stablehlo.constant dense<0.0033691164782556943> : tensor<f64>
    %137 = stablehlo.convert %cst_32 : (tensor<f64>) -> tensor<f32>
    %138 = stablehlo.broadcast_in_dim %137, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %139 = stablehlo.add %136, %138 : tensor<20x20xf32>
    %140 = stablehlo.multiply %106, %139 : tensor<20x20xf32>
    %141 = stablehlo.subtract %140, %134 : tensor<20x20xf32>
    %cst_33 = stablehlo.constant dense<0.80449041101410879> : tensor<f64>
    %142 = stablehlo.convert %cst_33 : (tensor<f64>) -> tensor<f32>
    %143 = stablehlo.broadcast_in_dim %142, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %144 = stablehlo.add %141, %143 : tensor<20x20xf32>
    %145 = stablehlo.subtract %144, %134 : tensor<20x20xf32>
    %cst_34 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %146 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %147 = stablehlo.multiply %146, %145 : tensor<20x20xf32>
    %148 = stablehlo.sqrt %3 : tensor<20x20xf32>
    %149 = stablehlo.divide %147, %148 : tensor<20x20xf32>
    %cst_35 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %150 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %151 = stablehlo.compare  LE, %3, %150,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %152 = stablehlo.select %151, %104, %149 : tensor<20x20xi1>, tensor<20x20xf32>
    %153 = stablehlo.convert %152 : (tensor<20x20xf32>) -> tensor<20x20xbf16>
    stablehlo.custom_call @check.expect_close(%153, %1) {has_side_effect = true} : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> ()
    return %153 : tensor<20x20xbf16>
  }
  func.func private @inputs() -> (tensor<20x20xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x7CC0DD40A8C0903C8ABF6D3F023F90C00DC0FF4021C01BC05740A83F5D3F92409FBD8B4035C065BFE33E87C06B4088C09A3F603E19C0BC4081BFC4BE4CBED6BEF03F85C009408FC02A4024C091BF6B402540094087BE84BF8440CCBF8BC0E5BF9CC0C4C0E0BF3EBF58BE203F9C3F0441B6C081BFD8C073402FC049BE94C00B40A1BFF83EB9401B3F863FC7BF1EC063BF51408DBF2DC05540B43FEBBE21C0A1BF57C094C095C0B1BFAF3F86C02EC00840AABE16C1CB407840E13E46C0243DCEBFF7C035C0A3BF10C08A40EA3E134014BF88BFAFC06A3F40C059C041C06E40E8C081C04540413F84C0CCC0EDBEA1403BC05140DABF954034BF68C056C07BC09ABF1C403A400E3F95C0484047BF2A409D3F6DC03EC095C020401A416ABFD13FD4C0463FFB3EA2C0854001C17040DC40DABFA1C0C83E8E40A13F9A3F12C019C02240E5BE61C032BE4DC09CBFD1407840D13F52BEACC06CC0DA3F4640523F1B40C4C0D6401740EB3FC9BE733FC83F573FA4BF31C022C082408D3FE6BE04BF8840D13F71C0EBBE00C03E3E0C401B40D33C8BBF71BE71BFA3BE9ABF40C0AB3F0ABF2E402E404A40014047401EC09040E2BFFFBFAEC0CD3E8340593F24BF823D5F400CC0D43F85BF6BC018C00FBF98BF02C015C0C040E8C04340C040B1C0913F754055C01C3F43C041C031C05DC04E4088BE4A401F3F2DC0873FA0BF2940814094BFD5BF82BF9DBFFABEA6BE5C3F39C0313F5BBFE740273F3A40FBBE31C03F4000C0363F273FC83FC1BE30401341DC401E40F93E4940D63EC73F81C004C05940EBC08F40C93E98BE86BF0F3F0B401FC01540A03FCBC0ABC00A408A3F1B4028C094C038C0B340AB3E1240DBBF083F13C05A3FCA3DCF3F863F84C09DC00DC11FC08040ACBFB4C00E4097BF01C0C8BF5DC0C1C0FA3F9FC0FDC0F93F93BFA94089404140254081C00340C83F553F633F8FC007BFCCBF843F10C027C0394087BF23403CBFAAC060BF103E8A3FA03E4A40A34022C08A40473E99C088BE3DBFA64015BDD23F76BFAE3AB9BDED3F12BD2CC0BDC07CC0B1BF6EC051BFE44098401F40D13EC4BE283F8C4048C015C01D4068C06A4068400A409840463FFABFABBF9C4033C0823F4840"> : tensor<20x20xbf16>
    return %cst : tensor<20x20xbf16>
  }
  func.func private @expected() -> (tensor<20x20xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x563E1F3E373E7C3FE53EF93E243F473E953E133E8A3E8D3E693ECC3E013F453E6D3F4B3E813EFD3E2C3F4E3E5E3E4D3ED73E503F8E3E2D3EED3E353F543F303FA43E503E983E483E853E883EDE3E5E3E883E983E483FEA3E503EB53E4B3EA93E3F3E293EAB3E0B3F523F173FD53E113E303EED3E203E5A3E833E543F443E963ED13E273F2E3E193FE83EB83E8B3EFE3E6D3EE23E843E6B3EC43E2A3F8A3ED13E693E443E433EC63EC73E4F3E843E983E3D3F073E263E583E2D3F753E763FB43E163E813ED03E933E4B3E2B3F913E1C3FE73E333EFA3E793E683E783E5D3E1B3E533E753E0A3F503E253E2A3F3B3E7D3E6D3EAE3E433E0F3F603E6A3E563ED73E8C3E7D3E1F3F433E733E083F853ED43E5D3E7A3E433E8A3E053EFA3EB33E223E083F263F3B3E503E123E5C3E1F3EAE3E3B3E343F483ED13ED73E923E8E3E893E2C3F643E593F703ED53E233E583EB33E533F353E5E3EAE3E753E043F8D3E293E213E8F3EA63E343FF53EB83E033FCF3E823E893E523EE23E2C3F233F4D3EB33E5B3E2A3F9E3E563F963E8D3E7A3FE43E4D3FF63E3F3FD73E793ECA3E203F843E843E723E9D3E743E8B3E473EAA3E9E3E343E323F513E023F153F703F653E963EB13EE93E5E3E8F3E1E3FD83E9C3E903E2B3E1B3E773E2B3E323EDE3E593E6B3E183F773E783E823E663E6F3E483F723E173F843EE73ED23E863E533EDC3EB13EEC3ED43E273F3E3F013F7E3E103F023F1B3E143F7D3E263F823E7A3E9E3E0E3F143FB83E363F833E093E1F3E8B3E273F723E303FB83E533E9B3E683E1A3E483E343F423FE83E1E3F963E8B3E903ED23E263E353E973EE53E8D3E863E443E7F3E313E3C3F923EAE3E213F913E023F693FB43EE83E503E3E3E0C3E8B3E543EC93E313E943ED93E9D3EB83E663E2A3EA03E3D3E143EA13EDC3E373E4C3E783E883E533E9C3EB83E043FFE3E483E223FB53EEA3E933E873E7E3EE73E893E0C3F363E003F603FE53E403F723E3A3E893E4B3E553F403E483F0C3F383E773FB23EF43E803F6A3FA63E773F853E2C3E563EC63E5D3E053F1C3E413E8B3E313F353F133F4A3E733E903E8C3E603E5F3E603E973E413E083FA03ECA3E3F3E823EEC3E733E"> : tensor<20x20xbf16>
    return %cst : tensor<20x20xbf16>
  }
}
