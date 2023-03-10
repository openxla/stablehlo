// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xbf16>
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = stablehlo.convert %0 : (tensor<20x20xbf16>) -> tensor<20x20xf32>
    %3 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %4 = stablehlo.compare  LT, %2, %3 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %5 = stablehlo.negate %2 : tensor<20x20xf32>
    %6 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<20x20xf32>
    %8 = stablehlo.select %4, %5, %7 : tensor<20x20xi1>, tensor<20x20xf32>
    %9 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %10 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %11 = stablehlo.constant dense<676.520386> : tensor<20x20xf32>
    %12 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %13 = stablehlo.add %8, %12 : tensor<20x20xf32>
    %14 = stablehlo.multiply %13, %13 : tensor<20x20xf32>
    %15 = stablehlo.divide %11, %14 : tensor<20x20xf32>
    %16 = stablehlo.subtract %9, %15 : tensor<20x20xf32>
    %17 = stablehlo.divide %11, %13 : tensor<20x20xf32>
    %18 = stablehlo.add %10, %17 : tensor<20x20xf32>
    %19 = stablehlo.constant dense<-1259.13916> : tensor<20x20xf32>
    %20 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
    %21 = stablehlo.add %8, %20 : tensor<20x20xf32>
    %22 = stablehlo.multiply %21, %21 : tensor<20x20xf32>
    %23 = stablehlo.divide %19, %22 : tensor<20x20xf32>
    %24 = stablehlo.subtract %16, %23 : tensor<20x20xf32>
    %25 = stablehlo.divide %19, %21 : tensor<20x20xf32>
    %26 = stablehlo.add %18, %25 : tensor<20x20xf32>
    %27 = stablehlo.constant dense<771.323425> : tensor<20x20xf32>
    %28 = stablehlo.constant dense<3.000000e+00> : tensor<20x20xf32>
    %29 = stablehlo.add %8, %28 : tensor<20x20xf32>
    %30 = stablehlo.multiply %29, %29 : tensor<20x20xf32>
    %31 = stablehlo.divide %27, %30 : tensor<20x20xf32>
    %32 = stablehlo.subtract %24, %31 : tensor<20x20xf32>
    %33 = stablehlo.divide %27, %29 : tensor<20x20xf32>
    %34 = stablehlo.add %26, %33 : tensor<20x20xf32>
    %35 = stablehlo.constant dense<-176.615036> : tensor<20x20xf32>
    %36 = stablehlo.constant dense<4.000000e+00> : tensor<20x20xf32>
    %37 = stablehlo.add %8, %36 : tensor<20x20xf32>
    %38 = stablehlo.multiply %37, %37 : tensor<20x20xf32>
    %39 = stablehlo.divide %35, %38 : tensor<20x20xf32>
    %40 = stablehlo.subtract %32, %39 : tensor<20x20xf32>
    %41 = stablehlo.divide %35, %37 : tensor<20x20xf32>
    %42 = stablehlo.add %34, %41 : tensor<20x20xf32>
    %43 = stablehlo.constant dense<12.5073433> : tensor<20x20xf32>
    %44 = stablehlo.constant dense<5.000000e+00> : tensor<20x20xf32>
    %45 = stablehlo.add %8, %44 : tensor<20x20xf32>
    %46 = stablehlo.multiply %45, %45 : tensor<20x20xf32>
    %47 = stablehlo.divide %43, %46 : tensor<20x20xf32>
    %48 = stablehlo.subtract %40, %47 : tensor<20x20xf32>
    %49 = stablehlo.divide %43, %45 : tensor<20x20xf32>
    %50 = stablehlo.add %42, %49 : tensor<20x20xf32>
    %51 = stablehlo.constant dense<-0.138571098> : tensor<20x20xf32>
    %52 = stablehlo.constant dense<6.000000e+00> : tensor<20x20xf32>
    %53 = stablehlo.add %8, %52 : tensor<20x20xf32>
    %54 = stablehlo.multiply %53, %53 : tensor<20x20xf32>
    %55 = stablehlo.divide %51, %54 : tensor<20x20xf32>
    %56 = stablehlo.subtract %48, %55 : tensor<20x20xf32>
    %57 = stablehlo.divide %51, %53 : tensor<20x20xf32>
    %58 = stablehlo.add %50, %57 : tensor<20x20xf32>
    %59 = stablehlo.constant dense<9.98436917E-6> : tensor<20x20xf32>
    %60 = stablehlo.constant dense<7.000000e+00> : tensor<20x20xf32>
    %61 = stablehlo.add %8, %60 : tensor<20x20xf32>
    %62 = stablehlo.multiply %61, %61 : tensor<20x20xf32>
    %63 = stablehlo.divide %59, %62 : tensor<20x20xf32>
    %64 = stablehlo.subtract %56, %63 : tensor<20x20xf32>
    %65 = stablehlo.divide %59, %61 : tensor<20x20xf32>
    %66 = stablehlo.add %58, %65 : tensor<20x20xf32>
    %67 = stablehlo.constant dense<1.50563267E-7> : tensor<20x20xf32>
    %68 = stablehlo.constant dense<8.000000e+00> : tensor<20x20xf32>
    %69 = stablehlo.add %8, %68 : tensor<20x20xf32>
    %70 = stablehlo.multiply %69, %69 : tensor<20x20xf32>
    %71 = stablehlo.divide %67, %70 : tensor<20x20xf32>
    %72 = stablehlo.subtract %64, %71 : tensor<20x20xf32>
    %73 = stablehlo.divide %67, %69 : tensor<20x20xf32>
    %74 = stablehlo.add %66, %73 : tensor<20x20xf32>
    %75 = stablehlo.constant dense<7.500000e+00> : tensor<20x20xf32>
    %76 = stablehlo.add %75, %8 : tensor<20x20xf32>
    %77 = stablehlo.constant dense<2.01490307> : tensor<20x20xf32>
    %78 = stablehlo.divide %8, %75 : tensor<20x20xf32>
    %79 = stablehlo.log_plus_one %78 : tensor<20x20xf32>
    %80 = stablehlo.add %77, %79 : tensor<20x20xf32>
    %81 = stablehlo.divide %72, %74 : tensor<20x20xf32>
    %82 = stablehlo.constant dense<7.000000e+00> : tensor<20x20xf32>
    %83 = stablehlo.divide %82, %76 : tensor<20x20xf32>
    %84 = stablehlo.add %80, %81 : tensor<20x20xf32>
    %85 = stablehlo.subtract %84, %83 : tensor<20x20xf32>
    %86 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %87 = stablehlo.add %2, %86 : tensor<20x20xf32>
    %88 = stablehlo.floor %87 : tensor<20x20xf32>
    %89 = stablehlo.abs %88 : tensor<20x20xf32>
    %90 = stablehlo.add %2, %89 : tensor<20x20xf32>
    %91 = stablehlo.constant dense<3.14159274> : tensor<20x20xf32>
    %92 = stablehlo.multiply %91, %90 : tensor<20x20xf32>
    %93 = stablehlo.cosine %92 : tensor<20x20xf32>
    %94 = stablehlo.sine %92 : tensor<20x20xf32>
    %95 = stablehlo.multiply %91, %93 : tensor<20x20xf32>
    %96 = stablehlo.divide %95, %94 : tensor<20x20xf32>
    %97 = stablehlo.subtract %85, %96 : tensor<20x20xf32>
    %98 = stablehlo.select %4, %97, %85 : tensor<20x20xi1>, tensor<20x20xf32>
    %99 = stablehlo.compare  LE, %2, %9 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %100 = stablehlo.floor %2 : tensor<20x20xf32>
    %101 = stablehlo.compare  EQ, %2, %100 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %102 = stablehlo.and %99, %101 : tensor<20x20xi1>
    %103 = stablehlo.constant dense<0x7FC00000> : tensor<20x20xf32>
    %104 = stablehlo.select %102, %103, %98 : tensor<20x20xi1>, tensor<20x20xf32>
    %105 = stablehlo.convert %104 : (tensor<20x20xf32>) -> tensor<20x20xbf16>
    %106 = stablehlo.custom_call @check.eq(%105, %1) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> tensor<i1>
    return %106 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0x1E40D2BE00C051C02A4005C075C023C0F2BFAF3FA7BE21BFABC057BF753F01BF943E28C0843F903FFB3F11BE04400CBF3C40FFC0F2BF3E3F0CBDA4402BBF1340853F8CC04B3FE6BE81C043C0C8C0A9408D3ED4BF7B3FD63F2CC0693EC0BF50C027C014409440F2BF4FBF41400ABE14C00CC01CC0273F9D3E933F05408140EEBF76C0DC3F284014C045C07A4022BF50BEB6C0CD3FCDC088C0E23F13404FC0284058404840E93F75C068C03B3F6AC09D4008C0AD3F6440AC3F06C04CC04FC0A8BE5AC09FC064BF68400440A23E05C087C00B40E43D1BBFCC3DCFBF223E8D3FB23F58C0EA40F33F873F14BF49C05FC0B73E08401BC04DBFE9BE1E41063FE8BF90BFCBC081BF25BF8DBFAFBF9BBD28402F3EE1BF6CC033C0893F58BE174033BFC43F8EC0A7BD22BF8CC0103F1C40BE3F30BF8840BFC0B4BF9CC09140C5C018C00AC00E40093E333C63409BBF30C085C04340034029C0024060C031C05AC08CBF6EC0AF3F0BC0B53F82BF363F4040E53FAD3FA3BFC43F163F9F3F2DC009BF68401FC06B400940C0BF47C020C015C06EC0D0BF343F0EBFCAC05CC089C08F3F8D4094C0F83F5BBF56C0403F993F32406440D2C05CC0AD3F663F454015BF6AC04EBE2BBF57BE833F1240133F20403840984051C07140103F25402B3F994061C02DC08840A3C03C40A03E753F7E408240B5BFEBBF45BDD040FEC087402E4018C00440803E3CBF104007C1A9404140CEBF183F87C0C93EBA408E3F644040C0D4BE9A3FABBF03405C4058400B4052C09BBF8FBFF43F15C08CC0663FB33FE53F98BF8440C23E0E3F1CC01C3FAAC0A3BF8240C3BFA3C09AC0B8402C40FF3FCB3E15C078C057BEABC02040EFC0DD3ED93FE43F10400B404C3FF53F4740513FF6C0783FA83F93C0CB3FAE40CABCCDBF28400D4023C0A83FC5BF81C087C0F4BF653F6D402DC031405DC0B9BF14C043407D404B3F9E40864000BE873F51BFCF3FF7BE854001C063BFD6C0D6BF8CC0954050C027C0AC3E0A40FC3F673FB83F92BFA2BE23C0323F3AC01B40AD3C4DBB4C3F8F407EC03C40A3BD6EC0BDBFC4C03D408E3E9ABF4C3F9CBFB7C0EC3D044011C0453F6A3F6EBF8440C4BF51BF4CC0A0BE51C036408C404F3F"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @expected() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0x303F5C3FC07F8640473F584172C0273FFDC0C4BDE23F99BF5D40ADC026BFCF3A69C021BE07BFC7BECB3EC740ED3EC6BE663FEEC1FDC08EBFE541C43FD3BF193F04BF39407BBFFD3E0642B441A240C93F75C067BF1CBF3E3E70BF95C0343F8F406F3C1B3FB63FFDC089C06E3FD3404940B640DB3FAEBF5BC0B6BEF23EA23FB9C08DC0653E433F49405D419D3F9FBF7D408DBE023E39409640853E193F9940433F883F793F9A3E72C0F23D91BF81BEBE3F0941E6BD903FF7BD3541C0409940DF3F1440F2C107C1923FED3E54C05841AC40073F16C173BF27C1E8BED5C0D8BE91BD2A40F63FB63EFBBE2EBFFE40C53F3AC0003FEF3F82C0E13E0F40EDBF77C0014167400043AFBF204102404841433FC6C01FC02ABF43C0EFBE7040223F04C0863D0E4039419FBF3940D8BF2C3FB23CF3BFAA3FF0C1CE3FBDC0B33FF9401740DC400E3FFBC0B8C28F3F9240FBBFED40713FE83EADBEE33EB23F12C014402D4191BFC4BDC74041BD814298BF6C3F8E3EE6BD5340863DCCBF73BE95BF8FBE923FA03F943F023F343F21418D3F3B4091BF0ABF9BBFEBBE814000408540CCBEAF3FAB3EC33EC4C042408BBF97BE553F903FAA3F0040E6BD42BF743F38BF81BE8040D3BF72400ABF173FD2BF343F5F3FB93F8640983FD8BF3E3FA8BFBA3F9F3F95BFAA3F4141663F57C026BFA03FA33FC43F96C0A141E53F5BC1A83F4E3F1740ED3E87C027C0133F3440C93F6E3FBDBEC8BFAC4027C0D63FD2BE903FC07F533F92BE1A40E83E8B3F883F073F7A4092400A41B93E3B40394042BF81BD8E3EA840A53F2EC0DCBFDB3FC1BF77405340A33FF73E414142C0D43F4B3FD63E26C03B40C3C072405D40343F194016C0523E8B3E133F073F79BFBB3E773F6EBF863B21BF1FBE2D3FE93DCD3F204292BE433F0C3F273F1FBEAB3E0642AC4018C144BF963F95BF533FEC3F9C3F4940713F9F3F7BBFBF3FA73FE640FBBE91C0103E463EA63F824202C101BE8DBF3940B73F8F406F3C47C0053FCE3E40BFC1BCE540F23F273F9EBF12C12A3F40C2A04379BFB13FF3C1663F3E4191BF6C3F1841673F73C0994079BF8C4040BF11C1ED3E784085BF3ABF5AC1A53FD13E91C0C040F93F86405C3FAE3F72BF"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
}

