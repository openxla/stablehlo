// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf16>
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = stablehlo.convert %0 : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %3 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %4 = stablehlo.compare  LT, %2, %3 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %5 = stablehlo.negate %2 : tensor<20x20xf32>
    %6 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<20x20xf32>
    %8 = stablehlo.select %4, %5, %7 : tensor<20x20xi1>, tensor<20x20xf32>
    %9 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %10 = stablehlo.constant dense<676.520386> : tensor<20x20xf32>
    %11 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %12 = stablehlo.add %8, %11 : tensor<20x20xf32>
    %13 = stablehlo.divide %10, %12 : tensor<20x20xf32>
    %14 = stablehlo.add %9, %13 : tensor<20x20xf32>
    %15 = stablehlo.constant dense<-1259.13916> : tensor<20x20xf32>
    %16 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
    %17 = stablehlo.add %8, %16 : tensor<20x20xf32>
    %18 = stablehlo.divide %15, %17 : tensor<20x20xf32>
    %19 = stablehlo.add %14, %18 : tensor<20x20xf32>
    %20 = stablehlo.constant dense<771.323425> : tensor<20x20xf32>
    %21 = stablehlo.constant dense<3.000000e+00> : tensor<20x20xf32>
    %22 = stablehlo.add %8, %21 : tensor<20x20xf32>
    %23 = stablehlo.divide %20, %22 : tensor<20x20xf32>
    %24 = stablehlo.add %19, %23 : tensor<20x20xf32>
    %25 = stablehlo.constant dense<-176.615036> : tensor<20x20xf32>
    %26 = stablehlo.constant dense<4.000000e+00> : tensor<20x20xf32>
    %27 = stablehlo.add %8, %26 : tensor<20x20xf32>
    %28 = stablehlo.divide %25, %27 : tensor<20x20xf32>
    %29 = stablehlo.add %24, %28 : tensor<20x20xf32>
    %30 = stablehlo.constant dense<12.5073433> : tensor<20x20xf32>
    %31 = stablehlo.constant dense<5.000000e+00> : tensor<20x20xf32>
    %32 = stablehlo.add %8, %31 : tensor<20x20xf32>
    %33 = stablehlo.divide %30, %32 : tensor<20x20xf32>
    %34 = stablehlo.add %29, %33 : tensor<20x20xf32>
    %35 = stablehlo.constant dense<-0.138571098> : tensor<20x20xf32>
    %36 = stablehlo.constant dense<6.000000e+00> : tensor<20x20xf32>
    %37 = stablehlo.add %8, %36 : tensor<20x20xf32>
    %38 = stablehlo.divide %35, %37 : tensor<20x20xf32>
    %39 = stablehlo.add %34, %38 : tensor<20x20xf32>
    %40 = stablehlo.constant dense<9.98436917E-6> : tensor<20x20xf32>
    %41 = stablehlo.constant dense<7.000000e+00> : tensor<20x20xf32>
    %42 = stablehlo.add %8, %41 : tensor<20x20xf32>
    %43 = stablehlo.divide %40, %42 : tensor<20x20xf32>
    %44 = stablehlo.add %39, %43 : tensor<20x20xf32>
    %45 = stablehlo.constant dense<1.50563267E-7> : tensor<20x20xf32>
    %46 = stablehlo.constant dense<8.000000e+00> : tensor<20x20xf32>
    %47 = stablehlo.add %8, %46 : tensor<20x20xf32>
    %48 = stablehlo.divide %45, %47 : tensor<20x20xf32>
    %49 = stablehlo.add %44, %48 : tensor<20x20xf32>
    %50 = stablehlo.constant dense<7.500000e+00> : tensor<20x20xf32>
    %51 = stablehlo.add %50, %8 : tensor<20x20xf32>
    %52 = stablehlo.constant dense<2.01490307> : tensor<20x20xf32>
    %53 = stablehlo.divide %8, %50 : tensor<20x20xf32>
    %54 = stablehlo.log_plus_one %53 : tensor<20x20xf32>
    %55 = stablehlo.add %52, %54 : tensor<20x20xf32>
    %56 = stablehlo.divide %51, %55 : tensor<20x20xf32>
    %57 = stablehlo.add %8, %3 : tensor<20x20xf32>
    %58 = stablehlo.subtract %57, %56 : tensor<20x20xf32>
    %59 = stablehlo.multiply %58, %55 : tensor<20x20xf32>
    %60 = stablehlo.log %49 : tensor<20x20xf32>
    %61 = stablehlo.constant dense<0.918938517> : tensor<20x20xf32>
    %62 = stablehlo.add %61, %59 : tensor<20x20xf32>
    %63 = stablehlo.add %62, %60 : tensor<20x20xf32>
    %64 = stablehlo.abs %2 : tensor<20x20xf32>
    %65 = stablehlo.floor %64 : tensor<20x20xf32>
    %66 = stablehlo.subtract %64, %65 : tensor<20x20xf32>
    %67 = stablehlo.compare  LT, %3, %66 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %68 = stablehlo.subtract %6, %66 : tensor<20x20xf32>
    %69 = stablehlo.select %67, %68, %66 : tensor<20x20xi1>, tensor<20x20xf32>
    %70 = stablehlo.constant dense<3.14159274> : tensor<20x20xf32>
    %71 = stablehlo.multiply %70, %69 : tensor<20x20xf32>
    %72 = stablehlo.sine %71 : tensor<20x20xf32>
    %73 = stablehlo.log %72 : tensor<20x20xf32>
    %74 = stablehlo.constant dense<1.14472985> : tensor<20x20xf32>
    %75 = stablehlo.subtract %74, %73 : tensor<20x20xf32>
    %76 = stablehlo.subtract %75, %63 : tensor<20x20xf32>
    %77 = stablehlo.is_finite %73 : (tensor<20x20xf32>) -> tensor<20x20xi1>
    %78 = stablehlo.negate %73 : tensor<20x20xf32>
    %79 = stablehlo.select %77, %76, %78 : tensor<20x20xi1>, tensor<20x20xf32>
    %80 = stablehlo.select %4, %79, %63 : tensor<20x20xi1>, tensor<20x20xf32>
    %81 = stablehlo.abs %2 : tensor<20x20xf32>
    %82 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %83 = stablehlo.compare  EQ, %81, %82 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %84 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %85 = stablehlo.select %83, %84, %80 : tensor<20x20xi1>, tensor<20x20xf32>
    %86 = stablehlo.convert %85 : (tensor<20x20xf32>) -> tensor<20x20xf16>
    %87 = stablehlo.custom_call @check.eq(%86, %1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<i1>
    return %87 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0x1240B9457BC2BCC1FB385B42A5C23537EAC3074584360925963423C248BCA1B917C3B43E634458C3A6BCFDBA43C2AF45743F0A3795BA3C46A6B76C3F65C5A0B564C4AE403C45CE36103C8B30F2C1424511352D3942B4E4C34BACBD400C3E5F42424226C2F43EA4BC04C5D7443B3D61459CC59DC34FC44D424DBF26BFD5C022C40C3D30BCBDC1ACC4174162C450B8BCC573BDCD3C74C1342D6741703D2CC51EBE6DC004C4D3BA17B43C3B6E3F87479A3F5C448AC0FEC44EBE503F18C08C46B33D85383D34B6B795BE9334023E3B3009B86DBF93C2D83AECBF5B3F0F44E0BDF1C7EABB743C36402AB3C9BA4FB2394259B6DA31DABEB4BA1E3C2EC4F345BCB9F8AC863B6742BDBD26B45F40D0320D3F593824C2A7BB9438E7BE7BC51BB3C3BE2CBEEC425B3EF74063C156C4A6364041A03EA241F543F1B9E5C336C14A3CEDC527347F40BEBACCC066B911421ABA67C3C7C08BC1A4BFC3C6C5BF9F3C103FCF35B1BC97C072C20E4008C218BA0BC080C374BBD0C2783088421CB82D4062BA46B021410C43454433C5922DD83E0544C7BEFCB91643813E17390A4502443EB9D0C255B68EBECDBC8A4118439EC3193BC04204C1C2B6B6425F3E8644B3B982AFE43EBD3CCBBEEFC331B5F8C2A743C8AFA1B4A8B5263E41BC43BD24BFF4469146DB40C04045C2F8C00FB816C6E63B5BC2BA2EF84353C348BC013F39426E44BCC407BDA94054BCDFC198B547C63841513B634511C1674031BCC22DA3B83843993E283B963C53BD57471BC6FFB66DB5A9380CC239AC5CBDF9B98E3E41B61DC4D5467943B2C1EAC4174500C3C93C87C050B95741D4BE3CC4D925E6C3B5BABD3E874279BC64C1D8BFCE448B38293F9AB8F1A45FB0503C08392C376BC27E3C63BFF2422A3C79C733B5C54511C4844079BA062B74BF134600406E40D5448B42CC420C3C1DC5EFC595449FC765C16FC3E9C6EDBECAB0E0C1FDB88BBA8244D3C253BA20457DC89D3BA0421B454DB35DBEF4C269C4C34106BC744369461343CFC16F3D193F63BE84C19F2D313C5C3D77C53DC5D942223D553FC03B85BC50BCC8C257ACBAC0B74322C0A02F0E400BBDE4C09CB977C236C5D6455644883A27C54DBC4941DBC470C45333"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @expected() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0xD0235244A1B8BB36D535E83A36BB6939B52870423A3AD643933C803A4341D73D72BD71AE994097BD253F49409F2E42449BAA9939633F3145133DEFAA46C47E3D28C1B6311343E03984A08B3F5F3F2643283C47352E3EEBB17D414D32B3AFF83A863ACB3995AD323F3DB9E14101AF8743A8C4BEBCA2C0B13A073D783C822C54BD74AE1642EA36E1C111351DC1153DA5C4323C78ADA7A8F040F93671AF8CC2BF3ACB39A93B04404B3E0B2CDAAA9647F9A88640A537FD2CAC3A05ACA740BF45BCAF4637E83C123DE33A963CBAAFD83F103DA43D36BA862E86429CAB773F183BA3C78A4417AB282AB13EEB3F183F633A453D7A3E8A3BB13F27A4BABEB344F53D3641C928183B693B413EB42DD53D32ADDD37423A50421437B83B7BC4B73E443BB53AA23C4BAF5C3406ADD3C0103A0536ABAE3F380F3F383EDCB095AFCFA8A9C3FE3CEC2FCC3FC62E9E3DCB39753E84BD1030902A3F3F85C670408EAC26AD2C3BE43E3036E5B70B22B240723E424245BD70418BBC9D3F9D3B103D1229F43E36404C35E83C4840E7C2CB40FCAD443F4F3B483EFF3CFAAE84357942353F7E3D8BBC463DD93A503EDC37033DB8BCE02C433C22AC2F3D2E3C43AFF940EB3D7240D1AD2BAD5B3B7634AB3D25BD503E6140F43D7B3D9AAF7941A43C713C7E46C84586336C32862CD8A8103D46C4A41FAAB36740173F9BBD434163AD633AB740CBC15E3D8531F340263C813DB1C5D5351E2B8E439BAD3F2E0B42BA402A3D4B3DBEAE842C5BAC7B3C394783C4243D923DD036BB3F8541653C443EDAAE4B3D9BBC4446E23D10355AC0A1423CBD65AD01388C3D9436763BFABF884383AFB43F55AE993B3540E5AC2D41C7413237BBAC273DEB432C4027A9AF357339C0B696AB6E3DAF3CB7A52CC8AA3D66447AB82430243FBF41CD3DEA440000BA2EDB41A93B5D3CD09E91C179C323414BC8C3AC74BDECC5CF3B0340433C533D4C3FEE4099BCD73EBC427FC8A627003CAD42A23EAF3A18BD42C1B0382345D63D8145F83C83396FAF00ADB13A3925C74099A64CAF73C457C3793CBBAED8ABD42405400D4165BC7841F931763EDB3F24400B224F3DB724D23D55B80BC3834475405E3042C220413D3626C163C1873D"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
}

