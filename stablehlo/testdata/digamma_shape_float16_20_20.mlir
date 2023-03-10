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
    %105 = stablehlo.convert %104 : (tensor<20x20xf32>) -> tensor<20x20xf16>
    %106 = stablehlo.custom_call @check.eq(%105, %1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<i1>
    return %106 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0x3BB6844530C66F38C941003482AEECBF8DBE94B2B8C5F3BC20C79DBB7F3E6FBCEEBF43C175C288B825BCD0BEC6411A429BAD0A3C96AF08BEB740373821BF03C146BA894170C09EC4BCBE0DB31CBEAF36C036C4B5F9BFACC38EC49B3DFAC47B45884458BD123E353C1949F5C08F42834143C4573E9C397F4130B8FD3969441A43AE3D56410EB5E4BD4B42F63D36C4763C15BF0CBC8641513D8134AC361C344D2D96BAF93B8C3CBEB455398D3C1138ACC19BC2DFC4B3412649C63EAEB97C32B3C40B3EFEC1A043D53BDD40BB44F8C864C173B88FBEE1BFA4409DC1B2B2D9C4D8407B3C1FBFEB3DF63D5345A03C34A5C9C2B3C0E1BB803E25C10B3F182D7A3AD2B9EDBF9144C545053C3740F04457C41840874192C11AC40745DA45033CF83C4F4397B9022B043B2E3E81C155BDEE4318ABF54566BE2EB6A0C31E3C883D46C19FBCAB3DC5BDD0347844CBC2AFC02D3C6D40393573BD2BBA6BB69342304530B4D8C012B4763CFBC49CC14638D3BE1C40A6BBB0BD02C1B5BF8D4405424A4452450FB819C7A241633DECC0494099C6C9C684BB6B4373BCF4C1C84685C3173D3E27F7C2CE41243E68B98939A63E30BFFC40F7B820B885C203C02CC4533FB044CFB2D13A32419DBDE2BB93C15C2D00467DB109B810C13E4156C7AF3CC235F0C19AC4DE3E5DB9573C19B43FB18640503D673A2D43EB449BBD82C15EBDE1BD4DC31FB9F0BC8CC055B3BDBD0546FFBC45C626CB4F3D9CBF7EC53D44A7BE754021BCA3C5F1C60544E52F8BB9AEC1ED3F532CD83110C0F5BE804451B1482A10BD9BB46CBBA6C3EE3D7CC08EB714407440AF4165C6FF3FCAC4E6B350BA6840F6B85ABCF4BDEB39BE38EFBF114526B9DABA4D423F40313A7CC2F23EEDC4223E3BA62C402B4151C3BDB7A6BDA93BF53E78B503C14FC54D3F79BA65467CC22DB64F4361407AC274C3D63E4DBF223F49382EBC59C21941113F9B9C21C1F0C12E4045B717C4FEC1023F2AC1463FD4C3F02A87B481C0613569476B3B5EC5C73A3F32F045643E19C53E353842234051C01F434AC750C579C55DC050C36041734139443535E841DCC33EBF343F40411546FEB592C1644214B907B5763B7D343BC0F5C205B4AAB8A3C167C0"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @expected() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0x433C753E9A46E0BE093B3AC48A4847D275B9C643FFB9F043CF4809CDB030AE48FDD22CB3EB448CB8004F21BE043B8B3B58497DB8A5470A390C3959BF0AC2303C41C39D3AD544C23605BD0D432337E0C0D2C07C3D8BD814C44F3CAEAB1CD16E3E963DCC40C22AECB78B403C3D1C3C923A7B443C2FDEBC8B3A8EB15CBC773D7B3C44A9423A993FAE3BD53BFF2694456FB67FC15C55983A60B07CC3E2C01DC443CAC3C4AAB8F5B5544046BDF0B5B1BF56C42943A5C5E53A9140813271C038C5F7B8F329F6DBCE3CE6B85A39C73DB9CA30BC8FB7A3B901D0E338A4C2964361C4503953B6F2C16324FF264D3E89B51352EE4051411AD4B7306B36173482CA8CBBFEC09DD29F3DA73E8EB8D637F83D9E423E37993A77C185490C3EB73E94B894B39D3C1DC0B0CC67BAF92CFDBFDF40FC3C584CCB3E40B4643C7AC23EB811ADAAB47146A5A9FF3CFCC2863DD84082410FB86A3868C22B40A5C2963B1F3C303E6941753FAB416FB62ED286C237BF4DBE52378ECDCE3D433C42CA9B3D6A3B573D4C3E6CA7F949C83A85AFE93D1638B93BC1BFF7CBAE3C83482BD15B3F8EBF6EB27CD03C3E113B682CFDBEF9BCB331CBC298396EBC9DAE48445D59C946EC34BD3D6943CFBA003A8F3E3DD490C132CAD33EEE44B4996F3A163AAE433AB5C4C1AACFBF381933B7BE21B79B413745A23868B0B7BB873CF43DA33E12C0A740E73BC2B148BD08447843A9424D3DD73E7743B144064571B084C87A3F493DE5BB7C38D84F06366FCB0C3D40C8E5BF80C49236A3CBCAC51B503FC08E3D214538CDDB4294409DCA44C31A25504492342A377A38DE3A0642C136B8C0F2417FC35E3868BCC849813A73BC45BE68D3153E6FBD29C6D83BFD371BBC9F449533C5C94B2C0F51A137F3396DB4D831323E32B9A833533E303CD543DB344DC41A3F9F44673C9D3C4E38B44411BDE73248C45C3431BFA44D7946D1392934EE5AAB37AACFAB3725372E4AF6DBF833D434C634EFC8BCCCB9401F4435C2BC3FA1B95442E4BA6BC5C73EF02FD24962C2B83B7537C6467E3C8144B9431E40E34509B4543A763A453D6DC23C3B40CA97C392341A3AE23EE03C77C1FA3B0ABDB03F8DB983C3A248623EC841D1B962C34A45"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
}

