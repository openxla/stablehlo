// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf16>
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = call @bessel_i0e(%0) : (tensor<20x20xf16>) -> tensor<20x20xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0xABAC58407A40E3BF0644A4381CC28F4165AA643DAEBF12B6333B934698C176BE14C115C26CC45EC0D44053BF2839F2C20328553BFC40B1BED7361DBC07B7F2BC3FC08DC298323EBCAD403442213F4D426DC006B26D3D643E04C4D3C001B69E3D4F3CF73D2B38EFBEC2462DC35F3FF8424332EF40CD3BB1A1713C11C0353014BFE44442C19C406E4327441F40BFBD493736C8EA2AF5BBBE419C3FEEC031C4ABC4DF3C2C43F444C23B7D3D4B3BB5BBC6B3E342FE3844C3C2C09044D33DE0BA62B5A73BA23ECB3E44C07843623E534458432AC4C2C17AB9C6BA913D733FF2B8023A1AAE8946C345C04363C09FBE90BA5FBEAAC492C063415E43EA440AC094BC84441EC02A413C42253C0FADF3C04A452440C036BE3D1D3B3A4581C2A1388FBE622C4145D9BFDBC3E4BC48C232B823C2C1BC6140DBBE0FC6C1B8223E6044D8446FC77D363DC5B9411A3C22C39247304173417F3EE045323CEE3FF1B02EC5C8C46EC44BBC3BBE42C1DE40AE402BB6F73FA045C63A75BDA539FD3DF7C3A848AB4521C2F3C452C16E4040352B44EE4028B52E438CC47D3F2B3E1DC6D6B51ABFE3B684B6AB4054387643ABB3EE375B433D377D31D7C08C45E9C3C6C3863455B52AC509C6A73E40BDD84004C72125E2BD9FBF8A423F31CA402FC105C2B0C1CB42034012C56BBCF54091B711401037723264C1F037613F1CC547401E3FACC62A3D5EB52C2604438BB875B2CC3DCCBD8E3D873C22C54542F53CD7389FC37A3675BFE241EAB4613F81409E408D4587C157BE173CBF3E7443E7C22BC4BF3D0DAAC5B381BADDBC414370C4E2C21D3FB4BF1BC3BDBFC1C223C4AD43AF4089C0ED3630C32F407A41C5C0C53B5FB9663C7DB9A5BA7BC49E4549C450C37D3FE741D13D014000C7EBB4143CDA4081B8F43D6A4169C3FABB6AC706C3BEBC47C460420CBD8EB6BFBE35409A2A8DA1D64150BDD2BE8E4113C533C2A1440EB191BEF14474C5A040083CD04056418B450BB1FF3ADCBBA0B737C1FBC2863DD13C26BC68C392B03EBE2540A7C1D7BE86BC3EBF55C604456AC15CC35EC51F38583AB9475B38CE381C48731411C5E2BDBCBA5E4505BF08C043C5263395C64AC75834FC3D46C4BA4469C1BB4149BF"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @expected() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0x723BB2349D34FB349A32DD38B23310349D3B45360F35AC39E43715310C349F354834B7334632AF3469343535A6382933C13BD137543480357539563768399A36C3346633943A36377F34A1334B359033A534B03A3F36A8359C326A34B1391E362637E635143962350331083330352633A43A5B348E37E93B0637E3340B3B5135F13132348934E6327E32D934093657397830963B7937F93317355B3475321832A9360933E73194373436D6379B375E3A3233B738FC3273342B32FC350B38E139A33788357435C034E132AA355932F1327B32F6338538133827362735BC3854384A3B19317331BD32AB348A352438AB3519328E342334EE32EE31E834E7363432DA343D349C334E37673B5934B331D6347C390A36F137BD316E33DE3892357B3BB831FE34B132A53694331139AD33C136AD346C354E31D038CD354F32F931C5308E39BB31FD3359370E33B9303B341C349A3564314237F634E53AC431043244322A37BF35323464347E34A539F334843113383A367538E335A4323E307F31AF33E8312B34A434EB397A325B34F33908332E322335C8354831BE394E3572398C3980340139E232633A2D39F0325A39CA3A68348F31AA32BB32273AE539C631513186355F366734EA30D83BF33515356833D63A6F343B34C33302344033ED34D4310C3758344539E33466399B3A23342D392F35CE31BE344D350B316F36E239D03B1F33E8389B3A013601362936F336CB3196339736C738CD328F392735DD33063A2F35993487348E311334B0355C377A35E33230337A320936A33B5E3A2938AA36FD32433232334D350D351233093546338132C7327E3494346F390733CE341934713493379038113784381D383B3286316132F6322335D933FE35EF34EB30063A5F376634ED38E8352034E9327737C6301E33C43663328333853689397A35CA349A3BEA3BE633533670351034D431A2331F32E03A9135E9319C3186346C376B3429348F31E03A013887374139373424332E36B4364D37E932F83ABD35D53406346E35F4363E352F31DD312034EF32A8311A393638AD30FE38CB388730FE3BD531F3351638A8315835EA34B7317A3A1531D130363AE33564320D322034FC333935"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @bessel_i0e(%arg0: tensor<20x20xf16>) -> tensor<20x20xf16> {
    %0 = call @xla_fallback_bessel_i0e(%arg0) : (tensor<20x20xf16>) -> tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @xla_fallback_bessel_i0e(%arg0: tensor<20x20xf16>) -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.convert %arg0 : (tensor<20x20xf16>) -> tensor<20x20xf32>
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
    %164 = stablehlo.convert %163 : (tensor<20x20xf32>) -> tensor<20x20xf16>
    return %164 : tensor<20x20xf16>
  }
}
