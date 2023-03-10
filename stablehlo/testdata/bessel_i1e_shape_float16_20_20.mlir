// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf16>
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = stablehlo.convert %0 : (tensor<20x20xf16>) -> tensor<20x20xf32>
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
    %127 = stablehlo.convert %126 : (tensor<20x20xf32>) -> tensor<20x20xf16>
    %128 = stablehlo.custom_call @check.eq(%127, %1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<i1>
    return %128 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0x7C407F4425B1CABFCAC0BD321DC4FCBD4C4305416D3995C3B44245ABCE4054446041E1BF093547C18DBAABC03BBC8EC0BABB812C95C1FF44C2B4543CF7BD263C8440CAC57C3EE23D7ABF40C1B2B08E30A63E9546B5355BB8AFBA7AB8ADB908405ABD5643EB41E744413FCCBC52C55FC029BA6FC06DC3AE3DC6432EC3D8C5C042CCC024B475BDDE410345BC3E28C11A41E344BA45B13D09497BBE0A43CB399FC425BD734322B783AB942AD3BE7644DF3D74BE86C310B8804844BC32BEEFB929BF16C0A0BB85C1E73DBFC4D23EC3C5C62A97BB6ABCA8BC6DC02B3FAFC36BC4DFC1B5425C4096C19236E4C2BAC3163BEEC4C64301B8F8C0DC361436FC36503390C4034456C0774418C1BE4564C32742BB44343F42A90B3DFCBD13C01C46874176C02F44C5C03141E9C49F4183BC27406E3F68C309C3553A79BD85C12639CDC260AF663F4B3D653B13C645C4D7C616C422432737C9AB42BFD52C9E42A53452BBC53648354E3AE240B5BB363AF025C63077B399C1B440CD427341CA29B941F2C33D3CEAC124BFD0B53F3D71B5ADC415422D3F6FBA64C39846FA41D6BF9143C5436B4080416C3634BBC4412DC23D435BC49A4306422E47E73CA9C8E0B99ABE6CBDF2348DC2DDBCC73C58BD5CB93941B2C050BCAE3F3A3815C4523D2BBBB43A3B417CC49B402AC0AD43B8BCACC27743C044A74178410BBD0D3A403B9134C3388F40CF3C63394BBF8EBD5FBE89392646C4B5B2B2F14602C45FBDB4C096B66CC411C2F5380F4188BC213F6DB0FE442948D53C2240673C2541553AC84105B99E3E90447BB449B98CC4B74536C0AE3DC8425742FF420841003339BE35BBA6B8EF45083C4B42434609393E42BFC301458A45FD2FFB45D43F34B448B8C9C3B5B97CC6FF3D05413F2730458440B0B138C29B3FF53DD538D0382EBC0240D8BEBD41BD36743CB344533E3044813833434AC6674553362EC094AAE63E49C0ECC444C43CC34EC438C500BE9F36EF2C1A3F6BC4F93A38C47A400144F7BFBBC50D46FCC4B1BDCC3F974528C550C358BF8ABD7843E43D3BC071BE623D1C3D17BE5BC5483DA3BF6037653D51A4C9C17826774434466B386CBEDAC4CA42EF40E041D7447EC3ADB91EC1B936"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @expected() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0xC332793165ACE9B2ADB27D2DA9B102B3E9319B32D531D5B11632DFA6AC328E317F32E7B2712F86B246B2B6B2BDB2BEB298B233286EB2403125AFC53202B3B632C132F1B002330233F2B289B211ACEC2B0033AD300F303EB151B252B1F1B1E232F9B2E73153324A31F732E3B21EB1CBB223B2C7B2E0B1FF32C831F2B1ECB01232ACB272AEFBB257323E31FF3290B294324C31F630FF32BC2F02B3FC31FE316AB1F2B2DE31AEB016A74026FEB27D31023302B3D9B10DB11230C0B203B30DB2F8B2DEB293B273B202335CB1FE32F3B06D2691B2CBB2DBB2C8B2F832CEB182B157B21532CC326EB2733007B2CBB1703247B1C83102B19FB292303C309F30DB2D71B1B731CEB27D3195B2F530E3B140325D31F8320CA5EE3202B3DFB2D4307232C5B2A031AEB28D3249B16B32D2B2DA32F332E2B1FDB13332FCB273B2B2310EB295AAF332F7328532D7B095B199B0ADB1F531B03054A7F6B27C281C32062F80B28930B22F3132A53297B22832CD21202CF4AD6DB2B3320E32793289256332BCB1BD3253B2F9B21CB0F632DCAF64B14632F8323CB2E3B1AC304E32E8B2D631C831C8327432633078B25F323EB2EE318AB1D4314B328130E83202B007B201B3FAB2592F21B2E7B2E232F8B2CDB18B32B4B2C4B2EC322931ADB1F83276B253328A327AB1BB32D9B2CF31DFB218B2DD315B3168327732EEB219327B32EF2E7D31BE32E432D031F6B2FDB202B3E131D13016B076AD9230B7B1F9B2B3B275B082B147B298319832D4B2F932BAAB40313830E532DB32CB32913233325E32A1B101337131D7AEC3B173B1F730D6B2FF3210323232FF319A32A92D03B378B26CB1E330AA323532C730A3313932CAB13F310831112BDF30E83285AE32B1C7B1F5B1B5B002339B320B232B31C132C8AC3BB2EE32023387318431B8B2E332FEB261328530CE3261310333A0315631F131C5B015315830D8B240A6FD32D1B248B196B1EEB191B128B102B379309228FA3282B168329CB1C432B831E5B2F6B0D93041B1FFB2E93203312FB1E8B1F5B2FDB2DD310233D5B202B3F932F13203B31AB1F732EEB2C730FA323FA05EB24F227D31CC30483102B350B10F32A23256325131DBB1F1B193B28430"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
}

