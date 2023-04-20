// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<7x5x3xf32>, tensor<2x0x1xf32>, tensor<3xi32>)
    %1 = call @expected() : () -> tensor<7x5x3xf32>
    %2 = "stablehlo.slice"(%0#2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1xi32>
    %3 = stablehlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
    %4 = "stablehlo.slice"(%0#2) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1xi32>
    %5 = stablehlo.reshape %4 : (tensor<1xi32>) -> tensor<i32>
    %6 = "stablehlo.slice"(%0#2) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1xi32>
    %7 = stablehlo.reshape %6 : (tensor<1xi32>) -> tensor<i32>
    %8 = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.compare  LT, %3, %8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %10 = stablehlo.constant dense<7> : tensor<i32>
    %11 = stablehlo.add %3, %10 : tensor<i32>
    %12 = stablehlo.select %9, %11, %3 : tensor<i1>, tensor<i32>
    %13 = stablehlo.constant dense<0> : tensor<i32>
    %14 = stablehlo.compare  LT, %5, %13,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %15 = stablehlo.constant dense<5> : tensor<i32>
    %16 = stablehlo.add %5, %15 : tensor<i32>
    %17 = stablehlo.select %14, %16, %5 : tensor<i1>, tensor<i32>
    %18 = stablehlo.constant dense<0> : tensor<i32>
    %19 = stablehlo.compare  LT, %7, %18,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %20 = stablehlo.constant dense<3> : tensor<i32>
    %21 = stablehlo.add %7, %20 : tensor<i32>
    %22 = stablehlo.select %19, %21, %7 : tensor<i1>, tensor<i32>
    %23 = stablehlo.dynamic_update_slice %0#0, %0#1, %12, %17, %22 : (tensor<7x5x3xf32>, tensor<2x0x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<7x5x3xf32>
    %24 = stablehlo.custom_call @check.eq(%23, %1) : (tensor<7x5x3xf32>, tensor<7x5x3xf32>) -> tensor<i1>
    return %24 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x5x3xf32>, tensor<2x0x1xf32>, tensor<3xi32>) {
    %0 = stablehlo.constant dense<"0x37476E4080B5AFBF81FB963F32599F40255C0AC0014BE93FB9FCC5BF3ACF8B3EBC8C05C1DC7988BF29E3803F1308D73FCD617BBF7EF05C403A108DC0476472408627F03FA10F553FF72FB1BF4C300AC0C3E6A3C0EC3C9940B62BA5BF596655403B1A6C4048BA4FC020330940CF6ADEBEEC6C40C0EABE29BF54ECB2BFA7A205C0054E8DC0A6A7A23FC4D908BF67994F40CD28F5BF59C902C0F2EBD13E51EE194073F68EBF71B3763FF765A2BF69DF26C028C500C09B6BA240A242F23F79E98D40B84A28401DE8B13FE85BD24056992E3DB39CDEBE79F20A3F9C17A640EABA8CC052C73DBFD2D45D40E13A41C0DCC42ABF392980C0F7C2F43F5C72294078920040382322C0360648C0FA30DA3FD7DAB13F180E7740143993BED2E412BFAD5939405987F8BE76E96A4012F690BF0017043ED7333240E343923EFF6BF13F5A00EB3F3319823F34F97F402B62A1C057A50E3FDAF2E23F02EAF33F93EC1840511B1DC0F3AFD0BFAE8E52C01EBC8BC056F2F440AA4F5ABF4B298B40245D7EC08ED4603F675066BE895126C0B8B232BF9B3D6E3F5DFD32BE95A35540C50AFA3D690F6B3D09F98340"> : tensor<7x5x3xf32>
    %1 = stablehlo.constant dense<> : tensor<2x0x1xf32>
    %2 = stablehlo.constant dense<[4, 1, 0]> : tensor<3xi32>
    return %0, %1, %2 : tensor<7x5x3xf32>, tensor<2x0x1xf32>, tensor<3xi32>
  }
  func.func private @expected() -> tensor<7x5x3xf32> {
    %0 = stablehlo.constant dense<"0x37476E4080B5AFBF81FB963F32599F40255C0AC0014BE93FB9FCC5BF3ACF8B3EBC8C05C1DC7988BF29E3803F1308D73FCD617BBF7EF05C403A108DC0476472408627F03FA10F553FF72FB1BF4C300AC0C3E6A3C0EC3C9940B62BA5BF596655403B1A6C4048BA4FC020330940CF6ADEBEEC6C40C0EABE29BF54ECB2BFA7A205C0054E8DC0A6A7A23FC4D908BF67994F40CD28F5BF59C902C0F2EBD13E51EE194073F68EBF71B3763FF765A2BF69DF26C028C500C09B6BA240A242F23F79E98D40B84A28401DE8B13FE85BD24056992E3DB39CDEBE79F20A3F9C17A640EABA8CC052C73DBFD2D45D40E13A41C0DCC42ABF392980C0F7C2F43F5C72294078920040382322C0360648C0FA30DA3FD7DAB13F180E7740143993BED2E412BFAD5939405987F8BE76E96A4012F690BF0017043ED7333240E343923EFF6BF13F5A00EB3F3319823F34F97F402B62A1C057A50E3FDAF2E23F02EAF33F93EC1840511B1DC0F3AFD0BFAE8E52C01EBC8BC056F2F440AA4F5ABF4B298B40245D7EC08ED4603F675066BE895126C0B8B232BF9B3D6E3F5DFD32BE95A35540C50AFA3D690F6B3D09F98340"> : tensor<7x5x3xf32>
    return %0 : tensor<7x5x3xf32>
  }
}
