// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x5x3xf32>, tensor<3xi32>)
    %1 = call @expected() : () -> tensor<3x1x2xf32>
    %2 = "stablehlo.slice"(%0#1) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1xi32>
    %3 = stablehlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
    %4 = "stablehlo.slice"(%0#1) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1xi32>
    %5 = stablehlo.reshape %4 : (tensor<1xi32>) -> tensor<i32>
    %6 = "stablehlo.slice"(%0#1) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1xi32>
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
    %23 = stablehlo.dynamic_slice %0#0, %12, %17, %22, sizes = [3, 1, 2] : (tensor<7x5x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x1x2xf32>
    %24 = stablehlo.custom_call @check.eq(%23, %1) : (tensor<3x1x2xf32>, tensor<3x1x2xf32>) -> tensor<i1>
    return %24 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x5x3xf32>, tensor<3xi32>) {
    %0 = stablehlo.constant dense<"0xE21719405B41BE3FA8B05FBFEFDCA0C0B81F00C04E0ADBBEFC0581BF911B96BF78221B40E0BE9B3F42A688BF129ACB3FD99313BF48E852BF73450040B3CD84C008AC7F3FFF6D58BFE054154002690E40072F6FBFD60EF0BFA1AE75C0C23E9EBEBE197C404C4986BEFE6E06C157C5B23F248E554008C08ABE1AD890BFFE99DD3E181AC2C0EC76F93EE5D6AE3E0015E54009E8B3402E9F093F7920B0BF4201A2BF09B62B408890303F82E52040565E21BF577E0B404089713FE0EAE43FD56E353F62DF1E3FBB4FFF3FB863A4BFF740D240F45D27C0718197BE1A7CCF3F1635933FD0559CBFAE821440275D5EC0ED9615C0EE7DF9BE284CC9BF56F62AC0600124BF592345C0B5D35340B15CA8C053BEE93DA126A2C070F71240C33566C0610D0F409184C5BF4F5984BFE0FABA3E900FC840F64731C068AE12BF18744FBF49300F402A8AD33F966601C1051337BF0419B2402AEB8EBFCF32F5BFF63F5CBED4F6023D3F5C97C0FA83CC400DDDCEC04A4D0A407447AD40F515E13FDE99BD3F9330AB3C8CD5A8BF3FA6BD3FE5DDACBFF67191BEABBA9CBFAAB49ABF51079FBE06150F4025F186C0"> : tensor<7x5x3xf32>
    %1 = stablehlo.constant dense<[4, 0, 1]> : tensor<3xi32>
    return %0, %1 : tensor<7x5x3xf32>, tensor<3xi32>
  }
  func.func private @expected() -> tensor<3x1x2xf32> {
    %0 = stablehlo.constant dense<[[[-1.5726366, -2.67128515]], [[-2.77001715, -0.572973728]], [[2.16096735, 5.41497231]]]> : tensor<3x1x2xf32>
    return %0 : tensor<3x1x2xf32>
  }
}
