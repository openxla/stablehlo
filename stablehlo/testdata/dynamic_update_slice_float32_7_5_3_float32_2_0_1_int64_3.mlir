// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<7x5x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<7x5x3xf32>, tensor<2x0x1xf32>, tensor<3xi64>)
    %1 = call @expected() : () -> tensor<7x5x3xf32>
    %2 = stablehlo.slice %0#2 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %4 = stablehlo.slice %0#2 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %6 = stablehlo.slice %0#2 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %8 = stablehlo.compare  LT, %3, %c,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<7> : tensor<i64>
    %9 = stablehlo.add %3, %c_0 : tensor<i64>
    %10 = stablehlo.select %8, %9, %3 : tensor<i1>, tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %11 = stablehlo.compare  LT, %5, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_2 = stablehlo.constant dense<5> : tensor<i64>
    %12 = stablehlo.add %5, %c_2 : tensor<i64>
    %13 = stablehlo.select %11, %12, %5 : tensor<i1>, tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %14 = stablehlo.compare  LT, %7, %c_3,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_4 = stablehlo.constant dense<3> : tensor<i64>
    %15 = stablehlo.add %7, %c_4 : tensor<i64>
    %16 = stablehlo.select %14, %15, %7 : tensor<i1>, tensor<i64>
    %17 = stablehlo.dynamic_update_slice %0#0, %0#1, %10, %13, %16 : (tensor<7x5x3xf32>, tensor<2x0x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<7x5x3xf32>
    stablehlo.custom_call @check.expect_close(%17, %1) {has_side_effect = true} : (tensor<7x5x3xf32>, tensor<7x5x3xf32>) -> ()
    return %17 : tensor<7x5x3xf32>
  }
  func.func private @inputs() -> (tensor<7x5x3xf32> {mhlo.layout_mode = "default"}, tensor<2x0x1xf32> {mhlo.layout_mode = "default"}, tensor<3xi64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xC063C5BFB1CE253E70918BBE0615D3C0EAA797BF0A9D7540CAABA3BFF77C0A4025F37DC0CA8A434016954740329700C033DAF83F6D75883F032A273F5EDF303F028CE3BF946050C022EF4EBF75CDA83FD8BA88408C2AE0BE155D49C0ED9A84C0FDC28CBF815215C094B34F3FC49A35C08285D83F45CC92BF710553BF44AD6840A96551BE1ACD4A3E1752A1BFC3E533407344FF3FB7ADA4BFE1609BBF709BD9BFC7A9953F04276E4001DCB7BFD2CACE4033FABE3FF990943F65E1503E4A91B5BF873C4740143EE8BFA360263E3D3A6BC074E452BF70C4A340EFE6393F26DCF7BE9EB608C098526D3FF8B92740F431B93F02B44BBFDDBF73BFE2DF46BF932C90BD98C5AF3F4A89893F344F49C0CF157CBEBFDB9A400452CE3F43737C40573AD5C0ADCC27406787913DAD95E7BFA277E5BFC6BF99BF25194AC038996E4041A3A640246439C0B9A15F3E803CB1BFE2F2F0BE0023ED3FC5ADA4401CA9054082B00B40E1BC19BF60DE4AC0936B3BBFCA525DBE943151BFE6044940940ABFBFE12116C07F4D0FBF06F3B23ED81002C0E99894401D03733FA43F39C0D316393EDD1A66C02791963F"> : tensor<7x5x3xf32>
    %cst_0 = stablehlo.constant dense<> : tensor<2x0x1xf32>
    %c = stablehlo.constant dense<[4, 1, 0]> : tensor<3xi64>
    return %cst, %cst_0, %c : tensor<7x5x3xf32>, tensor<2x0x1xf32>, tensor<3xi64>
  }
  func.func private @expected() -> (tensor<7x5x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xC063C5BFB1CE253E70918BBE0615D3C0EAA797BF0A9D7540CAABA3BFF77C0A4025F37DC0CA8A434016954740329700C033DAF83F6D75883F032A273F5EDF303F028CE3BF946050C022EF4EBF75CDA83FD8BA88408C2AE0BE155D49C0ED9A84C0FDC28CBF815215C094B34F3FC49A35C08285D83F45CC92BF710553BF44AD6840A96551BE1ACD4A3E1752A1BFC3E533407344FF3FB7ADA4BFE1609BBF709BD9BFC7A9953F04276E4001DCB7BFD2CACE4033FABE3FF990943F65E1503E4A91B5BF873C4740143EE8BFA360263E3D3A6BC074E452BF70C4A340EFE6393F26DCF7BE9EB608C098526D3FF8B92740F431B93F02B44BBFDDBF73BFE2DF46BF932C90BD98C5AF3F4A89893F344F49C0CF157CBEBFDB9A400452CE3F43737C40573AD5C0ADCC27406787913DAD95E7BFA277E5BFC6BF99BF25194AC038996E4041A3A640246439C0B9A15F3E803CB1BFE2F2F0BE0023ED3FC5ADA4401CA9054082B00B40E1BC19BF60DE4AC0936B3BBFCA525DBE943151BFE6044940940ABFBFE12116C07F4D0FBF06F3B23ED81002C0E99894401D03733FA43F39C0D316393EDD1A66C02791963F"> : tensor<7x5x3xf32>
    return %cst : tensor<7x5x3xf32>
  }
}
