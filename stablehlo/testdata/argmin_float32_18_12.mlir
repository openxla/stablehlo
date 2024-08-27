// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<18xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<18x12xf32>
    %1 = call @expected() : () -> tensor<18xi32>
    %2 = call @argmin(%0) : (tensor<18x12xf32>) -> tensor<18xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<18xi32>, tensor<18xi32>) -> ()
    return %2 : tensor<18xi32>
  }
  func.func private @inputs() -> (tensor<18x12xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x0A589DC0E62D20BF97B9F2BF7CDA4C3E302D8FC0BE7F9CC026F117C0B09E0DBEE9E0CA3F463B893F686EFFBD27AE38BF64EFF9BFB465293F025F8240D501194092121EC00E95093F192D57BFC64525C0B8723E40089D8B3F26EA19BF750304C0CDCE01C0CDE6FF3F9BEA9FBEB39437C0298022BD76B36CC0EED186BFE0EB73404B2C1BC0292882BE26CB6FC088578E3E1761C9BFCF905BBDAE296A40675C7BC074F7F0BE667D4E4063E01DC027511D3F97847F40B13F074083F417BFC57490BFF36098C08C03D0404CEB0E4035E800C0D095973F42F93040D5062F401B9CB8BFA3CC7E3F30A928C0FA747A40BCBD87BE5CCA3BC04E7089C076617B40F9590340E095404024C767C0B700E4BF003E61C04D95BE3F7AD2513F6CCC81C01C708BBF54988AC092A6E040DA1E123F0C81A73FBEF23F3FCC7D1640DB7834C0420D66C064FD4CC0D4DC434045E9BA3FAF5F3CBDADD4B0BE966BE83E50CADEBF957962C02777A9C01DDC35C01EE3AF3F784C9340956F00C154F8EA3E020C8840087A62C0A3211BBF451D14C0D4C418BF64B8443FC4FD2FC0DC385BC0FFB87B4036083040AC117ABF4226CBBF68286740B78B2EC0BA794DBEA92580C0D6BDA3C0E83CCEC0BC21C3BE0B6B8E3FCC01C9BF647BF53EE51AB83F748499BF8499F83F6BBE2840224824406F53C9BF457639C0E98CFE3F1567023F67F6B03EA97B58C0677EA3BF63B48B3FCEF46240E79343C0BAEEC0BFDC0DF93EB9BB69C0425ED7C05914EAC0837B9DC04D6C6E3F49AC6EBF7D499B3DB0F0E83F52C566C0E56543C0CAA0F63F97224F40749D0E41019CFB3E540DB340FBE304C0EFD8A04034B26C3F93E6B43FF29A69BC899336C0BB1BCD3F9DDE7EC05F59E63F5331DD409A141EBF666DB23FF81684C0BFB1084057D8C4BF7568FCBF774765C0A61BA540C27A9E3FE5C50A4078F4CFBF8FA344C06E2097C0B5B22E40CEF51E3E770546C08A53A7BF4A0BA93F6985A93F177556BF034A8140BE4763C0DA6A84BFA441CD403007FEBF6811A6BF3EFD4F403D13933FDBAA723F75E45DBCAEAF7E3E2ADD32C0EA77E1BFCCB87C4086F682BFBAE602C083769C4053B16D3FCFFC3A3E300224BF0EF273BF1B5E27BF02A5C3C07D6E17C071DF10405950E8BE1D265AC0076F12402DE991BFAAB70F3FF41BBDBFBC3084BF3DD9B33FC3171B3FBEE3E2C09D611D40CA6EDABF4A7625C0"> : tensor<18x12xf32>
    return %cst : tensor<18x12xf32>
  }
  func.func private @expected() -> (tensor<18xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 7, 10, 3, 0, 1, 0, 8, 5, 3, 6, 3, 11, 4, 2, 9, 8, 8]> : tensor<18xi32>
    return %c : tensor<18xi32>
  }
  func.func private @argmin(%arg0: tensor<18x12xf32>) -> tensor<18xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<18x12xi32>
    %cst = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [1] : (tensor<18x12xf32>, tensor<18x12xi32>, tensor<f32>, tensor<i32>) -> (tensor<18xf32>, tensor<18xi32>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %2 = stablehlo.compare  LT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %3 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %4 = stablehlo.or %2, %3 : tensor<i1>
      %5 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %7 = stablehlo.and %5, %6 : tensor<i1>
      %8 = stablehlo.or %4, %7 : tensor<i1>
      %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %9, %10 : tensor<f32>, tensor<i32>
    }
    return %1#1 : tensor<18xi32>
  }
}
