// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<18xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<18x12xf32>
    %1 = call @expected() : () -> tensor<18xi32>
    %2 = call @argmax(%0) : (tensor<18x12xf32>) -> tensor<18xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<18xi32>, tensor<18xi32>) -> ()
    return %2 : tensor<18xi32>
  }
  func.func private @inputs() -> (tensor<18x12xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x1AF6E43E2A04293F083E6FBF444748C067994F4021728F3F53AC19C02A0DCA3F32FE933F1F16043FBE9CFB3F6C4CAF40667AF2BF4F7E3EC044B8803F85A3913E4829E94062EE5A40996D47C090669BBF9E3119C05D591B408F755D403A326D3E23A94F40A1D282C0F2FBBDC08FDCB3BE87AB0B40328BA240C5CB81C0612538C0B1D84D40C909C4402342FEBE3439D33F965227C01E380340A5D4BCBF6BF340405DF663C00C2AB2BFE2704540CECBC9BD4C5828400BB432C07AB2CABE6BED523FC2CD8DC022AC63408BF998C09640133F1E1C45BF1169DDBEE21724C0E6B5A8C0B652DFBE46C704BF68A7944064E1D8C03DCB384012CE253F1A506BBFA2DE2B403CA78940AE58E2C01E31843F70B268C00AF33FC035F27BC0983A3AC0D22A35BFB7F5354069A0C73F8497023F7E1B47C0D393EC3E4B7BA2BFF7458340134484C039E661C0BB78C7407FC90C40CC9186402E2D1841608590BE10D223BFD5A3713F23B075BEE5B2B03E5ED1C63F965C9E3FB46635C079E584BFB3E322408D29233FD78581BFE892A4BFCDDF9E3F53C037BFCCCC6EBF5D59BEBF086E3040C4D287BE6059C4C02D8F34C0A559BC4089A25DC0F2889B3D49BB6BC02F1C88BD9EB57F3F20CFB3C0CC2108C035938B3F9731A0C047030C409793A3BFF697E4BF57A5F4BFA5CAC7BD746C9F404D0B8140F016CAC07926893E5C23D83E86D63D3F563F98C0B971E9BF8E4BA23E4485B8BDC344803EC696273ED15531403D7E694092131E3DF17CA73F5DD6BC3F937272C01CA2503F7D2808BF3C39ED3D8EFF6E3EC8903740AC968DC059D577C087AC173EB6F5E33B140B41C0737166BF776EA1BFEBF7A7BD731985BFFFE92EBFB7241A40381252BE77F6AFBF88F0D53F3BEE03C09B00B1BF7E164F402B18F7BF0B0A15C0849F9C40843E2A3F0E47BD402E3008406B1B164028B18D3E9E04CA40A78B5DC0A3E95C40324E003F507A8840253AC3BFB37ACABED0A0E2BFC8C7C53FF33B10C07BA22BC0A534453F4FDAB0BFA265BEBF6B6C19403B13E7BFFD28A23F675D16C0779F2240576A54C0B9EC04BE05F8193F853BAEBF3AB694C0AF0017BF1D4FB0BD93363EC09B6129403731E83F948286BF3A572E3F03782840A5A006C012B129C0CDC884BFDA9887BFDC5C0A3E1389DCBF1EACA93E79292AC060D19640463C893E339E1C4001D6BEBFDA75E53FBFE6B2C052D7A1BF"> : tensor<18x12xf32>
    return %cst : tensor<18x12xf32>
  }
  func.func private @expected() -> (tensor<18xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[11, 4, 9, 6, 10, 4, 9, 0, 10, 8, 1, 2, 10, 9, 1, 7, 4, 5]> : tensor<18xi32>
    return %c : tensor<18xi32>
  }
  func.func private @argmax(%arg0: tensor<18x12xf32>) -> tensor<18xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<18x12xi32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [1] : (tensor<18x12xf32>, tensor<18x12xi32>, tensor<f32>, tensor<i32>) -> (tensor<18xf32>, tensor<18xi32>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %2 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
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
