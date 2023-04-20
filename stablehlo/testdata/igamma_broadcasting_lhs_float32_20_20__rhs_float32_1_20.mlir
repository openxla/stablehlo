// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xf32>, tensor<1x20xf32>)
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = call @igamma(%0#0, %0#1) : (tensor<20x20xf32>, tensor<1x20xf32>) -> tensor<20x20xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xf32>, tensor<1x20xf32>) {
    %0 = stablehlo.constant dense<"0xB0D14A404391774096A018BEE1EE97BF676001BE11DBB23F7CCB263A74B87EC0010AB1404251613FDBD434C0A93BA4BE5578B04096BF30C0437DBBBFD2F7E9BF4697063FD70916411C5AD03FAE6C1D40A3AEED408F665A4066FA64C0EA2C903D9DCA84400EBB493F189009BF0D268B3F18B117BE9288064098704BC0D3CB0DC0ABC136C0D46F56BFD7539940B1811A408D6596BF1D538C3F8FBF42C02C8E1D400DAEA540C0B101C0FD13E73FF3B29FC07092E73FD062E43F4C21DCBE93D529C0405E8BC0F785274034533A3F1B48DABE299A5E3EDE19374026C67A3E58461440AD1C0CC026F2CA3F70FDEC3F169B55403B22B63F3DCC8CC0BDEF733FB88537BFCE3636406BABBFBFDFDAC3C04274EBBD53C1824036758DBF181E233F01DA9BBF2F5195BFD5C23BC064F91240228E0341DE840D4125576B3F552674C03F509140256037C0DBA08840E3499B3FEEC778C0374EC4BF4125B6BF44C023BE4D16D2C0FA0A103F4B40A4BFB4E093C0817412C0CE3E733FA49E80BF16A5B7BFBB65D03E016CD640473F5F3FD317AF406E4B22C02C46F4BFEFC09840B176A2C0D43D09401C9912C0FBFD86BE5EDE84C082F00E409CB852C058740940387A954096DA8840A1F33FC0782F41BFF8043E41E574663FEAC7A33E05C72E3D1F60AABFED53883F13884040365BCA403DD6CDBF9ED02E400A482540057464BFF494F0BF924F0FC0F59803BF9F49B9BFFC32D6C0371D2CC0E8FE80C07C8DC63F3B70DA3FD6F6353F0878083F4313463EDF6DA640E93ACFBEBB3D79C0C9052AC0569AFBBFFBE98040E2EDEF3F0DE703C0DABD18BF0EF206C02CF99F3FBF09124022D168C0A7FDEF3FED980B40184333C0C21AA0C0BB1C8FC07796223D555E7440DAC6B4BF44FAD7BF1ECDA33ECD8EB7BF229F30BF57BC35C07DA262C0BF30063FBA66C7C050DA1DBF952E3AC033DA733F513BDC3EA7585BC0E7FB14403B1B08407A34463D0DBF0A3FB60C5D3F985BF23F1E4077BE778F6E3E4F51494050B1D0C02EBB1B41496967BD7FB4183F0CEE97C02F07014029423240EB9692BEEC4FED3F6C2310C0732ABD3F03FB22C023B42C404B79BEBF8339B7BE7B5723C0C20025C07F8B8B404276F53CA52A3CC0E26CC4BF3181204053E1F93FF50B26BF1CE00B3FFF14F33F5F531CBFC26BECBFA7CE54BEBCC1E6C03554DB3F8504A3C08D9EB7C0FB1E81C0FA6609C01450DD3F80661EC0B65C89407F21D3BF2B2C94C0E30A0240FFB8D9BFC10067C0C7EA13C0CE7928403FB607402C078B3F6EA9FE3F777478C02B4394C0CD279D407DC11EC09F56B14077CA9A40213037C0194196C0A45EEC3EFD6D7EBF39A2BE407A7667BEF2524C40D098C7BE921AC5BF3E481B3FF9AE02BF1886CABE89CCAC3FB0B589BE493F0AC0757A5140B96ACD3F88F549C0F68FDF40F1689140C55639400FA0353F3E31304050109ABF0AF1E63F121F33C0ED54F640092AF3BFCCB9E7BECF76B0C0E5398540D1ECCEC0ADBD76BFBE6CA93F03545F3FFD46A1C0DF98A0BFCE514E403CF572C0738795404D814A3F453A2B3FE51F1A4034562940BD0FB2C018A5BEC09560AA3F7326254033103AC010E9193FC2E80F40A7AFCCBE1A1E20BF43948BC00C3A133FB4FD98C05E6E52C04931D5C0D12A0940643CA9BF9F626940E572D2C0A0263DC0528090C0C885A13F54AB91C021815DC0475A04BFECF9283E65AA7CC0F361FF3E60F519C02355FE3DF8C64AC08A3714BF1C2D74C01F2B8E4084616D408881D2BF7001274114D1B3BE566D91BE16A1C7BFAA81173F5495A9BF259D24407AEFB93F2A3F44C0D05D50C0D4EFED3F42643340BA380040C54A41BFCAE2C13F2B118A409AC684C01ABC993E785BA83FCC46094086DD6440ABAC9ABF872B5D3FF16E97BFF1CB8AC098833CBED8955B3FB04E2A40B6838AC0085B6240D49AAB3FFD9140C0DFC7D8BFAAFE55C0077B0AC0A69A3FC00F6A6CC018A4193F47FDB93FA4B03F40C8C74CC094DF1C40FD2DD8BF97CDA440FCEF9E40A366FB3F1D3FDCBF3A1834C0EFC81DBF6FE70340A4BD60C047EA8D40299A86BF8DE618C087BA44C00347AFC02D551D409A1332BE20698D40370BACBF63DF3B3E77FCAF40E4083B408922C13FBAFA74BF71F0823F834AD7BF42DBC1BFA2E2D8BEF3B1DDBFA6CA2AC01FFAB63ED703B4BFE0B721400CF0CFC053852B3F5893F1BFDEED9240528ED9BECFA37BBF93AEFF3FCC8432C0830E94C0508885BF84CF9AC05C3E92C0"> : tensor<20x20xf32>
    %1 = stablehlo.constant dense<[[1.8600738, 4.82661438, 4.69002342, 3.33467817, 7.985380e-01, -1.21585286, -2.25772905, -0.698300242, 0.21446687, -1.24689233, 0.454263598, -2.05684304, -2.04581523, 5.72695827, 1.50605834, 3.43758726, -4.12240314, -5.6210475, -2.39660525, -1.316880e-01]]> : tensor<1x20xf32>
    return %0, %1 : tensor<20x20xf32>, tensor<1x20xf32>
  }
  func.func private @expected() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0xF39A7F3E7B6C3B3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FF3AC12350000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FB424D83A5FD04D3F0000C07FC2D27F3FE978E23B0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F91D1CB3CB443493F0000C07F0000C07F0000C07F0000C07F5157083D0000C07FF1CB753F0000C07FC40B763E0000C07F0000C07F0000C07F0000C07F0000C07FF967033F0000C07F0000C07FA1526F3F41BB773F8AD74D3F0000C07F0000C07F0000C07F0000C07FCDBA3A3F0000C07FDCE47D3F0000C07F70D1733D0000C07F0000C07F0000C07F87606F380000C07F25C5113F0000C07F0000C07F0000C07F6750B63E4008A13C0000C07F0000C07F0000C07F0000C07F0000C07FAD95293FAE487C3F0000C07F0000C07F0000C07F0000C07F0000C07F986DE03E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FCE687E3F0000C07F0000C07F0000C07F0000C07F0000C07FC903123F0000C07FD326523F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FDF0B83390000C07F0000C07F0000C07FC3E5C633F758793F0000C07F0000C07F0000C07F0000C07FDE3D913E96679B3E0000C07FFEA9343F275EB33D0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F17537D3F42360A3F52CB7B3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F11BBD73E93A7633E0000C07F0000C07F0000C07FCCB2EA3D0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FBBA7783F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F840B353F0000C07F0000C07F8941793F40B17E3FA86D7D3F0000C07F0000C07F0000C07F0000C07F6241823E0000C07F3060E13C0000C07F8D2C3F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F160F723F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FB884673FB9755A3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FEF85733F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F58D0A73C0000C07F0000C07F0000C07F0000C07F993E163FE4EDBF3C0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FF7C2553F0000C07F0000C07FBFE83D3F0000C07F0000C07F0000C07F0000C07F0000C07F2E68CD3B0000C07F0000C07F3233B43E33320C3D3E8C303F0000C07F0000C07F0000C07F0000C07F0000C07FEA19103E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F08E4C73D0000C07F0000C07F0000C07F0000C07F0000C07F59D5ED3C70D77A3F0000C07F0000C07F0000C07F0000C07F0000C07F3CE57B3F7296653F0000C07F24953E3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F9F18793F0000C07F239B063F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F1D827F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F19B8EF350000C07F0000C07F0000C07F0000C07F0000C07F0BC9C73E81DF7A3F0000C07F0000C07F0BF1673E0000C07F0000C07F0000C07F984C823D0000C07F0000C07F0000C07F0000C07F5E14793FA628DB3D0000C07F0000C07F0000C07F0000C07F0000C07F51B5603F2004663F0000C07F0018073F7632CD3E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F15D06C3F0000C07FC87B473F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F14DA553F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FA8E9063D0000C07F0000C07F0000C07F2A317A3FB389433E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F4AD4E33E0000C07F0000C07F0000C07F0000C07F0000C07F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @igamma(%arg0: tensor<20x20xf32>, %arg1: tensor<1x20xf32>) -> tensor<20x20xf32> {
    %0 = call @xla_fallback_igamma(%arg0, %arg1) : (tensor<20x20xf32>, tensor<1x20xf32>) -> tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @igammac_body.202(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>> {
    %0 = stablehlo.get_tuple_element %arg0[7] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %1 = stablehlo.get_tuple_element %arg0[4] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %2 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %3 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
    %4 = stablehlo.add %1, %3 : tensor<20x20xf32>
    %5 = stablehlo.multiply %0, %4 : tensor<20x20xf32>
    %6 = stablehlo.get_tuple_element %arg0[9] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %7 = stablehlo.get_tuple_element %arg0[3] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %9 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %10 = stablehlo.add %7, %9 : tensor<20x20xf32>
    %11 = stablehlo.get_tuple_element %arg0[5] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<f32>
    %12 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %13 = stablehlo.add %11, %12 : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %15 = stablehlo.multiply %10, %14 : tensor<20x20xf32>
    %16 = stablehlo.multiply %6, %15 : tensor<20x20xf32>
    %17 = stablehlo.subtract %5, %16 : tensor<20x20xf32>
    %18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %19 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %20 = stablehlo.compare  NE, %17, %19 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %21 = stablehlo.get_tuple_element %arg0[12] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %22 = stablehlo.multiply %21, %4 : tensor<20x20xf32>
    %23 = stablehlo.get_tuple_element %arg0[6] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %24 = stablehlo.subtract %22, %23 : tensor<20x20xf32>
    %25 = stablehlo.get_tuple_element %arg0[10] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %26 = stablehlo.multiply %25, %15 : tensor<20x20xf32>
    %27 = stablehlo.subtract %24, %26 : tensor<20x20xf32>
    %28 = stablehlo.get_tuple_element %arg0[8] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %29 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %30 = stablehlo.multiply %28, %29 : tensor<20x20xf32>
    %31 = stablehlo.add %27, %30 : tensor<20x20xf32>
    %32 = stablehlo.multiply %23, %4 : tensor<20x20xf32>
    %33 = stablehlo.multiply %28, %15 : tensor<20x20xf32>
    %34 = stablehlo.subtract %32, %33 : tensor<20x20xf32>
    %35 = stablehlo.divide %34, %17 : tensor<20x20xf32>
    %36 = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %37 = stablehlo.select %20, %35, %36 : tensor<20x20xi1>, tensor<20x20xf32>
    %38 = stablehlo.get_tuple_element %arg0[13] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %39 = stablehlo.multiply %38, %4 : tensor<20x20xf32>
    %40 = stablehlo.subtract %39, %0 : tensor<20x20xf32>
    %41 = stablehlo.get_tuple_element %arg0[11] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %42 = stablehlo.multiply %41, %15 : tensor<20x20xf32>
    %43 = stablehlo.subtract %40, %42 : tensor<20x20xf32>
    %44 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %45 = stablehlo.multiply %6, %44 : tensor<20x20xf32>
    %46 = stablehlo.add %43, %45 : tensor<20x20xf32>
    %47 = stablehlo.multiply %37, %46 : tensor<20x20xf32>
    %48 = stablehlo.subtract %31, %47 : tensor<20x20xf32>
    %49 = stablehlo.divide %48, %17 : tensor<20x20xf32>
    %50 = stablehlo.get_tuple_element %arg0[14] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %51 = stablehlo.select %20, %49, %50 : tensor<20x20xi1>, tensor<20x20xf32>
    %52 = stablehlo.subtract %51, %50 : tensor<20x20xf32>
    %53 = stablehlo.abs %52 : tensor<20x20xf32>
    %54 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %55 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %56 = stablehlo.select %20, %53, %55 : tensor<20x20xi1>, tensor<20x20xf32>
    %57 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xi1>
    %58 = stablehlo.subtract %36, %35 : tensor<20x20xf32>
    %59 = stablehlo.divide %58, %35 : tensor<20x20xf32>
    %60 = stablehlo.abs %59 : tensor<20x20xf32>
    %61 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %62 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %63 = stablehlo.select %20, %60, %62 : tensor<20x20xi1>, tensor<20x20xf32>
    %64 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %65 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %66 = stablehlo.compare  GT, %63, %65 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %67 = stablehlo.and %57, %66 : tensor<20x20xi1>
    %68 = stablehlo.select %57, %37, %36 : tensor<20x20xi1>, tensor<20x20xf32>
    %69 = stablehlo.get_tuple_element %arg0[2] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %70 = stablehlo.select %57, %63, %69 : tensor<20x20xi1>, tensor<20x20xf32>
    %71 = stablehlo.select %57, %10, %7 : tensor<20x20xi1>, tensor<20x20xf32>
    %72 = stablehlo.select %57, %4, %1 : tensor<20x20xi1>, tensor<20x20xf32>
    %73 = stablehlo.abs %34 : tensor<20x20xf32>
    %74 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %75 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %76 = stablehlo.constant dense<0x4B000000> : tensor<f32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %78 = stablehlo.compare  GT, %73, %77 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %79 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %80 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %81 = stablehlo.multiply %34, %80 : tensor<20x20xf32>
    %82 = stablehlo.select %78, %81, %34 : tensor<20x20xi1>, tensor<20x20xf32>
    %83 = stablehlo.select %57, %82, %23 : tensor<20x20xi1>, tensor<20x20xf32>
    %84 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %85 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %86 = stablehlo.multiply %17, %85 : tensor<20x20xf32>
    %87 = stablehlo.select %78, %86, %17 : tensor<20x20xi1>, tensor<20x20xf32>
    %88 = stablehlo.select %57, %87, %0 : tensor<20x20xi1>, tensor<20x20xf32>
    %89 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %90 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %91 = stablehlo.multiply %23, %90 : tensor<20x20xf32>
    %92 = stablehlo.select %78, %91, %23 : tensor<20x20xi1>, tensor<20x20xf32>
    %93 = stablehlo.select %57, %92, %28 : tensor<20x20xi1>, tensor<20x20xf32>
    %94 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %95 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %96 = stablehlo.multiply %0, %95 : tensor<20x20xf32>
    %97 = stablehlo.select %78, %96, %0 : tensor<20x20xi1>, tensor<20x20xf32>
    %98 = stablehlo.select %57, %97, %6 : tensor<20x20xi1>, tensor<20x20xf32>
    %99 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %100 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %101 = stablehlo.multiply %21, %100 : tensor<20x20xf32>
    %102 = stablehlo.select %78, %101, %21 : tensor<20x20xi1>, tensor<20x20xf32>
    %103 = stablehlo.select %57, %102, %25 : tensor<20x20xi1>, tensor<20x20xf32>
    %104 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %105 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %106 = stablehlo.multiply %38, %105 : tensor<20x20xf32>
    %107 = stablehlo.select %78, %106, %38 : tensor<20x20xi1>, tensor<20x20xf32>
    %108 = stablehlo.select %57, %107, %41 : tensor<20x20xi1>, tensor<20x20xf32>
    %109 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %110 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %111 = stablehlo.multiply %31, %110 : tensor<20x20xf32>
    %112 = stablehlo.select %78, %111, %31 : tensor<20x20xi1>, tensor<20x20xf32>
    %113 = stablehlo.select %57, %112, %21 : tensor<20x20xi1>, tensor<20x20xf32>
    %114 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %115 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %116 = stablehlo.multiply %46, %115 : tensor<20x20xf32>
    %117 = stablehlo.select %78, %116, %46 : tensor<20x20xi1>, tensor<20x20xf32>
    %118 = stablehlo.select %57, %117, %38 : tensor<20x20xi1>, tensor<20x20xf32>
    %119 = stablehlo.select %57, %51, %50 : tensor<20x20xi1>, tensor<20x20xf32>
    %120 = stablehlo.tuple %67, %68, %70, %71, %72, %13, %83, %88, %93, %98, %103, %108, %113, %118, %119 {xla_shape = "(pred[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, /*index=5*/f32[], f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, /*index=10*/f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0})"} : tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>
    return %120 : tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>
  }
  func.func private @or.325(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @igammac_condition.329(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<i1> {
    %0 = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %1 = stablehlo.get_tuple_element %arg0[2] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %2 = stablehlo.get_tuple_element %arg0[3] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %3 = stablehlo.get_tuple_element %arg0[4] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %4 = stablehlo.get_tuple_element %arg0[6] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %5 = stablehlo.get_tuple_element %arg0[7] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %6 = stablehlo.get_tuple_element %arg0[8] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %7 = stablehlo.get_tuple_element %arg0[9] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %8 = stablehlo.get_tuple_element %arg0[10] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %9 = stablehlo.get_tuple_element %arg0[11] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %10 = stablehlo.get_tuple_element %arg0[12] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %11 = stablehlo.get_tuple_element %arg0[13] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %12 = stablehlo.get_tuple_element %arg0[14] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %13 = stablehlo.get_tuple_element %arg0[5] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<f32>
    %14 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
    %15 = stablehlo.compare  LT, %13, %14 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %16 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xi1>
    %17 = stablehlo.constant dense<false> : tensor<i1>
    %18 = stablehlo.reduce(%16 init: %17) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
     reducer(%arg1: tensor<i1>, %arg2: tensor<i1>)  {
      %20 = stablehlo.or %arg1, %arg2 : tensor<i1>
      stablehlo.return %20 : tensor<i1>
    }
    %19 = stablehlo.and %15, %18 : tensor<i1>
    return %19 : tensor<i1>
  }
  func.func private @igamma_body.381(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>> {
    %0 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xi1>
    %1 = stablehlo.get_tuple_element %arg0[2] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %2 = stablehlo.get_tuple_element %arg0[4] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %3 = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %6 = stablehlo.add %3, %5 : tensor<20x20xf32>
    %7 = stablehlo.divide %2, %6 : tensor<20x20xf32>
    %8 = stablehlo.multiply %1, %7 : tensor<20x20xf32>
    %9 = stablehlo.get_tuple_element %arg0[3] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %10 = stablehlo.add %9, %8 : tensor<20x20xf32>
    %11 = stablehlo.divide %8, %10 : tensor<20x20xf32>
    %12 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %13 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %14 = stablehlo.compare  GT, %11, %13 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %15 = stablehlo.and %0, %14 : tensor<20x20xi1>
    %16 = stablehlo.select %0, %6, %3 : tensor<20x20xi1>, tensor<20x20xf32>
    %17 = stablehlo.select %0, %8, %1 : tensor<20x20xi1>, tensor<20x20xf32>
    %18 = stablehlo.select %0, %10, %9 : tensor<20x20xi1>, tensor<20x20xf32>
    %19 = stablehlo.get_tuple_element %arg0[5] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %20 = stablehlo.divide %2, %6 : tensor<20x20xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<20x20xf32>
    %22 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %23 = stablehlo.constant dense<-1.000000e+00> : tensor<20x20xf32>
    %24 = stablehlo.multiply %23, %1 : tensor<20x20xf32>
    %25 = stablehlo.multiply %24, %2 : tensor<20x20xf32>
    %26 = stablehlo.multiply %6, %6 : tensor<20x20xf32>
    %27 = stablehlo.divide %25, %26 : tensor<20x20xf32>
    %28 = stablehlo.add %21, %27 : tensor<20x20xf32>
    %29 = stablehlo.select %0, %28, %19 : tensor<20x20xi1>, tensor<20x20xf32>
    %30 = stablehlo.get_tuple_element %arg0[6] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %31 = stablehlo.add %30, %28 : tensor<20x20xf32>
    %32 = stablehlo.select %0, %31, %30 : tensor<20x20xi1>, tensor<20x20xf32>
    %33 = stablehlo.tuple %15, %16, %17, %18, %2, %29, %32 {xla_shape = "(pred[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, /*index=5*/f32[20,20]{1,0}, f32[20,20]{1,0})"} : tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>
    return %33 : tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>
  }
  func.func private @or.418(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @igamma_condition.422(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<i1> {
    %0 = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %1 = stablehlo.get_tuple_element %arg0[2] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %2 = stablehlo.get_tuple_element %arg0[3] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %3 = stablehlo.get_tuple_element %arg0[4] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %4 = stablehlo.get_tuple_element %arg0[5] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %5 = stablehlo.get_tuple_element %arg0[6] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %6 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xi1>
    %7 = stablehlo.constant dense<false> : tensor<i1>
    %8 = stablehlo.reduce(%6 init: %7) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
     reducer(%arg1: tensor<i1>, %arg2: tensor<i1>)  {
      %9 = stablehlo.or %arg1, %arg2 : tensor<i1>
      stablehlo.return %9 : tensor<i1>
    }
    return %8 : tensor<i1>
  }
  func.func private @xla_fallback_igamma(%arg0: tensor<20x20xf32>, %arg1: tensor<1x20xf32>) -> tensor<20x20xf32> {
    %0 = stablehlo.reshape %arg1 : (tensor<1x20xf32>) -> tensor<20xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<20xf32>) -> tensor<20x20xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %4 = stablehlo.compare  EQ, %1, %3 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %7 = stablehlo.compare  LT, %1, %6 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %10 = stablehlo.compare  LE, %arg0, %9 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %11 = stablehlo.or %7, %10 : tensor<20x20xi1>
    %12 = stablehlo.or %4, %11 : tensor<20x20xi1>
    %13 = stablehlo.log %1 : tensor<20x20xf32>
    %14 = stablehlo.multiply %arg0, %13 : tensor<20x20xf32>
    %15 = stablehlo.subtract %14, %1 : tensor<20x20xf32>
    %16 = stablehlo.abs %arg0 : tensor<20x20xf32>
    %17 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %18 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %19 = stablehlo.compare  EQ, %16, %18 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %20 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %21 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %22 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %23 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %24 = stablehlo.compare  LT, %arg0, %23 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %25 = stablehlo.constant dense<3.14159274> : tensor<f32>
    %26 = stablehlo.constant dense<3.14159274> : tensor<20x20xf32>
    %27 = stablehlo.abs %arg0 : tensor<20x20xf32>
    %28 = stablehlo.floor %27 : tensor<20x20xf32>
    %29 = stablehlo.subtract %27, %28 : tensor<20x20xf32>
    %30 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %31 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %32 = stablehlo.compare  GT, %29, %31 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %33 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %34 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %35 = stablehlo.subtract %34, %29 : tensor<20x20xf32>
    %36 = stablehlo.select %32, %35, %29 : tensor<20x20xi1>, tensor<20x20xf32>
    %37 = stablehlo.multiply %26, %36 : tensor<20x20xf32>
    %38 = stablehlo.sine %37 : tensor<20x20xf32>
    %39 = stablehlo.log %38 : tensor<20x20xf32>
    %40 = stablehlo.is_finite %39 : (tensor<20x20xf32>) -> tensor<20x20xi1>
    %41 = stablehlo.constant dense<1.14472985> : tensor<f32>
    %42 = stablehlo.constant dense<1.14472985> : tensor<20x20xf32>
    %43 = stablehlo.subtract %42, %39 : tensor<20x20xf32>
    %44 = stablehlo.constant dense<0.918938517> : tensor<f32>
    %45 = stablehlo.constant dense<0.918938517> : tensor<20x20xf32>
    %46 = stablehlo.negate %arg0 : tensor<20x20xf32>
    %47 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %48 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %49 = stablehlo.subtract %arg0, %48 : tensor<20x20xf32>
    %50 = stablehlo.select %24, %46, %49 : tensor<20x20xi1>, tensor<20x20xf32>
    %51 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %52 = stablehlo.add %50, %51 : tensor<20x20xf32>
    %53 = stablehlo.constant dense<7.500000e+00> : tensor<f32>
    %54 = stablehlo.constant dense<7.500000e+00> : tensor<20x20xf32>
    %55 = stablehlo.add %54, %50 : tensor<20x20xf32>
    %56 = stablehlo.constant dense<2.01490307> : tensor<f32>
    %57 = stablehlo.constant dense<2.01490307> : tensor<20x20xf32>
    %58 = stablehlo.constant dense<7.500000e+00> : tensor<20x20xf32>
    %59 = stablehlo.divide %50, %58 : tensor<20x20xf32>
    %60 = stablehlo.log_plus_one %59 : tensor<20x20xf32>
    %61 = stablehlo.add %57, %60 : tensor<20x20xf32>
    %62 = stablehlo.divide %55, %61 : tensor<20x20xf32>
    %63 = stablehlo.subtract %52, %62 : tensor<20x20xf32>
    %64 = stablehlo.multiply %63, %61 : tensor<20x20xf32>
    %65 = stablehlo.add %45, %64 : tensor<20x20xf32>
    %66 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %67 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %68 = stablehlo.constant dense<676.520386> : tensor<f32>
    %69 = stablehlo.constant dense<676.520386> : tensor<20x20xf32>
    %70 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %71 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %72 = stablehlo.add %50, %71 : tensor<20x20xf32>
    %73 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %74 = stablehlo.add %72, %73 : tensor<20x20xf32>
    %75 = stablehlo.divide %69, %74 : tensor<20x20xf32>
    %76 = stablehlo.add %67, %75 : tensor<20x20xf32>
    %77 = stablehlo.constant dense<-1259.13916> : tensor<f32>
    %78 = stablehlo.constant dense<-1259.13916> : tensor<20x20xf32>
    %79 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %80 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %81 = stablehlo.add %50, %80 : tensor<20x20xf32>
    %82 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %83 = stablehlo.add %81, %82 : tensor<20x20xf32>
    %84 = stablehlo.divide %78, %83 : tensor<20x20xf32>
    %85 = stablehlo.add %76, %84 : tensor<20x20xf32>
    %86 = stablehlo.constant dense<771.323425> : tensor<f32>
    %87 = stablehlo.constant dense<771.323425> : tensor<20x20xf32>
    %88 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %89 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
    %90 = stablehlo.add %50, %89 : tensor<20x20xf32>
    %91 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %92 = stablehlo.add %90, %91 : tensor<20x20xf32>
    %93 = stablehlo.divide %87, %92 : tensor<20x20xf32>
    %94 = stablehlo.add %85, %93 : tensor<20x20xf32>
    %95 = stablehlo.constant dense<-176.615036> : tensor<f32>
    %96 = stablehlo.constant dense<-176.615036> : tensor<20x20xf32>
    %97 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %98 = stablehlo.constant dense<3.000000e+00> : tensor<20x20xf32>
    %99 = stablehlo.add %50, %98 : tensor<20x20xf32>
    %100 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %101 = stablehlo.add %99, %100 : tensor<20x20xf32>
    %102 = stablehlo.divide %96, %101 : tensor<20x20xf32>
    %103 = stablehlo.add %94, %102 : tensor<20x20xf32>
    %104 = stablehlo.constant dense<12.5073433> : tensor<f32>
    %105 = stablehlo.constant dense<12.5073433> : tensor<20x20xf32>
    %106 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %107 = stablehlo.constant dense<4.000000e+00> : tensor<20x20xf32>
    %108 = stablehlo.add %50, %107 : tensor<20x20xf32>
    %109 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %110 = stablehlo.add %108, %109 : tensor<20x20xf32>
    %111 = stablehlo.divide %105, %110 : tensor<20x20xf32>
    %112 = stablehlo.add %103, %111 : tensor<20x20xf32>
    %113 = stablehlo.constant dense<-0.138571098> : tensor<f32>
    %114 = stablehlo.constant dense<-0.138571098> : tensor<20x20xf32>
    %115 = stablehlo.constant dense<5.000000e+00> : tensor<f32>
    %116 = stablehlo.constant dense<5.000000e+00> : tensor<20x20xf32>
    %117 = stablehlo.add %50, %116 : tensor<20x20xf32>
    %118 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %119 = stablehlo.add %117, %118 : tensor<20x20xf32>
    %120 = stablehlo.divide %114, %119 : tensor<20x20xf32>
    %121 = stablehlo.add %112, %120 : tensor<20x20xf32>
    %122 = stablehlo.constant dense<9.98436917E-6> : tensor<f32>
    %123 = stablehlo.constant dense<9.98436917E-6> : tensor<20x20xf32>
    %124 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %125 = stablehlo.constant dense<6.000000e+00> : tensor<20x20xf32>
    %126 = stablehlo.add %50, %125 : tensor<20x20xf32>
    %127 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %128 = stablehlo.add %126, %127 : tensor<20x20xf32>
    %129 = stablehlo.divide %123, %128 : tensor<20x20xf32>
    %130 = stablehlo.add %121, %129 : tensor<20x20xf32>
    %131 = stablehlo.constant dense<1.50563267E-7> : tensor<f32>
    %132 = stablehlo.constant dense<1.50563267E-7> : tensor<20x20xf32>
    %133 = stablehlo.constant dense<7.000000e+00> : tensor<f32>
    %134 = stablehlo.constant dense<7.000000e+00> : tensor<20x20xf32>
    %135 = stablehlo.add %50, %134 : tensor<20x20xf32>
    %136 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %137 = stablehlo.add %135, %136 : tensor<20x20xf32>
    %138 = stablehlo.divide %132, %137 : tensor<20x20xf32>
    %139 = stablehlo.add %130, %138 : tensor<20x20xf32>
    %140 = stablehlo.log %139 : tensor<20x20xf32>
    %141 = stablehlo.add %65, %140 : tensor<20x20xf32>
    %142 = stablehlo.subtract %43, %141 : tensor<20x20xf32>
    %143 = stablehlo.negate %39 : tensor<20x20xf32>
    %144 = stablehlo.select %40, %142, %143 : tensor<20x20xi1>, tensor<20x20xf32>
    %145 = stablehlo.select %24, %144, %141 : tensor<20x20xi1>, tensor<20x20xf32>
    %146 = stablehlo.select %19, %21, %145 : tensor<20x20xi1>, tensor<20x20xf32>
    %147 = stablehlo.subtract %15, %146 : tensor<20x20xf32>
    %148 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
    %149 = stablehlo.constant dense<88.7228394> : tensor<f32>
    %150 = stablehlo.negate %149 : tensor<f32>
    %151 = stablehlo.broadcast_in_dim %150, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %152 = stablehlo.compare  LT, %147, %151 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %153 = stablehlo.or %12, %152 : tensor<20x20xi1>
    %154 = stablehlo.compare  NE, %arg0, %arg0 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %155 = stablehlo.compare  NE, %1, %1 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %156 = stablehlo.or %154, %155 : tensor<20x20xi1>
    %157 = stablehlo.or %153, %156 : tensor<20x20xi1>
    %158 = stablehlo.not %157 : tensor<20x20xi1>
    %159 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %160 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %161 = stablehlo.compare  GT, %1, %160 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %162 = stablehlo.compare  GT, %1, %arg0 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %163 = stablehlo.and %161, %162 : tensor<20x20xi1>
    %164 = stablehlo.and %158, %163 : tensor<20x20xi1>
    %165 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %166 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %167 = stablehlo.add %1, %166 : tensor<20x20xf32>
    %168 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %169 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %170 = stablehlo.subtract %169, %arg0 : tensor<20x20xf32>
    %171 = stablehlo.add %1, %170 : tensor<20x20xf32>
    %172 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %173 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %174 = stablehlo.add %171, %173 : tensor<20x20xf32>
    %175 = stablehlo.multiply %174, %1 : tensor<20x20xf32>
    %176 = stablehlo.divide %167, %175 : tensor<20x20xf32>
    %177 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %178 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %179 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %180 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %181 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %182 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %183 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %184 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %185 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %186 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %187 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %188 = stablehlo.negate %1 : tensor<20x20xf32>
    %189 = stablehlo.multiply %176, %188 : tensor<20x20xf32>
    %190 = stablehlo.subtract %187, %189 : tensor<20x20xf32>
    %191 = stablehlo.divide %190, %175 : tensor<20x20xf32>
    %192:15 = stablehlo.while(%iterArg = %164, %iterArg_0 = %176, %iterArg_1 = %178, %iterArg_2 = %170, %iterArg_3 = %174, %iterArg_4 = %179, %iterArg_5 = %167, %iterArg_6 = %175, %iterArg_7 = %181, %iterArg_8 = %1, %iterArg_9 = %183, %iterArg_10 = %185, %iterArg_11 = %187, %iterArg_12 = %188, %iterArg_13 = %191) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %225 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
      %226 = stablehlo.compare  LT, %iterArg_4, %225 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %227 = stablehlo.constant dense<false> : tensor<i1>
      %228 = stablehlo.reduce(%iterArg init: %227) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %230 = stablehlo.or %arg2, %arg3 : tensor<i1>
        stablehlo.return %230 : tensor<i1>
      }
      %229 = stablehlo.and %226, %228 : tensor<i1>
      stablehlo.return %229 : tensor<i1>
    } do {
      %225 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %226 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
      %227 = stablehlo.add %iterArg_3, %226 : tensor<20x20xf32>
      %228 = stablehlo.multiply %iterArg_6, %227 : tensor<20x20xf32>
      %229 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %230 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %231 = stablehlo.add %iterArg_2, %230 : tensor<20x20xf32>
      %232 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %233 = stablehlo.add %iterArg_4, %232 : tensor<f32>
      %234 = stablehlo.broadcast_in_dim %233, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %235 = stablehlo.multiply %231, %234 : tensor<20x20xf32>
      %236 = stablehlo.multiply %iterArg_8, %235 : tensor<20x20xf32>
      %237 = stablehlo.subtract %228, %236 : tensor<20x20xf32>
      %238 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %239 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
      %240 = stablehlo.compare  NE, %237, %239 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %241 = stablehlo.multiply %iterArg_11, %227 : tensor<20x20xf32>
      %242 = stablehlo.subtract %241, %iterArg_5 : tensor<20x20xf32>
      %243 = stablehlo.multiply %iterArg_9, %235 : tensor<20x20xf32>
      %244 = stablehlo.subtract %242, %243 : tensor<20x20xf32>
      %245 = stablehlo.broadcast_in_dim %233, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %246 = stablehlo.multiply %iterArg_7, %245 : tensor<20x20xf32>
      %247 = stablehlo.add %244, %246 : tensor<20x20xf32>
      %248 = stablehlo.multiply %iterArg_5, %227 : tensor<20x20xf32>
      %249 = stablehlo.multiply %iterArg_7, %235 : tensor<20x20xf32>
      %250 = stablehlo.subtract %248, %249 : tensor<20x20xf32>
      %251 = stablehlo.divide %250, %237 : tensor<20x20xf32>
      %252 = stablehlo.select %240, %251, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %253 = stablehlo.multiply %iterArg_12, %227 : tensor<20x20xf32>
      %254 = stablehlo.subtract %253, %iterArg_6 : tensor<20x20xf32>
      %255 = stablehlo.multiply %iterArg_10, %235 : tensor<20x20xf32>
      %256 = stablehlo.subtract %254, %255 : tensor<20x20xf32>
      %257 = stablehlo.broadcast_in_dim %233, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %258 = stablehlo.multiply %iterArg_8, %257 : tensor<20x20xf32>
      %259 = stablehlo.add %256, %258 : tensor<20x20xf32>
      %260 = stablehlo.multiply %252, %259 : tensor<20x20xf32>
      %261 = stablehlo.subtract %247, %260 : tensor<20x20xf32>
      %262 = stablehlo.divide %261, %237 : tensor<20x20xf32>
      %263 = stablehlo.select %240, %262, %iterArg_13 : tensor<20x20xi1>, tensor<20x20xf32>
      %264 = stablehlo.subtract %263, %iterArg_13 : tensor<20x20xf32>
      %265 = stablehlo.abs %264 : tensor<20x20xf32>
      %266 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %267 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %268 = stablehlo.select %240, %265, %267 : tensor<20x20xi1>, tensor<20x20xf32>
      %269 = stablehlo.subtract %iterArg_0, %251 : tensor<20x20xf32>
      %270 = stablehlo.divide %269, %251 : tensor<20x20xf32>
      %271 = stablehlo.abs %270 : tensor<20x20xf32>
      %272 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %273 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %274 = stablehlo.select %240, %271, %273 : tensor<20x20xi1>, tensor<20x20xf32>
      %275 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %276 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %277 = stablehlo.compare  GT, %274, %276 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %278 = stablehlo.and %iterArg, %277 : tensor<20x20xi1>
      %279 = stablehlo.select %iterArg, %252, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %280 = stablehlo.select %iterArg, %274, %iterArg_1 : tensor<20x20xi1>, tensor<20x20xf32>
      %281 = stablehlo.select %iterArg, %231, %iterArg_2 : tensor<20x20xi1>, tensor<20x20xf32>
      %282 = stablehlo.select %iterArg, %227, %iterArg_3 : tensor<20x20xi1>, tensor<20x20xf32>
      %283 = stablehlo.abs %250 : tensor<20x20xf32>
      %284 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %285 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %286 = stablehlo.constant dense<0x4B000000> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %286, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %288 = stablehlo.compare  GT, %283, %287 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %289 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %290 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %291 = stablehlo.multiply %250, %290 : tensor<20x20xf32>
      %292 = stablehlo.select %288, %291, %250 : tensor<20x20xi1>, tensor<20x20xf32>
      %293 = stablehlo.select %iterArg, %292, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      %294 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %295 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %296 = stablehlo.multiply %237, %295 : tensor<20x20xf32>
      %297 = stablehlo.select %288, %296, %237 : tensor<20x20xi1>, tensor<20x20xf32>
      %298 = stablehlo.select %iterArg, %297, %iterArg_6 : tensor<20x20xi1>, tensor<20x20xf32>
      %299 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %300 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %301 = stablehlo.multiply %iterArg_5, %300 : tensor<20x20xf32>
      %302 = stablehlo.select %288, %301, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      %303 = stablehlo.select %iterArg, %302, %iterArg_7 : tensor<20x20xi1>, tensor<20x20xf32>
      %304 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %305 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %306 = stablehlo.multiply %iterArg_6, %305 : tensor<20x20xf32>
      %307 = stablehlo.select %288, %306, %iterArg_6 : tensor<20x20xi1>, tensor<20x20xf32>
      %308 = stablehlo.select %iterArg, %307, %iterArg_8 : tensor<20x20xi1>, tensor<20x20xf32>
      %309 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %310 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %311 = stablehlo.multiply %iterArg_11, %310 : tensor<20x20xf32>
      %312 = stablehlo.select %288, %311, %iterArg_11 : tensor<20x20xi1>, tensor<20x20xf32>
      %313 = stablehlo.select %iterArg, %312, %iterArg_9 : tensor<20x20xi1>, tensor<20x20xf32>
      %314 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %315 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %316 = stablehlo.multiply %iterArg_12, %315 : tensor<20x20xf32>
      %317 = stablehlo.select %288, %316, %iterArg_12 : tensor<20x20xi1>, tensor<20x20xf32>
      %318 = stablehlo.select %iterArg, %317, %iterArg_10 : tensor<20x20xi1>, tensor<20x20xf32>
      %319 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %320 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %321 = stablehlo.multiply %247, %320 : tensor<20x20xf32>
      %322 = stablehlo.select %288, %321, %247 : tensor<20x20xi1>, tensor<20x20xf32>
      %323 = stablehlo.select %iterArg, %322, %iterArg_11 : tensor<20x20xi1>, tensor<20x20xf32>
      %324 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %325 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %326 = stablehlo.multiply %259, %325 : tensor<20x20xf32>
      %327 = stablehlo.select %288, %326, %259 : tensor<20x20xi1>, tensor<20x20xf32>
      %328 = stablehlo.select %iterArg, %327, %iterArg_12 : tensor<20x20xi1>, tensor<20x20xf32>
      %329 = stablehlo.select %iterArg, %263, %iterArg_13 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %278, %279, %280, %281, %282, %233, %293, %298, %303, %308, %313, %318, %323, %328, %329 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %193 = stablehlo.not %163 : tensor<20x20xi1>
    %194 = stablehlo.and %158, %193 : tensor<20x20xi1>
    %195 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %196 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %197 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %198 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %199 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %200 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %201 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %202 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %203:7 = stablehlo.while(%iterArg = %194, %iterArg_0 = %arg0, %iterArg_1 = %196, %iterArg_2 = %198, %iterArg_3 = %1, %iterArg_4 = %200, %iterArg_5 = %202) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %225 = stablehlo.constant dense<false> : tensor<i1>
      %226 = stablehlo.reduce(%iterArg init: %225) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %227 = stablehlo.or %arg2, %arg3 : tensor<i1>
        stablehlo.return %227 : tensor<i1>
      }
      stablehlo.return %226 : tensor<i1>
    } do {
      %225 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %226 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %227 = stablehlo.add %iterArg_0, %226 : tensor<20x20xf32>
      %228 = stablehlo.divide %iterArg_3, %227 : tensor<20x20xf32>
      %229 = stablehlo.multiply %iterArg_1, %228 : tensor<20x20xf32>
      %230 = stablehlo.add %iterArg_2, %229 : tensor<20x20xf32>
      %231 = stablehlo.divide %229, %230 : tensor<20x20xf32>
      %232 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %233 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %234 = stablehlo.compare  GT, %231, %233 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %235 = stablehlo.and %iterArg, %234 : tensor<20x20xi1>
      %236 = stablehlo.select %iterArg, %227, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %237 = stablehlo.select %iterArg, %229, %iterArg_1 : tensor<20x20xi1>, tensor<20x20xf32>
      %238 = stablehlo.select %iterArg, %230, %iterArg_2 : tensor<20x20xi1>, tensor<20x20xf32>
      %239 = stablehlo.divide %iterArg_3, %227 : tensor<20x20xf32>
      %240 = stablehlo.multiply %iterArg_4, %239 : tensor<20x20xf32>
      %241 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
      %242 = stablehlo.constant dense<-1.000000e+00> : tensor<20x20xf32>
      %243 = stablehlo.multiply %242, %iterArg_1 : tensor<20x20xf32>
      %244 = stablehlo.multiply %243, %iterArg_3 : tensor<20x20xf32>
      %245 = stablehlo.multiply %227, %227 : tensor<20x20xf32>
      %246 = stablehlo.divide %244, %245 : tensor<20x20xf32>
      %247 = stablehlo.add %240, %246 : tensor<20x20xf32>
      %248 = stablehlo.select %iterArg, %247, %iterArg_4 : tensor<20x20xi1>, tensor<20x20xf32>
      %249 = stablehlo.add %iterArg_5, %247 : tensor<20x20xf32>
      %250 = stablehlo.select %iterArg, %249, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %235, %236, %237, %238, %iterArg_3, %248, %250 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %204 = stablehlo.or %11, %156 : tensor<20x20xi1>
    %205 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %206 = stablehlo.constant dense<0x7FC00000> : tensor<20x20xf32>
    %207 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %208 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %209 = stablehlo.compare  EQ, %1, %208 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %210 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %211 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %212 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %213 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %214 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %215 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %216 = stablehlo.exponential %147 : tensor<20x20xf32>
    %217 = stablehlo.multiply %192#1, %216 : tensor<20x20xf32>
    %218 = stablehlo.subtract %215, %217 : tensor<20x20xf32>
    %219 = stablehlo.multiply %203#3, %216 : tensor<20x20xf32>
    %220 = stablehlo.divide %219, %arg0 : tensor<20x20xf32>
    %221 = stablehlo.select %163, %218, %220 : tensor<20x20xi1>, tensor<20x20xf32>
    %222 = stablehlo.select %4, %213, %221 : tensor<20x20xi1>, tensor<20x20xf32>
    %223 = stablehlo.select %209, %211, %222 : tensor<20x20xi1>, tensor<20x20xf32>
    %224 = stablehlo.select %204, %206, %223 : tensor<20x20xi1>, tensor<20x20xf32>
    return %224 : tensor<20x20xf32>
  }
}
