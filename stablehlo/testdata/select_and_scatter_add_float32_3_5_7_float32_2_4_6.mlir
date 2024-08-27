// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<3x5x7xf32>, tensor<2x4x6xf32>)
    %1 = call @expected() : () -> tensor<2x4x6xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.pad %0#1, %cst, low = [1, 1, 1], high = [1, 1, 1], interior = [0, 0, 0] : (tensor<2x4x6xf32>, tensor<f32>) -> tensor<4x6x8xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %cst_0) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<4x6x8xf32>, tensor<3x5x7xf32>, tensor<f32>) -> tensor<4x6x8xf32>
    %4 = stablehlo.slice %3 [1:3, 1:5, 1:7] : (tensor<4x6x8xf32>) -> tensor<2x4x6xf32>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> ()
    return %4 : tensor<2x4x6xf32>
  }
  func.func private @inputs() -> (tensor<3x5x7xf32> {mhlo.layout_mode = "default"}, tensor<2x4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x14BA8140EE7C8C40B72358BF44F2E4BF768518BEFDA53C40AD568A3F3AF382C0DA30B23FA30A4DBE19FA58406E2AA74061EF30C0060415C0C09C093F8ED504C18517CB3FEA12E0400A5C86C08ECE883F0A351F40784FB1409D6C8EC04119DABFC2628840E683B3C0E4CC7FC08D6F32402114844040901D4097D415BF3C7C1BC0C67094BF4F06574002358340AD4DA83FCAD2E43F1FA72CC0D739C9BE5229A1BF658E34BFB0CAF4BD184EC13F3EF3F1BF0B17DAC092412E403192934065E454C0FEAC61C063E230402A4929BF6D831ABE2E819F40F99E71BFFB91E93FCD43B5BF22B39240FF729CBFAFA8174055165CC0DEC273406E969F4067F6C1BF9AD4633FA240603F6EC866BFD8AFF8BF3906633E2E7A08405F6D7B3F1B2933BCE0971840D74E42407C9A2C3FD193BC3F8B51D7BEB7905EBF5FE08740622732C051FE1B4013D5A840B1C09DBEED4DCDBEA58C14BF4A1E20C0D75E073FA9E378C098635240D9D3E63FBBEABCBFEB839EC052D8B7C0836C3940DDF29CC0BCD0F23F4C9BC63FE211A73FA55E2E3FE3082E4033559340ADCF07C08751AD40003F2BC074D45FC00FC319C0"> : tensor<3x5x7xf32>
    %cst_0 = stablehlo.constant dense<[[[9.620100e+00, -2.12728262, -1.56567204, 3.59518933, 1.462800e+00, -4.86497736], [1.74625087, -0.00808556098, -0.0342763178, 2.07804894, -0.964987456, -5.13031816], [0.0146070309, -1.24919343, -5.25079107, 1.74715388, 0.203711271, 0.950955867], [0.189528704, -1.79846704, -4.15770769, -3.86397576, 3.24413562, 1.84910381]], [[1.46117151, -4.5444808, -1.31956887, -0.0305220447, 2.35218406, -3.15633678], [-1.86811352, 1.90593982, 0.378701568, -1.22142124, -1.6682936, 5.19656277], [1.08109593, 2.81859517, -3.05214119, 2.40367198, 1.07703853, 2.27809072], [-2.89373779, 0.616681277, 0.84218657, 2.28567815, 4.25000191, 3.33190656]]]> : tensor<2x4x6xf32>
    return %cst, %cst_0 : tensor<3x5x7xf32>, tensor<2x4x6xf32>
  }
  func.func private @expected() -> (tensor<2x4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[8.46665287, 0.000000e+00, -0.844294965, 12.3588591, 0.183020592, -1.24760044], [-5.00077152, 1.38641942, 0.000000e+00, 2.80357361, 0.000000e+00, 0.000000e+00], [0.000000e+00, -1.70389569, 0.000000e+00, 4.2620554, 0.000000e+00, 3.5564158], [8.56954193, -0.585275114, 0.000000e+00, -2.42945766, -7.40666294, 6.88827896]], [[6.61947345, 0.000000e+00, 0.33836174, 0.674232244, 0.0393084884, -0.988922894], [0.000000e+00, -7.1615696, 5.276010e+00, 0.000000e+00, 0.000000e+00, -13.8547611], [-3.66263819, -5.03225613, 0.000000e+00, 7.58948802, 0.000000e+00, 0.000000e+00], [2.7192924, 5.48013639, -3.02354622, 3.47333574, 7.83382701, -2.25459337]]]> : tensor<2x4x6xf32>
    return %cst : tensor<2x4x6xf32>
  }
}
