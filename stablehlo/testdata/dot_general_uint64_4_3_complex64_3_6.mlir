// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui64>, tensor<3x6xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x6xcomplex<f32>>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui64>) -> tensor<4x3xcomplex<f32>>
    %3 = stablehlo.convert %0#1 : tensor<3x6xcomplex<f32>>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()
    return %4 : tensor<4x6xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<4x3xui64> {mhlo.layout_mode = "default"}, tensor<3x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 0, 0], [0, 2, 4], [1, 1, 1], [0, 4, 0]]> : tensor<4x3xui64>
    %cst = stablehlo.constant dense<[[(2.45522833,-4.48552513), (-0.100219555,7.55077075), (1.84159267,0.578810215), (-2.86075354,1.12263703), (5.32669067,-2.02719879), (0.657366395,-2.13350892)], [(0.543746829,-2.79866958), (0.785029351,0.807303249), (-1.21870899,-3.00168967), (3.06053472,0.790934085), (1.3685559,-1.02736819), (-2.42934918,0.662117362)], [(0.0635634735,-0.945136666), (0.851711571,-3.79603148), (2.30837083,2.51139235), (-0.537010252,2.204260e+00), (0.855342984,0.925947964), (2.66539192,1.88715553)]]> : tensor<3x6xcomplex<f32>>
    return %c, %cst : tensor<4x3xui64>, tensor<3x6xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<4x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(2.45522833,-4.48552513), (-0.100219555,7.55077075), (1.84159267,0.578810215), (-2.86075354,1.12263703), (5.32669067,-2.02719879), (0.657366395,-2.13350892)], [(1.34174752,-9.37788581), (4.97690487,-13.569519), (6.79606533,4.042190e+00), (3.97302842,10.3989086), (6.15848351,1.64905548), (5.80286932,8.87285709)], [(3.06253886,-8.22933197), (1.53652143,4.56204271), (2.93125439,0.0885128974), (-0.337229073,4.11783123), (7.55058956,-2.12861896), (0.893409132,0.415763974)], [(2.17498732,-11.1946783), (3.14011741,3.229213), (-4.87483597,-12.0067587), (12.2421389,3.16373634), (5.47422361,-4.10947275), (-9.71739673,2.64846945)]]> : tensor<4x6xcomplex<f32>>
    return %cst : tensor<4x6xcomplex<f32>>
  }
}
