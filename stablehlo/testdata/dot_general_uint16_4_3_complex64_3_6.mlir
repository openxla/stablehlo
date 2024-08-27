// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui16>, tensor<3x6xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x6xcomplex<f32>>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui16>) -> tensor<4x3xcomplex<f32>>
    %3 = stablehlo.convert %0#1 : tensor<3x6xcomplex<f32>>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()
    return %4 : tensor<4x6xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<4x3xui16> {mhlo.layout_mode = "default"}, tensor<3x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 3, 3], [0, 2, 7], [2, 3, 6], [0, 2, 2]]> : tensor<4x3xui16>
    %cst = stablehlo.constant dense<[[(-3.16872978,1.637980e+00), (2.32991648,3.38587379), (-2.51480651,-2.17165756), (-3.58092475,1.74823511), (-3.11148381,-0.345993072), (-0.779480874,-3.14056802)], [(-5.21330357,6.43571901), (5.2494235,0.898137152), (-1.53369141,-0.149325684), (-0.575195372,-5.33157587), (-1.30546784,1.69562232), (-7.01016188,1.90953791)], [(-3.01676202,-4.999380e+00), (-3.27753401,-8.176140e-01), (-4.92207956,-3.18152237), (-5.74547911,-3.48363876), (-4.9207921,-5.89285707), (6.88681697,-1.48247635)]]> : tensor<3x6xcomplex<f32>>
    return %c, %cst : tensor<4x3xui16>, tensor<3x6xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<4x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-24.690197,4.30901718), (5.91566849,0.241569519), (-19.3673134,-9.99254417), (-18.9620228,-26.4456444), (-18.6787796,-12.5917053), (-0.370035172,1.28118467)], [(-31.5439415,-22.1242218), (-12.4438906,-3.92702389), (-37.5219383,-22.5693073), (-41.3687439,-35.0486221), (-37.0564804,-37.858757), (34.1873932,-6.55825901)], [(-40.0779419,-7.41316223), (0.742900848,4.56047487), (-39.163166,-23.8804264), (-43.3603096,-33.4000893), (-39.6641235,-30.9622631), (18.7314529,-9.447380e+00)], [(-16.4601307,2.8726778), (3.94377899,0.161046267), (-12.9115419,-6.66169596), (-12.6413488,-17.6304283), (-12.4525204,-8.39446926), (-0.246689796,0.854123115)]]> : tensor<4x6xcomplex<f32>>
    return %cst : tensor<4x6xcomplex<f32>>
  }
}
