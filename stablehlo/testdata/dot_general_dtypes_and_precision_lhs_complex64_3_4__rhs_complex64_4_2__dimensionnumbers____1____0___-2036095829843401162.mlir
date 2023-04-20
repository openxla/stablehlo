// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<3x2xcomplex<f32>>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>) -> tensor<3x2xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xcomplex<f32>>, tensor<3x2xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[(-1.99905837,-4.22710276), (-0.722109556,-1.42607856), (7.09746885,-0.639070749), (-1.40431321,5.90976572)], [(-0.975965261,5.401520e-01), (0.0463883951,1.5270108), (-3.43256855,-0.762728691), (-1.44157827,-6.888140e+00)], [(-0.587710321,0.0108269472), (-0.730207205,3.39075422), (-0.87460184,-1.51291549), (-4.94340229,-0.735580265)]]> : tensor<3x4xcomplex<f32>>
    %1 = stablehlo.constant dense<[[(-2.43169522,2.93219852), (-4.08006859,-1.28455973)], [(1.11888099,4.39408875), (0.760131955,0.884005486)], [(0.734864473,-4.11158705), (0.0675758794,-0.920152604)], [(-0.0545436963,-1.71550322), (1.17385483,1.77352107)]]> : tensor<4x2xcomplex<f32>>
    return %0, %1 : tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x2xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(35.517067,-27.9159641), (-8.7998867,15.9651127)], [(-23.2649956,14.1387072), (12.9515114,-7.28383446)], [(-22.1743603,9.8404026), (-7.09017658,-6.2854743)]]> : tensor<3x2xcomplex<f32>>
    return %0 : tensor<3x2xcomplex<f32>>
  }
}
