// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi32>, tensor<3x6xf64>)
    %1 = call @expected() : () -> tensor<4x6xf64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi32>) -> tensor<4x3xf64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    return %4 : tensor<4x6xf64>
  }
  func.func private @inputs() -> (tensor<4x3xi32> {mhlo.layout_mode = "default"}, tensor<3x6xf64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 2, -1], [-3, 0, 0], [0, 3, 6], [2, -1, 2]]> : tensor<4x3xi32>
    %cst = stablehlo.constant dense<[[3.4659555552847219, 4.2255337100745543, -5.4673025341106625, 3.4800095548462737, -4.012818706512439, 1.5126177330295851], [0.70199582847948072, 4.4209751754148696, -2.9100556724790274, -2.0857092183319894, -2.1443107051698429, -0.61941513355702638], [-4.6295346350155251, 1.090658511076384, 2.1762306249534613, -3.0100553769102412, 0.11623367309129488, -1.570001568863195]]> : tensor<3x6xf64>
    return %c, %cst : tensor<4x3xi32>, tensor<3x6xf64>
  }
  func.func private @expected() -> (tensor<4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[6.0335262919744865, 7.751291839753355, -7.9963419699115157, -1.1613630597537377, -4.4048550834309808, 0.33117130174914222], [-10.397866665854165, -12.676601130223663, 16.401907602331988, -10.440028664538822, 12.038456119537317, -4.5378531990887554], [-25.671220324654708, 19.806876592702913, 4.3272167322836852, -24.317459916457416, -5.7355300769617585, -11.27825481385025], [-3.0291539879410871, 6.2114092668870065, -3.6720881458353754, 3.0256175742040554, -5.6488593616724447, 0.50464746188980669]]> : tensor<4x6xf64>
    return %cst : tensor<4x6xf64>
  }
}
