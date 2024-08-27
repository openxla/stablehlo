// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xf16>, tensor<3x6xf64>)
    %1 = call @expected() : () -> tensor<4x6xf64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xf16>) -> tensor<4x3xf64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    return %4 : tensor<4x6xf64>
  }
  func.func private @inputs() -> (tensor<4x3xf16> {mhlo.layout_mode = "default"}, tensor<3x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-3.595700e+00, 3.619140e+00, -5.261720e+00], [1.973880e-01, -1.737300e+00, -2.189940e-01], [-2.433590e+00, 7.119140e-01, -4.019530e+00], [9.355460e-01, 2.970700e+00, -1.253910e+00]]> : tensor<4x3xf16>
    %cst_0 = stablehlo.constant dense<[[1.7644197472699148, 1.7043742348703241, 1.5261589163458105, 0.29420515636325822, 2.3672846682690594, -2.6927424430293074], [-0.40181932535446785, -4.7279690310955527, 4.3225875343381333, 1.5654165620697107, -5.338042688148974, -3.05218271806035], [-0.6860180832954107, -3.7113500994186488, 4.2941046651038839, 0.36092700378597814, 2.5590305312603814, -2.7780545373475642]]> : tensor<3x6xf64>
    return %cst, %cst_0 : tensor<4x3xf16>, tensor<3x6xf64>
  }
  func.func private @expected() -> (tensor<4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.1889360316560662, -3.7115281507470428, -12.4379332649491, 2.7084918915058407, -41.296078958279011, 13.253365595049736], [1.1965912855483649, 9.3631091878288916, -8.1487903552448131, -2.7405739524327122, 9.1806667569880531, 5.3794347843088435], [-1.8224705737657463, 7.4042255783609887, -17.897027840399264, -1.0522911362221743, -19.847340019194167, 15.546626408482044], [1.3172138174878851, -7.7971853007161682, 8.8845128235571309, 4.4730619616707674, -16.851818698521239, -8.1028955691478401]]> : tensor<4x6xf64>
    return %cst : tensor<4x6xf64>
  }
}
