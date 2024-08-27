// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<100xi32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<100xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<100xf32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<100xi32>, tensor<100xi32>, tensor<100xf32>)
    %1:3 = call @expected() : () -> (tensor<100xi32>, tensor<100xi32>, tensor<100xf32>)
    %2:3 = "stablehlo.sort"(%0#0, %0#1, %0#2) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<f32>, %arg5: tensor<f32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3 = stablehlo.compare  EQ, %arg4, %cst,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4 = stablehlo.select %3, %cst_0, %arg4 : tensor<i1>, tensor<f32>
      %5 = stablehlo.compare  NE, %arg4, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_1 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %6 = stablehlo.select %5, %cst_1, %4 : tensor<i1>, tensor<f32>
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %7 = stablehlo.compare  EQ, %arg5, %cst_2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %8 = stablehlo.select %7, %cst_3, %arg5 : tensor<i1>, tensor<f32>
      %9 = stablehlo.compare  NE, %arg5, %arg5,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %10 = stablehlo.select %9, %cst_4, %8 : tensor<i1>, tensor<f32>
      %11 = stablehlo.compare  LT, %6, %10,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %12 = stablehlo.compare  LT, %arg2, %arg3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %13 = stablehlo.compare  EQ, %arg2, %arg3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %14 = stablehlo.and %13, %11 : tensor<i1>
      %15 = stablehlo.or %12, %14 : tensor<i1>
      %16 = stablehlo.compare  LT, %arg0, %arg1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %17 = stablehlo.compare  EQ, %arg0, %arg1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %18 = stablehlo.and %17, %15 : tensor<i1>
      %19 = stablehlo.or %16, %18 : tensor<i1>
      stablehlo.return %19 : tensor<i1>
    }) : (tensor<100xi32>, tensor<100xi32>, tensor<100xf32>) -> (tensor<100xi32>, tensor<100xi32>, tensor<100xf32>)
    stablehlo.custom_call @check.expect_eq(%2#0, %1#0) {has_side_effect = true} : (tensor<100xi32>, tensor<100xi32>) -> ()
    stablehlo.custom_call @check.expect_eq(%2#1, %1#1) {has_side_effect = true} : (tensor<100xi32>, tensor<100xi32>) -> ()
    stablehlo.custom_call @check.expect_close(%2#2, %1#2) {has_side_effect = true} : (tensor<100xf32>, tensor<100xf32>) -> ()
    return %2#0, %2#1, %2#2 : tensor<100xi32>, tensor<100xi32>, tensor<100xf32>
  }
  func.func private @inputs() -> (tensor<100xi32> {mhlo.layout_mode = "default"}, tensor<100xi32> {mhlo.layout_mode = "default"}, tensor<100xf32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1]> : tensor<100xi32>
    %c_0 = stablehlo.constant dense<[1, 5, 2, 3, 4, 2, -5, 0, 2, 2, 0, -1, -1, 0, 0, -5, 4, -3, 3, 0, 0, 1, 0, 5, 3, -3, 0, 1, 0, 0, 0, -2, 0, 4, -2, -1, -3, 0, -1, 3, 1, -4, -4, -2, 3, 5, 1, 7, 2, 2, 1, 0, -3, 1, -1, -1, -3, -3, -1, -2, 2, -5, 0, 0, -1, -1, -1, -2, 1, -2, 2, 3, 0, -2, 4, -3, 0, 0, -4, 0, 1, 0, -2, 3, 0, -2, -1, -5, 1, 6, -2, 1, -3, 1, 0, 0, 0, 3, -1, 0]> : tensor<100xi32>
    %cst = stablehlo.constant dense<[6.20356131, 5.65439892, 2.38549161, -3.96550655, -3.96124077, 1.99819422, -4.68467474, -2.49656415, -2.54703045, 0.0197906531, -0.740306317, -3.19704342, 1.27773118, -1.63176405, -1.01926863, 1.9354341, -2.4787395, 2.25621748, -1.48426414, -0.342727482, -0.651743412, 1.14231288, 2.13769197, 5.721990e-01, 3.78986549, 4.27968693, 0.455839902, -1.66110146, -2.93755436, -5.64413548, 2.54588413, 8.145430e-01, 0.273220271, 0.788339495, 2.00295901, -1.60751247, -1.70372462, -1.1993798, 3.84779954, -0.852602064, 3.09254956, 1.50013828, 0.799096703, -3.54357862, 0.281735957, 1.62879372, 1.44648623, 0.415672541, -4.84510756, 3.98790479, -3.5780375, -6.54678869, 2.99260116, -1.87401211, -0.621982455, 1.01716506, 2.95774937, 3.05121779, -0.028854521, -2.60105753, 1.91555417, -7.83330297, -4.62947416, -0.03703003, 3.06019616, -0.196689233, -2.59691501, -2.38492537, 4.45763779, -0.077415429, -3.38808942, -1.14041936, -0.225846633, -3.38728166, 4.22935104, 2.17284632, -3.660960e+00, -1.2449944, 2.02476263, -2.42962742, 2.82782459, 1.50918257, -2.22166085, -2.52914572, 1.04657984, -1.03708935, -1.80974913, 1.55136991, -0.658415257, 1.07587576, 5.95514774, -0.463521421, 5.02912045, -1.08982968, -0.883667945, 1.4935534, -0.537910938, -3.090550e-01, 2.59730196, -1.88129139]> : tensor<100xf32>
    return %c, %c_0, %cst : tensor<100xi32>, tensor<100xi32>, tensor<100xf32>
  }
  func.func private @expected() -> (tensor<100xi32> {mhlo.layout_mode = "default"}, tensor<100xi32> {mhlo.layout_mode = "default"}, tensor<100xf32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<100xi32>
    %c_0 = stablehlo.constant dense<[-5, -4, -3, -3, -3, -3, -2, -2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5, 5, 7, -5, -5, -5, -4, -4, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6]> : tensor<100xi32>
    %cst = stablehlo.constant dense<[1.55136991, 0.799096703, 2.17284632, 2.25621748, 3.05121779, 5.02912045, -2.60105753, -2.38492537, -1.03708935, 5.95514774, -0.028854521, 1.01716506, 1.27773118, -4.62947416, -2.49656415, -2.42962742, -1.63176405, -0.883667945, -0.740306317, -0.651743412, -0.342727482, -0.225846633, 0.273220271, 0.455839902, 1.4935534, 2.54588413, -3.5780375, -1.66110146, -1.08982968, -0.658415257, 1.14231288, 2.82782459, 4.45763779, -4.84510756, 1.91555417, 2.38549161, -3.96550655, -2.52914572, -1.14041936, 0.281735957, 4.22935104, 5.721990e-01, 1.62879372, 5.65439892, 0.415672541, -7.83330297, -4.68467474, 1.9354341, 1.50013828, 2.02476263, -1.70372462, 2.95774937, 2.99260116, 4.27968693, -3.54357862, -3.38728166, -2.22166085, -0.077415429, 8.145430e-01, 2.00295901, -3.19704342, -2.59691501, -1.80974913, -1.60751247, -0.621982455, -0.196689233, 2.59730196, 3.06019616, 3.84779954, -6.54678869, -5.64413548, -3.660960e+00, -2.93755436, -1.88129139, -1.2449944, -1.1993798, -1.01926863, -0.537910938, -0.03703003, 1.04657984, 1.50918257, 2.13769197, -1.87401211, -0.463521421, 1.44648623, 3.09254956, 6.20356131, -3.38808942, -2.54703045, 0.0197906531, 1.99819422, 3.98790479, -1.48426414, -0.852602064, -3.090550e-01, 3.78986549, -3.96124077, -2.4787395, 0.788339495, 1.07587576]> : tensor<100xf32>
    return %c, %c_0, %cst : tensor<100xi32>, tensor<100xi32>, tensor<100xf32>
  }
}
