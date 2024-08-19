// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xcomplex<f32>>
    %1 = call @expected() : () -> tensor<5x7xcomplex<f32>>
    %2 = "stablehlo.sort"(%0) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
      %3 = stablehlo.real %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4 = stablehlo.compare  EQ, %3, %cst,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5 = stablehlo.select %4, %cst_0, %3 : tensor<i1>, tensor<f32>
      %6 = stablehlo.compare  NE, %3, %3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_1 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %7 = stablehlo.select %6, %cst_1, %5 : tensor<i1>, tensor<f32>
      %8 = stablehlo.imag %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %9 = stablehlo.compare  EQ, %8, %cst_2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %10 = stablehlo.select %9, %cst_3, %8 : tensor<i1>, tensor<f32>
      %11 = stablehlo.compare  NE, %8, %8,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %12 = stablehlo.select %11, %cst_4, %10 : tensor<i1>, tensor<f32>
      %13 = stablehlo.real %arg1 : (tensor<complex<f32>>) -> tensor<f32>
      %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %14 = stablehlo.compare  EQ, %13, %cst_5,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %15 = stablehlo.select %14, %cst_6, %13 : tensor<i1>, tensor<f32>
      %16 = stablehlo.compare  NE, %13, %13,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_7 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %17 = stablehlo.select %16, %cst_7, %15 : tensor<i1>, tensor<f32>
      %18 = stablehlo.imag %arg1 : (tensor<complex<f32>>) -> tensor<f32>
      %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %19 = stablehlo.compare  EQ, %18, %cst_8,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %20 = stablehlo.select %19, %cst_9, %18 : tensor<i1>, tensor<f32>
      %21 = stablehlo.compare  NE, %18, %18,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_10 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %22 = stablehlo.select %21, %cst_10, %20 : tensor<i1>, tensor<f32>
      %23 = stablehlo.compare  LT, %12, %22,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %24 = stablehlo.compare  LT, %7, %17,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %25 = stablehlo.compare  EQ, %7, %17,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %26 = stablehlo.and %25, %23 : tensor<i1>
      %27 = stablehlo.or %24, %26 : tensor<i1>
      stablehlo.return %27 : tensor<i1>
    }) : (tensor<5x7xcomplex<f32>>) -> tensor<5x7xcomplex<f32>>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x7xcomplex<f32>>, tensor<5x7xcomplex<f32>>) -> ()
    return %2 : tensor<5x7xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<5x7xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-4.230590e+00,-0.931078791), (2.437567,6.385600e+00), (-3.22736883,1.83972192), (-2.1802094,0.889342188), (2.12194395,-0.191629708), (5.18555069,-4.99879742), (3.67927694,-2.2557056)], [(0.379014701,0.71961534), (3.83116126,-0.202661604), (-1.31466472,-0.621450722), (-0.886600852,-3.84756875), (-0.77832818,-2.22487688), (-0.806218206,-2.79777122), (-1.17338157,-0.0607901253)], [(-1.40947139,0.967940688), (3.73917222,1.84880078), (0.479227781,3.7598331), (-0.0535792634,-1.28269935), (-1.52044451,0.930376708), (-4.65916061,3.13317561), (-1.79412854,-2.32595181)], [(0.751593947,-2.26882839), (-1.97411048,-5.05067635), (-2.80797291,0.160113797), (0.709535956,-0.731431484), (-0.599625289,1.46191716), (-3.644970e+00,-0.755360901), (-0.0846063494,-2.87114239)], [(3.9562633,-2.32443857), (2.38367414,1.16753387), (-1.4873457,-1.95769823), (-1.39860535,2.15638304), (-1.28114533,-3.99063683), (4.16027069,-5.125950e+00), (-2.47642064,0.641588687)]]> : tensor<5x7xcomplex<f32>>
    return %cst : tensor<5x7xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<5x7xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-4.230590e+00,-0.931078791), (-1.97411048,-5.05067635), (-3.22736883,1.83972192), (-2.1802094,0.889342188), (-1.52044451,0.930376708), (-4.65916061,3.13317561), (-2.47642064,0.641588687)], [(-1.40947139,0.967940688), (2.38367414,1.16753387), (-2.80797291,0.160113797), (-1.39860535,2.15638304), (-1.28114533,-3.99063683), (-3.644970e+00,-0.755360901), (-1.79412854,-2.32595181)], [(0.379014701,0.71961534), (2.437567,6.385600e+00), (-1.4873457,-1.95769823), (-0.886600852,-3.84756875), (-0.77832818,-2.22487688), (-0.806218206,-2.79777122), (-1.17338157,-0.0607901253)], [(0.751593947,-2.26882839), (3.73917222,1.84880078), (-1.31466472,-0.621450722), (-0.0535792634,-1.28269935), (-0.599625289,1.46191716), (4.16027069,-5.125950e+00), (-0.0846063494,-2.87114239)], [(3.9562633,-2.32443857), (3.83116126,-0.202661604), (0.479227781,3.7598331), (0.709535956,-0.731431484), (2.12194395,-0.191629708), (5.18555069,-4.99879742), (3.67927694,-2.2557056)]]> : tensor<5x7xcomplex<f32>>
    return %cst : tensor<5x7xcomplex<f32>>
  }
}
