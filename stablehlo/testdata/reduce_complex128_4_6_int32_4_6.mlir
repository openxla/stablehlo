// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<6xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x6xcomplex<f64>>, tensor<4x6xi32>)
    %1:2 = call @expected() : () -> (tensor<6xcomplex<f64>>, tensor<6xi32>)
    %cst = stablehlo.constant dense<(3.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %2:2 = stablehlo.reduce(%0#0 init: %cst), (%0#1 init: %c) across dimensions = [0] : (tensor<4x6xcomplex<f64>>, tensor<4x6xi32>, tensor<complex<f64>>, tensor<i32>) -> (tensor<6xcomplex<f64>>, tensor<6xi32>)
     reducer(%arg0: tensor<complex<f64>>, %arg2: tensor<complex<f64>>) (%arg1: tensor<i32>, %arg3: tensor<i32>)  {
      %3 = stablehlo.real %arg0 : (tensor<complex<f64>>) -> tensor<f64>
      %4 = stablehlo.real %arg2 : (tensor<complex<f64>>) -> tensor<f64>
      %5 = stablehlo.compare  EQ, %3, %4,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %6 = stablehlo.compare  GT, %3, %4,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %7 = stablehlo.imag %arg0 : (tensor<complex<f64>>) -> tensor<f64>
      %8 = stablehlo.imag %arg2 : (tensor<complex<f64>>) -> tensor<f64>
      %9 = stablehlo.compare  GT, %7, %8,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %10 = stablehlo.select %5, %9, %6 : tensor<i1>, tensor<i1>
      %11 = stablehlo.select %10, %arg0, %arg2 : tensor<i1>, tensor<complex<f64>>
      %12 = stablehlo.minimum %arg1, %arg3 : tensor<i32>
      stablehlo.return %11, %12 : tensor<complex<f64>>, tensor<i32>
    }
    stablehlo.custom_call @check.expect_close(%2#0, %1#0) {has_side_effect = true} : (tensor<6xcomplex<f64>>, tensor<6xcomplex<f64>>) -> ()
    stablehlo.custom_call @check.expect_eq(%2#1, %1#1) {has_side_effect = true} : (tensor<6xi32>, tensor<6xi32>) -> ()
    return %2#0, %2#1 : tensor<6xcomplex<f64>>, tensor<6xi32>
  }
  func.func private @inputs() -> (tensor<4x6xcomplex<f64>> {mhlo.layout_mode = "default"}, tensor<4x6xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-3.5085616894229723,0.37771718126801901), (-2.6985540122986622,-1.4782004149161674), (0.62479206768550799,1.0586492429165335), (-2.3514919467382724,0.34557962370836737), (1.1464624851688603,0.4073511449993249), (-4.98981787587027,-2.438433551129239)], [(-0.47688863862893083,1.3733011132611237), (0.36009476239198224,3.6652743319245462), (-0.52919555276589503,0.23277998372208186), (1.8347095423651467,-5.6685758720071817), (-5.7238188678998041,0.74610260039763876), (-2.0871431923644752,-1.5240444142189526)], [(-0.20308947479433498,1.3333373991376525), (2.0292540853553609,4.5846595150301273), (-1.6085746372381227,-0.40074266238460032), (-0.7316608630100403,-1.7644331560978213), (0.66876813798591839,4.7574329564216438), (2.5401825890278653,-0.66730971867746591)], [(3.139282464616052,0.3035813674320208), (-1.3525077737008202,-0.20649912308411184), (-0.93348857716511779,2.1604736280326318), (-0.97039543791560811,3.5291903112511829), (-0.068116426920902293,0.21319493179069926), (2.2223481075546907,3.2137199975415718)]]> : tensor<4x6xcomplex<f64>>
    %c = stablehlo.constant dense<[[1, 2, 3, 0, 0, 2], [0, 0, 0, 3, 1, -4], [6, 2, -1, 0, 4, 0], [1, -1, -1, -1, -3, -3]]> : tensor<4x6xi32>
    return %cst, %c : tensor<4x6xcomplex<f64>>, tensor<4x6xi32>
  }
  func.func private @expected() -> (tensor<6xcomplex<f64>> {mhlo.layout_mode = "default"}, tensor<6xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(3.139282464616052,0.3035813674320208), (3.000000e+00,0.000000e+00), (3.000000e+00,0.000000e+00), (3.000000e+00,0.000000e+00), (3.000000e+00,0.000000e+00), (3.000000e+00,0.000000e+00)]> : tensor<6xcomplex<f64>>
    %c = stablehlo.constant dense<[0, -1, -1, -1, -3, -4]> : tensor<6xi32>
    return %cst, %c : tensor<6xcomplex<f64>>, tensor<6xi32>
  }
}
