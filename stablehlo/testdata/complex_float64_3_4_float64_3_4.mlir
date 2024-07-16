// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<3x4xf64>, tensor<3x4xf64>)
    %1 = call @expected() : () -> tensor<3x4xcomplex<f64>>
    %2 = stablehlo.complex %0#0, %0#1 : tensor<3x4xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x4xcomplex<f64>>, tensor<3x4xcomplex<f64>>) -> ()
    return %2 : tensor<3x4xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<3x4xf64> {mhlo.layout_mode = "default"}, tensor<3x4xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.2446081079638374, 2.9311191956533249, -0.80297253142501623, 1.3327935458596931], [-2.3572290064600101, -0.038978970864541988, 3.39249420391236, 1.7492081468625311], [7.1506041002014626, -4.5479984958432951, -5.7099287996096058, -2.1279896891312085]]> : tensor<3x4xf64>
    %cst_0 = stablehlo.constant dense<[[-4.7595820050969397, -1.6095319627071554, -0.39933304773392725, 5.1633630930350929], [0.95945138201202718, 0.6913424204045242, 0.11489614345782669, 1.0447167440530702], [1.3667622863058879, -0.55920604571652877, 3.4785332690751671, -2.2838064843987271]]> : tensor<3x4xf64>
    return %cst, %cst_0 : tensor<3x4xf64>, tensor<3x4xf64>
  }
  func.func private @expected() -> (tensor<3x4xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(2.2446081079638374,-4.7595820050969397), (2.9311191956533249,-1.6095319627071554), (-0.80297253142501623,-0.39933304773392725), (1.3327935458596931,5.1633630930350929)], [(-2.3572290064600101,0.95945138201202718), (-0.038978970864541988,0.6913424204045242), (3.39249420391236,0.11489614345782669), (1.7492081468625311,1.0447167440530702)], [(7.1506041002014626,1.3667622863058879), (-4.5479984958432951,-0.55920604571652877), (-5.7099287996096058,3.4785332690751671), (-2.1279896891312085,-2.2838064843987271)]]> : tensor<3x4xcomplex<f64>>
    return %cst : tensor<3x4xcomplex<f64>>
  }
}
