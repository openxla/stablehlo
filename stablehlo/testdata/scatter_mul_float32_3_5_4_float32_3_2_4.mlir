// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5x4xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<1> : tensor<2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<3x5x4xf32>, tensor<3x2x4xf32>)
    %1 = call @expected() : () -> tensor<3x5x4xf32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<3x5x4xf32>, tensor<2x1xi64>, tensor<3x2x4xf32>) -> tensor<3x5x4xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5x4xf32>, tensor<3x5x4xf32>) -> ()
    return %2 : tensor<3x5x4xf32>
  }
  func.func private @inputs() -> (tensor<3x5x4xf32> {mhlo.layout_mode = "default"}, tensor<3x2x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[5.44057608, 3.04158258, 1.60004854, 5.56463432], [0.788330435, -1.09290743, 0.220675215, -4.76314497], [-1.69774103, 2.15626073, 1.18520045, -2.03143215], [1.25943065, -2.0910728, 0.337215304, -1.79655027], [2.33973241, 2.52476811, 6.00997305, -0.593710601]], [[0.473636419, 1.09154403, 0.188831165, 5.46831347E-4], [2.9006021, -5.42880535, 1.26965332, 2.39274287], [-1.60081124, 0.405228615, -2.02989745, -0.477075666], [1.50640273, -2.5767622, 1.09634233, 7.43059301], [-2.59851336, -0.821191489, -0.3641195, 0.341712803]], [[2.6365726, -2.47409654, -1.0037576, 0.598663806], [1.92197287, 0.550307691, -1.07548773, -0.460019678], [0.755633771, 2.74935317, 1.41705239, -7.92743063], [0.633334577, -4.81241655, -1.79825795, -0.129450306], [4.59869957, -3.46550488, -0.243491575, -1.94660282]]]> : tensor<3x5x4xf32>
    %cst_0 = stablehlo.constant dense<[[[0.650848746, -0.0424347632, 0.606700182, 1.65023088], [-6.72013664, 3.00883317, -1.01683223, 0.27341491]], [[0.304164976, -2.22471237, -2.51236558, 2.81838965], [1.08790135, -1.82119572, -6.89793539, 1.19621718]], [[-4.33912086, 4.46709251, -1.69811583, -0.258259743], [-0.516513109, -2.73177814, -3.40098524, 4.14861298]]]> : tensor<3x2x4xf32>
    return %cst, %cst_0 : tensor<3x5x4xf32>, tensor<3x2x4xf32>
  }
  func.func private @expected() -> (tensor<3x5x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[5.44057608, 3.04158258, 1.60004854, 5.56463432], [-3.44799376, 0.139541462, -0.136137262, -2.14912033], [-1.69774103, 2.15626073, 1.18520045, -2.03143215], [1.25943065, -2.0910728, 0.337215304, -1.79655027], [2.33973241, 2.52476811, 6.00997305, -0.593710601]], [[0.473636419, 1.09154403, 0.188831165, 5.46831347E-4], [0.959813535, -21.9955482, 22.0032654, 8.06690788], [-1.60081124, 0.405228615, -2.02989745, -0.477075666], [1.50640273, -2.5767622, 1.09634233, 7.43059301], [-2.59851336, -0.821191489, -0.3641195, 0.341712803]], [[2.6365726, -2.47409654, -1.0037576, 0.598663806], [4.30755043, -6.71546268, -6.21122885, 0.492874175], [0.755633771, 2.74935317, 1.41705239, -7.92743063], [0.633334577, -4.81241655, -1.79825795, -0.129450306], [4.59869957, -3.46550488, -0.243491575, -1.94660282]]]> : tensor<3x5x4xf32>
    return %cst : tensor<3x5x4xf32>
  }
}
