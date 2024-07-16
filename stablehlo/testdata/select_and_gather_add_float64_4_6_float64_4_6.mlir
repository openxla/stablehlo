// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x6xf64>, tensor<4x6xf64>)
    %1 = call @expected() : () -> tensor<3x5xf64>
    %cst = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2:2 = "stablehlo.reduce_window"(%0#1, %0#0, %cst, %cst_0) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>):
      %3 = stablehlo.compare  LE, %arg0, %arg2,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %4 = stablehlo.select %3, %arg0, %arg2 : tensor<i1>, tensor<f64>
      %5 = stablehlo.select %3, %arg1, %arg3 : tensor<i1>, tensor<f64>
      stablehlo.return %4, %5 : tensor<f64>, tensor<f64>
    }) : (tensor<4x6xf64>, tensor<4x6xf64>, tensor<f64>, tensor<f64>) -> (tensor<3x5xf64>, tensor<3x5xf64>)
    stablehlo.custom_call @check.expect_close(%2#1, %1) {has_side_effect = true} : (tensor<3x5xf64>, tensor<3x5xf64>) -> ()
    return %2#1 : tensor<3x5xf64>
  }
  func.func private @inputs() -> (tensor<4x6xf64> {mhlo.layout_mode = "default"}, tensor<4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-7.1149372968688649, 0.06320878551452172, -2.7274963731057098, 2.2385959323403424, 0.1852081754565747, -0.34050181539936386], [1.218581858737378, -1.8100311668811839, -2.0416048520467962, 3.3229527312744809, -1.5791817244234267, 0.41975500990071585], [-0.39454487826059792, 2.4831068786241488, -0.59727638631645219, -3.1629397784485924, 2.7649750466923506, 6.4629911464084318], [-1.3053018707290454, -1.0661663537704877, -1.9961123005977237, 2.057529159538126, -1.2635029031873009, 3.5539599255940049]]> : tensor<4x6xf64>
    %cst_0 = stablehlo.constant dense<[[2.6871646805053282, 1.1079280583580291, 0.071354124721101739, 2.366803437360061, -0.24529864383761962, -3.5082838238461784], [0.2256227690737253, -2.9742316811601768, 2.5802444178171275, 1.624968440403348, -1.332422678704781, 2.9098649030965662], [1.861052554504345, 0.75490544268091297, 4.0689720446555757, -2.51724204911234, -0.41140104575078318, 1.0564941083029569], [4.4932466380783289, -2.5840047455330049, -2.3001326558887865, 3.9169581915115663, 0.17115619727584172, -1.8850388829842393]]> : tensor<4x6xf64>
    return %cst, %cst_0 : tensor<4x6xf64>, tensor<4x6xf64>
  }
  func.func private @expected() -> (tensor<3x5xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.8100311668811839, -1.8100311668811839, -2.7274963731057098, -1.5791817244234267, -0.34050181539936386], [-1.8100311668811839, -1.8100311668811839, -3.1629397784485924, -3.1629397784485924, -1.5791817244234267], [-1.0661663537704877, -1.0661663537704877, -3.1629397784485924, -3.1629397784485924, 3.5539599255940049]]> : tensor<3x5xf64>
    return %cst : tensor<3x5xf64>
  }
}
