// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect --stablehlo-legalize-quantized-op-to-qdq | FileCheck %s --check-prefixes=CHECK


// -----
// Tests for StableHLO OPs supporting per-axis quantization. These OPs also support per-tensor quantization.

// CHECK-LABEL: @ops_per_axis_quantization
func.func @ops_per_axis_quantization(
  %arg0: tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>,
  %arg1: tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>,
  %shape: tensor<3xi64>,
  %token0: !stablehlo.token) -> (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>, tensor<1x2x3x2x!quant.uniform<i8<-128:127>:f32:3, {0.1:-30, 0.5:-20}>>, tensor<2x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30, 0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>, tensor<i32>, !stablehlo.token, tensor<2x2x!quant.uniform<i8<-128:127>:f32:1, {0.1:-30, 0.5:-20}>>, !stablehlo.token, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:1, {0.1:-30, 0.5:-20}>>, tuple<tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>>, tensor<1x2x2xf32>, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) {
  %bitcast_convert = "stablehlo.bitcast_convert"(%arg0) : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>
  %broadcast_in_dim_1 = "stablehlo.broadcast_in_dim" (%arg0) {broadcast_dimensions = array<i64: 0, 1, 3>} : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<1x2x3x2x!quant.uniform<i8<-128:127>:f32:3, {0.1:-30, 0.5:-20}>>
  %broadcast_in_dim_2 = "stablehlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>) -> tensor<2x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30, 0.1:-30}>>
  %custom_call = "stablehlo.custom_call" (%arg0) {call_target_name = "foo"} : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>
  %dynamic_broadcast_in_dim = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>, tensor<3xi64>) -> tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>
  %get_dimension_size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<i32>
  %outfeed = "stablehlo.outfeed"(%arg0, %token0) {outfeed_config = ""} : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>, !stablehlo.token) -> !stablehlo.token
  %reshape = "stablehlo.reshape" (%arg0) : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<2x2x!quant.uniform<i8<-128:127>:f32:1, {0.1:-30, 0.5:-20}>>
  %send = "stablehlo.send"(%arg0, %token0) {channel_handle = #stablehlo.channel_handle<handle = 5, type = 2>, is_host_transfer = true} : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>, !stablehlo.token) -> !stablehlo.token
  %transpose = "stablehlo.transpose"(%arg0) {permutation = array<i64: 0, 2, 1>}: (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:1, {0.1:-30, 0.5:-20}>>
  %tuple = "stablehlo.tuple"(%arg1, %arg1) : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>) -> tuple<tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>>
  %uniform_dequantize = "stablehlo.uniform_dequantize" (%arg0) : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<1x2x2xf32>
  %uniform_quantize = "stablehlo.uniform_quantize" (%arg0) : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>
  func.return %bitcast_convert, %broadcast_in_dim_1, %broadcast_in_dim_2, %custom_call, %dynamic_broadcast_in_dim, %get_dimension_size, %outfeed, %reshape, %send, %transpose, %tuple, %uniform_dequantize, %uniform_quantize : tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>, tensor<1x2x3x2x!quant.uniform<i8<-128:127>:f32:3, {0.1:-30, 0.5:-20}>>, tensor<2x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30, 0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>, tensor<i32>, !stablehlo.token, tensor<2x2x!quant.uniform<i8<-128:127>:f32:1, {0.1:-30, 0.5:-20}>>, !stablehlo.token, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:1, {0.1:-30, 0.5:-20}>>, tuple<tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>>, tensor<1x2x2xf32>, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>
}

// CHECK:      stablehlo.bitcast_convert %arg0
// CHECK-NEXT: stablehlo.broadcast_in_dim %arg0
// CHECK-NEXT: stablehlo.broadcast_in_dim %arg1
// CHECK-NEXT: stablehlo.custom_call @foo(%arg0)
// CHECK-NEXT: stablehlo.dynamic_broadcast_in_dim %arg0
// CHECK-NEXT: stablehlo.get_dimension_size %arg0
// CHECK-NEXT: stablehlo.outfeed"(%arg0, %arg3)
// CHECK-NEXT: stablehlo.reshape %arg0
// CHECK-NEXT: stablehlo.send"(%arg0, %arg3)
// CHECK-NEXT: stablehlo.transpose %arg0
// CHECK-NEXT: stablehlo.tuple %arg1, %arg1
// CHECK-NEXT: stablehlo.uniform_dequantize %arg0
// CHECK-NEXT: stablehlo.uniform_quantize %arg0

// -----
// Tests for StableHLO OPs supporting per-tensor quantization. These OPs may or may not support per-axis quantization

// CHECK-LABEL: @ops_per_tensor_quantization
func.func @ops_per_tensor_quantization(
  %arg0: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>,
  %arg1: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>,
  %arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>,
  %arg3: tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>,
  %shape: tensor<3xi64>, %token0: !stablehlo.token) -> (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2xi1>, tensor<2x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<i32>, tensor<1x2x2xi1>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, !stablehlo.token, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, !stablehlo.token, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32, 1.0:17>>, tuple<tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>>, tensor<1x2x2xf32>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) {

  %abs = "stablehlo.abs"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %add = "stablehlo.add"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %all_gather = "stablehlo.all_gather"(%arg3) { all_gather_dim = 1 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>
  %all_to_all = "stablehlo.all_to_all"(%arg3) { split_dimension = 1 : i64, concat_dimension = 1 : i64, split_count = 2 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>} : (tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>
  %atan2 = "stablehlo.atan2"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %bitcast_convert = "stablehlo.bitcast_convert"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %broadcast_in_dim = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %cbrt = "stablehlo.cbrt"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %ceil = "stablehlo.ceil"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %cholesky = "stablehlo.cholesky"(%arg0) { lower = true } : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %collective_permute = "stablehlo.collective_permute"(%arg0) { source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>, channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %compare = "stablehlo.compare"(%arg0, %arg1) { comparison_direction = #stablehlo<comparison_direction LT>, compare_type = #stablehlo<comparison_type FLOAT> } : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2xi1>
  %concatenate = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %cosine = "stablehlo.cosine"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %custom_call = "stablehlo.custom_call" (%arg0) {call_target_name = "foo"} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %divide = "stablehlo.divide"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %dynamic_broadcast_in_dim = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<3xi64>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %exponential = "stablehlo.exponential"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %exponential_minus_one = "stablehlo.exponential_minus_one"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %floor = "stablehlo.floor"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %get_dimension_size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<i32>
  %is_finite = "stablehlo.is_finite"(%arg0) {} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2xi1>
  %log = "stablehlo.log"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %log_plus_one = "stablehlo.log_plus_one"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %logistic = "stablehlo.logistic"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %maximum = "stablehlo.maximum"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %minimum = "stablehlo.minimum"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %multiply = "stablehlo.multiply"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %negate = "stablehlo.negate"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %outfeed = "stablehlo.outfeed"(%arg0, %token0) {outfeed_config = ""} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, !stablehlo.token) -> !stablehlo.token
  %optimization_barrier = "stablehlo.optimization_barrier"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>)
  %power = "stablehlo.power"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %remainder = "stablehlo.remainder"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %reshape = "stablehlo.reshape" (%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %rsqrt = "stablehlo.rsqrt"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %send = "stablehlo.send"(%arg0, %token0) {channel_handle = #stablehlo.channel_handle<handle = 5, type = 2>, is_host_transfer = true} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, !stablehlo.token) -> !stablehlo.token
  %sign = "stablehlo.sign"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %sine = "stablehlo.sine"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %sqrt = "stablehlo.sqrt"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %subtract = "stablehlo.subtract"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %tan = "stablehlo.tan"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %tanh = "stablehlo.tanh"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %transpose = "stablehlo.transpose"(%arg0) {permutation = array<i64: 0, 2, 1>}: (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8<-128:127>:f32, 1.0:17>>
  %tuple = "stablehlo.tuple"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tuple<tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>>
  %uniform_dequantize = "stablehlo.uniform_dequantize" (%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2xf32>
  %uniform_quantize = "stablehlo.uniform_quantize" (%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>

 func.return %abs, %add, %all_gather, %all_to_all, %atan2, %bitcast_convert, %broadcast_in_dim, %cbrt, %ceil, %cholesky, %collective_permute, %compare, %concatenate, %cosine, %custom_call, %divide, %dynamic_broadcast_in_dim, %exponential, %exponential_minus_one, %floor, %get_dimension_size, %is_finite, %log, %log_plus_one, %logistic, %maximum, %minimum, %multiply, %negate, %outfeed, %power, %remainder, %reshape, %rsqrt, %send, %sign, %sine, %sqrt, %subtract, %tanh, %transpose, %tuple, %uniform_dequantize, %uniform_quantize : tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2xi1>, tensor<2x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<i32>, tensor<1x2x2xi1>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, !stablehlo.token, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, !stablehlo.token, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32, 1.0:17>>, tuple<tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>>, tensor<1x2x2xf32>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK-NEXT: %[[UDQ_0:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[UDQ_0]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[ABS]]
// CHECK-NEXT: %[[UDQ_ADD_0:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[UDQ_ADD_1:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[UDQ_ADD_0]], %[[UDQ_ADD_1]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[ADD]]
// CHECK-NEXT: "stablehlo.all_gather"(%arg3)
// CHECK-NEXT: "stablehlo.all_to_all"(%arg3)
// CHECK-NEXT: %[[UDQ_1:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[UDQ_2:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[ATAN2:.*]] = stablehlo.atan2 %[[UDQ_1]], %[[UDQ_2]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[ATAN2]]
// CHECK-NEXT: stablehlo.bitcast_convert %arg0
// CHECK-NEXT: stablehlo.broadcast_in_dim %arg0
// CHECK-NEXT: %[[UDQ_3:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[CBRT:.*]] = stablehlo.cbrt %[[UDQ_3]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[CBRT]]
// CHECK-NEXT: %[[UDQ_4:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[CEIL:.*]] = stablehlo.ceil %[[UDQ_4]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[CEIL]]
// CHECK-NEXT: %[[UDQ_41:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[UDQ_41]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[CHOLESKY]]
// CHECK-NEXT: stablehlo.collective_permute"(%arg0)
// CHECK-NEXT: %[[UDQ_5:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[UDQ_6:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: stablehlo.compare  LT, %[[UDQ_5]], %[[UDQ_6]]
// CHECK-NEXT: stablehlo.concatenate %arg0, %arg1
// CHECK-NEXT: %[[UDQ_7:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[COSINE:.*]] = stablehlo.cosine %[[UDQ_7]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[COSINE]]
// CHECK-NEXT: stablehlo.custom_call @foo(%arg0)
// CHECK-NEXT: %[[UDQ_8:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[UDQ_9:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[DIVIDE:.*]] = stablehlo.divide %[[UDQ_8]], %[[UDQ_9]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[DIVIDE]]
// CHECK-NEXT: stablehlo.dynamic_broadcast_in_dim %arg0, %arg4
// CHECK-NEXT: %[[UDQ_10:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[EXPONENTIAL:.*]] = stablehlo.exponential %[[UDQ_10]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[EXPONENTIAL]]
// CHECK-NEXT: %[[UDQ_11:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[EXPONENTIAL_MINUS_ONE:.*]] = stablehlo.exponential_minus_one %[[UDQ_11]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[EXPONENTIAL_MINUS_ONE]]
// CHECK-NEXT: %[[UDQ_12:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[FLOOR:.*]] = stablehlo.floor %[[UDQ_12]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[FLOOR]]
// CHECK-NEXT: stablehlo.get_dimension_size %arg0
// CHECK-NEXT: stablehlo.is_finite %arg0
// CHECK-NEXT: %[[UDQ_13:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[LOG:.*]] = stablehlo.log %[[UDQ_13]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[LOG]]
// CHECK-NEXT: %[[UDQ_14:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[LOG_PLUS_ONE:.*]] = stablehlo.log_plus_one %[[UDQ_14]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[LOG_PLUS_ONE]]
// CHECK-NEXT: %[[UDQ_15:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[LOGISTIC:.*]] = stablehlo.logistic %[[UDQ_15]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[LOGISTIC]]
// CHECK-NEXT: %[[UDQ_16:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[UDQ_17:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[MAXIMUM:.*]] = stablehlo.maximum %[[UDQ_16]], %[[UDQ_17]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[MAXIMUM]]
// CHECK-NEXT: %[[UDQ_18:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[UDQ_19:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[MINIMUM:.*]] = stablehlo.minimum %[[UDQ_18]], %[[UDQ_19]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[MINIMUM]]
// CHECK-NEXT: %[[UDQ_20:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[UDQ_21:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[MULTIPLY:.*]] = stablehlo.multiply %[[UDQ_20]], %[[UDQ_21]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[MULTIPLY]]
// CHECK-NEXT: %[[UDQ_22:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[UDQ_22]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[NEGATE]]
// CHECK-NEXT: stablehlo.outfeed"(%arg0, %arg5)
// CHECK-NEXT: %[[UDQ_23:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[UDQ_24:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[POWER:.*]] = stablehlo.power %[[UDQ_23]], %[[UDQ_24]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[POWER]]
// CHECK-NEXT: %[[UDQ_25:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[UDQ_26:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[REMAINDER:.*]] = stablehlo.remainder %[[UDQ_25]], %[[UDQ_26]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[REMAINDER]]
// CHECK-NEXT: stablehlo.reshape %arg0
// CHECK-NEXT: %[[UDQ_27:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[RSQRT:.*]] = stablehlo.rsqrt %[[UDQ_27]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[RSQRT]]
// CHECK-NEXT: stablehlo.send"(%arg0, %arg5)
// CHECK-NEXT: %[[UDQ_28:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[SIGN:.*]] = stablehlo.sign %[[UDQ_28]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[SIGN]]
// CHECK-NEXT: %[[UDQ_29:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[SINE:.*]] = stablehlo.sine %[[UDQ_29]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[SINE]]
// CHECK-NEXT: %[[UDQ_30:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[SQRT:.*]] = stablehlo.sqrt %[[UDQ_30]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[SQRT]]
// CHECK-NEXT: %[[UDQ_31:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[UDQ_32:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[SUBTRACT:.*]] = stablehlo.subtract %[[UDQ_31]], %[[UDQ_32]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[SUBTRACT]]
// CHECK-NEXT: %[[UDQ_33:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[TANH:.*]] = stablehlo.tanh %[[UDQ_33]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[TANH]]
// CHECK-NEXT: stablehlo.transpose %arg0
// CHECK-NEXT: stablehlo.tuple %arg0, %arg1
// CHECK-NEXT: stablehlo.uniform_dequantize %arg0
// CHECK-NEXT: stablehlo.uniform_quantize %arg0

// -----

// CHECK-LABEL:  func.func @batch_norm_grad_per_tensor_quantization
func.func @batch_norm_grad_per_tensor_quantization(%input: tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, %scale: tensor<2x!quant.uniform<i8:f32, 1.0:17>>, %mean: tensor<2x!quant.uniform<i8:f32, 1.0:17>>, %variance: tensor<2x!quant.uniform<i8:f32, 1.0:17>>, %grad_output: tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>> {
  %0:3 = "stablehlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output)
   {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>)
   -> (tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>)
  func.return %0#0 : tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK:      %[[OPR0:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OPR1:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OPR2:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OPR3:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OPR4:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[GRAD_OPERAND:.*]], %[[GRAD_SCALE:.*]], %[[GRAD_OFFSET:.*]] = "stablehlo.batch_norm_grad"(%[[OPR0]], %[[OPR1]], %[[OPR2]], %[[OPR3]], %[[OPR4]])
// CHECK-NEXT: stablehlo.uniform_quantize %[[GRAD_OPERAND]]


// -----

// CHECK-LABEL: @batch_norm_inference_per_tensor_quantization
func.func @batch_norm_inference_per_tensor_quantization(%input: tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>, %scale: tensor<256x!quant.uniform<i8:f32, 1.0:17>>, %offset: tensor<256x!quant.uniform<i8:f32, 1.0:17>>, %mean: tensor<256x!quant.uniform<i8:f32, 1.0:17>>, %variance: tensor<256x!quant.uniform<i8:f32, 1.0:17>>) -> (tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>) {
  %0 = "stablehlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {
    epsilon = 1.001000e-05 : f32,
    feature_index = 1 : i64
  } : (tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK:      %[[OPR0:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OPR1:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OPR2:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OPR3:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OPR4:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[RES:.*]] = "stablehlo.batch_norm_inference"(%[[OPR0]], %[[OPR1]], %[[OPR2]], %[[OPR3]], %[[OPR4]])
// CHECK-NEXT: stablehlo.uniform_quantize %[[RES]]

// -----

// CHECK-LABEL: @batch_norm_training_per_tensor_quantization
func.func @batch_norm_training_per_tensor_quantization(%input: tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, %scale: tensor<2x!quant.uniform<i8:f32, 1.0:17>>, %offset: tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>> {
  %0:3 = "stablehlo.batch_norm_training" (%input, %scale, %offset) {
    epsilon = 0.001 : f32,
    feature_index = 1 : i64
  } : (tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>) ->
      (tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>)
  func.return %0#0 : tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK:      %[[OPR0:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OPR1:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OPR2:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OUTPUT:.*]], %[[GRAD_SCALE:.*]], %[[GRAD_OFFSET:.*]] = "stablehlo.batch_norm_training"(%[[OPR0]], %[[OPR1]], %[[OPR2]])
// CHECK-NEXT: stablehlo.uniform_quantize %[[OUTPUT]]

// -----

// CHECK-LABEL: @dynamic_slice_per_tensor_quantization
func.func @dynamic_slice_per_tensor_quantization(%arg0: tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 1, 4>} : (tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<i64>, tensor<i64>) -> tensor<1x4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<1x4x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: stablehlo.dynamic_slice %arg0, %arg1, %arg2

// -----

// CHECK-LABEL: @dynamic_update_slice_per_tensor_quantization
func.func @dynamic_update_slice_per_tensor_quantization(%operand: tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>, %update: tensor<1x4x!quant.uniform<i8:f32, 1.0:17>>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<i64>, tensor<i64>) -> tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3

// -----

// CHECK-LABEL: @gather_per_tensor_quantization
func.func @gather_per_tensor_quantization(%operand : tensor<?x?x?x?x?x?x?x?x!quant.uniform<i8:f32, 1.0:17>>, %start_indices : tensor<1x5x2xi32>) -> tensor<8x?x7x1x6x1x?x!quant.uniform<i8:f32, 1.0:17>> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0, 2, 3, 4, 5],
      collapsed_slice_dims = [0, 1, 3],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8, 1, 7, 1, 6, 1>,
    indices_are_sorted = false
  } : (tensor<?x?x?x?x?x?x?x?x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x5x2xi32>) -> tensor<8x?x7x1x6x1x?x!quant.uniform<i8:f32, 1.0:17>>
  func.return %res : tensor<8x?x7x1x6x1x?x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: "stablehlo.gather"(%arg0, %arg1)

// -----

// CHECK-LABEL: @map_per_tensor_quantization
func.func @map_per_tensor_quantization(%arg0: tensor<4x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>):
    "stablehlo.return"(%arg2) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<4x!quant.uniform<i8:f32, 1.0:17>>, tensor<4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<4x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: "stablehlo.map"(%arg0, %arg1)

// -----

// CHECK-LABEL: @pad_per_tensor_quantization
func.func @pad_per_tensor_quantization(%arg0: tensor<1x2x3x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x7x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 0, 1, 2>,
    edge_padding_high = array<i64: 1, 1, 0>,
    interior_padding = array<i64: 0, 0, 1>
  } : (tensor<1x2x3x!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x7x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x4x7x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: stablehlo.pad %arg0, %arg1

// -----

// CHECK-LABEL: @reduce_per_tensor_quantization
func.func @reduce_per_tensor_quantization(%arg0: tensor<16x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>):
      %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {
    dimensions = array<i64: 0>
  } : (tensor<16x!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: stablehlo.reduce(%arg0 init: %arg1)
// CHECK: stablehlo.uniform_dequantize
// CHECK-NEXT: stablehlo.uniform_dequantize
// CHECK-NEXT: stablehlo.add
// CHECK-NEXT: stablehlo.uniform_quantize

// -----

// CHECK-LABEL: @reduce_per_tensor_precision_quantization
func.func @reduce_per_tensor_precision_quantization(%arg0: tensor<6x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<6x!quant.uniform<i8:f32, 1.0:17>> {
  %output = "stablehlo.reduce_precision"(%arg0) {
    exponent_bits = 5 : i32,
    mantissa_bits = 10 : i32
  } : (tensor<6x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<6x!quant.uniform<i8:f32, 1.0:17>>
  func.return %output : tensor<6x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: %[[UDQ:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[RES:.*]] = stablehlo.reduce_precision %[[UDQ]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[RES]]

// -----

// CHECK-LABEL: @reduce_scatter_per_tensor_quantization
func.func @reduce_scatter_per_tensor_quantization(%data: tensor<4x16x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x4x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<!quant.uniform<i8:f32, 1.0:17>>
    "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids} : (tensor<4x16x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<4x4x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: "stablehlo.reduce_scatter"(%arg0)

// -----

// CHECK-LABEL: @op_reduce_window_per_tensor_quantization
func.func @op_reduce_window_per_tensor_quantization(%arg0: tensor<2x17x31x7x!quant.uniform<i8:f32, 0.1:-30>>, %arg1: tensor<!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<2x9x16x7x!quant.uniform<i8:f32, 0.1:-30>> {
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 0.1:-30>>, %arg3: tensor<!quant.uniform<i8:f32, 0.1:-30>>):
      %1 = "stablehlo.maximum"(%arg2, %arg3) : (tensor<!quant.uniform<i8:f32, 0.1:-30>>, tensor<!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<!quant.uniform<i8:f32, 0.1:-30>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 0.1:-30>>) -> ()
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>,
    window_strides = array<i64: 1, 4, 4, 1>,
    base_dilations = array<i64: 1, 2, 2, 1>,
    window_dilations = array<i64: 1, 2, 2, 1>,
    padding = dense<[[0, 0], [2, 0], [0, 2], [0, 0]]> : tensor<4x2xi64>
  } : (tensor<2x17x31x7x!quant.uniform<i8:f32, 0.1:-30>>, tensor<!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<2x9x16x7x!quant.uniform<i8:f32, 0.1:-30>>
  func.return %0 : tensor<2x9x16x7x!quant.uniform<i8:f32, 0.1:-30>>
}

// CHECK: "stablehlo.reduce_window"(%arg0, %arg1)
// CHECK: %[[OPR0:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OPR1:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[RES:.*]] = stablehlo.maximum %[[OPR0]], %[[OPR1]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[RES]]

// -----

// CHECK-LABEL: @reverse_per_tensor_quantization
func.func @reverse_per_tensor_quantization(%operand: tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>> {
  %result = "stablehlo.reverse"(%operand) {
    dimensions = array<i64: 1>
  } : (tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>>
  func.return %result : tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>>
}

// CHECK: stablehlo.reverse %arg0

// -----

// CHECK-LABEL: @round_afz_per_tensor_quantization
func.func @round_afz_per_tensor_quantization(%arg0: tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.round_nearest_afz"(%arg0) {} : (tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: %[[OPR0:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[RES:.*]] = stablehlo.round_nearest_afz %[[OPR0]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[RES]]

// -----

// CHECK-LABEL: @round_even_per_tensor_quantization
func.func @round_even_per_tensor_quantization(%arg0: tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.round_nearest_even"(%arg0) {} : (tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: %[[OPR0:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[RES:.*]] = stablehlo.round_nearest_even %[[OPR0]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[RES]]

// -----

// CHECK-LABEL: @scatter_per_tensor_quantization
func.func @scatter_per_tensor_quantization(%arg0: tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<10x2xi32>, %arg2: tensor<10x300x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg4: tensor<!quant.uniform<i8:f32, 1.0:17>>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >
  } : (tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>>, tensor<10x2xi32>, tensor<10x300x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: "stablehlo.scatter"(%arg0, %arg1, %arg2)
// CHECK: stablehlo.uniform_dequantize
// CHECK-NEXT: stablehlo.uniform_dequantize
// CHECK-NEXT: stablehlo.add
// CHECK-NEXT: stablehlo.uniform_quantize

// -----

// CHECK-LABEL: @select_per_tensor_quantization
func.func @select_per_tensor_quantization(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>, %arg2: tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x3x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<2x3xi1>, tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: %[[OPR0:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OPR1:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[RES:.*]] = stablehlo.select %arg0, %[[OPR0]], %[[OPR1]]
// CHECK-NEXT: stablehlo.uniform_quantize %[[RES]]

// -----

// CHECK-LABEL: @select_and_scatter_per_tensor_quantization
func.func @select_and_scatter_per_tensor_quantization(%arg0: tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<10x23x23x64x!quant.uniform<i8:f32, 1.0:17>>, %arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg4: tensor<!quant.uniform<i8:f32, 1.0:17>>):
      %1 = "stablehlo.compare"(%arg3, %arg4) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<i1>
      "stablehlo.return"(%1) : (tensor<i1>) -> ()
  }, {
    ^bb0(%arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg4: tensor<!quant.uniform<i8:f32, 1.0:17>>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>
  } : (tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>>, tensor<10x23x23x64x!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2)
// CHECK: %[[OPR0:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OPR1:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: stablehlo.compare GE, %[[OPR0]], %[[OPR1]]
// CHECK: stablehlo.uniform_dequantize
// CHECK-NEXT: stablehlo.uniform_dequantize
// CHECK-NEXT: stablehlo.add
// CHECK-NEXT: stablehlo.uniform_quantize

// -----

// CHECK-LABEL: @slice_per_tensor_qunatization
func.func @slice_per_tensor_qunatization(%arg0: tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.slice"(%arg0) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 2>} : (tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<1x2x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: stablehlo.slice %arg0

// -----

// CHECK-LABEL: @sort_per_tensor_quantization
func.func @sort_per_tensor_quantization(%input0: tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>, %input1: tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>) -> (tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>, tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>) {
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>, tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>) -> (tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>, tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>)
  func.return %0#0, %0#1: tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>, tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK-NOT: stablehlo.uniform_dequantize
// CHECK: "stablehlo.sort"(%arg0, %arg1)
// CHECK: %[[OPR0:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: %[[OPR1:.*]] = stablehlo.uniform_dequantize
// CHECK-NEXT: stablehlo.compare GT, %[[OPR0]], %[[OPR1]]

// -----

// CHECK-LABEL: @while_per_tensor_quantization
func.func @while_per_tensor_quantization(%arg0: tensor<4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<?x!quant.uniform<i8:f32, 1.0:17>> {
  %while = "stablehlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<?x!quant.uniform<i8:f32, 1.0:17>>):
    %1 = stablehlo.constant dense<true> : tensor<i1>
    stablehlo.return %1 : tensor<i1>
  },  {
  ^bb0(%arg1: tensor<?x!quant.uniform<i8:f32, 1.0:17>>):
    stablehlo.return %arg1 : tensor<?x!quant.uniform<i8:f32, 1.0:17>>
  }) : (tensor<4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<?x!quant.uniform<i8:f32, 1.0:17>>
  func.return %while : tensor<?x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: stablehlo.while(%iterArg = %arg0)
