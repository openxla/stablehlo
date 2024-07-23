func.func @all_gather(%arg3: tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.all_gather' that was explicitly marked illegal}}
  %0 = "stablehlo.all_gather"(%arg3) { all_gather_dim = 1 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>
}

// --- all_to_all ---
func.func @all_to_all(%arg3: tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.all_to_all' that was explicitly marked illegal}}
  %0 = "stablehlo.all_to_all"(%arg3) { split_dimension = 1 : i64, concat_dimension = 1 : i64, split_count = 2 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>} : (tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>
}

// --- collective_permute ---
func.func @collective_permute(%arg0: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.collective_permute' that was explicitly marked illegal}}
  %0 = "stablehlo.collective_permute"(%arg0) { source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>, channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
}

// --- custom_call ---
func.func @custom_call(%arg0: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.custom_call' that was explicitly marked illegal}}
  %0 = "stablehlo.custom_call" (%arg0) {call_target_name = "foo"} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
}

// --- is_finite ---
func.func @is_finite(%arg0: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2xi1> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.is_finite' that was explicitly marked illegal}}
  %0 = "stablehlo.is_finite"(%arg0) {} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2xi1>
  func.return %0 : tensor<1x2x2xi1>
}


