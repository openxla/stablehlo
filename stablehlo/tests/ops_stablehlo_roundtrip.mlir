// RUN: stablehlo-opt %s | stablehlo-opt
// RUN: diff <(stablehlo-opt %s) <(stablehlo-opt -emit-bytecode %s | stablehlo-opt)
// RUN: stablehlo-opt -emit-bytecode -debug-only=stablehlo-bytecode %s 2>&1 | (! grep 'Not Implemented')
// RUN: stablehlo-opt -emit-bytecode %s | stablehlo-opt -debug-only=stablehlo-bytecode 2>&1 | (! grep 'Not Implemented')

// This test compares the output from `stablehlo-opt` of this file, to a round
// trip of the a bytecoded version of this file. If the outputs do not match,
// the test will fail.
//
// Additionally this test will fail if any ops are not implemented on read or
// write. This is accomplished by calling `stablehlo-opt` with the
// `-debug-only=stablehlo-bytecode` trace enabled. If any type or attr is not
// implemented, a message '***Not Implemented' is logged. If there are no logs
// containing 'Not Implemented' the test will pass. This is tested for read and
// write.

func.func @test_add(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

func.func @test_after_all(%arg0: !stablehlo.token, %arg1: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.after_all"(%arg0, %arg1) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

func.func @test_all_gather(%arg0: tensor<128x32xf32>) -> tensor<128x128xf32> {
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    shard_count = 4,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<128x32xf32>) -> tensor<128x128xf32>
  func.return %0 : tensor<128x128xf32>
}

func.func @test_all_reduce(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "stablehlo.all_reduce"(%arg0) ({
  // Perform max reduction inside the region
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = stablehlo.maximum %lhs, %rhs : tensor<f32>
    "stablehlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 2
    >
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// Test non-uniform sized replica groups.
func.func @test_all_reduce2(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "stablehlo.all_reduce"(%arg0) ({
  // Perform max reduction inside the region
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = stablehlo.maximum %lhs, %rhs : tensor<f32>
    "stablehlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, -1], [1, 3, 5, 6]]> : tensor<2x4xi64>,
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 2
    >
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

func.func @test_all_reduce3(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "stablehlo.all_reduce"(%arg0) ({
  // Perform max reduction inside the region
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = stablehlo.maximum %lhs, %rhs : tensor<f32>
    "stablehlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 2
    >,
    use_global_device_ids
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

func.func @test_batch_norm_grad(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>> {
  %0:3 = "stablehlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  %1 = "stablehlo.tuple"(%0#0, %0#1, %0#2) : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
  func.return %1 : tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
}

func.func @test_batch_norm_training(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>> {
  %0:3 = "stablehlo.batch_norm_training" (%input, %scale, %offset) {epsilon = 0.001 : f32, feature_index = 3 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  %1 = "stablehlo.tuple"(%0#0, %0#1, %0#2) : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
  func.return %1 : tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
}

func.func @test_bitcast_convert(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  %0 = "stablehlo.bitcast_convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

func.func @test_broadcast(%arg0: tensor<4xi32>) -> tensor<1x2x3x4xi32> {
      %0 = "stablehlo.broadcast"(%arg0) {broadcast_sizes = dense<[1,2,3]> : tensor<3xi64>} : (tensor<4xi32>) -> tensor<1x2x3x4xi32>
  func.return %0 : tensor<1x2x3x4xi32>
}

func.func @test_broadcast_in_dim(%arg0: tensor<1xf32>) -> tensor<1x10xf32> {
  %result = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<1xf32>) -> tensor<1x10xf32>
  func.return %result : tensor<1x10xf32>
}

func.func @test_call_add(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = func.call @callee_add(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = func.call @callee_add(%0, %0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

func.func @callee_add(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

func.func @test_call_add_mult(%arg0: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
  %0:2 = func.call @callee_add_mult(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)
  func.return %0#0, %0#1 : tensor<4xi32>, tensor<4xi32>
}

func.func @callee_add_mult(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0, %1 : tensor<4xi32>, tensor<4xi32>
}

func.func @test_cast(%arg0: tensor<4xi32>) -> tensor<*xi32> {
  %0 = "stablehlo.not"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  %1 = tensor.cast %0 : tensor<4xi32> to tensor<*xi32>
  func.return %1 : tensor<*xi32>
}

func.func @test_collective_permute(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %0 = "stablehlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}

func.func @test_collective_permute_channel_handle(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %0 = "stablehlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}

func.func @test_concatenate(%arg0 : tensor<5x2xf32>,
           %arg1 : tensor<5x5xf32>,
           %arg2 : tensor<5x7xf32>) -> tensor<5x14xf32> {
  %result = "stablehlo.concatenate"(%arg0, %arg1, %arg2) {
    dimension = 1 : i64
  } : (tensor<5x2xf32>, tensor<5x5xf32>, tensor<5x7xf32>) -> tensor<5x14xf32>
  func.return %result : tensor<5x14xf32>
}

func.func @test_constants() {
  %cst = arith.constant dense<1> : tensor<i64>
  %cst_0 = arith.constant dense<
    [[[[1.000000e+00]], [[2.000000e+00]]], [[[3.000000e+00]], [[4.000000e+00]]]]
    > : tensor<2x2x1x1xf32>
  %cst_1 = arith.constant dense<1> : tensor<1xi32>
  %cst_2 = arith.constant dense<1> : tensor<10xi32>
  %cst_3 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %cst_4 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %cst_5 = arith.constant dense<[[3, 2], [1, 4]]> : tensor<2x2xi32>
  %cst_6 = arith.constant dense<[[1, 2], [4, 8]]> : tensor<2x2xui32>
  %cst_7 = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xbf16>
  %cst_8 = arith.constant dense<[1.0e+00, -4.0e+00, -65504.0e+00, 1.5625e-02]> : tensor<4xf16>
  %cst_9 = arith.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
  %cst_10 = arith.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
  func.return
}

func.func @test_convolution(%arg0 : tensor<100x26x26x32xf32>, %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  %result = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// Test convolution i8xi8 -> i32.
func.func @test_convolution2(%arg0 : tensor<100x26x26x32xi8>, %arg1 : tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32> {
  %result = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<100x26x26x32xi8>, tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32>
  func.return %result : tensor<100x28x28x1xi32>
}

// Test convolution with window reversal.
func.func @test_convolution3(%arg0 : tensor<100x26x26x32xi8>, %arg1 : tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32> {
  %result = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>,
    window_reversal = dense<1> : tensor<2xi1>
  } : (tensor<100x26x26x32xi8>, tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32>
  func.return %result : tensor<100x28x28x1xi32>
}

func.func @test_convert(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

func.func @test_create_token() -> !stablehlo.token {
  %0 = "stablehlo.create_token"() : () -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

func.func @test_cross_replica_sum(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = stablehlo.constant dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi32>
  %1 = "stablehlo.cross-replica-sum"(%arg0) {replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>} : (tensor<10xf32>) -> tensor<10xf32>
  func.return %1 : tensor<10xf32>
}

func.func @test_custom_call(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    has_side_effect = false,
    backend_config = "",
    api_version = 2 : i32,
    called_computations = [@foo],
    operand_layouts = [dense<> : tensor<0xindex>],
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [],
                                 operand_index = 0,
                                 operand_tuple_indices = []>],
    result_layouts = [dense<> : tensor<0xindex>]
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// The following op sharding is used:
// Proto debug string:
//   type: OTHER
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
// Serialized string:
//   "\08\03\1A\02\01\02\22\02\00\01"
func.func @test_custom_call2(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = "stablehlo.custom_call"(%arg0) {backend_config = "", call_target_name = "Sharding", stablehlo.sharding = "\08\03\1A\02\01\02\22\02\00\01"} : (tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

func.func @test_custom_call3(%arg0: tensor<2x3xf32>, %arg1: tensor<5x5xf32>) -> tensor<2x3xf32> {
        %0 = "stablehlo.custom_call"(%arg0, %arg1) {
    call_target_name = "callee_foo",
    called_computations = [@callee_foo]
  } : (tensor<2x3xf32>, tensor<5x5xf32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

func.func @callee_foo (%arg0: tensor<2x3xf32>, %arg1: tensor<5x5xf32>) -> tensor<2x3xf32> {
  func.return %arg0 : tensor<2x3xf32>
}

// Test dot i8xi8 -> i64
func.func @test_dot(%arg0: tensor<3xi8>, %arg1: tensor<3xi8>) -> tensor<i64> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<3xi8>, tensor<3xi8>) -> tensor<i64>
  func.return %0 : tensor<i64>
}

// Test dot i8xi8 -> i32.
func.func @test_dot_general(%arg0: tensor<2x2x2xi8>, %arg1: tensor<2x2x3xi8>) -> tensor<2x2x3xi32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = []} : (tensor<2x2x2xi8>, tensor<2x2x3xi8>) -> tensor<2x2x3xi32>
  func.return %0 : tensor<2x2x3xi32>
}

func.func @test_dynamic_slice(%arg: tensor<3x4xi32>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<1x4xi32> {
  %0 = "stablehlo.dynamic_slice"(%arg, %start1, %start2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

func.func @test_einsum(%arg0: tensor<3x4xi32>, %arg1: tensor<4x5xi32>) -> tensor<3x5xi32> {
  // Simple einsum is lowered to HLO dot op.
    %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "ab,bc->ac"} : (tensor<3x4xi32>, tensor<4x5xi32>) -> tensor<3x5xi32>
  func.return %0 : tensor<3x5xi32>
}

func.func @test_fft(%arg0: tensor<3x9xf32>) -> tensor<3x5xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {fft_length = dense<9> : tensor<1xi64>, fft_type = #stablehlo<fft_type RFFT>} : (tensor<3x9xf32>) -> tensor<3x5xcomplex<f32>>
  func.return %0 : tensor<3x5xcomplex<f32>>
}

func.func @test_gather(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>) -> tensor<10x300xf32> {
                    %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 1,
      offset_dims = [1],
      start_index_map = [0,1],
    >,
    indices_are_sorted = true,
    slice_sizes = dense<[1, 1, 300]> : tensor<3xi64>
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>) -> tensor<10x300xf32>
  func.return %0 : tensor<10x300xf32>
}

func.func @test_set_get_dimension_size(%arg: tensor<4x2xf32>, %size: tensor<i32>) -> tensor<i32> {
  %0 = "stablehlo.set_dimension_size"(%arg, %size) {dimension = 1 : i64} : (tensor<4x2xf32>, tensor<i32>) -> tensor<4x2xf32>
  %1 = "stablehlo.get_dimension_size"(%0) {dimension = 1 : i64} : (tensor<4x2xf32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

func.func @test_get_tuple_element(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  %0 = "stablehlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

func.func @test_infeed(%arg0: !stablehlo.token) -> tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !stablehlo.token> {
  %0:3 = "stablehlo.infeed"(%arg0) {infeed_config = "foobar", layout=[[0, 1], [0]]} : (!stablehlo.token) -> (tensor<3x3xi32>, tensor<i1>, !stablehlo.token)
  %1 = "stablehlo.tuple"(%0#0, %0#1) : (tensor<3x3xi32>, tensor<i1>) -> tuple<tensor<3x3xi32>, tensor<i1>>
  %2 = "stablehlo.tuple"(%1, %0#2) : (tuple<tensor<3x3xi32>, tensor<i1>>, !stablehlo.token) -> tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !stablehlo.token>
  func.return %2 : tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !stablehlo.token>
}

func.func @infeed2(%arg0: !stablehlo.token) -> tensor<3x3xi32> {
  %0:2 = "stablehlo.infeed"(%arg0) {infeed_config = "foobar", layout=[[0,1]]} : (!stablehlo.token) -> (tensor<3x3xi32>, !stablehlo.token)
  func.return %0#0 : tensor<3x3xi32>
}

func.func @test_infeed3(%arg0: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.infeed"(%arg0) {infeed_config = "foobar", layout = [], xla_shape = "((), token[])"} : (!stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

func.func @test_iota() -> tensor<1x10xf32> {
  %result = "stablehlo.iota"() {
    iota_dimension = 1 : i64
  } : () -> tensor<1x10xf32>
  func.return %result : tensor<1x10xf32>
}

func.func @test_map(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

func.func @test_map2(%arg0: tensor<4xf32>, %arg1: tensor<4xi32>) -> tensor<4xf32> {
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<i32>):
    "stablehlo.return"(%arg2) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

func.func @test_optimization_barrier(%arg0: tensor<4x4xf32>, %arg1: tensor<3x4xf32>) -> (tensor<4x4xf32>, tensor<3x4xf32>) {
  %0, %1 = "stablehlo.optimization_barrier"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<3x4xf32>) -> (tensor<4x4xf32>, tensor<3x4xf32>)
  func.return %0, %1 : tensor<4x4xf32>, tensor<3x4xf32>
}

func.func @test_outfeed(%data: tensor<3xi32>, %token: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.outfeed"(%data, %token) {outfeed_config = "foobar"} : (tensor<3xi32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

func.func @test_outfeed2(%data1: tensor<3xi32>, %data2: tensor<3xi32>, %token: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.outfeed"(%data1, %data2,  %token) {outfeed_config = "foobar"} : (tensor<3xi32>, tensor<3xi32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

func.func @test_outfeed3(%token: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.outfeed"(%token) {outfeed_config = "foobar"} : (!stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

func.func @test_pad(%arg: tensor<4x6xf32>, %pad: tensor<f32>) -> tensor<13x19xf32> {
  %0 = "stablehlo.pad"(%arg, %pad) {edge_padding_high = dense<[4,5]> : tensor<2xi64>, edge_padding_low = dense<[2,3]> : tensor<2xi64>, interior_padding = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<13x19xf32>
  func.return %0 : tensor<13x19xf32>
}

func.func @test_recv(%token: !stablehlo.token) -> tuple<tensor<3x4xi32>, !stablehlo.token> {
  %0:2 = "stablehlo.recv"(%token) {
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 3  // Host to device channel
    >,
    is_host_transfer = true
  } : (!stablehlo.token) -> (tensor<3x4xi32>, !stablehlo.token)
  %1 = "stablehlo.tuple"(%0#0, %0#1) : (tensor<3x4xi32>, !stablehlo.token) -> tuple<tensor<3x4xi32>, !stablehlo.token>
  func.return %1 : tuple<tensor<3x4xi32>, !stablehlo.token>
}

func.func @test_recv2(%token: !stablehlo.token) -> tuple<tensor<3x4xi32>, !stablehlo.token> {
  %0:2 = "stablehlo.recv"(%token) {
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 1  // Device to device channel
    >,
    is_host_transfer = false
  } : (!stablehlo.token) -> (tensor<3x4xi32>, !stablehlo.token)
  %1 = "stablehlo.tuple"(%0#0, %0#1) : (tensor<3x4xi32>, !stablehlo.token) -> tuple<tensor<3x4xi32>, !stablehlo.token>
  func.return %1 : tuple<tensor<3x4xi32>, !stablehlo.token>
}

func.func @test_recv3(%token: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.recv"(%token) {
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 1  // Device to device channel
    >,
    is_host_transfer = false
  } : (!stablehlo.token) -> (!stablehlo.token)
  func.return %0 : !stablehlo.token
}

func.func @test_reduce(%arg0 : tensor<1x10xf32>, %arg1 : tensor<1x10xi32>, %arg2 : tensor<f32>, %arg3 : tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %result0, %result1 = "stablehlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({
    ^bb0(%fa: tensor<f32>, %ia : tensor<i32>, %fb: tensor<f32>, %ib: tensor<i32>):
      %fmax = "stablehlo.maximum"(%fa, %fb) {} : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %imax = "stablehlo.maximum"(%ia, %ib) {} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%fmax, %imax) : (tensor<f32>, tensor<i32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<1x10xi32>, tensor<f32>, tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>)
  func.return %result0, %result1 : tensor<1xf32>, tensor<1xi32>
}

func.func @test_reduce_precision(%arg: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "stablehlo.reduce_precision"(%arg) {exponent_bits = 8 : i32, mantissa_bits = 10 : i32} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}

func.func @test_reduce_scatter(%arg0: tensor<10xf32>) -> tensor<5xf32> {
  %0 = "stablehlo.reduce_scatter"(%arg0) ({
  // Perform max reduction inside the region
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = stablehlo.maximum %lhs, %rhs : tensor<f32>
    "stablehlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>,
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 2
    >,
    scatter_dimension = 0 : i64
  } : (tensor<10xf32>) -> tensor<5xf32>
  func.return %0 : tensor<5xf32>
}

func.func @test_reduce_scatter2(%arg0: tensor<10xf32>) -> tensor<5xf32> {
  %0 = "stablehlo.reduce_scatter"(%arg0) ({
  // Perform max reduction inside the region
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = stablehlo.maximum %lhs, %rhs : tensor<f32>
    "stablehlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>,
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 2
    >,
    scatter_dimension = 0 : i64
  } : (tensor<10xf32>) -> tensor<5xf32>
  func.return %0 : tensor<5xf32>
}

func.func @test_reduce_window(%arg0: tensor<2x17x31x7xi32>) -> tensor<2x5x8x7xi32> {
  %0 = stablehlo.constant dense<-2147483648> : tensor<i32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %2 = stablehlo.maximum %arg1, %arg2 : tensor<i32>
    "stablehlo.return"(%2) : (tensor<i32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 4, 4, 1]> : tensor<4xi64>,
    padding = dense<[[0, 0], [2, 0], [0, 2], [0, 0]]> : tensor<4x2xi64>,
    base_dilations = dense<[1, 1, 1, 1]> : tensor<4xi64>,
    window_dilations = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<2x17x31x7xi32>, tensor<i32>) -> tensor<2x5x8x7xi32>
  func.return %1 : tensor<2x5x8x7xi32>
}

func.func @test_reduce_window2(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) -> (tensor<2x2xf32>, tensor<2x2xi32>) {
  %0:2 = "stablehlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>, %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = stablehlo.add %a0, %b0 : tensor<f32>
              %3 = stablehlo.add %a1, %b1 : tensor<i32>
              "stablehlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> } : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

func.func @test_reshape(%arg0: tensor<2xf32>) -> tensor<1x2xf32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<2xf32>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
}

func.func @test_reverse(%arg0 : tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %result = "stablehlo.reverse"(%arg0) {
    dimensions = dense<[1,2]> : tensor<2xi64>
  } : (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>
  func.return %result : tensor<10x11x12x13xf32>
}

func.func @test_rng(%mu: tensor<f32>, %sigma: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = stablehlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  %0 = "stablehlo.rng"(%mu, %sigma, %shape) {rng_distribution = #stablehlo<rng_distribution NORMAL>} : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

func.func @test_rng_uniform() -> tensor<2x3x5xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %2 = stablehlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  %3 = "stablehlo.rng"(%0, %1, %2) {rng_distribution = #stablehlo<rng_distribution UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %3 : tensor<2x3x5xf32>
}

func.func @test_rng_bit_generator(%arg: tensor<3xui64>) -> tuple<tensor<3xui64>, tensor<2x2xui32>> {
  %0:2 = "stablehlo.rng_bit_generator"(%arg) {rng_algorithm = #stablehlo<rng_algorithm PHILOX>} : (tensor<3xui64>) -> (tensor<3xui64>, tensor<2x2xui32>)
  %1 = "stablehlo.tuple"(%0#0, %0#1) : (tensor<3xui64>, tensor<2x2xui32>) -> tuple<tensor<3xui64>, tensor<2x2xui32>>
  func.return %1 : tuple<tensor<3xui64>, tensor<2x2xui32>>
}

func.func @test_scatter(%input_tensor: tensor<200x100x300xf32>, %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) -> tensor<200x100x300xf32> {
  %0 = "stablehlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = stablehlo.add %lhs, %rhs : tensor<f32>
    "stablehlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

func.func @test_scatter2(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi64>, %arg2: tensor<10x300xf32>) -> (tensor<200x100x300xf32>, tensor<200x100x300xf32>) {
    %0:2 = "stablehlo.scatter"(%arg0, %arg0, %arg1, %arg2, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
      %3 = stablehlo.add %arg5, %arg6 : tensor<f32>
      "stablehlo.return"(%2, %3) : (tensor<f32>, tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false} : (tensor<200x100x300xf32>, tensor<200x100x300xf32>, tensor<10x2xi64>, tensor<10x300xf32>, tensor<10x300xf32>) -> (tensor<200x100x300xf32>, tensor<200x100x300xf32>)
    return %0#0, %0#1 : tensor<200x100x300xf32>, tensor<200x100x300xf32>
  }

func.func @test_select(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

func.func @test_select_and_scatter(%arg0: tensor<10x24x24x64xf32>, %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "stablehlo.select_and_scatter"(%arg0, %arg1, %0) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = "stablehlo.compare"(%arg3, %arg4) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  func.return %1 : tensor<10x24x24x64xf32>
}

func.func @test_send(%arg: tensor<3x4xi32>, %token: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.send"(%arg, %token) {
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 2  // Device to host channel
    >,
    is_host_transfer = true
  } : (tensor<3x4xi32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

func.func @test_send2(%arg: tensor<3x4xi32>, %token: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.send"(%arg, %token) {
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 1  // Device to device channel
    >,
    is_host_transfer = false
  } : (tensor<3x4xi32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

func.func @test_send3(%token: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.send"(%token) {
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 1
    >,
    is_host_transfer = false
  } : (!stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

func.func @test_set_dimension_size(%arg: tensor<4x4xf32>, %size: tensor<i32>) -> tensor<4x4xf32> {
  %0 = "stablehlo.set_dimension_size"(%arg, %size) {dimension = 1 : i64} : (tensor<4x4xf32>, tensor<i32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

func.func @test_shift(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  %0 = stablehlo.shift_left %arg0, %arg1 : tensor<4xi32>
  %1 = stablehlo.shift_right_arithmetic %arg0, %arg1 : tensor<4xi32>
  %2 = stablehlo.shift_right_logical %arg0, %arg1 : tensor<4xi32>
  func.return %0, %1, %2 : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

func.func @test_slice(%arg: tensor<3x4xi32>) -> tensor<1x2xi32> {
  %0 = "stablehlo.slice"(%arg) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

func.func @test_sort(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

func.func @test_sort2(%input0: tensor<16x16xf32>) {
  %0 = "stablehlo.sort"(%input0) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>) -> (tensor<16x16xf32>)
  func.return
}

func.func @test_transpose(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0 : tensor<2x1x4x3xi32>
}

func.func @test_triangular_solve(%arg0: tensor<4x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  func.return %0 : tensor<4x3xf32>
}

func.func @test_triangular_solve2(%arg0: tensor<4x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {left_side = false, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false} : (tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  func.return %0: tensor<3x4xf32>
}

func.func @test_tuple(%arg0: tensor<f32>, %arg1 : tensor<i32>) -> tuple<tensor<f32>, tensor<i32>> {
  %result = "stablehlo.tuple"(%arg0, %arg1) {} : (tensor<f32>, tensor<i32>) -> tuple<tensor<f32>, tensor<i32>>
  func.return %result : tuple<tensor<f32>, tensor<i32>>
}

func.func @test_unary(%arg_f32: tensor<4xf32>, %arg_i32: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xi32>, tensor<4xi32>) {
  %expm1 = "stablehlo.exponential_minus_one"(%arg_f32) : (tensor<4xf32>) -> tensor<4xf32>
  %log1p = "stablehlo.log_plus_one"(%arg_f32) : (tensor<4xf32>) -> tensor<4xf32>
  %not = "stablehlo.not"(%arg_i32) : (tensor<4xi32>) -> tensor<4xi32>
  %popcnt = "stablehlo.popcnt"(%arg_i32) : (tensor<4xi32>) -> tensor<4xi32>
  func.return %expm1, %log1p, %not, %popcnt : tensor<4xf32>, tensor<4xf32>, tensor<4xi32>, tensor<4xi32>
}

func.func @test_unary_abs(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f64>>) -> (tensor<2xf32>, tensor<2xf64>) {
  %0 = "stablehlo.abs"(%arg0) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %1 = "stablehlo.abs"(%arg1) : (tensor<2xcomplex<f64>>) -> (tensor<2xf64>)
  func.return %0, %1 : tensor<2xf32>, tensor<2xf64>
}

func.func @test_unary_cbrt(%arg: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "stablehlo.cbrt"(%arg) : (tensor<3x4xf32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}

func.func @test_unary_round_nearest_even(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "stablehlo.round_nearest_even"(%arg0) {} : (tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
}

func.func @test_xor(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
      %0 = stablehlo.xor %arg0, %arg1 : tensor<4xi1>
    func.return %0 : tensor<4xi1>
}

// Attribute Tests:
func.func @test_attr_frontend_attributes(%arg: tensor<3x4xf32>, %token: !stablehlo.token) -> tuple<tensor<3x4xf32>, !stablehlo.token> {
  %0 = "stablehlo.send"(%arg, %token) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>, is_host_transfer = true, stablehlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "channel_dtoh_0"}} : (tensor<3x4xf32>, !stablehlo.token) -> !stablehlo.token
  %1:2 = "stablehlo.recv"(%0) {channel_handle = #stablehlo.channel_handle<handle = 2, type = 3>, is_host_transfer = true, stablehlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "channel_htod_0"}} : (!stablehlo.token) -> (tensor<3x4xf32>, !stablehlo.token)
  %2 = "stablehlo.tuple"(%1#0, %1#1) : (tensor<3x4xf32>, !stablehlo.token) -> tuple<tensor<3x4xf32>, !stablehlo.token>
  func.return %2 : tuple<tensor<3x4xf32>, !stablehlo.token>
}

func.func @test_attr_frontend_attributes_empty(%arg: tensor<3x4xf32>, %token: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.send"(%arg, %token) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>, is_host_transfer = true, stablehlo.frontend_attributes = {}} : (tensor<3x4xf32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

func.func @test_attr_result_alias (%arg0: tuple<tensor<f32>>
    {stablehlo.result_alias = #stablehlo.result_alias<
      tuple_indices = [0],
      result_index = [0, 0],
      must_alias>}
    ) -> (tuple<tensor<f32>>) {
  %0 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<f32>>) -> tensor<f32>
  %1 = stablehlo.tuple %0 : tuple<tensor<f32>>
  func.return %1 : tuple<tensor<f32>>
}

func.func @test_attr_type_extensions(%arg: tensor<4xf32>, %size: tensor<i32>) -> tensor<?xf32> {
  %0 = "stablehlo.set_dimension_size"(%arg, %size) {dimension = 0 : i64} : (tensor<4xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.type_extensions<bounds = [4]>>
  %1 = tensor.cast %0 : tensor<?xf32, #stablehlo.type_extensions<bounds = [4]>> to tensor<?xf32>
  func.return %1 : tensor<?xf32>
}
