// Test Zero Operand Ops
module {
  func.func @main() -> !stablehlo.token {
    %0 = "stablehlo.replica_id"() : () -> tensor<ui32>
    %2 = "stablehlo.create_token"() : () -> !stablehlo.token
    return %2 : !stablehlo.token
  }
}

// Test Zero Output Ops
module {
  func.func @main() -> !stablehlo.token {
    %1 = "stablehlo.constant"() {value = dense<[2, 3, 5]> : tensor<3xi64>} : () -> tensor<3xi64>
    "stablehlo.trace"(%1) {tag = "This is a random test"} : (tensor<3xi64>) -> ()
    "stablehlo.return"(%1) : (tensor<3xi64>) -> ()
  }
}

module {
  func.func @main() -> !stablehlo.token {
    %1 = "stablehlo.constant"() {value = dense<[2, 3, 5]> : tensor<3xi64>} : () -> tensor<3xi64>
    "stablehlo.return"(%1, %1) : (tensor<3xi64>, tensor<3xi64>) -> ()
  }
}

// Test Zero Attribute Ops
module {
  func.func @main(%arg0 : tensor<4xf32>,
                  %arg1 : !stablehlo.token,
		  %arg2 : tensor<4xi32>,
		  %arg3 : index) -> !stablehlo.token {
    %1 = "stablehlo.clamp"(%arg0, %arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %2 = "stablehlo.complex"(%arg0, %arg0) {} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
    %3 = "stablehlo.compute_reshape_shape"(%arg3, %arg2) : (index, tensor<4xi32>) -> tensor<4xi32>
    %5 = "stablehlo.cstr_reshapable"(%arg3, %arg2) : (index, tensor<4xi32>) -> !shape.witness
    %6:2 = "stablehlo.optimization_barrier"(%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
    %7 = "stablehlo.uniform_quantize"(%arg0) : (tensor<4xf32>) -> tensor<4x!quant.uniform<ui8:f32, 34.0:16>>
    %8 = "stablehlo.uniform_dequantize"(%7) : (tensor<4x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<4xf32>
    %9 = "stablehlo.after_all"(%arg1, %arg1) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token
    "stablehlo.return"(%arg1) : (!stablehlo.token) -> ()
  }

  func.func @dynamic(%arg0 : tensor<?x?xf32>,
                     %arg1 : tensor<f32>,
		     %arg2 : tensor<2xindex>,
		     %arg3 : tensor<?xf32>,
		     %arg4 : tensor<28x1x100xf32>,
		     %arg5 : tensor<1x1x100xf32>,
		     %arg6 : tensor<i32>,
		     %arg7 : tensor<10xi32>,
		     %arg8 : tensor<1xi32>,
		     %arg9 : tensor<1xindex>,
		     %arg10 : tensor<3x4xi32>) -> tensor<?x?xf32> {
    %10 = "stablehlo.dynamic_pad"(%arg0, %arg1, %arg2, %arg2, %arg2) : (tensor<?x?xf32>, tensor<f32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<?x?xf32>
    %11 = "stablehlo.dynamic_reshape"(%arg3, %arg2) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
    %12 = "stablehlo.dynamic_update_slice"(%arg4, %arg5, %arg6, %arg6, %arg6) : (tensor<28x1x100xf32>, tensor<1x1x100xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<28x1x100xf32>
    %13 = "stablehlo.real_dynamic_slice"(%arg7, %arg8, %arg8, %arg8) : (tensor<10xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>

    %19 = "stablehlo.dynamic_iota"(%arg9) {iota_dimension = 0 : i64} : (tensor<1xindex>) -> tensor<4xi32>
    %20 = "stablehlo.dynamic_slice"(%arg10, %arg6, %arg6) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i32>, tensor<i32>) -> tensor<1x4xi32>
    "stablehlo.return"(%arg0) : (tensor<?x?xf32>) -> ()

  }
}

// Test converter functions
module {
  func.func @main(%arg0 : tensor<2xf32>) -> () {
    %0 = "stablehlo.convert"(%arg0) : (tensor<2xf32>) -> tensor<2xf64>
    %1 = "stablehlo.reshape"(%arg0) : (tensor<2xf32>) -> tensor<1x2xf32>
    %2 = "stablehlo.bitcast_convert"(%arg0) : (tensor<2xf32>) -> tensor<2xi32>
    "stablehlo.return"() : () -> ()
  }
}

// Test unary operations
module {
  func.func @main(%arg0 : tensor<2xi32>, %arg1 : tensor<2xf32>) -> () {
    %0 = "stablehlo.abs"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    %1 = "stablehlo.ceil"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %2 = "stablehlo.count_leading_zeros"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    %3 = "stablehlo.convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
    %4 = "stablehlo.cosine"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %5 = "stablehlo.exponential"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %6 = "stablehlo.exponential_minus_one"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %7 = "stablehlo.floor"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %8 = "stablehlo.imag"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %9 = "stablehlo.is_finite"(%arg1) : (tensor<2xf32>) -> tensor<2xi1>
    %10 = "stablehlo.log"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %11 = "stablehlo.log_plus_one"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %12 = "stablehlo.logistic"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %13 = "stablehlo.not"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    %14 = "stablehlo.negate"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %15 = "stablehlo.popcnt"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    %16 = "stablehlo.real"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %17 = "stablehlo.round_nearest_afz"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %18 = "stablehlo.round_nearest_even"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %19 = "stablehlo.sign"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %20 = "stablehlo.sine"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %21 = "stablehlo.sqrt"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    %22 = "stablehlo.tanh"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    "stablehlo.return"(%0) : (tensor<2xi32>) -> ()
  }
}

// Test Binary ops
module {
  func.func @main(%arg0: tensor<2xi1>, %arg1 : tensor<2xf32>) -> tensor<2xi1> {
    %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    %1 = "stablehlo.and"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    %2 = "stablehlo.atan2"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    %3 = "stablehlo.divide"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    %4 = "stablehlo.maximum"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %5 = "stablehlo.minimum"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %6 = "stablehlo.multiply"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %7 = "stablehlo.or"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    %8 = "stablehlo.power"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %9 = "stablehlo.remainder"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %10 = "stablehlo.shift_left"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %11 = "stablehlo.shift_right_arithmetic"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %12 = "stablehlo.shift_right_logical"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %13 = "stablehlo.subtract"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %14 = "stablehlo.xor"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    func.return %0 : tensor<2xi1>
  }
}

// Single Attr Ops
module {
  func.func @main(%arg0 : tensor<1x2xf32>,
                  %arg1 : tensor<3xi32>,
                  %arg2 : tensor<2x2xf32>,
		  %arg3 : tensor<4xf32>,
		  %arg4 : tensor<i32>) -> () {
    %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x2xf32>) -> tensor<1x2x3xf32>
    %1 = "stablehlo.broadcast"(%arg1) {broadcast_sizes = dense<[1, 2]> : tensor<2xi64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
    %2 = "stablehlo.cholesky"(%arg2) { lower = true } : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %3 = "stablehlo.collective_permute"(%arg0) {source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %4 = "stablehlo.concatenate"(%arg0, %arg0) {dimension = 1 : i64} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x4xf32>
    %5 = "stablehlo.constant"() {value = dense<[[1,2], [3,4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %6 = "stablehlo.cross-replica-sum"(%arg0) {replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %7 = "stablehlo.dot"(%arg3, %arg3) {precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
    %71 = "stablehlo.dot"(%arg3, %arg3) {} : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
    %8 = "stablehlo.set_dimension_size"(%arg0, %arg4) {dimension = 1 : i64} : (tensor<1x2xf32>, tensor<i32>) -> tensor<1x2xf32>
    %9 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<1x2xf32>) -> tensor<i32>
    %t  = "stablehlo.tuple"(%arg4, %arg4) : (tensor<i32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>>
    %10 = "stablehlo.get_tuple_element"(%t) {index = 0 : i32} : (tuple<tensor<i32>, tensor<i32>>) -> tensor<i32>
    %11 = "stablehlo.iota"() {iota_dimension = 1 : i64}  : () -> tensor<1x10xf32>
    %12 = "stablehlo.reverse"(%arg0) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %13, %14 = "stablehlo.rng_bit_generator"(%arg0) {rng_algorithm = #stablehlo.rng_algorithm<PHILOX>} : (tensor<1x2xf32>) -> (tensor<1x2xf32>, tensor<2x2xui32>)
    %15 = "stablehlo.rng"(%7, %7, %arg1) {rng_distribution = #stablehlo.rng_distribution<NORMAL>} : (tensor<f32>, tensor<f32>, tensor<3xi32>) -> tensor<2x3x5xf32>
    %16 = "stablehlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1x2xf32>) -> tensor<2x1xf32>
    %17 = "stablehlo.einsum"(%arg0, %arg0) {einsum_config = "ab,bc->ac"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xi16>
    %18 = "stablehlo.unary_einsum"(%arg0) {einsum_config = "ab->aa"} : (tensor<1x2xf32>) -> tensor<1x1xf32>
     "stablehlo.return"() : () -> ()
  }
}

// Test Custom Calls
module {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<5x5xf32>) -> tensor<1x2x3xf32> {
    %0 = "stablehlo.custom_call"(%arg0, %arg1) {backend_config = "bar", call_target_name = "foo", has_side_effect = true} : (tensor<2x3xf32>, tensor<5x5xf32>) -> tensor<1x2x3xf32>
    %1 = "stablehlo.custom_call"(%arg0) {backend_config = "", call_target_name = "Sharding", stablehlo.sharding = "\08\03\1A\02\01\02\22\02\00\01"} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %3 = "stablehlo.custom_call"(%arg0, %arg1) {call_target_name = "foo", called_computations = [@foo] } : (tensor<2x3xf32>, tensor<5x5xf32>) -> tensor<2x3xf32>
    "stablehlo.return"(%0) : (tensor<1x2x3xf32>) -> ()
  }
}

// Multiple Attr Ops
module {
  func.func @main(%arg0 : tensor<3x9xf32>) -> () {
    %0 = "stablehlo.reduce_precision"(%arg0) {exponent_bits = 8 : i32, mantissa_bits = 10 : i32} : (tensor<3x9xf32>) -> tensor<3x9xf32>
    %1 = "stablehlo.fft"(%arg0) {fft_length = dense<9> : tensor<1xi64>, fft_type = #stablehlo<fft_type RFFT>} : (tensor<3x9xf32>) -> tensor<3x5xcomplex<f32>>
    "stablehlo.return"() : () -> ()
  }
}

// Test Select Op
module {
  func.func @main(%arg0 : tensor<i1>,
                  %arg1 : tensor<2x3xi32>,
		  %arg2 : tensor<2x3xi1>) -> () {
    %0 = "stablehlo.select"(%arg0, %arg1, %arg1) : (tensor<i1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
    %1 = "stablehlo.select"(%arg2, %arg1, %arg1) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
    "stablehlo.return"() : () -> ()
  }
}

// Test extension types
module {
  func.func @main(%arg0 : tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3, -1]>>,
		  %arg1 : tensor<i32>) -> () {
    %0 = "stablehlo.set_dimension_size"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3, -1]>>, tensor<i32>) -> tensor<*xf32>
    "stablehlo.return"() : () -> ()
  }
}
#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed"]
}>


module {

func.func @op_encodings(%arg0: tensor<10x20xf32, #CSR>,
                        %arg1: tensor<10x20xf32, #DCSR>) -> tensor<10x20xf32> {
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<10x20xf32, #CSR>,
                                   tensor<10x20xf32, #DCSR>) -> tensor<10x20xf32>
  %1 = "stablehlo.add"(%arg1, %arg1) : (tensor<10x20xf32, #DCSR>,
                                   tensor<10x20xf32, #DCSR>) -> tensor<10x20xf32, #DCSR>

  %2 = "stablehlo.abs"(%arg0) : (tensor<10x20xf32, #CSR>) -> tensor<10x20xf32>
  %3 = "stablehlo.abs"(%arg0) : (tensor<10x20xf32, #CSR>) -> tensor<10x20xf32, #CSR>
  func.return %0 : tensor<10x20xf32>
}
}

module {
  func.func @tuple_ops(%arg0 : tensor<4xf32>) -> () {
    "stablehlo.optimization_barrier"() : () -> ()
    %0 = "stablehlo.optimization_barrier"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
    %1:2 = "stablehlo.optimization_barrier"(%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
    "stablehlo.return"() : () -> ()
  }
func.func @zero_output_ret2(%arg0 : tensor<3xi64>) -> (tensor<3xi64>, tensor<3xi64>) {
  "stablehlo.trace"(%arg0) {tag = "This is a test"} : (tensor<3xi64>) -> ()
  "stablehlo.return"(%arg0, %arg0) : (tensor<3xi64>, tensor<3xi64>) -> ()
}

func.func @zero_output_ret1(%arg0 : tensor<3xi64>) -> (tensor<3xi64>) {
  "stablehlo.return"(%arg0) : (tensor<3xi64>) -> ()
}

func.func @zero_output_ret0(%arg0 : tensor<3xi64>) -> () {
  "stablehlo.return"() : () -> ()
}

func.func @compare_op(%arg0 : tensor<3xi32>) -> () {
   %0 = "stablehlo.compare"(%arg0, %arg0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
   %1 = "stablehlo.compare"(%arg0, %arg0) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  "stablehlo.return"() : () -> ()
}

}
