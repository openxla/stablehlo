// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file | FileCheck %s


// CHECK-LABEL: func @broadcast_add_quantized
func.func @broadcast_add_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>> {
  %0 = "chlo.broadcast_add"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
}

// -----

func.func @broadcast_add_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i16:f32, 2.0:15>> {
  // expected-error @+1{{'chlo.broadcast_add' op requires compatible element types for all operands and results}}
  %0 = "chlo.broadcast_add"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i16:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i16:f32, 2.0:15>>
}

// -----

// CHECK-LABEL: func @broadcast_max_quantized
func.func @broadcast_max_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>> {
  %0 = "chlo.broadcast_maximum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
}

// -----

func.func @broadcast_max_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i16:f32, 2.0:15>> {
  // expected-error @+1{{'chlo.broadcast_maximum' op requires compatible element types for all operands and results}}
  %0 = "chlo.broadcast_maximum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i16:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i16:f32, 2.0:15>>
}

// -----

func.func @broadcast_max_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f16, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>> {
  // expected-error @+1{{'chlo.broadcast_maximum' op requires compatible element types for all operands and results}}
  %0 = "chlo.broadcast_maximum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f16, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
}

// -----

// CHECK-LABEL: func @broadcast_min_quantized
func.func @broadcast_min_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>> {
  %0 = "chlo.broadcast_minimum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
}

// -----

func.func @broadcast_min_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i16:f32, 2.0:15>> {
  // expected-error @+1{{'chlo.broadcast_minimum' op requires compatible element types for all operands and results}}
  %0 = "chlo.broadcast_minimum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i16:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i16:f32, 2.0:15>>
}

// -----

func.func @broadcast_min_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f16, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>> {
  // expected-error @+1{{'chlo.broadcast_minimum' op requires compatible element types for all operands and results}}
  %0 = "chlo.broadcast_minimum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f16, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
}

// -----

func.func @constant_like(%arg0: tensor<1x2xi64>) -> (tensor<1x2xi32>) {
  // expected-error @+1 {{'chlo.constant_like' op value's type doesn't match element return type}}
  %0 = "chlo.constant_like"(%arg0) { value = 3.2 : f32 } : (tensor<1x2xi64>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func @minimum_broadcast_shapes
func.func @minimum_broadcast_shapes(%lhs: tensor<?xindex>, %rhs: tensor<?xindex>)
    -> (tensor<?xindex>, tensor<?xindex>) {
  %0, %1 = chlo.minimum_broadcast_shapes %lhs, %rhs :
      tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>, tensor<?xindex>
  func.return %0, %1 : tensor<?xindex>, tensor<?xindex>
}

// -----

func.func @minimum_broadcast_shapes_mismatch_operand_and_result_count(%lhs: tensor<?xindex>, %rhs: tensor<?xindex>) {
  // expected-error @+1{{number of operand shapes (2) does not match number of result shapes (1)}}
  %0 = chlo.minimum_broadcast_shapes %lhs, %rhs :
      tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
  func.return
}

// -----

func.func @minimum_broadcast_shapes_one_operand(%arg: tensor<?xindex>) {
  // expected-error @+1{{number of operand shapes (1) should be >= 2}}
  %0 = chlo.minimum_broadcast_shapes %arg : tensor<?xindex> -> tensor<?xindex>
  func.return
}

// -----

func.func @rank_specialization_cluster(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>,
    %arg2 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "chlo.rank_specialization_cluster"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg0_ : tensor<*xf32>, %arg1_ : tensor<*xf32>, %arg2_ : tensor<*xf32>):
    %1 = chlo.broadcast_multiply %arg0_, %arg1_
        : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %2 = chlo.broadcast_add %1, %arg2_
        : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "chlo.rank_specialization_cluster_yield"(%2) : (tensor<*xf32>) -> ()
  }) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @rank_specialization_cluster(%arg0 : tensor<*xf32>,
    %arg1 : tensor<*xf32>) -> tensor<*xf32> {
  // expected-error @+1{{source has 2 operands, but target successor needs 1}}
  %0 = "chlo.rank_specialization_cluster"(%arg0, %arg1) ({
  ^bb0(%arg0_ : tensor<*xf32>, %arg1_ : tensor<*xf32>):
    "chlo.rank_specialization_cluster_yield"(%arg0_, %arg1_)
        : (tensor<*xf32>, tensor<*xf32>) -> ()
  }) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @rank_specialization_cluster(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  // expected-error @+1{{block argument types must match operand types}}
  %0 = "chlo.rank_specialization_cluster"(%arg0) ({
  ^bb0(%arg0_ : tensor<*xf32>, %arg1_ : tensor<*xf32>):
    "chlo.rank_specialization_cluster_yield"(%arg0_) : (tensor<*xf32>) -> ()
  }) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @rank_specialization_cluster(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>,
    %arg2 : tensor<*xf32>) -> tensor<*xf32> {
  // expected-error @+1{{nested ops must not depend on implicit operands}}
  %0 = "chlo.rank_specialization_cluster"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg0_ : tensor<*xf32>, %arg1_ : tensor<*xf32>, %arg2_ : tensor<*xf32>):
    %1 = chlo.broadcast_multiply %arg0_, %arg1_
        : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %2 = chlo.broadcast_add %1, %arg2
        : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "chlo.rank_specialization_cluster_yield"(%2) : (tensor<*xf32>) -> ()
  }) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @top_k(%arg0 : tensor<f32>) {
  // expected-error @+2 {{failed to infer returned types}}
  // @expected-error @+1{{operand's rank must be at least 1}}
  %0:2 = chlo.top_k(%arg0, k=8) : tensor<f32> -> (tensor<8xf32>, tensor<8xi32>)
  return
}

// -----

func.func @top_k(%arg0 : tensor<4xf32>) {
  // expected-error @+2 {{failed to infer returned types}}
  // @expected-error @+1{{operand's last dimension must be at least 8}}
  %0:2 = chlo.top_k(%arg0, k=8) : tensor<4xf32> -> (tensor<8xf32>, tensor<8xi32>)
  return
}

// -----

func.func @top_k_1d(%arg0 : tensor<16xf32>) {
  %0:2 = chlo.top_k(%arg0, k=8) : tensor<16xf32> -> (tensor<8xf32>, tensor<8xi32>)
  return
}

// -----

func.func @top_k_nd(%arg0 : tensor<16x16xf32>) {
  %0:2 = chlo.top_k(%arg0, k=8) : tensor<16x16xf32> -> (tensor<16x8xf32>, tensor<16x8xi32>)
  return
}

// -----

func.func @top_k_unbounded(%arg0 : tensor<?x16x?xf32>) {
  %0:2 = chlo.top_k(%arg0, k=8) : tensor<?x16x?xf32> -> (tensor<?x16x8xf32>, tensor<?x16x8xi32>)
  return
}

// -----

func.func @top_k_bounded(%arg0 : tensor<?x?x?xf32, #stablehlo.bounds<?, 16, 16>>) {
  %0:2 = chlo.top_k(%arg0, k=8) : tensor<?x?x?xf32, #stablehlo.bounds<?, 16, 16>> -> (tensor<16x?x8xf32, #stablehlo.bounds<?, 16, ?>>, tensor<16x?x8xi32, #stablehlo.bounds<?, 16, ?>>)
  return
}

// -----

func.func @top_k_unranked(%arg0 : tensor<*xf32>) {
  %0:2 = chlo.top_k(%arg0, k=8) : tensor<*xf32> -> (tensor<*xf32>, tensor<*xi32>)
  return
}

// -----

func.func @erf_inv(%arg0 : tensor<16x16xf32>) {
  %0 = chlo.erf_inv %arg0 : tensor<16x16xf32> -> tensor<16x16xf32>
  return
}
