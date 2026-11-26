// RUN: stablehlo-opt %s --stablehlo-legalize-to-linalg --split-input-file --canonicalize | FileCheck %s

func.func @matching_update_tensor(%arg0: tensor<1x32x32x128xf32>, %arg1: tensor<1x32x1x128xf32>, %arg2: tensor<1x1xi64>) -> tensor<1x32x32x128xf32> {
  // CHECK-NOT: stablehlo.scatter
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[EXT:.*]] = tensor.extract %arg2[%[[ZERO]], %[[ZERO]]] : tensor<1x1xi64>
  // CHECK: %[[IDX:.*]] = arith.index_cast %[[EXT]] : i64 to index
  // CHECK: tensor.insert_slice %arg1 into %arg0[0, 0, %[[IDX]], 0] [1, 32, 1, 128] [1, 1, 1, 1] : tensor<1x32x1x128xf32> into tensor<1x32x32x128xf32>
  %0 = "stablehlo.scatter"(%arg0, %arg2, %arg1) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [0, 1, 3],
      inserted_window_dims = [2],
      scatter_dims_to_operand_dims = [2],
      index_vector_dim = 1>,
    unique_indices = false}> ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    stablehlo.return %arg4 : tensor<f32>
  }) : (tensor<1x32x32x128xf32>, tensor<1x1xi64>, tensor<1x32x1x128xf32>) -> tensor<1x32x32x128xf32>
  return %0 : tensor<1x32x32x128xf32>


}

// -----

func.func @smaller_update_tensor() -> tensor<9x7x5xf64> {
  // CHECK-DAG: %[[scatter_indices:.*]] = tensor.empty() : tensor<1xi32>
  // CHECK-DAG: %[[inputs:.*]] = tensor.empty() : tensor<9x[[dim1:.*]]x[[dim0:.*]]xf64>
  // CHECK-DAG: %[[updates:.*]] = tensor.empty() : tensor<[[dim1]]x[[dim0]]xf64>
  // CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
  %scatter_indices = tensor.empty() : tensor<1xi32>
  %inputs = tensor.empty() : tensor<9x7x5xf64>
  %updates = tensor.empty() : tensor<7x5xf64>

  // CHECK-DAG: %[[ext:.*]] = tensor.extract %[[scatter_indices]][%[[zero]]] : tensor<1xi32>
  // CHECK-DAG: %[[idx:.*]] = arith.index_cast %[[ext]] : i32 to index
  // CHECK-DAG: %[[inserted_slice:.*]] = tensor.insert_slice %[[updates]] into %[[inputs]][%[[idx]], 0, 0] [1, [[dim1]], [[dim0]]] [1, 1, 1] : tensor<7x5xf64> into tensor<9x7x5xf64>

  %3 = "stablehlo.scatter"(%inputs, %scatter_indices, %updates) <{
    indices_are_sorted = true,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [0, 1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0]>,
    unique_indices = true}> ({
  ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
    stablehlo.return %arg1 : tensor<f64>
  }) : (tensor<9x7x5xf64>, tensor<1xi32>, tensor<7x5xf64>) -> tensor<9x7x5xf64>
  return %3 : tensor<9x7x5xf64>
}

// -----

func.func @non_matching_scatter(%arg0: tensor<2x3x4x2xi64>, %arg1: tensor<2x2x3x2xi64>, %arg2: tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64> {
  // CHECK: stablehlo.scatter
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [3, 4], inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [1], scatter_dims_to_operand_dims = [2, 1], index_vector_dim = 3>, unique_indices = false}> ({
  ^bb0(%arg3: tensor<i64>, %arg4: tensor<i64>):
    %1 = stablehlo.add %arg3, %arg4 : tensor<i64>
    stablehlo.return %1 : tensor<i64>
  }) : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
  return %0 : tensor<2x3x4x2xi64>
}

// -----

func.func @scatter_with_batching_dims(%input_tensor: tensor<5x200x100x300xf32>,
    %scatter_indices: tensor<5x10x2xi32>, %updates: tensor<5x10x300xf32>) ->
      tensor<5x200x100x300xf32> {
  // CHECK: stablehlo.scatter
  %0 = "stablehlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = stablehlo.add %lhs, %rhs : tensor<f32>
    "stablehlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2],
      inserted_window_dims = [1, 2],
      input_batching_dims = [0],
      scatter_indices_batching_dims = [0],
      scatter_dims_to_operand_dims = [1, 2],
      index_vector_dim = 2
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<5x200x100x300xf32>, tensor<5x10x2xi32>, tensor<5x10x300xf32>) ->
      tensor<5x200x100x300xf32>
  func.return %0 : tensor<5x200x100x300xf32>
}

// -----

func.func @valid_scatter_dimensions_with_dynamic_index_vector_dim(
    %input_tensor: tensor<?x?x?xf32>, %scatter_indices: tensor<10x?xi32>,
    %updates: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "stablehlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = stablehlo.add %lhs, %rhs : tensor<f32>
    "stablehlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1, 2],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<?x?x?xf32>, tensor<10x?xi32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}
