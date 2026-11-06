// RUN: stablehlo-opt %s --stablehlo-legalize-to-linalg --split-input-file --canonicalize | FileCheck %s --check-prefixes=CHECK,GENERIC
// RUN: stablehlo-opt %s --stablehlo-legalize-to-linalg="enable-primitive-ops=true" --split-input-file --canonicalize | FileCheck %s --check-prefixes=CHECK,PRIMITIVE
// RUN: stablehlo-opt %s --stablehlo-legalize-to-linalg --split-input-file --canonicalize --one-shot-bufferize="bufferize-function-boundaries" | FileCheck %s --check-prefix=BUFFERIZED --implicit-check-not=stablehlo.dynamic_broadcast_in_dim --implicit-check-not=tensor.extract
// RUN: stablehlo-opt %s --stablehlo-legalize-to-linalg="enable-primitive-ops=true" --split-input-file --canonicalize --one-shot-bufferize="bufferize-function-boundaries" | FileCheck %s --check-prefix=BUFFERIZED --implicit-check-not=stablehlo.dynamic_broadcast_in_dim --implicit-check-not=tensor.extract

// Regression for #2833: the leading dimension may expand even when the input
// and output types are identical. The output shape must come from %shape.
// CHECK-LABEL: func.func @dynamic_leading_dimension(
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x4096xf16>, %[[SHAPE:.*]]: tensor<2xi64>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[SIZE:.*]] = tensor.dim %[[INPUT]], %[[ZERO]]
// CHECK: %[[EXPANDS:.*]] = arith.cmpi eq, %[[SIZE]], %[[ONE]]
// CHECK: %[[EXTENT:.*]] = tensor.extract %[[SHAPE]][%[[ZERO]]]
// CHECK: %[[DIM:.*]] = arith.index_cast %[[EXTENT]] : i64 to index
// CHECK: %[[EMPTY:.*]] = tensor.empty(%[[DIM]]) : tensor<?x4096xf16>
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [{{.*}}], iterator_types = ["parallel", "parallel"]
// CHECK-SAME: outs(%[[EMPTY]] : tensor<?x4096xf16>)
// CHECK: %[[I:.*]] = linalg.index 0 : index
// CHECK: %[[ROW:.*]] = arith.select %[[EXPANDS]], %[[ZERO]], %[[I]] : index
// CHECK: %[[COL:.*]] = linalg.index 1 : index
// CHECK: %[[ELEMENT:.*]] = tensor.extract %[[INPUT]][%[[ROW]], %[[COL]]]
// CHECK: linalg.yield %[[ELEMENT]] : f16
// CHECK: return %[[RESULT]] : tensor<?x4096xf16>
// BUFFERIZED-LABEL: func.func @dynamic_leading_dimension(
// BUFFERIZED: arith.select
// BUFFERIZED: memref.load
func.func @dynamic_leading_dimension(%input: tensor<?x4096xf16>, %shape: tensor<2xi64>) -> tensor<?x4096xf16> {
  %0 = stablehlo.dynamic_broadcast_in_dim %input, %shape, dims = [0, 1] : (tensor<?x4096xf16>, tensor<2xi64>) -> tensor<?x4096xf16>
  return %0 : tensor<?x4096xf16>
}

// -----

// Cover rank expansion, a nonidentity dimension mapping, and index shapes.
// CHECK-LABEL: func.func @rank_expansion(
// CHECK-SAME: %[[INPUT:.*]]: tensor<?xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[EXPANDS:.*]] = arith.cmpi eq
// CHECK: linalg.generic
// CHECK: %[[I:.*]] = linalg.index 1 : index
// CHECK: %[[INDEX:.*]] = arith.select %[[EXPANDS]], %[[ZERO]], %[[I]] : index
// CHECK: %[[ELEMENT:.*]] = tensor.extract %[[INPUT]][%[[INDEX]]]
// CHECK: linalg.yield %[[ELEMENT]] : f32
// BUFFERIZED-LABEL: func.func @rank_expansion(
// BUFFERIZED: memref.load
func.func @rank_expansion(%input: tensor<?xf32>, %shape: tensor<3xindex>) -> tensor<?x?x?xf32> {
  %0 = stablehlo.dynamic_broadcast_in_dim %input, %shape, dims = [1] : (tensor<?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

// Each dynamic operand dimension must be checked independently. Also cover a
// static result and permuted broadcast dimensions.
// CHECK-LABEL: func.func @permuted_dimensions(
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x?xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[SIZE0:.*]] = tensor.dim %[[INPUT]], %[[ZERO]]
// CHECK: %[[EXPANDS0:.*]] = arith.cmpi eq, %[[SIZE0]], %[[ONE]]
// CHECK: %[[SIZE1:.*]] = tensor.dim %[[INPUT]], %[[ONE]]
// CHECK: %[[EXPANDS1:.*]] = arith.cmpi eq, %[[SIZE1]], %[[ONE]]
// CHECK: linalg.generic
// CHECK: %[[I:.*]] = linalg.index 2 : index
// CHECK: %[[INDEX0:.*]] = arith.select %[[EXPANDS0]], %[[ZERO]], %[[I]] : index
// CHECK: %[[J:.*]] = linalg.index 0 : index
// CHECK: %[[INDEX1:.*]] = arith.select %[[EXPANDS1]], %[[ZERO]], %[[J]] : index
// CHECK: %[[ELEMENT:.*]] = tensor.extract %[[INPUT]][%[[INDEX0]], %[[INDEX1]]]
// CHECK: linalg.yield %[[ELEMENT]] : f32
// BUFFERIZED-LABEL: func.func @permuted_dimensions(
// BUFFERIZED: memref.load
func.func @permuted_dimensions(%input: tensor<?x?xf32>, %shape: tensor<3xi64>) -> tensor<3x4x5xf32> {
  %0 = stablehlo.dynamic_broadcast_in_dim %input, %shape, dims = [2, 0] : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<3x4x5xf32>
  return %0 : tensor<3x4x5xf32>
}

// -----

// Preserve known expansion information when only some dimensions are unknown.
// CHECK-LABEL: func.func @partial_annotations(
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x?x?x1x4xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[SIZE:.*]] = tensor.dim %[[INPUT]], %[[ZERO]]
// CHECK: %[[EXPANDS:.*]] = arith.cmpi eq, %[[SIZE]], %[[ONE]]
// CHECK-NOT: arith.cmpi
// CHECK: linalg.generic
// CHECK: %[[I:.*]] = linalg.index 0 : index
// CHECK: %[[INDEX:.*]] = arith.select %[[EXPANDS]], %[[ZERO]], %[[I]] : index
// CHECK: %[[J:.*]] = linalg.index 2 : index
// CHECK: %[[K:.*]] = linalg.index 4 : index
// CHECK-NOT: arith.select
// CHECK: %[[ELEMENT:.*]] = tensor.extract %[[INPUT]][%[[INDEX]], %[[ZERO]], %[[J]], %[[ZERO]], %[[K]]]
// CHECK: linalg.yield %[[ELEMENT]] : f32
// BUFFERIZED-LABEL: func.func @partial_annotations(
// BUFFERIZED: memref.load
func.func @partial_annotations(%input: tensor<?x?x?x1x4xf32>, %shape: tensor<5xi64>) -> tensor<?x?x?x?x4xf32> {
  %0 = stablehlo.dynamic_broadcast_in_dim %input, %shape, dims = [0, 1, 2, 3, 4] {known_expanding_dimensions = array<i64: 1>, known_nonexpanding_dimensions = array<i64: 2>} : (tensor<?x?x?x1x4xf32>, tensor<5xi64>) -> tensor<?x?x?x?x4xf32>
  return %0 : tensor<?x?x?x?x4xf32>
}

// -----

// Fully known broadcasts should continue using the existing affine/primitive
// lowering, without runtime indexing.
// CHECK-LABEL: func.func @known_nonexpanding(
// CHECK-NOT: arith.cmpi
// GENERIC: linalg.generic
// PRIMITIVE: linalg.broadcast
// CHECK-NOT: arith.select
// CHECK: return
// BUFFERIZED-LABEL: func.func @known_nonexpanding(
func.func @known_nonexpanding(%input: tensor<?x4xf32>, %shape: tensor<3xi64>) -> tensor<?x?x4xf32> {
  %0 = stablehlo.dynamic_broadcast_in_dim %input, %shape, dims = [1, 2] {known_nonexpanding_dimensions = array<i64: 0>} : (tensor<?x4xf32>, tensor<3xi64>) -> tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}
