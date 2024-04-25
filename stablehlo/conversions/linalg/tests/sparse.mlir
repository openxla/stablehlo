// RUN: stablehlo-opt %s --stablehlo-legalize-to-linalg=enable-sparse-ops --split-input-file --canonicalize | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

func.func @dot_matmul(%arg0: tensor<2x3xf32, #CSR>,
                 %arg1: tensor<3x?xf32, #CSR>) -> tensor<2x?xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {someattr}
           : (tensor<2x3xf32, #CSR>, tensor<3x?xf32, #CSR>) -> tensor<2x?xf32, #CSR>
  %2 = sparse_tensor.convert %0 : tensor<2x?xf32, #CSR> to tensor<2x?xf32>
  func.return %2 : tensor<2x?xf32>
}
// CHECK-LABEL: func @dot_matmul
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<2x3xf32, #sparse>, %[[ARG1:.*]]: tensor<3x?xf32, #sparse>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<3x?xf32, #sparse>
// CHECK: %[[INIT:.*]] = bufferization.alloc_tensor(%[[D1]]) : tensor<2x?xf32, #sparse>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: %[[OUT:.*]] = linalg.matmul
// CHECK-SAME: {someattr}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xf32, #sparse>, tensor<3x?xf32, #sparse>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xf32, #sparse>)
// CHECK: %[[RETURN:.*]] = sparse_tensor.convert %[[OUT]] : tensor<2x?xf32, #sparse> to tensor<2x?xf32>
// CHECK: %[[RETURN]]

// -----

#CCC = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : compressed)
}>

func.func @dot_general(%arg0: tensor<?x?x?xf32, #CCC>,
                  %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    someattr
  } : (tensor<?x?x?xf32, #CCC>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}
// The iterations are (Batch Dim, LHS Other Dim, RHS Other dim, Contracting Dim)
// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0)>
// Output is the iterators excluding contracting
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: func @dot_general(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32, #sparse>, %[[ARG1:.*]]: tensor<?x?x?xf32>)
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C0]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]])
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: %[[RETURN:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// Only contracting dims are reductions
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?x?x?xf32, #sparse>, tensor<?x?x?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?x?x?xf32>)
// CHECK-SAME: {someattr}
// CHECK:   ^bb0(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
// CHECK:     %[[MUL:.*]] = arith.mulf %[[ARG2]], %[[ARG3]] : f32
// CHECK:     %[[SUM:.*]] = arith.addf %[[ARG4]], %[[MUL]] : f32
// CHECK:     linalg.yield %[[SUM]] : f32
// CHECK: } -> tensor<?x?x?xf32>
// CHECK: %[[RETURN]] : tensor<?x?x?xf32>

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK:     @reduce_add
// CHECK-PRIMITIVE-LABEL: @reduce_add
func.func @reduce_add(%arg0: tensor<5x4xi32, #CSR>, %arg1: tensor<i32>) -> tensor<5xi32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = stablehlo.add %arg3, %arg4 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 1>, someattr} : (tensor<5x4xi32, #CSR>, tensor<i32>) -> tensor<5xi32>
  func.return %0 : tensor<5xi32>
}
// CHECK-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-DAG: %[[INIT_TENSOR:.*]] = tensor.empty()
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4xi32, #sparse>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<5xi32>)
// CHECK-SAME: {someattr}
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.addi %[[RHS_IN]], %[[LHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// CHECK-PRIMITIVE-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-PRIMITIVE-DAG: %[[INIT_TENSOR:.*]] = tensor.empty()
// CHECK-PRIMITIVE-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK-PRIMITIVE: linalg.reduce { arith.addi {overflowFlags = #arith.overflow<none>} }
// CHECK-PRIMITIVE-SAME: ins(%{{.*}}tensor<5x4xi32, #sparse>)
// CHECK-PRIMITIVE-SAME: outs(%[[FILL_TENSOR]] : tensor<5xi32>)
// CHECK-PRIMITIVE-SAME: dimensions = [1]  {someattr}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @float_add
// CHECK-PRIMITIVE-LABEL: func @float_add
func.func @float_add(%lhs: tensor<2x2xf32, #CSR>,
                     %rhs: tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR> {
  // CHECK: linalg.generic
  // CHECK-SAME: {someattr}
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[RESULT:[a-zA-Z0-9_]*]] = arith.addf %[[ARG0]], %[[ARG1]]
  // CHECK: linalg.yield %[[RESULT]]
  // CHECK: #sparse

  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.addf
  // CHECK-PRIMITIVE: #sparse
  %0 = "stablehlo.add"(%lhs, %rhs) {someattr}
      : (tensor<2x2xf32, #CSR>, tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR>
  func.return %0 : tensor<2x2xf32, #CSR>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func @integer_sub
// CHECK-PRIMITIVE-LABEL: func @integer_sub
func.func @integer_sub(%lhs: tensor<2x2xi32, #CSR>,
                  %rhs: tensor<2x2xi32, #CSR>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: subi
  // CHECK: #sparse
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: subi
  // CHECK-PRIMITIVE: #sparse
  %0 = "stablehlo.subtract"(%lhs, %rhs) : (tensor<2x2xi32, #CSR>,
                                    tensor<2x2xi32, #CSR>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func @float_mul
// CHECK-PRIMITIVE-LABEL: func @float_mul
func.func @float_mul(%lhs: tensor<2x2xf32, #CSR>,
                     %rhs: tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR> {
  // CHECK: linalg.generic
  // CHECK: mulf
  // CHECK: #sparse
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: mulf
  // CHECK-PRIMITIVE: #sparse
  %0 = "stablehlo.multiply"(%lhs, %rhs) : (tensor<2x2xf32, #CSR>,
                                           tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR>
  func.return %0 : tensor<2x2xf32, #CSR>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func @float_neg
// CHECK-PRIMITIVE-LABEL: func @float_neg
func.func @float_neg(%arg0: tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR> {
  // CHECK: linalg.generic
  // CHECK: negf
  // CHECK: #sparse
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: negf
  // CHECK-PRIMITIVE: #sparse
  %0 = "stablehlo.negate"(%arg0) : (tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR>
  func.return %0 : tensor<2x2xf32, #CSR>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func @float_abs
// CHECK-PRIMITIVE-LABEL: func @float_abs
func.func @float_abs(%arg0: tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR> {
  // CHECK: linalg.generic
  // CHECK-SAME: {someattr}
  // CHECK: math.absf
  // CHECK: #sparse
  // CHECK-PRIMITIVE: linalg.map { math.absf }
  // CHECK-PRIMITIVE-SAME: ins(
  // CHECK-PRIMITIVE-SAME: outs(
  // CHECK-PRIMITIVE-SAME: {someattr}
  // CHECK-PRIMITIVE: #sparse
  %0 = "stablehlo.abs"(%arg0) {someattr} : (tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR>
  func.return %0 : tensor<2x2xf32, #CSR>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func @complex_sqrt
// CHECK-PRIMITIVE-LABEL: func @complex_sqrt
func.func @complex_sqrt(%operand: tensor<2x2xcomplex<f32>, #CSR>) -> tensor<2x2xcomplex<f32>, #CSR> {
  %tensor_result = "stablehlo.sqrt"(%operand)
      : (tensor<2x2xcomplex<f32>, #CSR>) -> tensor<2x2xcomplex<f32>, #CSR>
  // CHECK: linalg.generic
  // CHECK: complex.sqrt
  // CHECK: #sparse
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.sqrt
  // CHECK-PRIMITIVE: #sparse
  func.return %tensor_result : tensor<2x2xcomplex<f32>, #CSR>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func @float_ceil
// CHECK-PRIMITIVE-LABEL: func @float_ceil
func.func @float_ceil(%arg0: tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR> {
  // CHECK: linalg.generic
  // CHECK: math.ceil
  // CHECK: #sparse
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.ceil
  // CHECK-PRIMITIVE: #sparse
  %0 = "stablehlo.ceil"(%arg0) : (tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR>
  func.return %0 : tensor<2x2xf32, #CSR>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func @reshape_3D_2D
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2]]
// CHECK: #sparse
func.func @reshape_3D_2D(%arg0: tensor<12x1x42xi32>) -> tensor<12x42xi32, #CSR> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<12x1x42xi32>) -> tensor<12x42xi32, #CSR>
  func.return %0 : tensor<12x42xi32, #CSR>
}

// -----

#CCC = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : compressed)
}>

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, d2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: func @broadcast_in_dim
func.func @broadcast_in_dim(%operand: tensor<5x7x6xf32, #CCC>) -> tensor<7x10x6x4x5xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = array<i64: 4, 0, 2>}
         : (tensor<5x7x6xf32, #CCC>) -> tensor<7x10x6x4x5xf32>
  func.return %0 : tensor<7x10x6x4x5xf32>
}
// CHECK: tensor.empty() : tensor<7x10x6x4x5xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK: #sparse
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim
// CHECK-PRIMITIVE: tensor.collapse_shape
// CHECK-PRIMITIVE: linalg.transpose
// CHECK-PRIMITIVE:   permutation = [1, 0]
// CHECK-PRIMITIVE: tensor.empty() : tensor<7x10x6x4x5xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [1, 2, 3]

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func @concatenate
// CHECK: sparse_tensor.concatenate
func.func @concatenate(%a: tensor<2x2xi32, #CSR>, %b: tensor<2x2xi32, #CSR>, %c: tensor<2x2xi32, #CSR>) -> tensor<?x?xi32, #CSR> {
    %concat = "stablehlo.concatenate"(%a, %b, %c) {
      dimension = 1
    } : (tensor<2x2xi32, #CSR>, tensor<2x2xi32, #CSR>, tensor<2x2xi32, #CSR>) -> tensor<?x?xi32, #CSR>
    func.return %concat : tensor<?x?xi32, #CSR>
}

// -----

#CCCC = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (d0 : compressed, d1 : compressed, d2 : compressed, d3 : compressed)
}>

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func @transpose
func.func @transpose(%arg0: tensor<2x3x9x5xi32, #CCCC>) -> tensor<3x2x5x9xi32, #CCCC> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0, 3, 2>}
        : (tensor<2x3x9x5xi32, #CCCC>) -> tensor<3x2x5x9xi32, #CCCC>
  func.return %0 : tensor<3x2x5x9xi32, #CCCC>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK: #sparse

// CHECK-PRIMITIVE-LABEL: func @transpose
// CHECK-PRIMITIVE: linalg.transpose

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: convert
func.func @convert(%in: tensor<2x2xi32, #CSR>) -> tensor<2x2xi64, #CSR> {
  // CHECK: linalg.generic
  // CHECK: #sparse
  // CHECK: arith.extsi
  %0 = "stablehlo.convert"(%in) : (tensor<2x2xi32, #CSR>) -> tensor<2x2xi64, #CSR>
  func.return %0 : tensor<2x2xi64, #CSR>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func @float_expm1
func.func @float_expm1(%arg0: tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR> {
  // CHECK: linalg.generic
  // CHECK: #sparse
  // CHECK: expm1
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: expm1
  %0 = "stablehlo.exponential_minus_one"(%arg0) : (tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR>
  func.return %0 : tensor<2x2xf32, #CSR>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func @float_log1p
// CHECK-PRIMITIVE-LABEL: func @float_log1p
func.func @float_log1p(%arg0: tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR> {
  // CHECK: linalg.generic
  // CHECK: #sparse
  // CHECK: math.log1p
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.log1p
  %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR>
  func.return %0 : tensor<2x2xf32, #CSR>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func @complex_sign
// CHECK-PRIMITIVE-LABEL: func @complex_sign
func.func @complex_sign(
    %arg0: tensor<2x2xcomplex<f32>, #CSR>) -> tensor<2x2xcomplex<f32>, #CSR> {
  // CHECK: linalg.generic
  // CHECK: #sparse
  // CHECK: complex.sign
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.sign
  %0 = "stablehlo.sign"(%arg0) : (tensor<2x2xcomplex<f32>, #CSR>)
                          -> tensor<2x2xcomplex<f32>, #CSR>
  func.return %0 : tensor<2x2xcomplex<f32>, #CSR>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func @float_sin
// CHECK-PRIMITIVE-LABEL: func @float_sin
func.func @float_sin(%arg0: tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR> {
  // CHECK: linalg.generic
  // CHECK: #sparse
  // CHECK: math.sin
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.sin
  %0 = "stablehlo.sine"(%arg0) : (tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR>
  func.return %0 : tensor<2x2xf32, #CSR>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func @float_tanh
// CHECK-PRIMITIVE-LABEL: func @float_tanh
func.func @float_tanh(%arg0: tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR> {
  // CHECK: linalg.generic
  // CHECK: #sparse
  // CHECK: tanh
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: tanh
  %0 = "stablehlo.tanh"(%arg0) : (tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR>
  func.return %0 : tensor<2x2xf32, #CSR>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func @floor
// CHECK-PRIMITIVE-LABEL: func @floor
func.func @floor(%input: tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR> {
  // CHECK: linalg.generic
  // CHECK: #sparse
  // CHECK: math.floor
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.floor
  %0 = "stablehlo.floor"(%input) : (tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR>
  func.return %0 : tensor<2x2xf32, #CSR>
}

// -----

#SV = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

// CHECK-LABEL: @real_real
// CHECK-SAME: (%[[ARG0:.*]]:
func.func @real_real(%arg0: tensor<?xf32, #SV>) -> tensor<?xf32, #SV> {
  %1 = "stablehlo.real"(%arg0) : (tensor<?xf32, #SV>) -> (tensor<?xf32, #SV>)
  // CHECK: #sparse
  // CHECK: return %[[ARG0]]
  func.return %1 : tensor<?xf32, #SV>
}

// -----

#SV = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

// CHECK-LABEL: @imag_real
func.func @imag_real(%arg0: tensor<?xf32, #SV>) -> tensor<?xf32, #SV> {
  %1 = "stablehlo.imag"(%arg0) : (tensor<?xf32, #SV>) -> (tensor<?xf32, #SV>)
  // CHECK: %[[CST:.*]] = arith.constant 0
  // CHECK: linalg.generic
  // CHECK: #sparse
  // CHECK: yield %[[CST]]
  func.return %1 : tensor<?xf32, #SV>
}
