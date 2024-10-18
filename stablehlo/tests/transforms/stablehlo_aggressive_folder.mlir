// RUN: stablehlo-opt --stablehlo-aggressive-folder --split-input-file --verify-diagnostics %s | FileCheck %s

////////
// AddOp

// CHECK-LABEL: @add_fold_cst
func.func @add_fold_cst() -> (tensor<i32>, tensor<f32>) {
  %cst = stablehlo.constant dense<1> : tensor<i32>
  %cst_1 = stablehlo.constant dense<1.0> : tensor<f32>
  // CHECK: stablehlo.constant dense<2> : tensor<i32>
  // CHECK: stablehlo.constant dense<2.0{{.*}}> : tensor<f32>
  %0 = stablehlo.add %cst, %cst : tensor<i32>
  %1 = stablehlo.add %cst_1, %cst_1 : tensor<f32>
  return %0, %1 : tensor<i32>, tensor<f32>
}

// -----

////////
// BroadcastInDimOp

// CHECK-LABEL: func.func @broadcast_in_dim_fold_splat
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<3x3xi32>)
func.func @broadcast_in_dim_fold_splat(%arg0: tensor<3x3xi32>)
  -> (tensor<6xi32>, tensor<3xf32>, tensor<3x3xi32>) {
  %c0 = stablehlo.constant dense<5> : tensor<i32>
  %c1 = stablehlo.constant dense<3.0> : tensor<f32>
  %c2 = stablehlo.constant dense<1> : tensor<1x3xi32>

  %0 = stablehlo.broadcast_in_dim %c0, dims = [] : (tensor<i32>) -> tensor<6xi32>
  %1 = stablehlo.broadcast_in_dim %c1, dims = [] : (tensor<f32>) -> tensor<3xf32>
  %2 = stablehlo.broadcast_in_dim %c2, dims = [1, 0] : (tensor<1x3xi32>) -> tensor<3x3xi32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.constant dense<5> : tensor<6xi32>
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.constant dense<3.000000e+00> : tensor<3xf32>
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.constant dense<1> : tensor<3x3xi32>

  // CHECK-NEXT: return [[R0]], [[R1]], [[R2]]
  return %0, %1, %2 : tensor<6xi32>, tensor<3xf32>, tensor<3x3xi32>
}

// -----

////////
// CompareOp

// CHECK-LABEL: func.func @compare_folds
func.func @compare_folds()
  -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
  %cn1 = stablehlo.constant dense<-1> : tensor<i32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %c5 = stablehlo.constant dense<5> : tensor<i32>

  %0 = stablehlo.compare EQ, %cn1, %cn1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = stablehlo.compare GT, %c5, %c5, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = stablehlo.compare GE, %c4, %cn1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %3 = stablehlo.compare LE, %c4, %c5, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  %4 = stablehlo.compare EQ, %cn1, %cn1, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %5 = stablehlo.compare GT, %c5, %cn1, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %6 = stablehlo.compare GE, %c5, %c4, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %7 = stablehlo.compare LE, %cn1, %c5, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  // CHECK-DAG:  [[FALSE:%.+]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-DAG:  [[TRUE:%.+]] = stablehlo.constant dense<true> : tensor<i1>

  // CHECK-NEXT: return [[TRUE]], [[FALSE]], [[TRUE]], [[TRUE]], [[TRUE]], [[FALSE]], [[TRUE]], [[FALSE]]
  return %0, %1, %2, %3, %4, %5, %6, %7 :
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
}


// -----

////////
// ConcatenateOp

// CHECK-LABEL: func.func @concatenate_fold
func.func @concatenate_fold() -> (tensor<6xi32>, tensor<3xi32>, tensor<3x3xi32>, tensor<2x5xi32>) {
  %c0 = stablehlo.constant dense<[0, 1]> : tensor<2xi32>
  %c1 = stablehlo.constant dense<[2, 3, 4]> : tensor<3xi32>
  %c2 = stablehlo.constant dense<[5]> : tensor<1xi32>

  %c3 = stablehlo.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %c4 = stablehlo.constant dense<[[6, 7, 8]]> : tensor<1x3xi32>
  %c5 = stablehlo.constant dense<[[11, 12], [13, 14]]> : tensor<2x2xi32>

  %0 = stablehlo.concatenate %c0, %c1, %c2, dim = 0 : (tensor<2xi32>, tensor<3xi32>, tensor<1xi32>) -> tensor<6xi32>
  %1 = stablehlo.concatenate %c0, %c2, dim = 0 : (tensor<2xi32>, tensor<1xi32>) -> tensor<3xi32>

  %2 = stablehlo.concatenate %c3, %c4, dim = 0 : (tensor<2x3xi32>, tensor<1x3xi32>) -> tensor<3x3xi32>
  %3 = stablehlo.concatenate %c3, %c5, dim = 1 : (tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x5xi32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.constant dense<[0, 1, 2, 3, 4, 5]> : tensor<6xi32>
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.constant dense<[0, 1, 5]> : tensor<3xi32>
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.constant dense<{{\[\[0, 1, 2\], \[3, 4, 5\], \[6, 7, 8\]\]}}> : tensor<3x3xi32>
  // CHECK-DAG:  [[R3:%.+]] = stablehlo.constant dense<{{\[\[0, 1, 2, 11, 12\], \[3, 4, 5, 13, 14\]\]}}> : tensor<2x5xi32>
  // CHECK-NEXT: return [[R0]], [[R1]], [[R2]], [[R3]]
  return %0, %1, %2, %3 : tensor<6xi32>, tensor<3xi32>, tensor<3x3xi32>, tensor<2x5xi32>
}

// -----

////////
// MulOp

// CHECK-LABEL: @mul_fold_cst
func.func @mul_fold_cst() -> (tensor<i32>, tensor<f32>) {
  %cst = stablehlo.constant dense<2> : tensor<i32>
  %cst_1 = stablehlo.constant dense<2.0> : tensor<f32>
  // CHECK: stablehlo.constant dense<4> : tensor<i32>
  // CHECK: stablehlo.constant dense<4.0{{.*}}> : tensor<f32>
  %0 = stablehlo.multiply %cst, %cst : tensor<i32>
  %1 = stablehlo.multiply %cst_1, %cst_1 : tensor<f32>
  return %0, %1 : tensor<i32>, tensor<f32>
}

// -----

////////
// SubtractOp

// CHECK-LABEL: @subtract_fold_cst
func.func @subtract_fold_cst() -> (tensor<i32>, tensor<f32>) {
  %cst = stablehlo.constant dense<1> : tensor<i32>
  %cst_1 = stablehlo.constant dense<3> : tensor<i32>
  %cst_2 = stablehlo.constant dense<1.0> : tensor<f32>
  %cst_3 = stablehlo.constant dense<3.0> : tensor<f32>
  // CHECK: stablehlo.constant dense<2> : tensor<i32>
  // CHECK: stablehlo.constant dense<2.0{{.*}}> : tensor<f32>
  %0 = stablehlo.subtract %cst_1, %cst : tensor<i32>
  %1 = stablehlo.subtract %cst_3, %cst_2 : tensor<f32>
  return %0, %1 : tensor<i32>, tensor<f32>
}

// -----

////////
// IotaOp

// CHECK-LABEL: func @eval_iota
func.func @eval_iota() -> (tensor<3x4x5xi32>, tensor<3x4x5xi32>, tensor<3x4x5xi32>) {
  // CHECK-NOT: stablehlo.iota
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<
  // CHECK-SAME: {{\[\[}}[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
  // CHECK-SAME: {{\[}}[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
  // CHECK-SAME: {{\[}}[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]]> : tensor<3x4x5xi32>

  // CHECK: [[RESULT1:%.*]] = stablehlo.constant dense<
  // CHECK-SAME: {{\[\[}}[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
  // CHECK-SAME: {{\[}}[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
  // CHECK-SAME: {{\[}}[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]]> : tensor<3x4x5xi32>

  // CHECK: [[RESULT2:%.*]] = stablehlo.constant dense<
  // CHECK-SAME: {{\[\[}}[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
  // CHECK-SAME: {{\[}}[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
  // CHECk-SAME: {{\[}}[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]]> : tensor<3x4x5xi32>

  // CHECK: return [[RESULT0]], [[RESULT1]], [[RESULT2]]
  %0 = stablehlo.iota dim = 0 : tensor<3x4x5xi32>
  %1 = stablehlo.iota dim = 1 : tensor<3x4x5xi32>
  %2 = stablehlo.iota dim = 2 : tensor<3x4x5xi32>
  func.return %0, %1, %2 : tensor<3x4x5xi32>, tensor<3x4x5xi32>, tensor<3x4x5xi32>
}

// -----

// CHECK-LABEL: func @eval_iota_zero_dimension
func.func @eval_iota_zero_dimension() -> (tensor<0xi32>, tensor<5x0x2xi32>) {
  // CHECK-NOT: stablehlo.iota
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<> : tensor<0xi32>
  // CHECK: [[RESULT1:%.*]] = stablehlo.constant dense<> : tensor<5x0x2xi32>
  // CHECK: return [[RESULT0]], [[RESULT1]]
  %0 = stablehlo.iota dim = 0 : tensor<0xi32>
  %1 = stablehlo.iota dim = 2 : tensor<5x0x2xi32>
  func.return %0, %1 : tensor<0xi32>, tensor<5x0x2xi32>
}

// -----

////////
// ReshapeOp

// CHECK-LABEL: func @reshape
func.func @reshape_fold() -> (tensor<1xi32>, tensor<2x2xi32>) {
  %c0 = stablehlo.constant dense<2> : tensor<i32>
  %c1 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %0 = stablehlo.reshape %c0 : (tensor<i32>) -> tensor<1xi32>
  %1 = stablehlo.reshape %c1 : (tensor<4xi32>) -> tensor<2x2xi32>

  // CHECK-DAG:  [[CST1:%.+]] = stablehlo.constant dense<2> : tensor<1xi32>
  // CHECK-DAG:  [[CST2:%.+]] = stablehlo.constant dense<{{\[\[1, 2\], \[3, 4\]\]}}> : tensor<2x2xi32>
  // CHECK-NEXT: return [[CST1]], [[CST2]]
  return %0, %1 : tensor<1xi32>, tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @eval_convert_f32_to_i64
func.func @eval_convert_f32_to_i64() -> tensor<2xi64> {
  // CHECK-NOT: stablehlo.convert
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %1 = stablehlo.convert %0 : (tensor<2xf32>) -> tensor<2xi64>
  func.return %1 : tensor<2xi64>
}

// -----

// CHECK-LABEL: func @eval_convert_f32_non_convertable
func.func @eval_convert_f32_non_convertable() -> (tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) {
  // CHECK: [[RESULT0:%.*]] = stablehlo.convert
  // CHECK: [[RESULT1:%.*]] = stablehlo.convert
  // CHECK: [[RESULT2:%.*]] = stablehlo.convert
  // CHECK: return [[RESULT0]], [[RESULT1]], [[RESULT2]]
  %pinf = stablehlo.constant dense<[1.0, 0x7F800000]> : tensor<2xf32>
  %ninf = stablehlo.constant dense<[2.0, 0xFF800000]> : tensor<2xf32>
  %nzero = stablehlo.constant dense<[3.0, 0x80000000]> : tensor<2xf32>
  %0 = stablehlo.convert %pinf : (tensor<2xf32>) -> tensor<2xi64>
  %1 = stablehlo.convert %ninf : (tensor<2xf32>) -> tensor<2xi64>
  %2 = stablehlo.convert %nzero : (tensor<2xf32>) -> tensor<2xi64>
  func.return %0, %1, %2 : tensor<2xi64>, tensor<2xi64>, tensor<2xi64>
}

// -----

// CHECK-LABEL: func @eval_convert_f32_non_fittable
func.func @eval_convert_f32_non_fittable() -> (tensor<1xi32>, tensor<1xi32>) {
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<2.14748365E+9> : tensor<1xf32>
  // CHECK: [[RESULT1:%.*]] = stablehlo.constant dense<2147483520> : tensor<1xi32>
  // CHECK: [[RESULT2:%.*]] = stablehlo.convert [[RESULT0]]
  // CHECK: return [[RESULT1]], [[RESULT2]]
  %twopow30 = stablehlo.constant dense<[2147483583.0]> : tensor<1xf32>
  %twopow31 = stablehlo.constant dense<[2147483584.0]> : tensor<1xf32>
  %1 = stablehlo.convert %twopow30 : (tensor<1xf32>) -> tensor<1xi32>
  %2 = stablehlo.convert %twopow31 : (tensor<1xf32>) -> tensor<1xi32>
  func.return %1, %2 : tensor<1xi32>, tensor<1xi32>
}

// -----

// CHECK-LABEL: func @eval_convert_i32_non_exact
func.func @eval_convert_i32_non_exact() -> (tensor<1xf32>, tensor<1xf32>) {
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<0x4B7FFFFF> : tensor<1xf32>
  // 0x4B800000 = 16777216, error due to conversion -1
  // CHECK: [[RESULT1:%.*]] = stablehlo.constant dense<0x4B800000> : tensor<1xf32>
  // CHECK: return [[RESULT0]], [[RESULT1]]
  %pow23 = stablehlo.constant dense<[16777215]> : tensor<1xi32>
  %pow24 = stablehlo.constant dense<[16777217]> : tensor<1xi32>
  %1 = stablehlo.convert %pow23 : (tensor<1xi32>) -> tensor<1xf32>
  %2 = stablehlo.convert %pow24 : (tensor<1xi32>) -> tensor<1xf32>
  func.return %1, %2 : tensor<1xf32>, tensor<1xf32>
}

// -----

// CHECK-LABEL: func @eval_convert_f64_precision_loss
func.func @eval_convert_f64_precision_loss() -> (tensor<1xf32>, tensor<f32>) {
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<9.99999996E-13> : tensor<1xf32>
  // CHECK: return [[RESULT0]]
  %0 = arith.constant dense<9.9999999999999998E-13> : tensor<1xf64>
  %1 = stablehlo.constant dense<8.000000e+00> : tensor<f64>
  %2 = stablehlo.convert %0 : (tensor<1xf64>) -> tensor<1xf32>
  %3 = stablehlo.convert %1 : (tensor<f64>) -> tensor<f32>
  func.return %2, %3 : tensor<1xf32>, tensor<f32>
}

// -----

// CHECK-LABEL: func @eval_transpose
func.func @eval_transpose() -> (tensor<2x3x2xi32>, tensor<2x4x3xi32>, tensor<4x3x2xi32>) {
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<
  // CHECK-SAME: {{\[\[}}[1, 7], [3, 9], [5, 11]],
  // CHECK-SAME:   {{\[}}[2, 8], [4, 10], [6, 12]]]> : tensor<2x3x2xi32>
  //
  // CHECK: [[RESULT1:%.*]] = stablehlo.constant dense<
  // CHECK-SAME: {{\[\[}}[1, 3, 5], [7, 9, 11], [13, 15, 17], [19, 21, 23]],
  // CHECK-SAME:   {{\[}}[2, 4, 6], [8, 10, 12], [14, 16, 18], [20, 22, 24]]]> : tensor<2x4x3xi32>
  //
  // CHECK: [[RESULT2:%.*]] = stablehlo.constant dense<
  // CHECK-SAME: {{\[\[}}[1, 2],  [3, 4],  [5, 6]]
  // CHECK-SAME:   {{\[}}[7, 8],  [9, 10], [11, 12]],
  // CHECK-SAME:   {{\[}}[13, 14], [15, 16], [17, 18]],
  // CHECK-SAME:   {{\[}}[19, 20], [21, 22], [23, 24]]]> : tensor<4x3x2xi32>
  //
  // CHECK: return [[RESULT0]], [[RESULT1]], [[RESULT2]]
  %0 = stablehlo.constant dense<[[[1,2], [3,4], [5,6]],
                                 [[7,8], [9,10], [11,12]]]> : tensor<2x3x2xi32>
  %1 = stablehlo.constant dense<[[[1, 2],  [3, 4],  [5, 6]],
                                 [[7, 8],  [9, 10], [11,12]],
                                 [[13,14], [15,16], [17,18]],
                                 [[19,20], [21,22], [23,24]]]> : tensor<4x3x2xi32>
  %2 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
  %3 = stablehlo.transpose %1, dims = [2, 0, 1] : (tensor<4x3x2xi32>) -> tensor<2x4x3xi32>
  %4 = stablehlo.transpose %3, dims = [1, 2, 0] : (tensor<2x4x3xi32>) -> tensor<4x3x2xi32>
  func.return %2, %3, %4 : tensor<2x3x2xi32>, tensor<2x4x3xi32>, tensor<4x3x2xi32>
}

// -----

// CHECK-LABEL: func @eval_transpose_zerodim
func.func @eval_transpose_zerodim() -> (tensor<10x3x0xf32>) {
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<> : tensor<10x3x0xf32>
  // CHECK: return [[RESULT0]]
  %0 = stablehlo.constant dense<> : tensor<3x0x10xf32>
  %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<3x0x10xf32>) -> tensor<10x3x0xf32>
  func.return %1 : tensor<10x3x0xf32>
}

// -----

// CHECK-LABEL: func @eval_transpose_zerorank
func.func @eval_transpose_zerorank() -> tensor<i32> {
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK: return [[RESULT0]]
  %0 = stablehlo.constant dense<1> : tensor<i32>
  %1 = stablehlo.transpose %0, dims = [] : (tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// -----

// CHECK-LABEL: func @eval_transpose_splat
func.func @eval_transpose_splat() -> (tensor<10x3x1xi32>) {
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<1> : tensor<10x3x1xi32>
  // CHECK: return [[RESULT0]]
  %0 = stablehlo.constant dense<1> : tensor<3x1x10xi32>
  %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<3x1x10xi32>) -> tensor<10x3x1xi32>
  func.return %1 : tensor<10x3x1xi32>
}
