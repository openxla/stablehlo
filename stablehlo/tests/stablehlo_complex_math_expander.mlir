// RUN: stablehlo-opt --stablehlo-complex-math-expander --split-input-file --verify-diagnostics %s | FileCheck %s --check-prefix=CHECK-PRECISION
// RUN: stablehlo-opt --stablehlo-complex-math-expander=mode=all --split-input-file --verify-diagnostics %s | FileCheck %s --check-prefix=CHECK-ALL

//////// Precision mode (default): transcendental ops are expanded

// CHECK-PRECISION-LABEL: func.func @log_plus_one_complex_f32(
// CHECK-PRECISION-NOT:     stablehlo.log_plus_one {{.*}} tensor<complex<f32>>
// CHECK-PRECISION:         stablehlo.real
// CHECK-PRECISION:         stablehlo.imag
// CHECK-PRECISION:         stablehlo.complex

// CHECK-ALL-LABEL: func.func @log_plus_one_complex_f32(
// CHECK-ALL-NOT:     stablehlo.log_plus_one {{.*}} tensor<complex<f32>>
func.func @log_plus_one_complex_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "stablehlo.log_plus_one"(%arg) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

//////// Precision mode: non-default result_accuracy prevents expansion

// CHECK-PRECISION-LABEL: func.func @log_plus_one_complex_result_accuracy_f32(
// CHECK-PRECISION:         stablehlo.log_plus_one
// CHECK-PRECISION-SAME:    result_accuracy

// CHECK-ALL-LABEL: func.func @log_plus_one_complex_result_accuracy_f32(
// CHECK-ALL:         stablehlo.log_plus_one
// CHECK-ALL-SAME:    result_accuracy
func.func @log_plus_one_complex_result_accuracy_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "stablehlo.log_plus_one"(%arg) {
    result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e-05, rtol = 0.000000e+00, ulps = 1, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  } : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

//////// AddOp: precision mode leaves complex add unchanged

// CHECK-PRECISION-LABEL: func.func @add_complex_f32(
// CHECK-PRECISION:         stablehlo.add %arg0, %arg1 : tensor<4xcomplex<f32>>
// CHECK-PRECISION:         return

// CHECK-ALL-LABEL: func.func @add_complex_f32(
// CHECK-ALL-SAME:    %[[LHS:.*]]: tensor<4xcomplex<f32>>, %[[RHS:.*]]: tensor<4xcomplex<f32>>
// CHECK-ALL-DAG:     %[[LR:.*]] = stablehlo.real %[[LHS]]
// CHECK-ALL-DAG:     %[[LI:.*]] = stablehlo.imag %[[LHS]]
// CHECK-ALL-DAG:     %[[RR:.*]] = stablehlo.real %[[RHS]]
// CHECK-ALL-DAG:     %[[RI:.*]] = stablehlo.imag %[[RHS]]
// CHECK-ALL-DAG:     %[[ADDR:.*]] = stablehlo.add %[[LR]], %[[RR]] : tensor<4xf32>
// CHECK-ALL-DAG:     %[[ADDI:.*]] = stablehlo.add %[[LI]], %[[RI]] : tensor<4xf32>
// CHECK-ALL:         %[[RES:.*]] = stablehlo.complex %[[ADDR]], %[[ADDI]]
// CHECK-ALL:         return %[[RES]]
func.func @add_complex_f32(%lhs: tensor<4xcomplex<f32>>, %rhs: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %result = stablehlo.add %lhs, %rhs : tensor<4xcomplex<f32>>
  func.return %result : tensor<4xcomplex<f32>>
}

// -----

//////// AddOp: f64

// CHECK-ALL-LABEL: func.func @add_complex_f64(
// CHECK-ALL-SAME:    %[[LHS:.*]]: tensor<2xcomplex<f64>>, %[[RHS:.*]]: tensor<2xcomplex<f64>>
// CHECK-ALL-DAG:     %[[LR:.*]] = stablehlo.real %[[LHS]]
// CHECK-ALL-DAG:     %[[LI:.*]] = stablehlo.imag %[[LHS]]
// CHECK-ALL-DAG:     %[[RR:.*]] = stablehlo.real %[[RHS]]
// CHECK-ALL-DAG:     %[[RI:.*]] = stablehlo.imag %[[RHS]]
// CHECK-ALL-DAG:     %[[ADDR:.*]] = stablehlo.add %[[LR]], %[[RR]] : tensor<2xf64>
// CHECK-ALL-DAG:     %[[ADDI:.*]] = stablehlo.add %[[LI]], %[[RI]] : tensor<2xf64>
// CHECK-ALL:         %[[RES:.*]] = stablehlo.complex %[[ADDR]], %[[ADDI]]
// CHECK-ALL:         return %[[RES]]
func.func @add_complex_f64(%lhs: tensor<2xcomplex<f64>>, %rhs: tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>> {
  %result = stablehlo.add %lhs, %rhs : tensor<2xcomplex<f64>>
  func.return %result : tensor<2xcomplex<f64>>
}

// -----

//////// AddOp: real types are not expanded even in mode=all

// CHECK-ALL-LABEL: func.func @add_real_f32(
// CHECK-ALL:         stablehlo.add %arg0, %arg1 : tensor<4xf32>
// CHECK-ALL-NOT:     stablehlo.real
func.func @add_real_f32(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
  %result = stablehlo.add %lhs, %rhs : tensor<4xf32>
  func.return %result : tensor<4xf32>
}

// -----

//////// SubtractOp

// CHECK-PRECISION-LABEL: func.func @subtract_complex_f32(
// CHECK-PRECISION:         stablehlo.subtract %arg0, %arg1 : tensor<4xcomplex<f32>>

// CHECK-ALL-LABEL: func.func @subtract_complex_f32(
// CHECK-ALL-SAME:    %[[LHS:.*]]: tensor<4xcomplex<f32>>, %[[RHS:.*]]: tensor<4xcomplex<f32>>
// CHECK-ALL-DAG:     %[[LR:.*]] = stablehlo.real %[[LHS]]
// CHECK-ALL-DAG:     %[[LI:.*]] = stablehlo.imag %[[LHS]]
// CHECK-ALL-DAG:     %[[RR:.*]] = stablehlo.real %[[RHS]]
// CHECK-ALL-DAG:     %[[RI:.*]] = stablehlo.imag %[[RHS]]
// CHECK-ALL-DAG:     %[[SUBR:.*]] = stablehlo.subtract %[[LR]], %[[RR]] : tensor<4xf32>
// CHECK-ALL-DAG:     %[[SUBI:.*]] = stablehlo.subtract %[[LI]], %[[RI]] : tensor<4xf32>
// CHECK-ALL:         %[[RES:.*]] = stablehlo.complex %[[SUBR]], %[[SUBI]]
// CHECK-ALL:         return %[[RES]]
func.func @subtract_complex_f32(%lhs: tensor<4xcomplex<f32>>, %rhs: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %result = stablehlo.subtract %lhs, %rhs : tensor<4xcomplex<f32>>
  func.return %result : tensor<4xcomplex<f32>>
}

// -----

//////// MultiplyOp: (a+bi)(c+di) = (ac-bd) + (ad+bc)i

// CHECK-PRECISION-LABEL: func.func @multiply_complex_f32(
// CHECK-PRECISION:         stablehlo.multiply %arg0, %arg1 : tensor<4xcomplex<f32>>

// CHECK-ALL-LABEL: func.func @multiply_complex_f32(
// CHECK-ALL-SAME:    %[[LHS:.*]]: tensor<4xcomplex<f32>>, %[[RHS:.*]]: tensor<4xcomplex<f32>>
// CHECK-ALL-DAG:     %[[A:.*]] = stablehlo.real %[[LHS]]
// CHECK-ALL-DAG:     %[[B:.*]] = stablehlo.imag %[[LHS]]
// CHECK-ALL-DAG:     %[[C:.*]] = stablehlo.real %[[RHS]]
// CHECK-ALL-DAG:     %[[D:.*]] = stablehlo.imag %[[RHS]]
// CHECK-ALL-DAG:     %[[AC:.*]] = stablehlo.multiply %[[A]], %[[C]]
// CHECK-ALL-DAG:     %[[BD:.*]] = stablehlo.multiply %[[B]], %[[D]]
// CHECK-ALL-DAG:     %[[AD:.*]] = stablehlo.multiply %[[A]], %[[D]]
// CHECK-ALL-DAG:     %[[BC:.*]] = stablehlo.multiply %[[B]], %[[C]]
// CHECK-ALL-DAG:     %[[REAL:.*]] = stablehlo.subtract %[[AC]], %[[BD]]
// CHECK-ALL-DAG:     %[[IMAG:.*]] = stablehlo.add %[[AD]], %[[BC]]
// CHECK-ALL:         %[[RES:.*]] = stablehlo.complex %[[REAL]], %[[IMAG]]
// CHECK-ALL:         return %[[RES]]
func.func @multiply_complex_f32(%lhs: tensor<4xcomplex<f32>>, %rhs: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %result = stablehlo.multiply %lhs, %rhs : tensor<4xcomplex<f32>>
  func.return %result : tensor<4xcomplex<f32>>
}

// -----

//////// DivideOp: Smith's method for numerical stability

// CHECK-PRECISION-LABEL: func.func @divide_complex_f32(
// CHECK-PRECISION:         stablehlo.divide %arg0, %arg1 : tensor<4xcomplex<f32>>

// CHECK-ALL-LABEL: func.func @divide_complex_f32(
// CHECK-ALL-SAME:    %[[LHS:.*]]: tensor<4xcomplex<f32>>, %[[RHS:.*]]: tensor<4xcomplex<f32>>
// CHECK-ALL-DAG:     stablehlo.real %[[LHS]]
// CHECK-ALL-DAG:     stablehlo.imag %[[LHS]]
// CHECK-ALL-DAG:     stablehlo.real %[[RHS]]
// CHECK-ALL-DAG:     stablehlo.imag %[[RHS]]
// CHECK-ALL:         stablehlo.abs
// CHECK-ALL:         stablehlo.abs
// CHECK-ALL:         stablehlo.compare LE
// CHECK-ALL:         stablehlo.select
// CHECK-ALL:         stablehlo.select
// CHECK-ALL:         stablehlo.complex
// CHECK-ALL-NOT:     stablehlo.divide {{.*}} tensor<4xcomplex<f32>>
func.func @divide_complex_f32(%lhs: tensor<4xcomplex<f32>>, %rhs: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %result = stablehlo.divide %lhs, %rhs : tensor<4xcomplex<f32>>
  func.return %result : tensor<4xcomplex<f32>>
}

// -----

//////// DivideOp: f64

// CHECK-ALL-LABEL: func.func @divide_complex_f64(
// CHECK-ALL-SAME:    %[[LHS:.*]]: tensor<2xcomplex<f64>>, %[[RHS:.*]]: tensor<2xcomplex<f64>>
// CHECK-ALL:         stablehlo.compare LE
// CHECK-ALL:         stablehlo.select
// CHECK-ALL:         stablehlo.complex
// CHECK-ALL-NOT:     stablehlo.divide {{.*}} tensor<2xcomplex<f64>>
func.func @divide_complex_f64(%lhs: tensor<2xcomplex<f64>>, %rhs: tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>> {
  %result = stablehlo.divide %lhs, %rhs : tensor<2xcomplex<f64>>
  func.return %result : tensor<2xcomplex<f64>>
}

// -----

//////// NegateOp

// CHECK-PRECISION-LABEL: func.func @negate_complex_f32(
// CHECK-PRECISION:         stablehlo.negate %arg0 : tensor<4xcomplex<f32>>

// CHECK-ALL-LABEL: func.func @negate_complex_f32(
// CHECK-ALL-SAME:    %[[ARG:.*]]: tensor<4xcomplex<f32>>
// CHECK-ALL-DAG:     %[[R:.*]] = stablehlo.real %[[ARG]]
// CHECK-ALL-DAG:     %[[I:.*]] = stablehlo.imag %[[ARG]]
// CHECK-ALL-DAG:     %[[NR:.*]] = stablehlo.negate %[[R]] : tensor<4xf32>
// CHECK-ALL-DAG:     %[[NI:.*]] = stablehlo.negate %[[I]] : tensor<4xf32>
// CHECK-ALL:         %[[RES:.*]] = stablehlo.complex %[[NR]], %[[NI]]
// CHECK-ALL:         return %[[RES]]
func.func @negate_complex_f32(%arg: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %result = stablehlo.negate %arg : tensor<4xcomplex<f32>>
  func.return %result : tensor<4xcomplex<f32>>
}

// -----

//////// NegateOp: real types are not expanded

// CHECK-ALL-LABEL: func.func @negate_real_f32(
// CHECK-ALL:         stablehlo.negate %arg0 : tensor<4xf32>
// CHECK-ALL-NOT:     stablehlo.real
func.func @negate_real_f32(%arg: tensor<4xf32>) -> tensor<4xf32> {
  %result = stablehlo.negate %arg : tensor<4xf32>
  func.return %result : tensor<4xf32>
}

// -----

//////// Dynamic shapes: patterns work with dynamic dimensions

// CHECK-ALL-LABEL: func.func @add_complex_dynamic(
// CHECK-ALL-SAME:    %[[LHS:.*]]: tensor<?xcomplex<f32>>, %[[RHS:.*]]: tensor<?xcomplex<f32>>
// CHECK-ALL-DAG:     %[[LR:.*]] = stablehlo.real %[[LHS]]
// CHECK-ALL-DAG:     %[[LI:.*]] = stablehlo.imag %[[LHS]]
// CHECK-ALL-DAG:     %[[RR:.*]] = stablehlo.real %[[RHS]]
// CHECK-ALL-DAG:     %[[RI:.*]] = stablehlo.imag %[[RHS]]
// CHECK-ALL:         stablehlo.add %[[LR]], %[[RR]] : tensor<?xf32>
// CHECK-ALL:         stablehlo.add %[[LI]], %[[RI]] : tensor<?xf32>
// CHECK-ALL:         stablehlo.complex
func.func @add_complex_dynamic(%lhs: tensor<?xcomplex<f32>>, %rhs: tensor<?xcomplex<f32>>) -> tensor<?xcomplex<f32>> {
  %result = stablehlo.add %lhs, %rhs : tensor<?xcomplex<f32>>
  func.return %result : tensor<?xcomplex<f32>>
}

// -----

//////// Scalar (0-d) tensors

// CHECK-ALL-LABEL: func.func @multiply_complex_scalar(
// CHECK-ALL-SAME:    %[[LHS:.*]]: tensor<complex<f32>>, %[[RHS:.*]]: tensor<complex<f32>>
// CHECK-ALL:         stablehlo.real %[[LHS]]
// CHECK-ALL:         stablehlo.imag %[[LHS]]
// CHECK-ALL:         stablehlo.multiply
// CHECK-ALL:         stablehlo.complex
func.func @multiply_complex_scalar(%lhs: tensor<complex<f32>>, %rhs: tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = stablehlo.multiply %lhs, %rhs : tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

//////// Multi-dimensional tensors

// CHECK-ALL-LABEL: func.func @subtract_complex_2d(
// CHECK-ALL:         stablehlo.subtract {{.*}} : tensor<3x4xf32>
// CHECK-ALL:         stablehlo.subtract {{.*}} : tensor<3x4xf32>
// CHECK-ALL:         stablehlo.complex
func.func @subtract_complex_2d(%lhs: tensor<3x4xcomplex<f32>>, %rhs: tensor<3x4xcomplex<f32>>) -> tensor<3x4xcomplex<f32>> {
  %result = stablehlo.subtract %lhs, %rhs : tensor<3x4xcomplex<f32>>
  func.return %result : tensor<3x4xcomplex<f32>>
}
