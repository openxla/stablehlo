// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect --stablehlo-create-compatibility-expander='target=1.0.0' | FileCheck %s --check-prefixes=CHECK
// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file --stablehlo-create-compatibility-expander='target=1.6.0' | FileCheck %s --check-prefixes=CHECK-NO-DOWNGRADE

// -----

// CHECK-LABEL @tan_op_non_complex
// CHECK: %[[sine0:.*]] = stablehlo.sine %arg0 : tensor<4xf64>
// CHECK-NEXT: %[[cosine1:.*]] = stablehlo.cosine %arg0 : tensor<4xf64>
// CHECK-NEXT: %[[div2:.*]] = stablehlo.divide %[[sine0]], %[[cosine1]] : tensor<4xf64>
// CHECK-NEXT: return %[[div2]] : tensor<4xf64>
func.func @tan_op_non_complex(%arg0: tensor<4xf64>) -> tensor<4xf64> {
  // CHECK-NO-DOWNGRADE: stablehlo.tan %arg0 : tensor<4xf64>
  %1 = stablehlo.tan %arg0 : tensor<4xf64>
  func.return %1 : tensor<4xf64>
}

// -----

// CHECK-LABEL: @tan_op_complex
// CHECK: %[[cst:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<4xf64>
// CHECK: %[[complex:.*]] = stablehlo.complex %arg0, %arg1 : tensor<4xcomplex<f64>>
// CHECK: %[[real:.*]] = stablehlo.real %[[complex]] : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
// CHECK: %[[sine:.*]] = stablehlo.sine %[[real]] : tensor<4xf64>
// CHECK: %[[cosine:.*]] = stablehlo.cosine %[[real]] : tensor<4xf64>
// CHECK: %[[divide1:.*]] = stablehlo.divide %[[sine]], %[[cosine]] : tensor<4xf64>
// CHECK: %[[imag:.*]] = stablehlo.imag %[[complex]] : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
// CHECK: %[[tanh:.*]] = stablehlo.tanh %[[imag]] : tensor<4xf64>
// CHECK: %[[complex2:.*]] = stablehlo.complex %[[divide1]], %[[tanh]] : tensor<4xcomplex<f64>>
// CHECK: %[[multiply:.*]] = stablehlo.multiply %[[divide1]], %[[tanh]] : tensor<4xf64>
// CHECK: %[[negate:.*]] = stablehlo.negate %[[multiply]] : tensor<4xf64>
// CHECK: %[[complex3:.*]] = stablehlo.complex %[[cst]], %[[negate]] : tensor<4xcomplex<f64>>
// CHECK: %[[divide2:.*]] = stablehlo.divide %[[complex2]], %[[complex3]] : tensor<4xcomplex<f64>>
// CHECK: %[[real2:.*]] = stablehlo.real %[[divide2]] : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
// CHECK: %[[imag2:.*]] = stablehlo.imag %[[divide2]] : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
// CHECK: return %[[real2]], %[[imag2]] : tensor<4xf64>, tensor<4xf64>
func.func @tan_op_complex(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
  %0 = stablehlo.complex %arg0, %arg1 : tensor<4xcomplex<f64>>
  // CHECK-NO-DOWNGRADE: stablehlo.tan %0 : tensor<4xcomplex<f64>>
  %1 = stablehlo.tan %0 : tensor<4xcomplex<f64>>
  %2 = stablehlo.real %1 : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
  %3 = stablehlo.imag %1 : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
  func.return %2, %3 : tensor<4xf64>, tensor<4xf64>
}
