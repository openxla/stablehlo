// RUN: stablehlo-opt --stablehlo-check-shape-assertions --split-input-file --verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: func.func @assertion_succeeds
// CHECK-NOT: stablehlo.custom_call @shape_assertion
// CHECK: return
func.func @assertion_succeeds() {
  %c1 = stablehlo.constant dense<true> : tensor<i1>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  stablehlo.custom_call @shape_assertion(%c1, %c0) {
    api_version = 2 : i32,
    error_message = "should not fire",
    has_side_effect = true
  } : (tensor<i1>, tensor<i32>) -> ()
  return
}

// -----

// ERR-LABEL: func.func @assertion_fails
func.func @assertion_fails() {
  %c1 = stablehlo.constant dense<false> : tensor<i1>
  %c0 = stablehlo.constant dense<7> : tensor<i32>
  // expected-error@+1 {{should fire}}
  stablehlo.custom_call @shape_assertion(%c1, %c0) {
    api_version = 2 : i32,
    error_message = "should fire",
    has_side_effect = true
  } : (tensor<i1>, tensor<i32>) -> ()
  return
}

// -----

// ERR-LABEL: func.func @assertion_fails_not_constant
func.func @assertion_fails_not_constant(%arg0 : tensor<i1>) {
  %c0 = stablehlo.constant dense<7> : tensor<i32>
  // expected-error@+1 {{expects static assert_what (operand #0)}}
  stablehlo.custom_call @shape_assertion(%arg0, %c0) {
    api_version = 2 : i32,
    error_message = "not firing",
    has_side_effect = true
  } : (tensor<i1>, tensor<i32>) -> ()
  return
}
