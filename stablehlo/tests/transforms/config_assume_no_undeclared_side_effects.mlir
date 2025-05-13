// RUN: stablehlo-opt --stablehlo-aggressive-folder=assume-no-undeclared-side-effects=true --split-input-file --verify-diagnostics %s | FileCheck %s

func.func @foo() -> tensor<i64> {
  %1 = stablehlo.custom_call @something() : () -> tensor<i64>
  return %1 : tensor<i64>
}

// CHECK-LABEL: dce_while_false_condition
func.func @dce_while_false_condition() -> tensor<i64> {
  %0 = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NOT: stablehlo.while
  %1 = stablehlo.while(%iterArg = %0) : tensor<i64>
    cond {
    %2 = stablehlo.compare LT, %0, %0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  } do {
    %2 = func.call @foo() : () -> tensor<i64>
    stablehlo.return %2 : tensor<i64>
  }
  return %1 : tensor<i64>
}
