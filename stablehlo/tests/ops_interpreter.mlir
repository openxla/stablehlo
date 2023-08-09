// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// CHECK-LABEL: func @main
module @run_parallel_success {
  func.func @foo() {
    func.return
  }
  func.func @main() {
    "interpreter.run_parallel"() {
      programs=[["foo"]]
    } : () -> ()
    func.return
  }
}

// -----

module @run_parallel_invalid_programs_size {
  func.func @foo() {
    func.return
  }
  func.func @main() {
    // expected-error@+1 {{`programs` attribute cannot be empty}}
    "interpreter.run_parallel"() {
      programs=[]
    } : () -> ()
    func.return
  }
}

// -----

module @run_parallel_invalid_programs_shape {
  func.func @foo() {
    func.return
  }
  func.func @main() {
    // expected-error@+1 {{Sizes of second dimension of `programs` should all match 1 but got 2}}
    "interpreter.run_parallel"() {
      programs=[["foo"], ["bar", "baz"]]
    } : () -> ()
    func.return
  }
}

// -----

module @run_parallel_invalid_function {
  func.func @foo() {
    func.return
  }
  func.func @main() {
    // expected-error@+1 {{Function "bar" not found}}
    "interpreter.run_parallel"() {
      programs=[["bar"], ["bar"]]
    } : () -> ()
    func.return
  }
}

// -----

module @run_parallel_invalid_arg_size {
  func.func @foo(%arg0 : tensor<i64>) {
    func.return
  }
  func.func @main() {
    %inputs = stablehlo.constant dense<0> : tensor<i64>
    // expected-error@+1 {{Number of inputs (2) should match the sum of the number of inputs of all programs (1)}}
    "interpreter.run_parallel"(%inputs, %inputs) {
      programs=[["foo"]]
    } : (tensor<i64>, tensor<i64>) -> ()
    func.return
  }
}

// -----

module @run_parallel_invalid_results_size {
  func.func @foo(%arg0 : tensor<i64>) -> tensor<i64> {
    func.return %arg0 : tensor<i64>
  }
  func.func @main() {
    %inputs = stablehlo.constant dense<0> : tensor<i64>
    // expected-error@+1 {{Number of results (0) should match the sum of the number of results of all programs (1)}}
    "interpreter.run_parallel"(%inputs) {
      programs=[["foo"]]
    } : (tensor<i64>) -> ()
    func.return
  }
}
