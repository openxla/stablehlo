// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// CHECK-LABEL: func @main
module @run_parallel_success {
  func.func @foo() {
    func.return
  }
  func.func @main() {
    "interpreter.run_parallel"() {
      programs=[[@foo]]
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
      programs=[[@foo], [@bar, @baz]]
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
      programs=[[@bar], [@bar]]
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
      programs=[[@foo]]
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
      programs=[[@foo]]
    } : (tensor<i64>) -> ()
    func.return
  }
}

// -----

module @run_parallel_empty_infeed {
  func.func @foo() {
    func.return
  }
  func.func @main() {
    // expected-error@+1 {{infeed attribute is optional or should not be empty}}
    "interpreter.run_parallel"() {
      infeed=[],
      programs=[[@foo]]
    } : () -> ()
    func.return
  }
}

// -----

module @run_parallel_invalid_infeed {
  func.func @foo() {
    func.return
  }
  func.func @main() {
    // expected-error@+1 {{Function "bar" not found}}
    "interpreter.run_parallel"() {
      infeed=[@bar],
      programs=[[@foo]]
    } : () -> ()
    func.return
  }
}

// -----

module @run_parallel_invalid_infeed_return_count {
  func.func @infeed(%token : !stablehlo.token) -> (tensor<i64>, !stablehlo.token) {
    %results0:2 = "stablehlo.infeed"(%token) :
        (!stablehlo.token) -> (tensor<i64>, !stablehlo.token)
    func.return %results0#0, %results0#1 : tensor<i64>, !stablehlo.token
  }
  func.func @infeed_queue() -> (tensor<i64>, tensor<i64>) {
    %result = stablehlo.constant dense<1> : tensor<i64>
    func.return %result, %result : tensor<i64>, tensor<i64>
  }
  func.func @main() {
    %token = stablehlo.after_all : !stablehlo.token
    // expected-error@+1 {{Function "infeed_queue" should return 1 tensor but returns 2}}
    "interpreter.run_parallel"(%token) {
      infeed=[@infeed_queue],
      programs=[[@infeed]]
    } : (!stablehlo.token) -> (tensor<i64>, !stablehlo.token)
    func.return
  }
}

// -----

module @run_parallel_invalid_infeed_return_type {
  func.func @infeed(%token : !stablehlo.token) -> (tensor<i64>, !stablehlo.token) {
    %results0:2 = "stablehlo.infeed"(%token) :
        (!stablehlo.token) -> (tensor<i64>, !stablehlo.token)
    func.return %results0#0, %results0#1 : tensor<i64>, !stablehlo.token
  }
  func.func @infeed_queue() -> !stablehlo.token {
    %token = stablehlo.after_all : !stablehlo.token
    func.return %token : !stablehlo.token
  }
  func.func @main() {
    %token = stablehlo.after_all : !stablehlo.token
    // expected-error@+1 {{Function "infeed_queue" should return a tensor type, but instead returns '!stablehlo.token'}}
    "interpreter.run_parallel"(%token) {
      infeed=[@infeed_queue],
      programs=[[@infeed]]
    } : (!stablehlo.token) -> (tensor<i64>, !stablehlo.token)
    func.return
  }
}
