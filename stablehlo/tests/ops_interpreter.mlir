// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @main
module @distribution_ops {
  func.func @foo() {
    func.return
  }
  func.func @main() {
    %results:2 = "interpreter.run_parallel"() {
      programs=["foo"],
      num_replicas=1,
      num_partitions=1
    } : () -> (tensor<ui32>, tensor<ui32>)
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
    %results:2 = "interpreter.run_parallel"() {
      programs=["bar"],
      num_replicas=2,
      num_partitions=1
    } : () -> (tensor<ui32>, tensor<ui32>)
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
    // expected-error@+1 {{The inputs size: 2 does not match sum of all inputs of programs: 1}}
    %results:2 = "interpreter.run_parallel"(%inputs, %inputs) {
      programs=["foo"],
      num_replicas=1,
      num_partitions=1
    } : (tensor<i64>, tensor<i64>) -> (tensor<ui32>, tensor<ui32>)
    func.return
  }
}

// -----

module @run_parallel_invalid_programs_size {
  func.func @foo() {
    func.return
  }
  func.func @main() {
    // expected-error@+1 {{Number of programs should match numReplicas * numPartitions (2 * 1) but got 1}}
    %results:2 = "interpreter.run_parallel"() {
      programs=["foo"],
      num_replicas=2,
      num_partitions=1
    } : () -> (tensor<ui32>, tensor<ui32>)
    func.return
  }
}
