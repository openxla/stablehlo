// RUN: stablehlo-opt --interpreter-instrument-with-probe="useDebugInfo=true" --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @instrument_basic_no_location
func.func @instrument_basic_no_location(%arg0: tensor<1x2xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  // CHECK: [[RESULT:%.*]] = interpreter.probe %0, probe_id = "probe1" : tensor<1x2xi32>
  // CHECK-NEXT: return [[RESULT]]
  %0 = stablehlo.add %arg0, %arg1 : tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func @instrument_basic_location
func.func @instrument_basic_location(%arg0: tensor<1x2xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  // CHECK: [[RESULT:%.*]] = interpreter.probe %0, probe_id = "named_location.1" : tensor<1x2xi32>
  // CHECK-NEXT: return [[RESULT]]
  %0 = stablehlo.add %arg0, %arg1 : tensor<1x2xi32> loc("named_location")
  func.return %0 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func @do_not_instrument_constant
func.func @do_not_instrument_constant() -> tensor<1xi64> {
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<0> : tensor<1xi64>
  // CHECK-NEXT: return [[RESULT]]
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  func.return %0 : tensor<1xi64>
}

// -----

// CHECK-LABEL: func @only_instrument_tensor_type
func.func @only_instrument_tensor_type(%arg0: tensor<f32>) -> (!stablehlo.token, tuple<tensor<f32>>, tensor<f32>) {
  // CHECK: stablehlo.create_token
  // CHECK-NEXT: stablehlo.tuple
  // CHECK-NEXT: [[SUM:%.*]] = stablehlo.add
  // CHECK-NEXT: interpreter.probe [[SUM]]
  // CHECK-NEXT: return
  %0 = "stablehlo.create_token"() : () -> !stablehlo.token
  %1 = "stablehlo.tuple"(%arg0) : (tensor<f32>) -> tuple<tensor<f32>>
  %2 = stablehlo.add %arg0, %arg0 : tensor<f32>
  func.return %0, %1, %2 : !stablehlo.token, tuple<tensor<f32>>, tensor<f32>
}

// -----

// CHECK-LABEL: func @instrument_if
func.func @instrument_if(%arg0: tensor<i1>, %arg1: tensor<2xi64>, %arg2: tensor<2xi64>) -> tensor<2xi64> {
  // CHECK: interpreter.probe {{.*}}, probe_id = "add.1" : tensor<2xi64>
  // CHECK: interpreter.probe {{.*}}, probe_id = "probe2" : tensor<2xi64>
  %result = "stablehlo.if"(%arg0) ({
    %0 = stablehlo.constant dense<0> : tensor<2xi64>
    stablehlo.return %0 : tensor<2xi64>
  }, {
    %0 = stablehlo.add %arg1, %arg2 : tensor<2xi64> loc("add")
    stablehlo.return %0 : tensor<2xi64>
  }) : (tensor<i1>) -> (tensor<2xi64>)
  func.return %result : tensor<2xi64>
}

// -----

// CHECK-LABEL: func @instrument_loop
func.func @instrument_loop() -> tensor<i64> {
  // Instrumented loop condition
  // CHECK: [[WHILE:%.*]]:2 = stablehlo.while
  // CHECK: [[COND:%.*]] = stablehlo.compare LT
  // CHECK-NEXT: [[PROBE1:%.*]] = interpreter.probe [[COND]]

  // Instrumented loop body
  // CHECK: interpreter.probe {{.*}}, probe_id = "add.2" : tensor<i64>
  // CHECK: interpreter.probe {{.*}}, probe_id = "add.3" : tensor<i64>

  // Instrumented loop return values
  // CHECK: interpreter.probe [[WHILE]]#1
  // CHECK-NEXT: interpreter.probe [[WHILE]]#0

  // int i = 0;
  // int sum = 0;
  // while (i < 2) {
  //   sum += 1;
  //   i += 1;
  // }
  %init_i = stablehlo.constant dense<0> : tensor<i64>
  %init_sum = stablehlo.constant dense<0> : tensor<i64>
  %one = stablehlo.constant dense<1> : tensor<i64>
  %two = stablehlo.constant dense<2> : tensor<i64>
  %results0, %results1 = stablehlo.while(%arg0 = %init_i, %arg1 = %init_sum) : tensor<i64>, tensor<i64>
  cond {
    %cond = stablehlo.compare LT, %arg0, %two : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cond : tensor<i1>
  } do {
    %new_sum = stablehlo.add %arg1, %one : tensor<i64> loc("add")
    %new_i = stablehlo.add %arg0, %one : tensor<i64> loc("add")
    stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
  }

  func.return %results1 : tensor<i64>
}
