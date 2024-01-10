// RUN: stablehlo-translate --interpret --probe-output-dir=%T -split-input-file %s

// Test an empty module

// -----

func.func @probe_i1() {
  %0 = stablehlo.constant dense<[[0], [0], [0]]> : tensor<3x1xi1>
  %1 = stablehlo.constant dense<[[1], [1], [1]]> : tensor<3x1xi1>
  %2 = stablehlo.add %0, %1 : tensor<3x1xi1>
  %3 = interpreter.probe %2, probe_id = "probe_i1" : tensor<3x1xi1>
  check.expect_serialized_eq %3, probe_id = "probe_i1" : tensor<3x1xi1>
  func.return
}

// -----

func.func @probe_si64() {
  %0 = stablehlo.constant dense<[[-127], [126], [0]]> : tensor<3x1xi64>
  %1 = stablehlo.constant dense<[[-1], [1], [1]]> : tensor<3x1xi64>
  %2 = stablehlo.add %0, %1 : tensor<3x1xi64>
  %3 = interpreter.probe %2, probe_id = "probe_si64" : tensor<3x1xi64>
  check.expect_serialized_eq %3, probe_id = "probe_si64" : tensor<3x1xi64>
  func.return
}

// -----

func.func @probe_ui64() {
  %0 = stablehlo.constant dense<[[32766, 0], [0, 0]]> : tensor<2x2xui64>
  %1 = stablehlo.constant dense<[[1, 1], [2, 2]]> : tensor<2x2xui64>
  %2 = stablehlo.add %0, %1 : tensor<2x2xui64>
  %3 = interpreter.probe %2, probe_id = "probe_ui64" : tensor<2x2xui64>
  check.expect_serialized_eq %3, probe_id = "probe_ui64" : tensor<2x2xui64>
  func.return
}

// -----

func.func @probe_f64() {
  %0 = stablehlo.constant dense<[3.402e+38, 1.175e-38, 0.0e+00]> : tensor<3xf64>
  %1 = stablehlo.constant dense<[1.0e+00, -1.0e+00, 1.0e+00]> : tensor<3xf64>
  %2 = stablehlo.add %0, %1 : tensor<3xf64>
  %3 = interpreter.probe %2, probe_id = "probe_f64" : tensor<3xf64>
  check.expect_serialized_eq %3, probe_id = "probe_f64" : tensor<3xf64>
  func.return
}

// -----

func.func @probe_c32() {
  %0 = stablehlo.constant dense<[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]> : tensor<3xcomplex<f32>>
  %1 = stablehlo.constant dense<[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]> : tensor<3xcomplex<f32>>
  %2 = stablehlo.add %0, %1 : tensor<3xcomplex<f32>>
  %3 = interpreter.probe %2, probe_id = "probe_c32" : tensor<3xcomplex<f32>>
  check.expect_serialized_eq %3, probe_id = "probe_c32" : tensor<3xcomplex<f32>>
  func.return
}

// -----

func.func @probe_sanitized_probe_id() {
  %0 = stablehlo.constant dense<[[1], [2], [3]]> : tensor<3x1xi64>
  %1 = stablehlo.constant dense<[[4], [5], [6]]> : tensor<3x1xi64>
  %2 = stablehlo.add %0, %1 : tensor<3x1xi64>
  %3 = interpreter.probe %2, probe_id = "probe/0" : tensor<3x1xi64>
  check.expect_serialized_eq %3, probe_id = "probe/0" : tensor<3x1xi64>
  func.return
}

// -----

func.func @probe_iterations() {
  // int i = 0;
  // int sum = 0;
  // while (i < 10) {
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
    %new_sum = stablehlo.add %arg1, %one : tensor<i64>
    %new_sum_instrumented = interpreter.probe %new_sum, probe_id = "probe_iterations" : tensor<i64>
    %new_i = stablehlo.add %arg0, %one : tensor<i64>
    stablehlo.return %new_i, %new_sum_instrumented : tensor<i64>, tensor<i64>
  }

  check.expect_eq_const %results0, dense<2> : tensor<i64>
  check.expect_serialized_eq %one, probe_id = "probe_iterations", iter = 0 : tensor<i64>
  check.expect_serialized_eq %two, probe_id = "probe_iterations", iter = 1 : tensor<i64>
  func.return
}
