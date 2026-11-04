// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.21.0' --verify-diagnostics --split-input-file %s

// expected-error @+1 {{failed to convert VHLO to v1.21.0}}
module {
  func.func @compare_weakorder(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.compare_v1' that was explicitly marked illegal}}
    %0 = "stablehlo.compare"(%arg0, %arg1) {
      comparison_direction = #stablehlo<comparison_direction LT>,
      compare_type = #stablehlo<comparison_type WEAKORDER>
    } : (tensor<f32>, tensor<f32>) -> tensor<i1>
    func.return %0 : tensor<i1>
  }
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.21.0}}
module {
  func.func @discardable_weakorder_attr(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.add_v1' that was explicitly marked illegal}}
    %0 = "stablehlo.add"(%arg0, %arg1) {
      some_attr = #stablehlo<comparison_type WEAKORDER>
    } : (tensor<f32>, tensor<f32>) -> tensor<f32>
    func.return %0 : tensor<f32>
  }
}

