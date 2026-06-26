// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.16.0' --verify-diagnostics --split-input-file %s

// expected-error @+1 {{failed to convert VHLO to v1.16.0}}
module {
  func.func @custom_call_with_future_type(%arg0: !stablehlo.future<tensor<f32>>) -> !stablehlo.future<tensor<f32>> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.custom_call_v2' that was explicitly marked illegal}}
    %0 = "stablehlo.custom_call"(%arg0) {
      call_target_name = "foo"
    } : (!stablehlo.future<tensor<f32>>) -> !stablehlo.future<tensor<f32>>
    func.return %0 : !stablehlo.future<tensor<f32>>
  }
}
