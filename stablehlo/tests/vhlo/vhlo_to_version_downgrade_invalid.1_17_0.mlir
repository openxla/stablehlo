// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.17.0' --verify-diagnostics --split-input-file %s

// expected-error @+1 {{failed to convert VHLO to v1.17.0}}
module {
  func.func @custom_call_with_result_tilings(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.custom_call_v2' that was explicitly marked illegal}}
    %0 = "stablehlo.custom_call"(%arg0) {
      call_target_name = "foo",
      operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
      result_layouts = [dense<[1, 0]> : tensor<2xindex>],
      result_tilings = [[dense<[8, 128]> : tensor<2xindex>]]
    } : (tensor<64x256xf32>) -> tensor<64x256xf32>
    func.return %0 : tensor<64x256xf32>
  }
}
