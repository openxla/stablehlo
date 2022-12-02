// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=0.3.0' --verify-diagnostics --split-input-file %s 

func.func @custom_call_v2_with_result_layout(%arg0 : tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+2 {{failed to downgrade vhlo.custom_call_v2, op has a non-empty result_layouts attribute}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.custom_call_v2' that was explicitly marked illegal}}
  %0 = stablehlo.custom_call @foo(%arg0) {
    operand_layouts = [dense<[0]> : tensor<1xindex>],
    result_layouts = [dense<[0]> : tensor<1xindex>]
  } : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// This test emulates two things:
//   1. A file that is too old and no longer supported on consumer.
//   2. A file that is too new and not yet supported on consumer.
// More work should be done to improve this error message.
// TODO: Make github issue to improve error handling in compatibility machinery.
func.func @invalid_program_unknown_op(%arg0 : tensor<f32>) -> (tensor<f32>) {
  // expected-error @+1 {{unregistered operation 'vhlo.unknown_op' found in dialect ('vhlo') that does not allow unknown operations}}
  %0 = "vhlo.unknown_op"(%arg0) : (tensor<f32>) -> tensor<f32> 
  func.return
}
