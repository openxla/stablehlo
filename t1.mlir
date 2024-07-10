func.func @attr_custom_call_api_version_typed_ffi(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    api_version = 4 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
