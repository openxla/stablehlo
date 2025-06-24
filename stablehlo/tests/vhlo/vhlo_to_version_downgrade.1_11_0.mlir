// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.11.0' %s | FileCheck %s

// SendOp and RecvOp were changed in v1.11.0 to have
// source_target_pair attribute. Ensure that serializing for 1.10.0 is valid and targets the
// v1.10.0 opset.
//
// This will catch issues in op `isLegal` checks:
//   op.minVersion() <= target <= op.maxVersion()

// CHECK-LABEL: vhlo.func_v1 @send_op
// CHECK-NEXT: "vhlo.send_v1"(%arg0, %arg1) <{
// CHECK-SAME:   channel_id = #vhlo.integer_v1<0 : i64>,
// CHECK-SAME:   channel_type = #vhlo.integer_v1<2 : i64>,
// CHECK-SAME:   is_host_transfer = #vhlo.bool_v1<true>
// CHECK-SAME:   !vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.token_v1) -> !vhlo.token_v1
func.func public @send_op(%arg0: tensor<f32>, %arg1: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.send"(%arg0, %arg1) {
    source_target_pairs = dense<[]> : tensor<0xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 2>,
    is_host_transfer = true
  } : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

// CHECK-LABEL: vhlo.func_v1 @recv_op
// CHECK-NEXT: "vhlo.recv_v1"(%arg0) <{
// CHECK-SAME:   channel_id = #vhlo.integer_v1<0 : i64>,
// CHECK-SAME:   channel_type = #vhlo.integer_v1<3 : i64>,
// CHECK-SAME:   is_host_transfer = #vhlo.bool_v1<true>
// CHECK-SAME:   !vhlo.token_v1) -> (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.token_v1)
func.func public @recv_op(%arg0: !stablehlo.token) -> (tensor<f32>, !stablehlo.token) {
  %0:2 = "stablehlo.recv"(%arg0) {
    source_target_pairs = dense<[]> : tensor<0xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 3>,
    is_host_transfer = true
  } : (!stablehlo.token) -> (tensor<f32>, !stablehlo.token)
  func.return %0#0, %0#1 : tensor<f32>, !stablehlo.token
}
