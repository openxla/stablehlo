// RUN: stablehlo-opt --vhlo-to-version='target=0.4.0' %s | FileCheck %s
// RUN: stablehlo-opt --vhlo-to-version='target=current' %s | FileCheck %s

// CHECK-LABEL: @all_gather_to_v2
func.func @all_gather_to_v2(%arg0: tensor<16x8xf32>) -> tensor<16x16xf32> {
  // CHECK-NEXT: %0 = "vhlo.all_gather_v2"(%arg0)
  %0 = "vhlo.all_gather"(%arg0) {all_gather_dim = 1 : i64, channel_handle = #vhlo.channel_handle<handle = 0, type = 0>, replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>} : (tensor<16x8xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK-LABEL: @collective_permute_to_v2
func.func @collective_permute_to_v2(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK-NEXT: %0 = "vhlo.collective_permute_v2"(%arg0)
  %0 = "vhlo.collective_permute"(%arg0) {source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>} : (tensor<16x8xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: @custom_call_to_v2
func.func @custom_call_to_v2(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  // CHECK-NEXT: %0 = "vhlo.custom_call_v2"(%arg0)
  %0 = "vhlo.custom_call"(%arg0) {backend_config = "", call_target_name = "foo"} : (tensor<2xi1>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}
