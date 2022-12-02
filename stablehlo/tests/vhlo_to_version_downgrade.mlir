// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=0.3.0' %s | FileCheck %s


// CHECK-LABEL: @all_gather_to_v1
func.func @all_gather_to_v1(%arg0: tensor<16x8xf32>) -> tensor<16x16xf32> {
  // CHECK-NEXT: %0 = "vhlo.all_gather"(%arg0)
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<16x8xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// CHECK-LABEL: @collective_permute_to_v1
func.func @collective_permute_to_v1(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK-NEXT: %0 = "vhlo.collective_permute"(%arg0)
  %0 = "stablehlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: @custom_call_to_v1
func.func @custom_call_to_v1(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  // CHECK-NEXT: %0 = "vhlo.custom_call"(%arg0)
  %0 = stablehlo.custom_call @foo(%arg0) : (tensor<2xi1>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK-LABEL: @custom_call_to_v1_empty_output_operand_alias
func.func @custom_call_to_v1_empty_output_operand_alias(%arg0 : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: %0 = "vhlo.custom_call"(%arg0)
  %0 = stablehlo.custom_call @foo(%arg0) {
    has_side_effect = false,
    operand_layouts = [dense<> : tensor<0xindex>],
    output_operand_aliases = [],
    result_layouts = [dense<> : tensor<0xindex>]
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
