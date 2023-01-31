// RUN: stablehlo-opt --vhlo-to-version='target=0.4.0' %s | FileCheck %s
// RUN: stablehlo-opt --vhlo-to-version='target=current' %s | FileCheck %s

// CHECK-LABEL: @all_to_all_to_v2
func.func @all_to_all_to_v2(%arg0: !vhlo.tensor<4x16x!vhlo.f32>) -> !vhlo.tensor<16x4x!vhlo.f32> {
  // CHECK-NEXT: %0 = "vhlo.all_to_all_v2"(%arg0)
  %0 = "vhlo.all_to_all"(%arg0) {
    split_dimension = #vhlo.integer<1 : i64>,
    concat_dimension = #vhlo.integer<0 : i64>,
    split_count = #vhlo.integer<4 : i64>,
    replica_groups = #vhlo.dense<dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>>
  } : (!vhlo.tensor<4x16x!vhlo.f32>) -> !vhlo.tensor<16x4x!vhlo.f32>
  func.return %0 : !vhlo.tensor<16x4x!vhlo.f32>
}

// CHECK-LABEL: @all_gather_to_v2
func.func @all_gather_to_v2(%arg0: !vhlo.tensor<16x8x!vhlo.f32>) -> !vhlo.tensor<16x16x!vhlo.f32> {
  // CHECK-NEXT: %0 = "vhlo.all_gather_v2"(%arg0)
  %0 = "vhlo.all_gather"(%arg0) {
    all_gather_dim = #vhlo.integer<1 : i64>,
    channel_handle = #vhlo.channel_handle<handle = 0, type = 0>,
    replica_groups = #vhlo.dense<dense<[[0], [1]]> : tensor<2x1xi64>>
  } : (!vhlo.tensor<16x8x!vhlo.f32>) -> !vhlo.tensor<16x16x!vhlo.f32>
  return %0 : !vhlo.tensor<16x16x!vhlo.f32>
}

// CHECK-LABEL: @collective_permute_to_v2
func.func @collective_permute_to_v2(%arg0: !vhlo.tensor<16x8x!vhlo.f32>) -> !vhlo.tensor<16x8x!vhlo.f32> {
  // CHECK-NEXT: %0 = "vhlo.collective_permute_v2"(%arg0)
  %0 = "vhlo.collective_permute"(%arg0) {
    source_target_pairs = #vhlo.dense<dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>>
  } : (!vhlo.tensor<16x8x!vhlo.f32>) -> !vhlo.tensor<16x8x!vhlo.f32>
  return %0 : !vhlo.tensor<16x8x!vhlo.f32>
}

// CHECK-LABEL: @custom_call_to_v2
func.func @custom_call_to_v2(%arg0: !vhlo.tensor<2x!vhlo.i1>) -> !vhlo.tensor<2x!vhlo.i1> {
  // CHECK-NEXT: %0 = "vhlo.custom_call_v2"(%arg0)
  %0 = "vhlo.custom_call"(%arg0) {
    backend_config = #vhlo.string<"">,
    call_target_name = #vhlo.string<"foo">
  } : (!vhlo.tensor<2x!vhlo.i1>) -> !vhlo.tensor<2x!vhlo.i1>
  return %0 : !vhlo.tensor<2x!vhlo.i1>
}
