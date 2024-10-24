// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.4.1' %s | FileCheck %s

// AllToAll was in the initial StableHLO opset, but changed in v1.5.0 to have
// tuple arguments. Ensure that serializing for 1.4.1 is valid and targets the
// v1.4.0 opset.
//
// This will catch issues in op `isLegal` checks:
//   op.minVersion() <= target <= op.maxVersion()

// CHECK-LABEL: vhlo.func_v1 @all_to_all
func.func public @all_to_all(%arg0: tensor<8x8x1xui16>) -> tensor<1x8x8xui16> {
  // CHECK: vhlo.all_to_all_v1
  %0 = "stablehlo.all_to_all"(%arg0) <{concat_dimension = 2 : i64, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, split_count = 8 : i64, split_dimension = 0 : i64}> : (tensor<8x8x1xui16>) -> tensor<1x8x8xui16>
  return %0 : tensor<1x8x8xui16>
}
