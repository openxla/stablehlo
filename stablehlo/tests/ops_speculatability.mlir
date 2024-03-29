// RUN: stablehlo-opt %s --hlo-test-speculatability --split-input-file --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @dynamic_iota
// CHECK-NEXT:  return
func.func @dynamic_iota(%unknown_shape: tensor<2xi32>) {
  %constant_shape = stablehlo.constant dense<[3, 4]> : tensor<2xi32>
  %speculatable_0 = stablehlo.dynamic_iota %constant_shape, dim = 0 : (tensor<2xi32>) -> tensor<3x4xi64>
  %speculatable_1 = stablehlo.dynamic_iota %constant_shape, dim = 0 : (tensor<2xi32>) -> tensor<3x?xi64>
  %speculatable_2 = stablehlo.dynamic_iota %constant_shape, dim = 0 : (tensor<2xi32>) -> tensor<?x4xi64>
  %speculatable_3 = stablehlo.dynamic_iota %constant_shape, dim = 0 : (tensor<2xi32>) -> tensor<?x?xi64>
  %not_speculatable_0 = stablehlo.dynamic_iota %unknown_shape, dim = 0 : (tensor<2xi32>) -> tensor<3x4xi64>
  %not_speculatable_1 = stablehlo.dynamic_iota %unknown_shape, dim = 0 : (tensor<2xi32>) -> tensor<3x?xi64>
  %not_speculatable_2 = stablehlo.dynamic_iota %unknown_shape, dim = 0 : (tensor<2xi32>) -> tensor<?x4xi64>
  %speculatable_4 = stablehlo.dynamic_iota %unknown_shape, dim = 0 : (tensor<2xi32>) -> tensor<?x?xi64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<3x4xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<3x?xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?x4xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_3) : (tensor<?x?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<3x4xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<3x?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<?x4xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_4) : (tensor<?x?xi64>) -> ()
  return
}
