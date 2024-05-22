// RUN: stablehlo-opt %s --stablehlo-convert-to-signless --canonicalize --split-input-file | FileCheck %s

func.func @uint16_to_int16(%arg0: memref<*xui16>) -> memref<ui16> {
  // CHECK-NOT: unrealized_conversion_cast
  // CHECK: %[[CAST:.*]] = memref.cast %arg0 : memref<*xi16> to memref<i16>
  // CHECK: return %[[CAST]] : memref<i16>
  %1 = builtin.unrealized_conversion_cast %arg0 : memref<*xui16> to memref<*xi16>
  %2 = memref.cast %1 : memref<*xi16> to memref<i16>
  %3 = builtin.unrealized_conversion_cast %2 : memref<i16> to memref<ui16>
  %4 = bufferization.to_tensor %3 : memref<ui16>
  %5 = builtin.unrealized_conversion_cast %4 : tensor<ui16> to tensor<i16>
  %6 = bufferization.to_memref %5 : memref<i16>
  %7 = builtin.unrealized_conversion_cast %6 : memref<i16> to memref<ui16>
  func.return %7 : memref<ui16>
}

// -----

func.func @stablehlo_uint8_add(%arg0: tensor<3xui8>) -> tensor<3xui8> {
  // CHECK: %[[CST:.*]] = arith.constant dense<[0, 1, -1]> : tensor<3xi8>
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %[[CST]] : tensor<3xi8>
  // CHECK: return %[[ADD]] : tensor<3xi8>
  %cst = arith.constant dense<[0, 1, 255]> : tensor<3xui8>
  %0 = stablehlo.add %arg0, %cst : tensor<3xui8>
  return %0 : tensor<3xui8>
}
