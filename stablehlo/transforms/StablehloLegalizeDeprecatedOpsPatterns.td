/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This is the canonicalize pattern definition file.

include "mlir/IR/OpBase.td"
include "stablehlo/dialect/StablehloOps.td"

def UnaryToBinaryEinsumEq : NativeCodeCall<
  "$_builder.getStringAttr(\",\" + $0.getValue().str())">;

def GetI64DenseElementsAttr : NativeCodeCall<
  "$0.mapValues($_builder.getI64Type(), [](llvm::APInt x) { return x.sext(64); })">;

def GetDefaultDotDimensionNumbers : NativeCodeCall<
  "getDefaultDotDimensionNumbers($0)">;

def GetDefaultBroadcastDimensions : NativeCodeCall<
  "getBroadcastDimensionsFromBroadcastSizes(cast<RankedTensorType>($0.getType()), cast<DenseI64ArrayAttr>($1))">;

// Here, the element type can be any integer or float type. But, note that only
// 32 bit integers are supported for the value.
class GetScalarOfType<int value> : NativeCodeCall<
  "getScalarOfType(getElementTypeOrSelf($0)," # value # ")">;

def BroadcastToBroadcastInDim : Pat<
  (StableHLO_BroadcastOp:$result $operand, $broadcast_sizes),
  (StableHLO_BroadcastInDimOp $operand, (GetDefaultBroadcastDimensions $result, $broadcast_sizes))>;

def CreateTokenToAfterAll : Pat<
  (StableHLO_CreateTokenOp),
  (StableHLO_AfterAllOp (NativeCodeCall<"ValueRange{}">))>;

// def CrossReplicaSumToAllReduce : <Declared in cpp>

def DotToDotGeneral : Pat<
  (StableHLO_DotOp $lhs, $rhs, $precision_config),
  (StableHLO_DotGeneralOp $lhs, $rhs, (GetDefaultDotDimensionNumbers $lhs), $precision_config, (NativeCodeCall<"Attribute{}">))>;

def UnaryEinsumToEinsum : Pat<
  (StableHLO_UnaryEinsumOp $operand, $equation),
  (StableHLO_EinsumOp (StableHLO_ConstantOp (GetScalarOfType<1> $operand)),
                $operand, (UnaryToBinaryEinsumEq $equation))>;
