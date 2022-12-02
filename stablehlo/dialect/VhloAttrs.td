/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_VHLO_ATTRS
#define STABLEHLO_DIALECT_VHLO_ATTRS

include "stablehlo/dialect/VhloBase.td"
include "stablehlo/dialect/VhloEnums.td"

include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def VHLO_Dims : ArrayRefParameter<"int64_t", "Dimension"> {
  let parser = "mlir::hlo::parseDimSizes($_parser)";
  let printer = "mlir::hlo::printDimSizes($_printer, $_self)";
}

def VHLO_ScatterDimensionNumbers : AttrDef<VHLO_Dialect, "ScatterDimensionNumbers"> {
  let cppNamespace = "::mlir::vhlo";
  let mnemonic = "scatter";
  let parameters = (ins
      VHLO_Dims:$updateWindowDims,
      VHLO_Dims:$insertedWindowDims,
      VHLO_Dims:$scatterDimsToOperandDims,
      "int64_t":$indexVectorDim
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def VHLO_GatherDimensionNumbers : AttrDef<VHLO_Dialect, "GatherDimensionNumbers"> {
  let cppNamespace = "::mlir::vhlo";
  let mnemonic = "gather";
  let parameters = (ins
      VHLO_Dims:$offsetDims,
      VHLO_Dims:$collapsedSliceDims,
      VHLO_Dims:$startIndexMap,
      "int64_t":$indexVectorDim
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def VHLO_DotDimensionNumbers : AttrDef<VHLO_Dialect, "DotDimensionNumbers"> {
  let cppNamespace = "::mlir::vhlo";
  let mnemonic = "dot";
  let parameters = (ins
      VHLO_Dims:$lhsBatchingDimensions,
      VHLO_Dims:$rhsBatchingDimensions,
      VHLO_Dims:$lhsContractingDimensions,
      VHLO_Dims:$rhsContractingDimensions
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def VHLO_OutputOperandAlias : AttrDef<VHLO_Dialect, "OutputOperandAlias"> {
  let cppNamespace = "::mlir::vhlo";
  let mnemonic = "output_operand_alias";
  let parameters = (ins
    VHLO_Dims:$outputTupleIndices,
    "int64_t":$operandIndex,
    VHLO_Dims:$operandTupleIndices
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def VHLO_ArgResultAlias : AttrDef<VHLO_Dialect, "ArgResultAlias"> {
  let cppNamespace = "::mlir::vhlo";
  let mnemonic = "result_alias";
  let parameters = (ins
    VHLO_Dims:$argTupleIndices,
    "int64_t":$resultIndex,
    VHLO_Dims:$resultTupleIndices,
    "bool":$isMustAlias
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def VHLO_ChannelHandle : AttrDef<VHLO_Dialect, "ChannelHandle"> {
  let cppNamespace = "::mlir::vhlo";
  let mnemonic = "channel_handle";
  let parameters = (ins "int64_t":$handle, "int64_t":$type);
  let assemblyFormat = "`<` struct(params) `>`";
}

def VHLO_TypeExtensions : AttrDef<VHLO_Dialect, "TypeExtensions"> {
  let cppNamespace = "::mlir::vhlo";
  let mnemonic = "type_extensions";
  let parameters = (ins VHLO_Dims:$bounds);
  let assemblyFormat = "`<` struct(params) `>`";
}

def VHLO_LayoutAttr : Attr<
  And<[IndexElementsAttr.predicate,
       CPred<[{$_self.cast<::mlir::DenseIntElementsAttr>().getType().getRank()
               == 1}]>]>,
  "A 1D tensor of index type (layout)"> {
  let storageType = IndexElementsAttr.storageType;
  let returnType = IndexElementsAttr.returnType;
  let convertFromStorage = IndexElementsAttr.convertFromStorage;
}

// An array of layout (1D tensor) attributes.
def VHLO_ArrayOfLayoutAttr : TypedArrayAttrBase<VHLO_LayoutAttr,
    "Array of layout (1D tensor of index type) attributes">;

// An array of FlatSymbolRef attributes that can be used as a default valued
// attribute.
def VHLO_FlatSymbolRefArrayAttr :
  TypedArrayAttrBase<FlatSymbolRefAttr, "flat symbol ref array attribute"> {
  let constBuilderCall = "::mlir::ArrayAttr::get($_builder.getContext(), $0)";
}

def VHLO_BoolElementsAttr :
    ElementsAttrBase<
      And<[CPred<"$_self.isa<::mlir::DenseIntOrFPElementsAttr>()">,
           CPred<"$_self.cast<::mlir::DenseIntOrFPElementsAttr>().getType().getElementType().isInteger(1)">]>,
      "constant boolean vector/tensor attribute"> {
  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  let convertFromStorage = "$_self";
}

def VHLO_ConvDimensionNumbers : AttrDef<VHLO_Dialect, "ConvDimensionNumbers"> {
  let cppNamespace = "::mlir::vhlo";
  let mnemonic = "conv";
  let parameters = (ins
    "int64_t":$inputBatchDimension,
    "int64_t":$inputFeatureDimension,
    VHLO_Dims:$inputSpatialDimensions,

    "int64_t":$kernelInputFeatureDimension,
    "int64_t":$kernelOutputFeatureDimension,
    VHLO_Dims:$kernelSpatialDimensions,

    "int64_t":$outputBatchDimension,
    "int64_t":$outputFeatureDimension,
    VHLO_Dims:$outputSpatialDimensions
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def VHLO_ConvolutionAttributes {
  dag attributes = (ins
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$padding,
    OptionalAttr<I64ElementsAttr>:$lhs_dilation,
    OptionalAttr<I64ElementsAttr>:$rhs_dilation,
    OptionalAttr<VHLO_BoolElementsAttr>:$window_reversal,
    VHLO_ConvDimensionNumbers:$dimension_numbers,
    I64Attr:$feature_group_count,
    I64Attr:$batch_group_count,
    VHLO_PrecisionConfigAttr:$precision_config
  );
}

#endif // STABLEHLO_DIALECT_VHLO_ATTRS
