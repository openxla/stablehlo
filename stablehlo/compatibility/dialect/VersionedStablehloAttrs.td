/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The VersionedHLO Authors.

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

#ifndef STABLEHLO_COMPATIBILITY_VERSIONEDHLO_ATTRS
#define STABLEHLO_COMPATIBILITY_VERSIONEDHLO_ATTRS

include "stablehlo/compatibility/dialect/VersionedStablehloBase.td"
include "stablehlo/compatibility/dialect/VersionedStablehloEnums.td"

include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// FIXME: This should be versioned differently
def VersionedHLO_Dim : ArrayRefParameter<"int64_t", "Dimension"> {
  let printer = [{ $_printer << '['; llvm::interleaveComma($_self, $_printer); $_printer << ']' }];
  let parser = [{
      [&]() -> FailureOr<SmallVector<int64_t>> {
      SmallVector<int64_t> dims;
      if ($_parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&](){
        return $_parser.parseInteger(dims.emplace_back());
      })) return failure();
      return dims;
    }();
  }];
}

def VersionedHLO_ScatterDimensionNumbers : AttrDef<VersionedHLO_Dialect, "ScatterDimensionNumbers"> {
  let cppNamespace = "::mlir::stablehlo";
  let mnemonic = "scatter";
  let summary = "Attribute that models the dimension information for scatter";
  let parameters = (ins
      VersionedHLO_Dim:$updateWindowDims,
      VersionedHLO_Dim:$insertedWindowDims,
      VersionedHLO_Dim:$scatterDimsToOperandDims,
      "int64_t":$indexVectorDim
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def VersionedHLO_GatherDimensionNumbers : AttrDef<VersionedHLO_Dialect, "GatherDimensionNumbers"> {
  let cppNamespace = "::mlir::stablehlo";
  let mnemonic = "gather";
  let parameters = (ins
      VersionedHLO_Dim:$offsetDims,
      VersionedHLO_Dim:$collapsedSliceDims,
      VersionedHLO_Dim:$startIndexMap,
      "int64_t":$indexVectorDim
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def VersionedHLO_DotDimensionNumbers : AttrDef<VersionedHLO_Dialect, "DotDimensionNumbers"> {
  let cppNamespace = "::mlir::stablehlo";
  let mnemonic = "dot";
  let summary = "Attribute that models the dimension information for dot.";
  let parameters = (ins
      VersionedHLO_Dim:$lhsBatchingDimensions,
      VersionedHLO_Dim:$rhsBatchingDimensions,
      VersionedHLO_Dim:$lhsContractingDimensions,
      VersionedHLO_Dim:$rhsContractingDimensions
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def VersionedHLO_ArgResultAlias : AttrDef<VersionedHLO_Dialect, "ArgResultAlias"> {
  let cppNamespace = "::mlir::stablehlo";
  let mnemonic = "result_alias";
  let parameters = (ins
    VersionedHLO_Dim:$argTupleIndices,
    "int64_t":$resultIndex,
    VersionedHLO_Dim:$resultTupleIndices,
    "bool":$isMustAlias
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def OutputOperandAlias : AttrDef<VersionedHLO_Dialect, "OutputOperandAlias"> {
  let cppNamespace = "::mlir::versionedhlo";
  let mnemonic = "output_operand_alias";
  let parameters = (ins
    VersionedHLO_Dim:$outputTupleIndices,
    "int64_t":$operandIndex,
    VersionedHLO_Dim:$operandTupleIndices
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def VersionedHLO_ChannelHandle : AttrDef<VersionedHLO_Dialect, "ChannelHandle"> {
  let cppNamespace = "::mlir::stablehlo";
  let mnemonic = "channel_handle";
  let parameters = (ins "int64_t":$handle, "int64_t":$type);
  let assemblyFormat = "`<` struct(params) `>`";
}

def VersionedHLO_TypeExtensions : AttrDef<VersionedHLO_Dialect, "TypeExtensions"> {
  let cppNamespace = "::mlir::stablehlo";
  let mnemonic = "type_extensions";
  let parameters = (ins ArrayRefParameter<"int64_t">:$bounds);
  let assemblyFormat = "`<` struct(params) `>`";
}

def VersionedHLO_LayoutAttr : Attr<
  And<[IndexElementsAttr.predicate,
       CPred<[{$_self.cast<::mlir::DenseIntElementsAttr>().getType().getRank()
               == 1}]>]>,
  "A 1D tensor of index type (layout)"> {
  let storageType = IndexElementsAttr.storageType;
  let returnType = IndexElementsAttr.returnType;
  let convertFromStorage = IndexElementsAttr.convertFromStorage;
}

// An array of layout (1D tensor) attributes.
def VersionedHLO_ArrayOfLayoutAttr : TypedArrayAttrBase<VersionedHLO_LayoutAttr,
    "Array of layout (1D tensor of index type) attributes">;

// An array of FlatSymbolRef attributes that can be used as a default valued
// attribute.
def VersionedHLO_FlatSymbolRefArrayAttr :
  TypedArrayAttrBase<FlatSymbolRefAttr, "flat symbol ref array attribute"> {
  let constBuilderCall = "::mlir::ArrayAttr::get($_builder.getContext(), $0)";
}

def VersionedHLO_BoolElementsAttr :
    ElementsAttrBase<
      And<[CPred<"$_self.isa<::mlir::DenseIntOrFPElementsAttr>()">,
           CPred<"$_self.cast<::mlir::DenseIntOrFPElementsAttr>().getType().getElementType().isInteger(1)">]>,
      "constant boolean vector/tensor attribute"> {
  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  let convertFromStorage = "$_self";
}

def VersionedHLO_ConvDimensionNumbers : AttrDef<VersionedHLO_Dialect, "ConvDimensionNumbers"> {
  let cppNamespace = "::mlir::stablehlo";
  let mnemonic = "conv";
  let parameters = (ins
    "int64_t":$inputBatchDimension,
    "int64_t":$inputFeatureDimension,
    VersionedHLO_Dim:$inputSpatialDimensions,

    "int64_t":$kernelInputFeatureDimension,
    "int64_t":$kernelOutputFeatureDimension,
    VersionedHLO_Dim:$kernelSpatialDimensions,

    "int64_t":$outputBatchDimension,
    "int64_t":$outputFeatureDimension,
    VersionedHLO_Dim:$outputSpatialDimensions
  );
  let assemblyFormat = "struct(params)";
}

def VersionedHLO_ConvolutionAttributes {
  dag attributes = (ins
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$padding,
    OptionalAttr<I64ElementsAttr>:$lhs_dilation,
    OptionalAttr<I64ElementsAttr>:$rhs_dilation,
    OptionalAttr<VersionedHLO_BoolElementsAttr>:$window_reversal,
    VersionedHLO_ConvDimensionNumbers:$dimension_numbers,
    I64Attr:$feature_group_count,
    I64Attr:$batch_group_count,
    VersionedHLO_PrecisionConfigAttr:$precision_config
  );
}

#endif // STABLEHLO_COMPATIBILITY_VERSIONEDHLO_ATTRS
