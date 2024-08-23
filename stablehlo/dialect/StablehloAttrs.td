/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#ifndef STABLEHLO_DIALECT_STABLEHLO_ATTRS
#define STABLEHLO_DIALECT_STABLEHLO_ATTRS

include "mlir/IR/OpBase.td"
include "mlir/IR/TensorEncoding.td"

def StableHLO_Dims : ArrayRefParameter<"int64_t", "Dimension"> {
  let parser = "parseDimSizes($_parser)";
  let printer = "printDimSizes($_printer, $_self)";
}

def StableHLO_ScatterDimensionNumbers : AttrDef<StableHLO_Dialect, "ScatterDimensionNumbers"> {
  let mnemonic = "scatter";
  let summary = "Attribute that models the dimension information for scatter";
  let parameters = (ins
      StableHLO_Dims:$updateWindowDims,
      StableHLO_Dims:$insertedWindowDims,
      StableHLO_Dims:$inputBatchingDims,
      StableHLO_Dims:$scatterIndicesBatchingDims,
      StableHLO_Dims:$scatterDimsToOperandDims,
      "int64_t":$indexVectorDim
  );
  let hasCustomAssemblyFormat = 1;
}

def StableHLO_GatherDimensionNumbers : AttrDef<StableHLO_Dialect, "GatherDimensionNumbers"> {
  let mnemonic = "gather";
  let summary = "Attribute that models the dimension information for gather";
  let parameters = (ins
      StableHLO_Dims:$offsetDims,
      StableHLO_Dims:$collapsedSliceDims,
      StableHLO_Dims:$operandBatchingDims,
      StableHLO_Dims:$startIndicesBatchingDims,
      StableHLO_Dims:$startIndexMap,
      "int64_t":$indexVectorDim
  );
  let hasCustomAssemblyFormat = 1;
}

def StableHLO_DotAlgorithm : AttrDef<StableHLO_Dialect, "DotAlgorithm"> {
  let mnemonic = "dot_algorithm";
  let summary = "Attribute that models the algorithm constraints to use for computing dot.";
  let parameters = (ins
      "Type":$lhsPrecisionType,
      "Type":$rhsPrecisionType,
      "Type":$accumulationType,
      "int64_t":$lhsComponentCount,
      "int64_t":$rhsComponentCount,
      "int64_t":$numPrimitiveOperations,
      "bool":$allowImpreciseAccumulation
  );
  let assemblyFormat = [{
    `<`
      `lhs_precision_type` `=` $lhsPrecisionType `,`
      `rhs_precision_type` `=` $rhsPrecisionType `,`
      `accumulation_type` `=` $accumulationType `,`
      `lhs_component_count` `=` $lhsComponentCount `,`
      `rhs_component_count` `=` $rhsComponentCount `,`
      `num_primitive_operations` `=` $numPrimitiveOperations `,`
      `allow_imprecise_accumulation` `=` $allowImpreciseAccumulation
    `>`
  }];
  let genVerifyDecl = 1;
}

def StableHLO_DotDimensionNumbers : AttrDef<StableHLO_Dialect, "DotDimensionNumbers"> {
  let mnemonic = "dot";
  let summary = "Attribute that models the dimension information for dot.";
  let parameters = (ins
      StableHLO_Dims:$lhsBatchingDimensions,
      StableHLO_Dims:$rhsBatchingDimensions,
      StableHLO_Dims:$lhsContractingDimensions,
      StableHLO_Dims:$rhsContractingDimensions
  );
  let hasCustomAssemblyFormat = 1;
}

def StableHLO_OutputOperandAlias : AttrDef<StableHLO_Dialect, "OutputOperandAlias"> {
  let mnemonic = "output_operand_alias";
  let summary =
    "Attribute that models the alias relationship of output and operand of a CustomCall op";
  let description = [{
    This attribute captures the alias relationship of the output to one of the
    operands for a CustomCall op, denoted by `operand_index`. The
    `output_tuple_indices` and `operand_tuple_indices` are used to index into
    output and operand types. These indices lists are empty if the corresponding
    types are not tuple types, and can be arbitrarily long in case of
    arbitrarily nested tuple types.

    See https://www.tensorflow.org/xla/aliasing.

    Example when used as array with in stablehlo.custom-call:

    ```mlir
    %0 = "stablehlo.custom_call"(%arg0, %arg1) {
      // other attributes
      output_operand_alias = [
        #stablehlo.output_operand_alias<output_tuple_indices = [0],
                                   operand_index = 0,
                                   operand_tuple_indices = [1]>
      ]
    } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tuple<tensor<2x3xf32>>

    The output and the 0th operand are both tuples. The aliasing shows the
    relationship between the 0th element in output tuple with the 1st element in
    the 0th operand. And both of them are of the same type: tensor<2x3xf32>.
    ```
  }];
  let parameters = (ins
    StableHLO_Dims:$outputTupleIndices,
    "int64_t":$operandIndex,
    StableHLO_Dims:$operandTupleIndices
  );
  let assemblyFormat = [{
    `<` `output_tuple_indices` `=` $outputTupleIndices `,`
        `operand_index` `=` $operandIndex `,`
        `operand_tuple_indices` `=` $operandTupleIndices `>`
  }];
}

// Represents a unique identifier for each Send/Recv instruction pair or
// optionally for collective instructions (AllToAll, AllReduce,
// CollectiveBroadcast, and CollectivePermute). Non-positive channel_id
// handle is equivalent to no channel id.
def StableHLO_ChannelHandle : AttrDef<StableHLO_Dialect, "ChannelHandle"> {
  let mnemonic = "channel_handle";
  let parameters = (ins "int64_t":$handle, "int64_t":$type);
  let summary = "two 64-bit integers 'handle' and 'type'";
  let assemblyFormat = "`<` struct(params) `>`";
}

def StableHLO_TypeExtensions : AttrDef<StableHLO_Dialect, "TypeExtensions", [
    DeclareAttrInterfaceMethods<VerifiableTensorEncoding>,
    DeclareAttrInterfaceMethods<HLO_BoundedAttrInterface>]> {
  let mnemonic = "type_extensions";

  // TODO(b/238903065): Move sparsity related info here from the standalone
  // attribute. That will allow composition of bounds and sparsity info.
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$bounds
  );

  let summary = "Attribute that extends tensor type with StableHLO type properties.";

  let description = [{
    This attribute is used to extend MLIR tensor type with StableHLO tensor
    specific properties. These properties aren't modeled in the MLIR type. This
    attribute is set in the `encoding` field of the tensor type.

    See `HLO_BoundedAttrInterface` for documentation for `bounds`.
  }];
  let assemblyFormat = "`<` `bounds` `=` ` ` custom<DimSizes>($bounds) `>`";
}

// A layout attribute (1D tensor of index type)
def StableHLO_LayoutAttr : Attr<
  And<[IndexElementsAttr.predicate,
       CPred<[{cast<::mlir::DenseIntElementsAttr>($_self).getType().getRank()
               == 1}]>]>,
  "A 1D tensor of index type (layout)"> {
  let storageType = IndexElementsAttr.storageType;
  let returnType = IndexElementsAttr.returnType;
  let convertFromStorage = IndexElementsAttr.convertFromStorage;
}

// An array of layout (1D tensor) attributes.
def StableHLO_ArrayOfLayoutAttr : TypedArrayAttrBase<StableHLO_LayoutAttr,
    "Array of layout (1D tensor of index type) attributes">;

// An array of FlatSymbolRef attributes that can be used as a default valued
// attribute.
def StableHLO_FlatSymbolRefArrayAttr :
  TypedArrayAttrBase<FlatSymbolRefAttr, "flat symbol ref array attribute"> {
  let constBuilderCall = "::mlir::ArrayAttr::get($_builder.getContext(), $0)";
}

def StableHLO_ConvDimensionNumbers : AttrDef<StableHLO_Dialect, "ConvDimensionNumbers"> {
  let mnemonic = "conv";
  let summary = "Structure of dimension information for conv op";
  let parameters = (ins
    "int64_t":$inputBatchDimension,
    "int64_t":$inputFeatureDimension,
    StableHLO_Dims:$inputSpatialDimensions,

    "int64_t":$kernelInputFeatureDimension,
    "int64_t":$kernelOutputFeatureDimension,
    StableHLO_Dims:$kernelSpatialDimensions,

    "int64_t":$outputBatchDimension,
    "int64_t":$outputFeatureDimension,
    StableHLO_Dims:$outputSpatialDimensions
  );
  let hasCustomAssemblyFormat = 1;
}

#endif // STABLEHLO_DIALECT_STABLEHLO_ATTRS
