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

include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Attribute Versioning
//===----------------------------------------------------------------------===//

def VHLO_VersionedAttrInterface : AttrInterface<"VersionedAttrInterface"> {
  let cppNamespace = "::mlir::vhlo";
  let methods = [
    InterfaceMethod<
      "Returns the minimum version of the VHLO dialect an attribute is supported in.",
      "mlir::vhlo::Version", "getMinVersion">,
    InterfaceMethod<
      "Returns the maximum version (inclusive) of the VHLO dialect an attribute is supported in.",
      "mlir::vhlo::Version", "getMaxVersion">,
  ];
}

class VHLO_AttrDef<string name,
                   string minVersion = "0.3.0",
                   string maxVersion = "current">
  : AttrDef<VHLO_Dialect, name, [VHLO_VersionedAttrInterface]> {
  let extraClassDeclaration = [{
    mlir::vhlo::Version getMinVersion() {
      auto version = mlir::vhlo::Version::fromString("}] #  minVersion # [{");
      if (failed(version)) llvm_unreachable("invalid version }] # minVersion # [{ in }] # name # [{");
      return *version;
    }
    mlir::vhlo::Version getMaxVersion() {
      if (!strcmp("}] # maxVersion # [{", "current")) return VhloDialect::getCurrentVersion();
      auto version = mlir::vhlo::Version::fromString("}] #  maxVersion # [{");
      if (failed(version)) llvm_unreachable("invalid version }] # maxVersion # [{ in }] # name # [{");
      return *version;
    }
  }];
}

//===----------------------------------------------------------------------===//
// VHLO Attributes
//===----------------------------------------------------------------------===//

def VHLO_ScatterDimensionNumbers: VHLO_AttrDef<"ScatterDimensionNumbers"> {
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

def VHLO_GatherDimensionNumbers : VHLO_AttrDef<"GatherDimensionNumbers"> {
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

def VHLO_DotDimensionNumbers : VHLO_AttrDef<"DotDimensionNumbers"> {
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

def VHLO_OutputOperandAlias : VHLO_AttrDef<"OutputOperandAlias", "0.4.0"> {
  let cppNamespace = "::mlir::vhlo";
  let mnemonic = "output_operand_alias";
  let parameters = (ins
    VHLO_Dims:$outputTupleIndices,
    "int64_t":$operandIndex,
    VHLO_Dims:$operandTupleIndices
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def VHLO_ArgResultAlias : VHLO_AttrDef<"ArgResultAlias"> {
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

def VHLO_ChannelHandle : VHLO_AttrDef<"ChannelHandle"> {
  let cppNamespace = "::mlir::vhlo";
  let mnemonic = "channel_handle";
  let parameters = (ins "int64_t":$handle, "int64_t":$type);
  let assemblyFormat = "`<` struct(params) `>`";
}

def VHLO_TypeExtensions : VHLO_AttrDef<"TypeExtensions"> {
  let cppNamespace = "::mlir::vhlo";
  let mnemonic = "type_extensions";
  let parameters = (ins VHLO_Dims:$bounds);
  let assemblyFormat = "`<` struct(params) `>`";
}

def VHLO_ConvDimensionNumbers : VHLO_AttrDef<"ConvDimensionNumbers"> {
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
    OptionalAttr<VHLO_AnyAttr>:$window_strides,
    OptionalAttr<VHLO_AnyAttr>:$padding,
    OptionalAttr<VHLO_AnyAttr>:$lhs_dilation,
    OptionalAttr<VHLO_AnyAttr>:$rhs_dilation,
    OptionalAttr<VHLO_AnyAttr>:$window_reversal,
    VHLO_AnyAttr:$dimension_numbers,
    VHLO_AnyAttr:$feature_group_count,
    VHLO_AnyAttr:$batch_group_count,
    VHLO_AnyAttr:$precision_config
  );
}

//===----------------------------------------------------------------------===//
// Forked Attributes
//===----------------------------------------------------------------------===//

def VHLO_ArrayAttr : VHLO_AttrDef<"ArrayV1"> {
  let mnemonic = "array";
  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$value);
  let assemblyFormat = "`<` custom<AttributeArray>($value) `>`";
}

def VHLO_DenseIntOrFPElements : VHLO_AttrDef<"DenseIntOrFPElementsV1"> {
  let mnemonic = "dense";
  let parameters = (ins "::mlir::Type":$type, "ArrayRef<char>":$raw_data);
  let hasCustomAssemblyFormat = 1;
}

def VHLO_DictionaryAttr : VHLO_AttrDef<"DictionaryV1"> {
  let mnemonic = "dict";
  let parameters = (ins ArrayRefParameter<"std::pair<mlir::Attribute, mlir::Attribute>", "">:$value);
  let assemblyFormat = "`<` custom<Dictionary>($value) `>`";
}

def VHLO_FlatSymbolRefAttr : VHLO_AttrDef<"FlatSymbolRefV1"> {
  let mnemonic = "sym";
  let parameters = (ins "::mlir::Attribute":$root_reference);
  let assemblyFormat = "`<` $root_reference `>`";
}

def VHLO_FloatAttr : VHLO_AttrDef<"FloatV1"> {
  let mnemonic = "float";
  let parameters = (ins "mlir::Type":$type, VHLO_Double:$value);
  let assemblyFormat = "`<` $value `:` $type `>`";
}

def VHLO_IntegerAttr  : VHLO_AttrDef<"IntegerV1"> {
  let mnemonic = "integer";
  let parameters = (ins "mlir::IntegerAttr":$value);
  let assemblyFormat = "`<` $value `>`";
}

def VHLO_StringAttr : VHLO_AttrDef<"StringV1"> {
  let mnemonic = "string";
  let parameters = (ins StringRefParameter<"">:$value);
  let assemblyFormat = "`<` $value `>`";
}

def VHLO_TypeAttr : VHLO_AttrDef<"TypeV1"> {
  let mnemonic = "type";
  let parameters = (ins "::mlir::Type":$value);
  let assemblyFormat = "`<` $value `>`";
}
/*
def VHLO_DictionaryAttr : VHLO_AttrDef<"DictionaryV1"> {
  let mnemonic = "dict";
  let parameters = (ins Variadic<VHLO_AnyType>:$entries); // NamedAttributes
  let assemblyFormat = "`<` $entries `>`";
}
*/

def VHLO_UnitAttr : VHLO_AttrDef<"UnitV1"> {
  let mnemonic = "unit";
  let storageType = [{ ::mlir::UnitAttr }];
  let constBuilderCall = "(($0) ? $_builder.getUnitAttr() : nullptr)";
  let convertFromStorage = "$_self != nullptr";
  let returnType = "bool";
  let defaultValue = "false";
  let valueType = NoneType;
  let isOptional = 1;
}

#endif // STABLEHLO_DIALECT_VHLO_ATTRS
