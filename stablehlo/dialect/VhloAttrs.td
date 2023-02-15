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
include "stablehlo/dialect/VhloDialect.td"
include "stablehlo/dialect/VhloTypes.td"

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

class VHLO_AttrDef<string name, string minVersion, string maxVersion>
  : AttrDef<VHLO_Dialect, name, [VHLO_VersionedAttrInterface]> {
  let extraClassDeclaration = [{
    mlir::vhlo::Version getMinVersion() {
      auto version = mlir::vhlo::Version::fromString("}] #  minVersion # [{");
      if (failed(version)) llvm_unreachable("invalid version }] # minVersion # [{ in }] # name # [{");
      return *version;
    }
    mlir::vhlo::Version getMaxVersion() {
      if (!strcmp("}] # maxVersion # [{", "current")) return Version::getCurrentVersion();
      auto version = mlir::vhlo::Version::fromString("}] #  maxVersion # [{");
      if (failed(version)) llvm_unreachable("invalid version }] # maxVersion # [{ in }] # name # [{");
      return *version;
    }
  }];
}

//===----------------------------------------------------------------------===//
// VHLO Attributes
//===----------------------------------------------------------------------===//

def VHLO_ScatterDimensionNumbersAttrV1 : VHLO_AttrDef<"ScatterDimensionNumbersV1", "0.3.0", "current"> {
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

def VHLO_GatherDimensionNumbersAttrV1 : VHLO_AttrDef<"GatherDimensionNumbersV1", "0.3.0", "current"> {
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

def VHLO_DotDimensionNumbersAttrV1 : VHLO_AttrDef<"DotDimensionNumbersV1", "0.3.0", "current"> {
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

def VHLO_OutputOperandAliasAttrV1 : VHLO_AttrDef<"OutputOperandAliasV1", "0.4.0", "current"> {
  let cppNamespace = "::mlir::vhlo";
  let mnemonic = "output_operand_alias";
  let parameters = (ins
    VHLO_Dims:$outputTupleIndices,
    "int64_t":$operandIndex,
    VHLO_Dims:$operandTupleIndices
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def VHLO_ArgResultAliasAttrV1 : VHLO_AttrDef<"ArgResultAliasV1", "0.3.0", "current"> {
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

def VHLO_ChannelHandleAttrV1 : VHLO_AttrDef<"ChannelHandleV1", "0.3.0", "current"> {
  let cppNamespace = "::mlir::vhlo";
  let mnemonic = "channel_handle";
  let parameters = (ins "int64_t":$handle, "int64_t":$type);
  let assemblyFormat = "`<` struct(params) `>`";
}

def VHLO_TypeExtensionsAttrV1 : VHLO_AttrDef<"TypeExtensionsV1", "0.3.0", "current"> {
  let cppNamespace = "::mlir::vhlo";
  let mnemonic = "type_extensions";
  let parameters = (ins VHLO_Dims:$bounds);
  let assemblyFormat = "`<` struct(params) `>`";
}

def VHLO_ConvDimensionNumbersAttrV1 : VHLO_AttrDef<"ConvDimensionNumbersV1", "0.3.0", "current"> {
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

def VHLO_ConvolutionAttributesV1 {
  dag attributes = (ins
    OptionalAttr<VHLO_AnyAttr>:$window_strides,
    OptionalAttr<VHLO_AnyAttr>:$padding,
    OptionalAttr<VHLO_AnyAttr>:$lhs_dilation,
    OptionalAttr<VHLO_AnyAttr>:$rhs_dilation,
    OptionalAttr<VHLO_AnyAttr>:$window_reversal,
    VHLO_AnyAttr:$dimension_numbers,
    VHLO_AnyAttr:$feature_group_count,
    VHLO_AnyAttr:$batch_group_count,
    OptionalAttr<VHLO_AnyAttr>:$precision_config
  );
}

//===----------------------------------------------------------------------===//
// Forked Attributes
//===----------------------------------------------------------------------===//

def VHLO_ArrayAttrV1 : VHLO_AttrDef<"ArrayV1", "0.3.0", "current"> {
  let mnemonic = "array";
  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$value);
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult ArrayV1Attr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, ArrayRef<mlir::Attribute> value) {
      if (!allFromVhlo(value)) return errFn() << "expected array of VHLO attriutes";
      return success();
    }
  }];
  let assemblyFormat = "`<` custom<AttributeArray>($value) `>`";
}

def VHLO_DenseIntOrFPElementsData : AttrParameter<"::llvm::ArrayRef<char>", "Array of int"> {
  // Custom allocator to copy dense elements data into MLIR Context
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}
def VHLO_DenseIntOrFPElementsAttrV1 : VHLO_AttrDef<"DenseIntOrFPElementsV1", "0.3.0", "current"> {
  let mnemonic = "dense";
  let parameters = (ins "::mlir::Type":$type, VHLO_DenseIntOrFPElementsData:$raw_data);
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult DenseIntOrFPElementsV1Attr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, mlir::Type type, ArrayRef<char>) {
      if (!isFromVhlo(type)) errFn() << "expected VHLO type";
      return success();
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

def VHLO_DictionaryAttr : VHLO_AttrDef<"DictionaryV1", "0.3.0", "current"> {
  let mnemonic = "dict";
  let parameters = (ins ArrayRefParameter<"std::pair<mlir::Attribute, mlir::Attribute>", "">:$value);
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult DictionaryV1Attr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn,
        ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value) {
      for (auto & entry : value)
        if (!isFromVhlo(entry.first) || !isFromVhlo(entry.second))
          errFn() << "expected VHLO attribute";
      return success();
    }
  }];
  let assemblyFormat = "`<` custom<AttributeDictionary>($value) `>`";
}

def VHLO_FlatSymbolRefAttrV1 : VHLO_AttrDef<"FlatSymbolRefV1", "0.3.0", "current"> {
  let mnemonic = "sym";
  let parameters = (ins "::mlir::Attribute":$root_reference);
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult FlatSymbolRefV1Attr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, mlir::Attribute rootReference) {
      if (!isFromVhlo(rootReference)) return errFn() << "expected VHLO attribute";
      return success();
    }
  }];
  let assemblyFormat = "`<` $root_reference `>`";
}

def VHLO_FloatAttrV1 : VHLO_AttrDef<"FloatV1", "0.3.0", "current"> {
  let mnemonic = "float";
  let parameters = (ins "mlir::Type":$type, VHLO_APFloatV1:$value);
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult FloatV1Attr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, mlir::Type type, APFloat value) {
      if (!isFromVhlo(type)) return errFn() << "expected VHLO type";
      return success();
    }
  }];
  let assemblyFormat = "`<` $value `:` $type `>`";
}

def VHLO_IntegerAttrV1  : VHLO_AttrDef<"IntegerV1", "0.3.0", "current"> {
  let mnemonic = "integer";
  let parameters = (ins "mlir::Type":$type, "APInt":$value);
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult IntegerV1Attr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, mlir::Type type, APInt value) {
      if (!isFromVhlo(type)) return errFn() << "expected VHLO type";
      return success();
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

def VHLO_StringAttrV1 : VHLO_AttrDef<"StringV1", "0.3.0", "current"> {
  let mnemonic = "string";
  let parameters = (ins StringRefParameter<"">:$value);
  let assemblyFormat = "`<` custom<EscapedString>($value) `>`";
}

def VHLO_TypeAttrV1 : VHLO_AttrDef<"TypeV1", "0.3.0", "current"> {
  let mnemonic = "type";
  let parameters = (ins "::mlir::Type":$value);
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult TypeV1Attr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, mlir::Type value) {
      if (!isFromVhlo(value)) return errFn() << "expected VHLO type";
      return success();
    }
  }];
  let assemblyFormat = "`<` $value `>`";
}

def VHLO_UnitAttrV1 : VHLO_AttrDef<"UnitV1", "0.3.0", "current"> {
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
