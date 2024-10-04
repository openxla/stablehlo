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

#ifndef STABLEHLO_DIALECT_VHLO_TYPES
#define STABLEHLO_DIALECT_VHLO_TYPES

include "mlir/IR/AttrTypeBase.td"
include "stablehlo/dialect/VhloBase.td"
include "stablehlo/dialect/VhloDialect.td"

def VHLO_VersionedTypeInterface : TypeInterface<"VersionedTypeInterface"> {
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

class VHLO_TypeDef<string cppName, string name, string minVersion, string maxVersion>
  : TypeDef<VHLO_Dialect, cppName, [VHLO_VersionedTypeInterface]> {
  let mnemonic = name;
  let extraClassDeclaration = [{
    mlir::vhlo::Version getMinVersion() {
      return mlir::vhlo::Version(}] # !subst(".", ", ", minVersion) # [{);
    }
    mlir::vhlo::Version getMaxVersion() {
      }] # !if(
        !eq(maxVersion, "current"),
        [{ return mlir::vhlo::Version::getCurrentVersion(); }],
        [{ return mlir::vhlo::Version("}] # !subst(".", ", ", maxVersion) # [{"); }]
      ) # [{
    }
  }];
}

// Corresponds to BooleanType from the StableHLO spec.
def VHLO_BooleanV1 : VHLO_TypeDef<"BooleanV1", "bool_v1", "0.9.0", "current">;

// Corresponds to ComplexType from the StableHLO spec.
def VHLO_ComplexV1 : VHLO_TypeDef<"ComplexV1", "complex_v1", "0.9.0", "current"> {
  let parameters = (ins "Type":$elementType);
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult ComplexV1Type::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, mlir::Type elementType) {
      if (!isFromVhlo(elementType)) return errFn() << "expected VHLO type";
      return success();
    }
  }];
  let assemblyFormat = "`<` $elementType `>`";
}

// Corresponds to the 'bf16' FloatType from the StableHLO spec.
def VHLO_FloatBF16V1 : VHLO_TypeDef<"FloatBF16V1", "bf16_v1", "0.9.0", "current">;

// Corresponds to the 'f16' FloatType from the StableHLO spec.
def VHLO_FloatF16V1 : VHLO_TypeDef<"FloatF16V1", "f16_v1", "0.9.0", "current">;

// Corresponds to the 'f32' FloatType from the StableHLO spec.
def VHLO_FloatF32V1 : VHLO_TypeDef<"FloatF32V1", "f32_v1", "0.9.0", "current">;

// Corresponds to the 'f64' FloatType from the StableHLO spec.
def VHLO_FloatF64V1 : VHLO_TypeDef<"FloatF64V1","f64_v1", "0.9.0", "current">;

// Corresponds to the 'f4E2M1FN' FloatType from the StableHLO spec.
def VHLO_FloatF4E2M1FNV1 : VHLO_TypeDef<"FloatF4E2M1FNV1", "f4E2M1FN_v1", "1.8.0", "current">;

// Corresponds to the 'f6E2M3FN' FloatType from the StableHLO spec.
def VHLO_FloatF6E2M3FNV1 : VHLO_TypeDef<"FloatF6E2M3FNV1", "f6E2M3FN_v1", "1.8.0", "current">;

// Corresponds to the 'f6E3M2FN' FloatType from the StableHLO spec.
def VHLO_FloatF6E3M2FNV1 : VHLO_TypeDef<"FloatF6E3M2FNV1", "f6E3M2FN_v1", "1.8.0", "current">;

// Corresponds to the 'f8E3M4' FloatType from the StableHLO spec.
def VHLO_FloatF8E3M4V1 : VHLO_TypeDef<"FloatF8E3M4V1", "f8E3M4_v1", "1.7.0", "current">;

// Corresponds to the 'f8E4M3' FloatType from the StableHLO spec.
def VHLO_FloatF8E4M3V1 : VHLO_TypeDef<"FloatF8E4M3V1", "f8E4M3_v1", "1.7.0", "current">;

// Corresponds to the 'f8E4M3FN' FloatType from the StableHLO spec.
def VHLO_FloatF8E4M3FNV1 : VHLO_TypeDef<"FloatF8E4M3FNV1", "f8E4M3FN_v1", "0.9.0", "current">;

// Corresponds to the 'f8E5M2' FloatType from the StableHLO spec.
def VHLO_FloatF8E5M2V1 : VHLO_TypeDef<"FloatF8E5M2V1", "f8E5M2_v1", "0.9.0", "current">;

// Corresponds to the 'f8E4M3FNUZ' FloatType from the StableHLO spec.
def VHLO_FloatF8E4M3FNUZV1 : VHLO_TypeDef<"FloatF8E4M3FNUZV1", "f8E4M3FNUZ_v1", "0.10.0", "current">;

// Corresponds to the 'f8E4M3B11FNUZ' FloatType from the StableHLO spec.
def VHLO_FloatF8E4M3B11FNUZV1 : VHLO_TypeDef<"FloatF8E4M3B11FNUZV1", "f8E4M3B11FNUZ_v1", "0.11.0", "current">;

// Corresponds to the 'f8E5M2FNUZ' FloatType from the StableHLO spec.
def VHLO_FloatF8E5M2FNUZV1 : VHLO_TypeDef<"FloatF8E5M2FNUZV1", "f8E5M2FNUZ_v1", "0.10.0", "current">;

// Corresponds to the 'f8E8M0FNU' FloatType from the StableHLO spec.
def VHLO_FloatF8E8M0FNUV1 : VHLO_TypeDef<"FloatF8E8M0FNUV1", "f8E8M0FNU_v1", "1.8.0", "current">;

// Corresponds to the 'tf32' FloatType from the StableHLO spec.
def VHLO_FloatTF32V1 : VHLO_TypeDef<"FloatTF32V1", "tf31_v1", "1.6.0", "current">;

// Corresponds to FunctionType from the StableHLO spec.
def VHLO_FunctionV1 : VHLO_TypeDef<"FunctionV1", "func_v1", "0.9.0", "current"> {
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$inputs,
    ArrayRefParameter<"mlir::Type">:$outputs
  );
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult FunctionV1Type::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn,
        ArrayRef<mlir::Type> inputs, ArrayRef<mlir::Type> outputs) {
      if (!allFromVhlo(inputs) || !allFromVhlo(outputs))
        return errFn() << "expected VHLO types";
      return success();
    }
  }];
  let assemblyFormat = "`<` `(` custom<TypeArray>($inputs) `)` `->` custom<TypeArray>($outputs) `>`";
}

// TODO(#8): Index is not part of the StableHLO spec.
// At the moment, it is used to represent values participating in shape
// computations, and we're planning to look into it as part of the work on the
// dynamism RFC.
def VHLO_IndexV1 : VHLO_TypeDef<"IndexV1", "index_v1", "0.9.0", "current">;

// Corresponds to the 'si2' IntegerType from the StableHLO spec.
def VHLO_IntegerSI2V1 : VHLO_TypeDef<"IntegerSI2V1", "i2_v1", "1.2.0", "current">;

// Corresponds to the 'si4' IntegerType from the StableHLO spec.
def VHLO_IntegerSI4V1 : VHLO_TypeDef<"IntegerSI4V1", "i4_v1", "0.9.0", "current">;

// Corresponds to the 'si8' IntegerType from the StableHLO spec.
def VHLO_IntegerSI8V1 : VHLO_TypeDef<"IntegerSI8V1", "i8_v1", "0.9.0", "current">;

// Corresponds to the 'si16' IntegerType from the StableHLO spec.
def VHLO_IntegerSI16V1 : VHLO_TypeDef<"IntegerSI16V1", "i16_v1", "0.9.0", "current">;

// Corresponds to the 'si32' IntegerType from the StableHLO spec.
def VHLO_IntegerSI32V1 : VHLO_TypeDef<"IntegerSI32V1", "i32_v1", "0.9.0", "current">;

// Corresponds to the 'si64' IntegerType from the StableHLO spec.
def VHLO_IntegerSI64V1 : VHLO_TypeDef<"IntegerSI64V1", "i64_v1", "0.9.0", "current">;

// Corresponds to the 'ui2' IntegerType from the StableHLO spec.
def VHLO_IntegerUI2V1 : VHLO_TypeDef<"IntegerUI2V1", "ui2_v1", "1.2.0", "current">;

// Corresponds to the 'ui4' IntegerType from the StableHLO spec.
def VHLO_IntegerUI4V1 : VHLO_TypeDef<"IntegerUI4V1", "ui4_v1", "0.9.0", "current">;

// Corresponds to the 'ui8' IntegerType from the StableHLO spec.
def VHLO_IntegerUI8V1 : VHLO_TypeDef<"IntegerUI8V1", "ui8_v1", "0.9.0", "current">;

// Corresponds to the 'ui16' IntegerType from the StableHLO spec.
def VHLO_IntegerUI16V1 : VHLO_TypeDef<"IntegerUI16V1", "ui16_v1", "0.9.0", "current">;

// Corresponds to the 'ui32' IntegerType from the StableHLO spec.
def VHLO_IntegerUI32V1 : VHLO_TypeDef<"IntegerUI32V1", "ui32_v1", "0.9.0", "current">;

// Corresponds to the 'ui64' IntegerType from the StableHLO spec.
def VHLO_IntegerUI64V1 : VHLO_TypeDef<"IntegerUI64V1", "ui64_v1", "0.9.0", "current">;

// The nonetype is used for optional values which have implementation specific
// default values, for attributes which must be specified as None, use TypedAttr
// of NoneType.
def VHLO_NoneTypeV1 : VHLO_TypeDef<"NoneV1", "none_v1", "1.6.0", "current">;

// Corresponds to TensorType from the StableHLO spec.
// TODO(#8): Encoding is not part of the StableHLO spec.
// At the moment, it is used to represent dimension bounds to support bounded
// dynamism, and we're planning to look into it as part of the work on the
// dynamism RFC.
def VHLO_RankedTensorV1 : VHLO_TypeDef<"RankedTensorV1", "tensor_v1", "0.9.0", "current"> {
  let parameters = (ins
    VHLO_Dims:$shape,
    "::mlir::Type":$elementType,
    "::mlir::Attribute":$encoding
  );
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult RankedTensorV1Type::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn,
        ArrayRef<int64_t> shape, ::mlir::Type elementType, ::mlir::Attribute encoding) {
      if (!isFromVhlo(elementType) || (encoding && !isFromVhlo(encoding)))
        return errFn() << "expected VHLO type or attribute";
      return success();
    }
  }];
  let assemblyFormat = "`<` custom<Shape>($shape) `` $elementType `` custom<Encoding>($encoding) `>`";
}

// Corresponds to TokenType from the StableHLO spec.
def VHLO_TokenV1 : VHLO_TypeDef<"TokenV1", "token_v1", "0.9.0", "current">;

// Corresponds to TupleType from the StableHLO spec.
def VHLO_TupleV1 : VHLO_TypeDef<"TupleV1", "tuple_v1", "0.9.0", "current"> {
  let parameters = (ins ArrayRefParameter<"::mlir::Type">:$types);
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult TupleV1Type::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, ArrayRef<mlir::Type> types) {
      if (!allFromVhlo(types)) return errFn() << "expected VHLO types";
      return success();
    }
  }];
  let assemblyFormat = "`<` $types `>`";
}

// TODO(#8): UniformQuantized is not part of the StableHLO spec.
// At the moment, it is used to represent uniform quantization, and we're
// planning to look into it as part of the work on speccing quantization.
def VHLO_APFloatV1 : APFloatParameter<""> {
  let parser = [{
    [&]() -> FailureOr<llvm::APFloat> {
      double value;
      if (failed($_parser.parseFloat(value))) {
        return failure();
      }
      return APFloat(value);
    }()
  }];
  let printer = "$_printer.printFloat($_self);";
}
def VHLO_UniformQuantizedV1 : VHLO_TypeDef<"UniformQuantizedV1", "quant_v1", "0.9.0", "current"> {
  let parameters = (ins
    "unsigned":$flags,
    "::mlir::Type":$storageType,
    "::mlir::Type":$expressedType,
    VHLO_APFloatV1:$scale,
    "int64_t":$zeroPoint,
    "int64_t":$storageTypeMin,
    "int64_t":$storageTypeMax
  );
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult UniformQuantizedV1Type::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn,
        unsigned int, mlir::Type storageType, mlir::Type expressedType,
        llvm::APFloat, int64_t, int64_t, int64_t) {
      if (!isFromVhlo(storageType) || !isFromVhlo(expressedType))
        return errFn() << "expected VHLO type";
      return success();
    }
  }];
  let assemblyFormat = "`<` $storageType `` `:` `` $expressedType `,` $scale `` `:` `` $zeroPoint `,` $storageTypeMin `` `:` `` $storageTypeMax `,` $flags `>`";
}
def VHLO_QuantizationScalesV1 : ArrayRefParameter<"::llvm::APFloat", "array of double scales"> {
  let parser = [{
    [&]() -> FailureOr<llvm::SmallVector<::llvm::APFloat>> {
      ::llvm::SmallVector<double> scales;

      auto parseResult = $_parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
        return $_parser.parseFloat(scales.emplace_back());
      });
      if(failed(parseResult)) return failure();
      return llvm::map_to_vector(
        scales, [](double scale) { return APFloat(scale); });
    }()
  }];
  let printer = [{
    $_printer << '[';
    llvm::interleaveComma($_self, $_printer, [&](APFloat scale) {
      $_printer << scale;
    });
    $_printer << ']';
  }];
}
def VHLO_UniformQuantizedPerAxisV1 : VHLO_TypeDef<"UniformQuantizedPerAxisV1", "quant_per_axis_v1", "0.18.0", "current"> {
  let parameters = (ins
    "unsigned":$flags,
    "::mlir::Type":$storageType,
    "::mlir::Type":$expressedType,
    "int32_t":$quantizedDimension,
    VHLO_QuantizationScalesV1:$scales,
    ArrayRefParameter<"int64_t">:$zeroPoints,
    "int64_t":$storageTypeMin,
    "int64_t":$storageTypeMax
  );
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult UniformQuantizedPerAxisV1Type::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn,
        unsigned int, mlir::Type storageType, mlir::Type expressedType,
        int32_t, ::llvm::ArrayRef<::llvm::APFloat>, ::llvm::ArrayRef<int64_t>, int64_t, int64_t) {
      if (!isFromVhlo(storageType) || !isFromVhlo(expressedType))
        return errFn() << "expected VHLO type";
      return success();
    }
  }];
  let assemblyFormat = "`<` $storageType `` `:` `` $expressedType `,` $quantizedDimension `,` $scales `,` `[` $zeroPoints `]` `,` $storageTypeMin `` `:` `` $storageTypeMax `,` $flags `>`";
}

// TODO(#8): UnrankedTensor is not part of the StableHLO spec.
// At the moment, it is used to represent unranked dynamism, and we will likely
// remove it as part of the work on the dynamism RFC.
def VHLO_UnrankedTensorV1 : VHLO_TypeDef<"UnrankedTensorV1", "unranked_tensor_v1", "0.9.0", "current"> {
  let parameters = (ins "::mlir::Type":$elementType);
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult UnrankedTensorV1Type::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, mlir::Type elementType) {
      if (!isFromVhlo(elementType)) return errFn() << "expected VHLO type";
      return success();
    }
  }];
  let assemblyFormat = "`<` $elementType `>`";
}

// TODO(#8): Witness is not part of the StableHLO spec.
// At the moment, it is used to represent constraints for dynamic shapes,
// and we're planning to look into it as part of the work on the dynamism RFC.
def VHLO_WitnessV1 : VHLO_TypeDef<"WitnessV1", "witness_v1", "0.9.0", "current">;

#endif  // STABLEHLO_DIALECT_VHLO_TYPES
