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

include "stablehlo/dialect/VhloBase.td"
include "stablehlo/dialect/VhloDialect.td"

include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// VHLO Type Versioning
//===----------------------------------------------------------------------===//

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

class VHLO_TypeDef<string cppName,
                   string name,
                   string minVersion = "0.3.0",
                   string maxVersion = "current">
  : TypeDef<VHLO_Dialect, cppName, [VHLO_VersionedTypeInterface]> {
  let mnemonic = name;
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
// VHLO Type Definitions.
//===----------------------------------------------------------------------===//

def VHLO_TokenV1 : VHLO_TypeDef<"TokenV1", "token">;

//===----------------------------------------------------------------------===//
// Forked Types
//===----------------------------------------------------------------------===//

def VHLO_BFloat16V1 : VHLO_TypeDef<"BFloat16V1", "bf16">;

def VHLO_ComplexV1 : VHLO_TypeDef<"ComplexV1", "complex"> {
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

def VHLO_Float16V1 : VHLO_TypeDef<"Float16V1", "f16">;
def VHLO_Float32V1 : VHLO_TypeDef<"Float32V1", "f32">;
def VHLO_Float64V1 : VHLO_TypeDef<"Float64V1","f64">;

def VHLO_Float8E4M3FNV1 : VHLO_TypeDef<"Float8E4M3FNV1", "f8E4M3FN", "0.4.0">;

def VHLO_Float8E5M2V1 : VHLO_TypeDef<"Float8E5M2V1", "f8E5M2", "0.4.0">;

def VHLO_FunctionV1 : VHLO_TypeDef<"FunctionV1", "func"> {
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$inputs,
    ArrayRefParameter<"mlir::Type">:$results
  );
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult FunctionV1Type::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn,
        ArrayRef<mlir::Type> inputs, ArrayRef<mlir::Type> results) {
      if (!allFromVhlo(inputs) || !allFromVhlo(results))
        return errFn() << "expected VHLO types";
      return success();
    }
  }];
  let assemblyFormat = "`<` `(` custom<TypeArray>($inputs) `)` `->` custom<TypeArray>($results) `>`";
}

def VHLO_IndexV1 : VHLO_TypeDef<"IndexV1", "index">;

def VHLO_IntegerI1V1 : VHLO_TypeDef<"IntegerI1V1", "i1">;
def VHLO_IntegerI4V1 : VHLO_TypeDef<"IntegerI4V1", "i4">;
def VHLO_IntegerI8V1 : VHLO_TypeDef<"IntegerI8V1", "i8">;
def VHLO_IntegerI16V1 : VHLO_TypeDef<"IntegerI16V1", "i16">;
def VHLO_IntegerI32V1 : VHLO_TypeDef<"IntegerI32V1", "i32">;
def VHLO_IntegerI64V1 : VHLO_TypeDef<"IntegerI64V1", "i64">;
def VHLO_IntegerUI4V1 : VHLO_TypeDef<"IntegerUI4V1", "ui4">;
def VHLO_IntegerUI8V1 : VHLO_TypeDef<"IntegerUI8V1", "ui8">;
def VHLO_IntegerUI16V1 : VHLO_TypeDef<"IntegerUI16V1", "ui16">;
def VHLO_IntegerUI32V1 : VHLO_TypeDef<"IntegerUI32V1", "ui32">;
def VHLO_IntegerUI64V1 : VHLO_TypeDef<"IntegerUI64V1", "ui64">;

def VHLO_RankedTensorV1 : VHLO_TypeDef<"RankedTensorV1", "tensor"> {
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

def VHLO_TupleV1 : VHLO_TypeDef<"TupleV1", "tuple"> {
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

def VHLO_APFloatV1 : APFloatParameter<""> {
  // Used by UniformQuantizedV1 to have APFloat with custom parser
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
def VHLO_UniformQuantizedV1 : VHLO_TypeDef<"UniformQuantizedV1", "quant"> {
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

def VHLO_UnrankedTensorV1 : VHLO_TypeDef<"UnrankedTensorV1", "unranked_tensor"> {
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

def VHLO_WitnessV1 : VHLO_TypeDef<"WitnessV1", "witness">;

#endif // STABLEHLO_DIALECT_VHLO_TYPES
