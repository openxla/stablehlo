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

#ifndef STABLEHLO_DIALECT_VHLO_BASE
#define STABLEHLO_DIALECT_VHLO_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// VHLO Type Definitions.
//===----------------------------------------------------------------------===//

// VHLO is intended to represent the layout only, as such uses AnyType everywhere.
def VHLO_AnyType : AnyTypeOf<[AnyType]>;
def VHLO_AnyAttr : AnyAttrOf<[AnyAttr]>;
def VHLO_AnyRegion : Region<CPred<"true">, "any region">;

def VersionedOpInterface : OpInterface<"VersionedOpInterface"> {
  let methods = [
    InterfaceMethod<
      "Returns the minimum version of the VHLO dialect an op is supported in.",
      "mlir::vhlo::Version", "getMinVersion">,
    InterfaceMethod<
      "Returns the maximum version (inclusive) of the VHLO dialect an op is supported in.",
      "mlir::vhlo::Version", "getMaxVersion">,
  ];
}

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

//===----------------------------------------------------------------------===//
// VHLO Type Versioning
//===----------------------------------------------------------------------===//

class VHLO_TypeDef<string cppName,
                   string name,
                   string cppBaseClass = "::mlir::Type",
                   string minVersion = "0.3.0",
                   string maxVersion = "current">
  : TypeDef<VHLO_Dialect, cppName, [VHLO_VersionedTypeInterface], cppBaseClass> {
  let mnemonic = name;
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
// VHLO Type Definitions.
//===----------------------------------------------------------------------===//

// VHLO is intended to represent the layout only, as such uses AnyType everywhere.
def VHLO_AnyType : AnyTypeOf<[AnyType]>;
def VHLO_AnyAttr : AnyAttrOf<[AnyAttr]>;
def VHLO_AnyRegion : Region<CPred<"true">, "any region">;

// Token type.
def VHLO_Token : VHLO_TypeDef<"Token", "token">;

// A type wrapper to conrtol serialization and deserialization
def VHLO_Wrapped : VHLO_TypeDef<"Wrapped", "wrapped"> {
  let mnemonic = "wrapped";
  let parameters = (ins "mlir::Type":$data);
  let assemblyFormat = "`<` $data `>`";
}

// Element Types
def VHLO_IntegerType : VHLO_TypeDef<"IntegerV1", "integer"> {
  let parameters = (ins "::mlir::IntegerType":$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    ::mlir::LogicalResult IntegerV1Type::verify(::llvm::function_ref<InFlightDiagnostic()> emitError,
                     ::mlir::IntegerType value) {
      llvm::SmallVector<unsigned> validWidths{4, 8, 16, 32, 64};
      bool isPred = (value.getWidth() == 1 && value.isSignless());
      if (!isPred && !llvm::is_contained(validWidths, value.getWidth())) {
        return emitError() << "invalid integer width " << value;
      }
      //if (!value.isSignless() && !value.isUnsigned()) {
      //  return emitError() << "invalid integer signedness " << value;
      //}
      return success();
    }
  }];
}
def VHLO_ComplexV1 : VHLO_TypeDef<"ComplexV1", "complex", "::mlir::ComplexType"> {
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}
def VHLO_BFloat16V1 : VHLO_TypeDef<"BFloat16V1", "bf16", "::mlir::BFloat16Type">;
def VHLO_Float16V1 : VHLO_TypeDef<"Float16V1", "f16", "::mlir::Float16Type">;
def VHLO_Float32V1 : VHLO_TypeDef<"Float32V1", "f32", "::mlir::Float32Type">;
def VHLO_Float64V1 : VHLO_TypeDef<"Float64V1","f64", "::mlir::Float64Type">;
def VHLO_IndexV1 : VHLO_TypeDef<"IndexV1", "index", "::mlir::IndexType">;

// Quantized Type
def VHLO_Double : APFloatParameter<""> {
  let parser = [{
    [&]() -> FailureOr<llvm::APFloat> {
      double value;
      if (failed($_parser.parseFloat(value))) {
        return failure();
      }
      return APFloat(value);
    }()
  }];
  let printer = "$_printer << $_self;";
}
def VHLO_UniformQuantizedV1 : VHLO_TypeDef<"UniformQuantizedV1", "quant"> {
  let parameters = (ins
    "unsigned":$flags,
    "::mlir::Type":$storageType,
    "::mlir::Type":$expressedType,
    VHLO_Double:$scale,
    "int64_t":$zeroPoint,
    "int64_t":$storageTypeMin,
    "int64_t":$storageTypeMax
  );
  let assemblyFormat = "`<` $storageType `` `:` `` $expressedType `,` $scale `` `:` `` $zeroPoint `,` $storageTypeMin `` `:` `` $storageTypeMax `,` $flags `>`";
}

//===----------------------------------------------------------------------===//
// VHLO traits
//===----------------------------------------------------------------------===//

// VHLO intentionally does not include traits since it is only used to represent
// the in-memory structure of the IR at a given version.
//
// This section is included for file parity between VhloBase.td and Base.td

#endif // STABLEHLO_DIALECT_VHLO_BASE
