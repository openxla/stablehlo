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

class VHLO_TypeDef<string name,
                   string minVersion = "0.3.0",
                   string maxVersion = "current">
  : TypeDef<VHLO_Dialect, name, [VHLO_VersionedTypeInterface]> {
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
def VHLO_Token : VHLO_TypeDef<"Token"> {
  let mnemonic = "token";
}

// A type wrapper to conrtol serialization and deserialization
def VHLO_Wrapped : VHLO_TypeDef<"Wrapped"> {
  let mnemonic = "wrapped";
  let parameters = (ins "mlir::Type":$data);
  let assemblyFormat = "`<` $data `>`";
}

//===----------------------------------------------------------------------===//
// VHLO traits
//===----------------------------------------------------------------------===//

// VHLO intentionally does not include traits since it is only used to represent
// the in-memory structure of the IR at a given version.
//
// This section is included for file parity between VhloBase.td and Base.td

#endif // STABLEHLO_DIALECT_VHLO_BASE
