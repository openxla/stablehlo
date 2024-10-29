/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_VHLO_DIALECT
#define STABLEHLO_DIALECT_VHLO_DIALECT

def VHLO_Dialect : Dialect {
  let name = "vhlo";
  let cppNamespace = "::mlir::vhlo";

  let description = [{
    A "shallow" versioned copy of the StableHLO dialect that has been simplified
    down to a bare minimum which is used for upgrades, downgrades, and
    serialization/deserialization.

    Version log:
      0.9.0: Initial stability guarantees.
      0.10.0: Introduce `f8E4M3FNUZ` and `f8E5M2FNUZ` types.
      0.11.0: Introduce `f8E4M3B11FNUZ` type.
      0.12.0: MLIR bytecode version 1 => 3.
      0.14.0: MLIR bytecode version 3 => 5 (revised to 4 in #1827).
      0.15.0: MLIR bytecode version 5 => 6, use properties in VHLO.
      0.16.0: Introduce `collective_broadcast` operation.
      0.17.0: Allow reduce operations to promote to higher bitwidth.
      0.18.0: Introduce `UniformQuantizedPerAxisType` type.
      0.19.0: Introduce `composite` operation.
      0.20.0: Remove `padding` attribute from `dynamic_conv`.
      1.0.0: Increase compatibility guarantees to 5 years backward, 2 years forward (no functional changes relative to 0.20.0).
      1.1.0: Add gather/scatter batching dimensions.
      1.2.0: Introduce `si2` and `ui2` types.
      1.3.0: Extend `custom_call` op `backend_config` to support `DictionaryAttr`.
      1.4.0: Add `tan` op to StableHLO opset.
      1.5.0: Make collective ops (`all_reduce`, `all_gather`, `all_to_all`) variadic.
      1.6.0: Add DotAlgorithm specificaiton to `dot_general`.
      1.7.0: Introduce `f8E4M3` and `f8E3M4` types.
      1.8.0: Introduce `f4E2M1FN`, `f6E2M3FN`, `f6E3M2FN` and `f8E8M0FNU` types.
  }];

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
  let usePropertiesForAttributes = 1;
}

#endif  // STABLEHLO_DIALECT_VHLO_DIALECT
