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
    A shim opset of Versioned StableHLO ops for versions 0.x.x and 1.x.x.

    Version log:
      0.3.0: Bootstrap from MHLO: https://github.com/openxla/stablehlo/pull/1.
      0.4.0: Add AllGatherOp::use_global_device_ids: https://github.com/openxla/stablehlo/pull/272.
             Add CollectivePermuteOp::channel_handle: https://github.com/openxla/stablehlo/pull/388.
             Add CustomCallOp::output_operand_aliases: https://github.com/openxla/stablehlo/pull/403.
             Add PartitionIdOp: https://github.com/openxla/stablehlo/pull/673.
             Add AllToAllOp::channel_handle: https://github.com/openxla/stablehlo/pull/794.
  }];

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif
