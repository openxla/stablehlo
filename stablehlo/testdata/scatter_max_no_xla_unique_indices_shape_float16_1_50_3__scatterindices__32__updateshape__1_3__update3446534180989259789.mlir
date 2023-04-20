// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xf16>, tensor<1x3xf16>)
    %2 = call @expected() : () -> tensor<1x50x3xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xf16>, tensor<1xi32>, tensor<1x3xf16>) -> tensor<1x50x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xf16>, tensor<1x50x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xf16>, tensor<1x3xf16>) {
    %0 = stablehlo.constant dense<"0x1A41BABCC7B1474088BF57BCFC41F1B8E338004515C7DDB94EBE52413547CFAEA13CC4444746AC4641C5CCBE2B424242C4420E384E3EB0C1D64350B8CF3D31C558BC4AC3963F2CB23CC4664235C194C145BCADB959427839A62DF7BCCBBE373CDD44EFC42CC130BBF43D7645F33E44C3DD44454827C27F4291BCC9C02A3D332CC03C98B9DB439F3B2A3EB738F442EA3E303FAB430AC6DD3F164287BD2FC409BB42B09BBFBE3E633E80333E3D68BC303B25BF7B40E5462D4212BB5541BC38DDBD6C3C74402BC40C3F9543F93FC1B820B8A4B8CD4229C119BD8B403EC3FF40F1A9F8BCBF4260BDAB41B23EB232AE45DB4161BEAF40D6C5753DDD38FEC129C39C34E2448E3C0DBAA731E5400EC1BD409733F9BA14BB40C172C1F9BFB3C2E5C378C7E42EFC476B4308BB67C22438"> : tensor<1x50x3xf16>
    %1 = stablehlo.constant dense<[[-3.699220e+00, 4.074220e+00, -1.113280e+00]]> : tensor<1x3xf16>
    return %0, %1 : tensor<1x50x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> tensor<1x50x3xf16> {
    %0 = stablehlo.constant dense<"0x1A41BABCC7B1474088BF57BCFC41F1B8E338004515C7DDB94EBE52413547CFAEA13CC4444746AC4641C5CCBE2B424242C4420E384E3EB0C1D64350B8CF3D31C558BC4AC3963F2CB23CC4664235C194C145BCADB959427839A62DF7BCCBBE373CDD44EFC42CC130BBF43D7645F33E44C3DD44454827C27F4291BCC9C02A3D332CC03C98B9DB439F3B2A3EB738F442EA3E303FAB430AC6DD3F164287BD2FC409BB42B09BBFBE3E633E80333E3D68BC303B25BF7B40E5462D4212BB5541BC38DDBD6C3C134474BC0C3F9543F93FC1B820B8A4B8CD4229C119BD8B403EC3FF40F1A9F8BCBF4260BDAB41B23EB232AE45DB4161BEAF40D6C5753DDD38FEC129C39C34E2448E3C0DBAA731E5400EC1BD409733F9BA14BB40C172C1F9BFB3C2E5C378C7E42EFC476B4308BB67C22438"> : tensor<1x50x3xf16>
    return %0 : tensor<1x50x3xf16>
  }
}

