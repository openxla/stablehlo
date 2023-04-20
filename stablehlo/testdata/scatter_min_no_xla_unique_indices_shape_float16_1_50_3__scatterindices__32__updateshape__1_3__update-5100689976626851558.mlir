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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xf16>, tensor<1xi32>, tensor<1x3xf16>) -> tensor<1x50x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xf16>, tensor<1x50x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xf16>, tensor<1x3xf16>) {
    %0 = stablehlo.constant dense<"0x1BAE50C71C45823E69A81D3B2E3743BE0548633E2ABD56BAFA455B400AB8FEBA01BE7B39FFC500B9F737AF3234C0E1C0AB463FC46AC47BBD07BBDB4183C130C0A0BD084410C0033738440FC83DC0B1BB4B3E3F44D5384146584435C3D7417EB2B3BE62413FC5E72A1AC2E4BDD03E32BCD5C474445EC461C0454458358BC655C1EB39C740FA438A40673E3BC276BDAEACA5C1A2BD623E3440A33625BFB5A4043C6437AE3E374527BB4244AC41953D00C267BF8E41644448395241F73E97BA3D4655C48DACA933B1C06C41CB3F1BBFF23F8CBD7C45ED4476C244B9543E94121635FC32524180417D4680BEAEC08CBB3DC0A93FC23A19B667C0E9C0D2C0993DEFBDA6BCE7403941FBB402BEE635D09D6ABEB242A6B22E37302E8DBED7453F43213F4446DD401DC4D94050C10EBC"> : tensor<1x50x3xf16>
    %1 = stablehlo.constant dense<[[2.638670e+00, -3.740230e+00, -3.445310e+00]]> : tensor<1x3xf16>
    return %0, %1 : tensor<1x50x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> tensor<1x50x3xf16> {
    %0 = stablehlo.constant dense<"0x1BAE50C71C45823E69A81D3B2E3743BE0548633E2ABD56BAFA455B400AB8FEBA01BE7B39FFC500B9F737AF3234C0E1C0AB463FC46AC47BBD07BBDB4183C130C0A0BD084410C0033738440FC83DC0B1BB4B3E3F44D5384146584435C3D7417EB2B3BE62413FC5E72A1AC2E4BDD03E32BCD5C474445EC461C0454458358BC655C1EB39C740FA438A40673E3BC276BDAEACA5C1A2BD623E3440A33625BFB5A4043C6437AE3E374527BB4244AC41953D00C267BF8E41644448395241F73E97BA3D4655C47BC3E4C2B1C06C41CB3F1BBFF23F8CBD7C45ED4476C244B9543E94121635FC32524180417D4680BEAEC08CBB3DC0A93FC23A19B667C0E9C0D2C0993DEFBDA6BCE7403941FBB402BEE635D09D6ABEB242A6B22E37302E8DBED7453F43213F4446DD401DC4D94050C10EBC"> : tensor<1x50x3xf16>
    return %0 : tensor<1x50x3xf16>
  }
}

