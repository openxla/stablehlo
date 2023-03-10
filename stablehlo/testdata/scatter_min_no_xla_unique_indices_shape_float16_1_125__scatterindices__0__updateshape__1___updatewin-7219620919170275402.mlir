// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xf16>, tensor<1xf16>)
    %2 = call @expected() : () -> tensor<1x125xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xf16>, tensor<1xi32>, tensor<1xf16>) -> tensor<1x125xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xf16>, tensor<1x125xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xf16>, tensor<1xf16>) {
    %0 = stablehlo.constant dense<"0x10408C38EC3D5C3DDE3173C52ABD0042543E6DC0BD371045503E67393A3D18BD60BD2945E4BFC63A0838F9BE4D47AE39A333BAC06AC5A3BC8141E53D3F391845A23C92430E3F65B96DBED337C4405CBD983F44BF1BBE43C1DA3FE23D93BEBD3E20405DBD5DC16CC1FCBBD33D69C19939A7B00F3EBEC10BC0123C34C382462132BEBF06BCB4BF6DC6E44400C652C17EC406BCF9AB31C3BD439C389743AB4193392ABE7D4133C002C6F7BD644155C44AC12A3A36C3D9C0D9BE893D5D3CADC45FBBD4BE334134C341C058440C2E12C6BCBE7540143CE5B886BF924800C210BABF393D408ABCB9AC203C1F450F2E134063388B438842D93B12C58B40"> : tensor<1x125xf16>
    %1 = stablehlo.constant dense<-9.545890e-01> : tensor<1xf16>
    return %0, %1 : tensor<1x125xf16>, tensor<1xf16>
  }
  func.func private @expected() -> tensor<1x125xf16> {
    %0 = stablehlo.constant dense<"0xA3BB8C38EC3D5C3DDE3173C52ABD0042543E6DC0BD371045503E67393A3D18BD60BD2945E4BFC63A0838F9BE4D47AE39A333BAC06AC5A3BC8141E53D3F391845A23C92430E3F65B96DBED337C4405CBD983F44BF1BBE43C1DA3FE23D93BEBD3E20405DBD5DC16CC1FCBBD33D69C19939A7B00F3EBEC10BC0123C34C382462132BEBF06BCB4BF6DC6E44400C652C17EC406BCF9AB31C3BD439C389743AB4193392ABE7D4133C002C6F7BD644155C44AC12A3A36C3D9C0D9BE893D5D3CADC45FBBD4BE334134C341C058440C2E12C6BCBE7540143CE5B886BF924800C210BABF393D408ABCB9AC203C1F450F2E134063388B438842D93B12C58B40"> : tensor<1x125xf16>
    return %0 : tensor<1x125xf16>
  }
}

