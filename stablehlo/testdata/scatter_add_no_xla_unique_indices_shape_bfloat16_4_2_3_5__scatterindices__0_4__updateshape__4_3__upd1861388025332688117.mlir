// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>)
    %2 = call @expected() : () -> tensor<4x2x3x5xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xbf16>, tensor<2xi32>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>) {
    %0 = stablehlo.constant dense<"0x88408A3F8F3F81BC1CC0B4BFA94002C01A3F94C0E1BFCC3E8A3F8A40FD3F8C3F9F40FDBFCEBF89BE9AC07E4022C053C093C01740653F52408540573F8F3FFD3F8E40F2BD8A3F3E3FBCBFC8BF63C0A33F20C0C6BE193F3D4053C0D3BF5FBFC0BF3940C63F2BBF08C0E9BF214040C0C93F073F8840E0402ABF3EC09DBD5EC0A4401AC084C0FDBF9EC02740823F69BF234022C035406D409AC0B0C016C09440E53F753FCEBF97BFB8BF344037C0933EF9BE1C40154007C18F4085408F40ADBFB1BF01C0033F30C091403F3F693F17C0A2BF33C02FC0DC40533E0EC0843F06C00E40BD4006C0E9BFA0BF834004BF8740B840"> : tensor<4x2x3x5xbf16>
    %1 = stablehlo.constant dense<[[-3.859380e+00, -2.437500e+00, 3.398440e-01], [-2.265630e+00, 7.812500e+00, -1.937500e+00], [3.296880e+00, 1.703130e+00, -2.406250e+00], [-2.000000e+00, -9.250000e+00, 5.039060e-01]]> : tensor<4x3xbf16>
    return %0, %1 : tensor<4x2x3x5xbf16>, tensor<4x3xbf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xbf16> {
    %0 = stablehlo.constant dense<"0x88408A3F8F3F81BCCAC0B4BFA94002C01A3FE2C0E1BFCC3E8A3F8A4014408C3F9F40FDBFCEBF89BE9AC07E4022C053C093C01740653F52408540573F8F3FFD3F8E40F2BD98BF3E3FBCBFC8BF63C0114120C0C6BE193F3D40A8C0D3BF5FBFC0BF3940C63F2BBF08C0E9BF214040C0C93F073F8840E0402ABF3EC09DBD5EC0A440643F84C0FDBF9EC027402E4069BF234022C03540A63F9AC0B0C016C09440E53F753FCEBF97BFB8BF344037C0933EF9BE1C40154007C18F4085408F4056C0B1BF01C0033F30C097C03F3F693F17C0A2BF13C02FC0DC40533E0EC0843F06C00E40BD4006C0E9BFA0BF834004BF8740B840"> : tensor<4x2x3x5xbf16>
    return %0 : tensor<4x2x3x5xbf16>
  }
}

