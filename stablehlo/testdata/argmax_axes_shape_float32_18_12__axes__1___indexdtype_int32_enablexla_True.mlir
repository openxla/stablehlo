// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<18x12xf32>
    %1 = call @expected() : () -> tensor<18xi32>
    %2 = call @argmax(%0) : (tensor<18x12xf32>) -> tensor<18xi32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<18xi32>, tensor<18xi32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<18x12xf32> {
    %0 = stablehlo.constant dense<"0x70F184C0DA5B664016DD7AC0660CAABED8599140F332674073FBD6BF2102B33FDDBB6EC0E5488EC02C00C5404505D03E0029954065D2653FD85D9BBF92B00041E6650A40E00BF43F7853C7BEA96940407FFE804033252D404EE88EBF5532BCBFF637AC3E39D00D41625B3AC03E9B84400FFD3CBF6C4E9B4057549FBE522C503FCFF1D6BE5854E1BFCEF162407F66A5407187403F202FF4BF04487B3F4BF74340B429D140D5D03E401E524B3F08B6ADBE86E13B3F86F644406DDC3AC0DE30C7C04D3D9540BDBEBBBF61D660400F072F40716E4740CB890C40AA8D5CBF4E99DFBC43EB28BFBA521D40A7EA3A409FB7A5405F4863C0C36B37405F3A87BEE6E522400A44014007A7763EECBA474068FF87BF29EE31BF0400DABDCD2DDBBFC77FDA3EEAE45D40EA37A43D858DF23FF8FD5D4007BA8FBF746E44C014B711C08C8FE2409C1832C0E4313CC09A5280BFB725C0BDFF0353405BA09940989D9CBF399395C04251B0BEB39D3D40844722C0FCA10FBFBBA984C0FFBFCC3F39FEB43F818506403281053F2D4AAE3E52E5B54038F13DBE419767405D6A093FAD457C3F228FCF3FC30AE8BFC7566D40C6B2BCBFB5C1423F06E442BF3F2516C0A248203FBAB52EC0C2AE1B3F75B30BC09786DCBFECF875C05C1E69BF49A7BCBF6F1F043FC7470FC1ABC48BBB827A5FBF914B60C0A54C89BF3600C2BF2D3618404A828640A35991BE42496DBFE2A1A5C0CE74F73FC2626B40C061CB40F9094140C5D7B14018564CBF2113E9BF675BD43E5B1657C00648FD3F0E0ADB3D59EE913E4DA677C0D8C9B240D5BF9B40FF83644038903ABFDC0A8DBF9CA8CA402363F5BFF8F0853F80F1F53D2F6ABA40E56EDA3F99BC1640E8B4F2BFBBB0A3BFE7D711C0EB6204C063A205407F562A40E6FC3140A782AE40E76505404C6884C08F2ABDBF2F4B03C04B1AB8BECDFD2440AB4CB1BF817F23C0F30D80C0A4A26D3F2594F3BF0893BBC05E2A1EC0EBDF7B3FAE620D40AF8FBD3EE30E29BF2D2DDCBF4078A7BFB5385E409795ED3F9B4D843F6E9883407BF0873E5B112CC074416C402A97CCC0002620C01A4A4FBEA4B80CBF6CA78ABF8FAB9EBF993BC240D28F66BFBE326BC09E8DB8BE08349940E662DC3DDBDA5E40125F44BF31E735C044E3F7BF34531A40A967A4C0C3986FC0537360C089BB3640F06881400C80EC3FCA47E6BFC2AB8840B5D6EC3FB9EA8C3F"> : tensor<18x12xf32>
    return %0 : tensor<18x12xf32>
  }
  func.func private @expected() -> tensor<18xi32> {
    %0 = stablehlo.constant dense<[10, 3, 1, 4, 11, 6, 7, 1, 2, 2, 6, 0, 4, 6, 0, 5, 3, 9]> : tensor<18xi32>
    return %0 : tensor<18xi32>
  }
  func.func private @argmax(%arg0: tensor<18x12xf32>) -> tensor<18xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<18x12xi32>
    %1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [1] : (tensor<18x12xf32>, tensor<18x12xi32>, tensor<f32>, tensor<i32>) -> (tensor<18xf32>, tensor<18xi32>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %4 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %5 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.or %4, %5 : tensor<i1>
      %7 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %8 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %9 = stablehlo.and %7, %8 : tensor<i1>
      %10 = stablehlo.or %6, %9 : tensor<i1>
      %11 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %12 = stablehlo.select %10, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %11, %12 : tensor<f32>, tensor<i32>
    }
    return %3#1 : tensor<18xi32>
  }
}
