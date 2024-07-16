// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3x5xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 4]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>)
    %1 = call @expected() : () -> tensor<4x2x3x5xbf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<4x2x3x5xbf16>, tensor<2xi64>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> ()
    return %2 : tensor<4x2x3x5xbf16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xbf16> {mhlo.layout_mode = "default"}, tensor<4x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xE43E46C0C8BF0441ADBE14BF8140AFC08CBC5340D0C079BF8BBF1F3F3CBF6C40D9C0F8BFC3BC0ABEB2BF4FBFE93F1CC021BF8A404BC015C00A40C43E2AC0C43FA04058BE92C0B73FCEBF2040DE3F9EC0723F1A40D6BF3AC089C074BE88C013C0004050408E40DDBF98BFA0BF063F7E40334012C0B6BE9B409C4019404A40A73F27403AC0A040ACBFFC3FD23F0E408CC04040543F4140443FE2BFA8C0DEBF43400FC0AB3E97400BC0864011C085407F405D3FAF3FCDC0EABF7ABFFC404D4032BF6FBF654088C0BEBF633F423DD73F0B40D63F023E00C0A33D48C0CBC015C00EC05FC010C0A33F90404640AA400140ACC0"> : tensor<4x2x3x5xbf16>
    %cst_0 = stablehlo.constant dense<[[-3.703130e+00, -1.062500e+00, -2.265630e+00], [-1.250000e+00, 1.101560e+00, 5.351560e-01], [-5.062500e+00, 2.187500e+00, -5.125000e+00], [-7.304680e-01, -3.000000e+00, -3.171880e+00]]> : tensor<4x3xbf16>
    return %cst, %cst_0 : tensor<4x2x3x5xbf16>, tensor<4x3xbf16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xE43E46C0C8BF04416DC014BF8140AFC08CBC88BFD0C079BF8BBF1F3F11C06C40D9C0F8BFC3BC0ABEB2BF4FBFE93F1CC021BF8A404BC015C00A40C43E2AC0C43FA04058BE92C0B73FCEBF2040DE3F9EC0723F1A40D6BF3AC089C074BE88C013C0004050408E40DDBF98BFA0BF063F7E40334012C0B6BE9B409C4019404A40A73FA2C03AC0A040ACBFFC3FD23F0E408CC04040543FA4C0443FE2BFA8C0DEBF43400FC0AB3E97400BC0864011C085407F405D3FAF3FCDC0EABF7ABFFC403BBF32BF6FBF654088C040C0633F423DD73F0B404BC0023E00C0A33D48C0CBC015C00EC05FC010C0A33F90404640AA400140ACC0"> : tensor<4x2x3x5xbf16>
    return %cst : tensor<4x2x3x5xbf16>
  }
}
