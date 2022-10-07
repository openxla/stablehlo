// RUN: stablehlo-translate --compat --use-test-converter --verify-diagnostics %s

// expected-warning @+1 {{file version 45 is greater than the StableHLO consumer version 40. Compatibility is not guaranteed.}}
func.func private @test_upgrade() -> ()
  attributes {stablehlo.compat_version = 45 : i32} {
  func.return
}
