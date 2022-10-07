// RUN: stablehlo-translate --compat --use-test-converter --verify-diagnostics %s

// expected-warning @+1 {{file version 32 is less than the minimum suported StableHLO file version 35. Compatibility is not guaranteed.}}
func.func private @test_upgrade() -> ()
  attributes {compat_version = 32 : i32} {
  func.return
}
