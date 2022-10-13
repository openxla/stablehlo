// RUN: stablehlo-translate --compat --use-test-converter --target=38 --verify-diagnostics %s 

// expected-error @+1 {{failed to apply downgrades}}
func.func private @test_upgrade_invliad(%arg0: tensor<2xi1>) -> tensor<2xi1> 
  attributes {stablehlo.compat_version = 39 : i32} {
  // expected-error @+1 {{expected version_39_attr for downgrade.}}
  %0 = stablehlo.add %arg0, %arg0 {} : tensor<2xi1>
  %1 = stablehlo.subtract %arg0, %arg0 : tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

