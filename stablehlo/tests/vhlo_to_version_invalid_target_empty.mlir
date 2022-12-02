// RUN: stablehlo-opt --vhlo-to-version --verify-diagnostics %s
// expected-error @-2 {{No target version specified. Specify target using: --vhlo-to-version='target=[targetVersion]'}}
