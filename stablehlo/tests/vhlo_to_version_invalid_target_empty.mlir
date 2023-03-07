// RUN: stablehlo-opt --vhlo-to-version --verify-diagnostics %s
// expected-error @-2 {{No target version specified.}}
