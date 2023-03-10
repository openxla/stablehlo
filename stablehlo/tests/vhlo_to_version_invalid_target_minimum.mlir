// RUN: stablehlo-opt --vhlo-to-version='target=0.0.0' --verify-diagnostics %s
// expected-error @-2 {{target version 0.0.0 is less than minimum supported}}
