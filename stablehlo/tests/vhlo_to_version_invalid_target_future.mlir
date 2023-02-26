// RUN: stablehlo-opt --vhlo-to-version='target=100.10.10' --verify-diagnostics %s
// expected-error @-2 {{target version 100.10.10 is greater than current version}}
