// RUN: stablehlo-opt --vhlo-to-version='target=x.y.z' --verify-diagnostics %s
// expected-error @-2 {{Invalid target version argument 'x.y.z'}}
