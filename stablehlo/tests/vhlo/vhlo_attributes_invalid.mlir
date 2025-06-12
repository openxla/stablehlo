// RUN: stablehlo-opt --vhlo-to-version=target=1.9.0 -verify-diagnostics --split-input-file %s

func.func @invalid_dict_element_value() -> () attributes {
  // expected-error @+1 {{expected VHLO key attribute}}
  vhlo.attr = #vhlo.dict_v1<{"attr1" = #vhlo.array_v1<[]>}>
} {
  return
}

// -----

func.func @invalid_result_accuracy() -> () attributes {
  // expected-error @+1 {{expected VHLO result accuracy mode}}
  vhlo.attr = #vhlo.result_accuracy_v1<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
} {
  return
}
