// RUN: stablehlo-opt %s --hlo-test-speculatability --split-input-file --allow-unregistered-dialect | FileCheck %s

// -----

// UnaryElementwise ops

// -----

// CHECK-LABEL: func @abs_multidim
// CHECK-NEXT:  return
func.func @abs_multidim(%dynamic_arg: tensor<?x?xf64>) {
  %not_speculatable = stablehlo.abs %dynamic_arg : (tensor<?x?xf64>) -> tensor<?x2xf64>
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable) : (tensor<?x2xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @abs
// CHECK-NEXT:  return
func.func @abs(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.abs %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.abs %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.abs %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.abs %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @cbrt
// CHECK-NEXT:  return
func.func @cbrt(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.cbrt %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.cbrt %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.cbrt %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.cbrt %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @ceil
// CHECK-NEXT:  return
func.func @ceil(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.ceil %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.ceil %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.ceil %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.ceil %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @convert
// CHECK-NEXT:  return
func.func @convert(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.convert %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.convert %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.convert %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.convert %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @count_leading_zeros
// CHECK-NEXT:  return
func.func @count_leading_zeros(%static_arg: tensor<2xi64>, %dynamic_arg: tensor<?xi64>) {
  %speculatable_0 = stablehlo.count_leading_zeros %static_arg : (tensor<2xi64>) -> tensor<2xi64>
  %speculatable_1 = stablehlo.count_leading_zeros %static_arg : (tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_0 = stablehlo.count_leading_zeros %dynamic_arg : (tensor<?xi64>) -> tensor<2xi64>
  %speculatable_2 = stablehlo.count_leading_zeros %dynamic_arg : (tensor<?xi64>) -> tensor<?xi64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @cosine
// CHECK-NEXT:  return
func.func @cosine(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.cosine %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.cosine %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.cosine %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.cosine %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @exponential
// CHECK-NEXT:  return
func.func @exponential(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.exponential %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.exponential %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.exponential %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.exponential %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @exponential_minus_one
// CHECK-NEXT:  return
func.func @exponential_minus_one(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.exponential_minus_one %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.exponential_minus_one %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.exponential_minus_one %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.exponential_minus_one %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @floor
// CHECK-NEXT:  return
func.func @floor(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.floor %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.floor %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.floor %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.floor %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @imag
// CHECK-NEXT:  return
func.func @imag(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.imag %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.imag %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.imag %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.imag %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @is_finite
// CHECK-NEXT:  return
func.func @is_finite(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.is_finite %static_arg : (tensor<2xf64>) -> tensor<2xi1>
  %speculatable_1 = stablehlo.is_finite %static_arg : (tensor<2xf64>) -> tensor<?xi1>
  %not_speculatable_0 = stablehlo.is_finite %dynamic_arg : (tensor<?xf64>) -> tensor<2xi1>
  %speculatable_2 = stablehlo.is_finite %dynamic_arg : (tensor<?xf64>) -> tensor<?xi1>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xi1>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xi1>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xi1>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xi1>) -> ()
  return
}

// -----

// CHECK-LABEL: func @log
// CHECK-NEXT:  return
func.func @log(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.log %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.log %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.log %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.log %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @log_plus_one
// CHECK-NEXT:  return
func.func @log_plus_one(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.log_plus_one %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.log_plus_one %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.log_plus_one %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.log_plus_one %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @logistic
// CHECK-NEXT:  return
func.func @logistic(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.logistic %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.logistic %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.logistic %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.logistic %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @not
// CHECK-NEXT:  return
func.func @not(%static_arg: tensor<2xi64>, %dynamic_arg: tensor<?xi64>) {
  %speculatable_0 = stablehlo.not %static_arg : (tensor<2xi64>) -> tensor<2xi64>
  %speculatable_1 = stablehlo.not %static_arg : (tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_0 = stablehlo.not %dynamic_arg : (tensor<?xi64>) -> tensor<2xi64>
  %speculatable_2 = stablehlo.not %dynamic_arg : (tensor<?xi64>) -> tensor<?xi64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @negate
// CHECK-NEXT:  return
func.func @negate(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.negate %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.negate %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.negate %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.negate %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @popcnt
// CHECK-NEXT:  return
func.func @popcnt(%static_arg: tensor<2xi64>, %dynamic_arg: tensor<?xi64>) {
  %speculatable_0 = stablehlo.popcnt %static_arg : (tensor<2xi64>) -> tensor<2xi64>
  %speculatable_1 = stablehlo.popcnt %static_arg : (tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_0 = stablehlo.popcnt %dynamic_arg : (tensor<?xi64>) -> tensor<2xi64>
  %speculatable_2 = stablehlo.popcnt %dynamic_arg : (tensor<?xi64>) -> tensor<?xi64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @real
// CHECK-NEXT:  return
func.func @real(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.real %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.real %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.real %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.real %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @round_nearest_afz
// CHECK-NEXT:  return
func.func @round_nearest_afz(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.round_nearest_afz %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.round_nearest_afz %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.round_nearest_afz %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.round_nearest_afz %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @round_nearest_even
// CHECK-NEXT:  return
func.func @round_nearest_even(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.round_nearest_even %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.round_nearest_even %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.round_nearest_even %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.round_nearest_even %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @rsqrt
// CHECK-NEXT:  return
func.func @rsqrt(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.rsqrt %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.rsqrt %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.rsqrt %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.rsqrt %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @sign
// CHECK-NEXT:  return
func.func @sign(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.sign %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.sign %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.sign %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.sign %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @sine
// CHECK-NEXT:  return
func.func @sine(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.sine %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.sine %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.sine %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.sine %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @sqrt
// CHECK-NEXT:  return
func.func @sqrt(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.sqrt %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.sqrt %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.sqrt %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.sqrt %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @tanh
// CHECK-NEXT:  return
func.func @tanh(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.tanh %static_arg : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.tanh %static_arg : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.tanh %dynamic_arg : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.tanh %dynamic_arg : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// Other unary ops

// -----

// CHECK-LABEL: func @reduce_precision
// CHECK-NEXT:  return
func.func @reduce_precision(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.reduce_precision %static_arg, format = e5m10 : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.reduce_precision %static_arg, format = e5m10 : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.reduce_precision %dynamic_arg, format = e5m10 : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.reduce_precision %dynamic_arg, format = e5m10 : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @reverse
// CHECK-NEXT:  return
func.func @reverse(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.reverse %static_arg, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.reverse %static_arg, dims = [0] : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.reverse %dynamic_arg, dims = [0] : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.reverse %dynamic_arg, dims = [0] : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @all_reduce
// CHECK-NEXT:  return
func.func @all_reduce(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = "stablehlo.all_reduce"(%static_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
      stablehlo.return %0 : tensor<f64>
  }) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = "stablehlo.all_reduce"(%static_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
      stablehlo.return %0 : tensor<f64>
  }) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = "stablehlo.all_reduce"(%dynamic_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
      stablehlo.return %0 : tensor<f64>
  }) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = "stablehlo.all_reduce"(%dynamic_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
      stablehlo.return %0 : tensor<f64>
  }) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @collective_broadcast
// CHECK-NEXT:  return
func.func @collective_broadcast(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = "stablehlo.collective_broadcast"(%static_arg) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = "stablehlo.collective_broadcast"(%static_arg) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = "stablehlo.collective_broadcast"(%dynamic_arg) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<?xf64>) -> tensor<2xf64>
  %speculatable_2 = "stablehlo.collective_broadcast"(%dynamic_arg) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @collective_permute
// CHECK-NEXT:  return
func.func @collective_permute(%static_arg: tensor<2x2xf64>, %dynamic_arg: tensor<?x?xf64>) {
  %speculatable_0 = "stablehlo.collective_permute"(%static_arg) {
    source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<2x2xf64>) -> tensor<2x2xf64>
  %speculatable_1 = "stablehlo.collective_permute"(%static_arg) {
    source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<2x2xf64>) -> tensor<?x?xf64>
  %not_speculatable_0 = "stablehlo.collective_permute"(%dynamic_arg) {
    source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<?x?xf64>) -> tensor<2x2xf64>
  %speculatable_2 = "stablehlo.collective_permute"(%dynamic_arg) {
    source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<?x?xf64>) -> tensor<?x?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2x2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2x2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?x?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @rng_bit_generator
// CHECK-NEXT:  return
func.func @rng_bit_generator(%static_arg: tensor<2xui64>, %dynamic_arg: tensor<?xui64>) {
  %speculatable_0:2 = "stablehlo.rng_bit_generator"(%static_arg) {
    rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>
  } : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x2xui64>)
  %speculatable_1:2 = "stablehlo.rng_bit_generator"(%static_arg) {
    rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>
  } : (tensor<2xui64>) -> (tensor<?xui64>, tensor<2x2xui64>)
  %not_speculatable_0:2 = "stablehlo.rng_bit_generator"(%dynamic_arg) {
    rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>
  } : (tensor<?xui64>) -> (tensor<2xui64>, tensor<2x2xui64>)
  %speculatable_2:2 = "stablehlo.rng_bit_generator"(%dynamic_arg) {
    rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>
  } : (tensor<?xui64>) -> (tensor<?xui64>, tensor<2x2xui64>)
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xui64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xui64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xui64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xui64>) -> ()
  return
}

// -----

// Unary ops with special restrictions

// -----

// CHECK-LABEL: func @all_gather
// CHECK-NEXT:  return
func.func @all_gather(%static_arg: tensor<2x2xf64>, %dynamic_arg: tensor<?x?xf64>) {
    %static_all_gather_dim = "stablehlo.all_gather"(%static_arg) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle=1, type=0>
    } : (tensor<2x2xf64>) -> tensor<2x8xf64>
    %dynamic_all_gather_dim = "stablehlo.all_gather"(%static_arg) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle=1, type=0>
    } : (tensor<2x2xf64>) -> tensor<2x?xf64>
    %dynamic_all_gather_dim_not_speculatable = "stablehlo.all_gather"(%dynamic_arg) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle=1, type=0>
    } : (tensor<?x?xf64>) -> tensor<2x?xf64>
    %result_dynamic_input_static = "stablehlo.all_gather"(%static_arg) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle=1, type=0>
    } : (tensor<2x2xf64>) -> tensor<?x?xf64>
    %result_dynamic_input_dynamic = "stablehlo.all_gather"(%dynamic_arg) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle=1, type=0>
    } : (tensor<?x?xf64>) -> tensor<?x?xf64>
    "hlo_test_speculatability.is_not_speculatable"(%static_all_gather_dim) : (tensor<2x8xf64>) -> ()
    "hlo_test_speculatability.is_speculatable"(%dynamic_all_gather_dim) : (tensor<2x?xf64>) -> ()
    "hlo_test_speculatability.is_not_speculatable"(%dynamic_all_gather_dim_not_speculatable) : (tensor<2x?xf64>) -> ()
    "hlo_test_speculatability.is_speculatable"(%result_dynamic_input_static) : (tensor<?x?xf64>) -> ()
    "hlo_test_speculatability.is_speculatable"(%result_dynamic_input_dynamic) : (tensor<?x?xf64>) -> ()
    return
}

// -----

// CHECK-LABEL: func @reduce_scatter
// CHECK-NEXT:  return
func.func @reduce_scatter(%static_arg: tensor<2x4xf64>, %dynamic_arg: tensor<?x?xf64>) {
    %static_scatter_dim = "stablehlo.reduce_scatter"(%static_arg) ({
      ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
        stablehlo.return %0 : tensor<f64>
    }) {
      scatter_dimension = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle=1, type=0>
    } : (tensor<2x4xf64>) -> tensor<2x2xf64>
    %dynamic_scatter_dim = "stablehlo.reduce_scatter"(%static_arg) ({
      ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
        stablehlo.return %0 : tensor<f64>
    }) {
      scatter_dimension = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle=1, type=0>
    } : (tensor<2x4xf64>) -> tensor<2x?xf64>
    %dynamic_scatter_dim_not_speculatable = "stablehlo.reduce_scatter"(%dynamic_arg) ({
      ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
        stablehlo.return %0 : tensor<f64>
    }) {
      scatter_dimension = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle=1, type=0>
    } : (tensor<?x?xf64>) -> tensor<2x?xf64>
    %result_dynamic_input_static = "stablehlo.reduce_scatter"(%static_arg) ({
      ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
        stablehlo.return %0 : tensor<f64>
    }) {
      scatter_dimension = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle=1, type=0>
    } : (tensor<2x4xf64>) -> tensor<?x?xf64>
    %result_dynamic_input_dynamic = "stablehlo.reduce_scatter"(%dynamic_arg) ({
      ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
        stablehlo.return %0 : tensor<f64>
    }) {
      scatter_dimension = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle=1, type=0>
    } : (tensor<?x?xf64>) -> tensor<?x?xf64>
    "hlo_test_speculatability.is_not_speculatable"(%static_scatter_dim) : (tensor<2x2xf64>) -> ()
    "hlo_test_speculatability.is_speculatable"(%dynamic_scatter_dim) : (tensor<2x?xf64>) -> ()
    "hlo_test_speculatability.is_not_speculatable"(%dynamic_scatter_dim_not_speculatable) : (tensor<2x?xf64>) -> ()
    "hlo_test_speculatability.is_speculatable"(%result_dynamic_input_static) : (tensor<?x?xf64>) -> ()
    "hlo_test_speculatability.is_speculatable"(%result_dynamic_input_dynamic) : (tensor<?x?xf64>) -> ()
    return
}

// -----

// CHECK-LABEL: func @all_to_all
// CHECK-NEXT:  return
func.func @all_to_all(%static_arg: tensor<2x4x8xf64>, %dynamic_arg: tensor<?x?x?xf64>) {
    %concat_and_split_are_static = "stablehlo.all_to_all"(%static_arg) {
      concat_dimension = 0 : i64,
      split_dimension = 1 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x4x8xf64>) -> tensor<4x2x8xf64>
    %concat_is_static = "stablehlo.all_to_all"(%static_arg) {
      concat_dimension = 0 : i64,
      split_dimension = 1 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x4x8xf64>) -> tensor<4x?x8xf64>
    %split_is_static = "stablehlo.all_to_all"(%static_arg) {
      concat_dimension = 0 : i64,
      split_dimension = 1 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x4x8xf64>) -> tensor<?x2x8xf64>
    %concat_and_split_are_dynamic = "stablehlo.all_to_all"(%static_arg) {
      concat_dimension = 0 : i64,
      split_dimension = 1 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x4x8xf64>) -> tensor<?x?x8xf64>
    %other_dim_static = "stablehlo.all_to_all"(%dynamic_arg) {
      concat_dimension = 0 : i64,
      split_dimension = 1 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<?x?x?xf64>) -> tensor<?x?x8xf64>
    %result_fully_dynamic_0 = "stablehlo.all_to_all"(%static_arg) {
      concat_dimension = 0 : i64,
      split_dimension = 1 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x4x8xf64>) -> tensor<?x?x?xf64>
    %result_fully_dynamic_1 = "stablehlo.all_to_all"(%dynamic_arg) {
      concat_dimension = 0 : i64,
      split_dimension = 1 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
    "hlo_test_speculatability.is_not_speculatable"(%concat_and_split_are_static) : (tensor<4x2x8xf64>) -> ()
    "hlo_test_speculatability.is_not_speculatable"(%concat_is_static) : (tensor<4x?x8xf64>) -> ()
    "hlo_test_speculatability.is_not_speculatable"(%split_is_static) : (tensor<?x2x8xf64>) -> ()
    "hlo_test_speculatability.is_speculatable"(%concat_and_split_are_dynamic) : (tensor<?x?x8xf64>) -> ()
    "hlo_test_speculatability.is_not_speculatable"(%other_dim_static) : (tensor<?x?x8xf64>) -> ()
    "hlo_test_speculatability.is_speculatable"(%result_fully_dynamic_0) : (tensor<?x?x?xf64>) -> ()
    "hlo_test_speculatability.is_speculatable"(%result_fully_dynamic_1) : (tensor<?x?x?xf64>) -> ()
    return
}

// -----

// CHECK-LABEL: func @transpose
// CHECK-NEXT:  return
func.func @transpose(
  %static: tensor<2x4xf64>, %first_dim_dynamic: tensor<?x4xf64>,
  %second_dim_dynamic: tensor<2x?xf64>, %dynamic: tensor<?x?xf64>,
  %three_d: tensor<1x2x3xf64>, %three_d_dynamic: tensor<1x2x?xf64>
) {
  %speculatable_0 = stablehlo.transpose %static, dims = [1, 0] : (tensor<2x4xf64>) -> tensor<4x2xf64>
  %not_speculatable_0 = stablehlo.transpose %second_dim_dynamic, dims = [1, 0] : (tensor<2x?xf64>) -> tensor<4x2xf64>
  %speculatable_1 = stablehlo.transpose %second_dim_dynamic, dims = [1, 0] : (tensor<2x?xf64>) -> tensor<?x2xf64>
  %not_speculatable_1 = stablehlo.transpose %first_dim_dynamic, dims = [1, 0] : (tensor<?x4xf64>) -> tensor<4x2xf64>
  %speculatable_2 = stablehlo.transpose %first_dim_dynamic, dims = [1, 0] : (tensor<?x4xf64>) -> tensor<4x?xf64>
  %not_speculatable_2 = stablehlo.transpose %dynamic, dims = [1, 0] : (tensor<?x?xf64>) -> tensor<4x2xf64>
  %not_speculatable_3 = stablehlo.transpose %dynamic, dims = [1, 0] : (tensor<?x?xf64>) -> tensor<?x2xf64>
  %not_speculatable_4 = stablehlo.transpose %dynamic, dims = [1, 0] : (tensor<?x?xf64>) -> tensor<4x?xf64>
  %speculatable_3 = stablehlo.transpose %dynamic, dims = [1, 0] : (tensor<?x?xf64>) -> tensor<?x?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<4x2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<4x2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?x2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<4x2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<4x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<4x2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?x2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<4x?xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_3) : (tensor<?x?xf64>) -> ()

  %speculatable_4 = stablehlo.transpose %three_d, dims = [1, 0, 2] : (tensor<1x2x3xf64>) -> tensor<2x1x3xf64>
  %not_speculatable_5 = stablehlo.transpose %three_d_dynamic, dims = [1, 0, 2] : (tensor<1x2x?xf64>) -> tensor<2x1x3xf64>
  %speculatable_5 = stablehlo.transpose %three_d_dynamic, dims = [1, 0, 2] : (tensor<1x2x?xf64>) -> tensor<2x1x?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_4) : (tensor<2x1x3xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<2x1x3xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_5) : (tensor<2x1x?xf64>) -> ()
  return
}

// -----

// BinaryElementwise and BinaryBitwiseOrLogicalElementwise ops

// -----

// CHECK-LABEL: func @add_multidim
// CHECK-NEXT: return
func.func @add_multidim(%static_arg: tensor<2x2xf64>, %partially_dynamic_arg: tensor<2x?xf64>) {
  %not_speculatable = stablehlo.add %static_arg, %partially_dynamic_arg : (tensor<2x2xf64>, tensor<2x?xf64>) -> tensor<?x?xf64>
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable) : (tensor<?x?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @add
// CHECK-NEXT: return
func.func @add(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.add %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.add %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.add %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_1 = stablehlo.add %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<?xf64>
  %not_speculatable_2 = stablehlo.add %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<2xf64>
  %not_speculatable_3 = stablehlo.add %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_4 = stablehlo.add %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_5 = stablehlo.add %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @atan2
// CHECK-NEXT: return
func.func @atan2(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.atan2 %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.atan2 %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.atan2 %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_1 = stablehlo.atan2 %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<?xf64>
  %not_speculatable_2 = stablehlo.atan2 %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<2xf64>
  %not_speculatable_3 = stablehlo.atan2 %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_4 = stablehlo.atan2 %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_5 = stablehlo.atan2 %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @complex
// CHECK-NEXT: return
func.func @complex(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.complex %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xcomplex<f64>>
  %speculatable_1 = stablehlo.complex %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<?xcomplex<f64>>
  %not_speculatable_0 = stablehlo.complex %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<2xcomplex<f64>>
  %not_speculatable_1 = stablehlo.complex %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<?xcomplex<f64>>
  %not_speculatable_2 = stablehlo.complex %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<2xcomplex<f64>>
  %not_speculatable_3 = stablehlo.complex %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<?xcomplex<f64>>
  %not_speculatable_4 = stablehlo.complex %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<2xcomplex<f64>>
  %not_speculatable_5 = stablehlo.complex %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xcomplex<f64>>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xcomplex<f64>>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xcomplex<f64>>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xcomplex<f64>>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xcomplex<f64>>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xcomplex<f64>>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xcomplex<f64>>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xcomplex<f64>>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xcomplex<f64>>) -> ()
  return
}

// -----

// CHECK-LABEL: func @divide
// CHECK-NEXT: return
func.func @divide(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.divide %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.divide %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.divide %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_1 = stablehlo.divide %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<?xf64>
  %not_speculatable_2 = stablehlo.divide %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<2xf64>
  %not_speculatable_3 = stablehlo.divide %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_4 = stablehlo.divide %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_5 = stablehlo.divide %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @maximum
// CHECK-NEXT: return
func.func @maximum(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.maximum %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.maximum %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.maximum %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_1 = stablehlo.maximum %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<?xf64>
  %not_speculatable_2 = stablehlo.maximum %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<2xf64>
  %not_speculatable_3 = stablehlo.maximum %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_4 = stablehlo.maximum %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_5 = stablehlo.maximum %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @minimum
// CHECK-NEXT: return
func.func @minimum(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.minimum %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.minimum %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.minimum %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_1 = stablehlo.minimum %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<?xf64>
  %not_speculatable_2 = stablehlo.minimum %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<2xf64>
  %not_speculatable_3 = stablehlo.minimum %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_4 = stablehlo.minimum %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_5 = stablehlo.minimum %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @multiply
// CHECK-NEXT: return
func.func @multiply(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.multiply %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.multiply %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.multiply %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_1 = stablehlo.multiply %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<?xf64>
  %not_speculatable_2 = stablehlo.multiply %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<2xf64>
  %not_speculatable_3 = stablehlo.multiply %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_4 = stablehlo.multiply %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_5 = stablehlo.multiply %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @power
// CHECK-NEXT: return
func.func @power(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.power %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.power %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.power %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_1 = stablehlo.power %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<?xf64>
  %not_speculatable_2 = stablehlo.power %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<2xf64>
  %not_speculatable_3 = stablehlo.power %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_4 = stablehlo.power %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_5 = stablehlo.power %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @remainder
// CHECK-NEXT: return
func.func @remainder(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.remainder %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.remainder %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.remainder %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_1 = stablehlo.remainder %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<?xf64>
  %not_speculatable_2 = stablehlo.remainder %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<2xf64>
  %not_speculatable_3 = stablehlo.remainder %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_4 = stablehlo.remainder %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_5 = stablehlo.remainder %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @shift_left
// CHECK-NEXT: return
func.func @shift_left(%static_arg: tensor<2xi64>, %dynamic_arg: tensor<?xi64>) {
  %speculatable_0 = stablehlo.shift_left %static_arg, %static_arg : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
  %speculatable_1 = stablehlo.shift_left %static_arg, %static_arg : (tensor<2xi64>, tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_0 = stablehlo.shift_left %static_arg, %dynamic_arg : (tensor<2xi64>, tensor<?xi64>) -> tensor<2xi64>
  %not_speculatable_1 = stablehlo.shift_left %static_arg, %dynamic_arg : (tensor<2xi64>, tensor<?xi64>) -> tensor<?xi64>
  %not_speculatable_2 = stablehlo.shift_left %dynamic_arg, %static_arg : (tensor<?xi64>, tensor<2xi64>) -> tensor<2xi64>
  %not_speculatable_3 = stablehlo.shift_left %dynamic_arg, %static_arg : (tensor<?xi64>, tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_4 = stablehlo.shift_left %dynamic_arg, %dynamic_arg : (tensor<?xi64>, tensor<?xi64>) -> tensor<2xi64>
  %not_speculatable_5 = stablehlo.shift_left %dynamic_arg, %dynamic_arg : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @shift_right_arithmetic
// CHECK-NEXT: return
func.func @shift_right_arithmetic(%static_arg: tensor<2xi64>, %dynamic_arg: tensor<?xi64>) {
  %speculatable_0 = stablehlo.shift_right_arithmetic %static_arg, %static_arg : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
  %speculatable_1 = stablehlo.shift_right_arithmetic %static_arg, %static_arg : (tensor<2xi64>, tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_0 = stablehlo.shift_right_arithmetic %static_arg, %dynamic_arg : (tensor<2xi64>, tensor<?xi64>) -> tensor<2xi64>
  %not_speculatable_1 = stablehlo.shift_right_arithmetic %static_arg, %dynamic_arg : (tensor<2xi64>, tensor<?xi64>) -> tensor<?xi64>
  %not_speculatable_2 = stablehlo.shift_right_arithmetic %dynamic_arg, %static_arg : (tensor<?xi64>, tensor<2xi64>) -> tensor<2xi64>
  %not_speculatable_3 = stablehlo.shift_right_arithmetic %dynamic_arg, %static_arg : (tensor<?xi64>, tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_4 = stablehlo.shift_right_arithmetic %dynamic_arg, %dynamic_arg : (tensor<?xi64>, tensor<?xi64>) -> tensor<2xi64>
  %not_speculatable_5 = stablehlo.shift_right_arithmetic %dynamic_arg, %dynamic_arg : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @shift_right_logical
// CHECK-NEXT: return
func.func @shift_right_logical(%static_arg: tensor<2xi64>, %dynamic_arg: tensor<?xi64>) {
  %speculatable_0 = stablehlo.shift_right_logical %static_arg, %static_arg : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
  %speculatable_1 = stablehlo.shift_right_logical %static_arg, %static_arg : (tensor<2xi64>, tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_0 = stablehlo.shift_right_logical %static_arg, %dynamic_arg : (tensor<2xi64>, tensor<?xi64>) -> tensor<2xi64>
  %not_speculatable_1 = stablehlo.shift_right_logical %static_arg, %dynamic_arg : (tensor<2xi64>, tensor<?xi64>) -> tensor<?xi64>
  %not_speculatable_2 = stablehlo.shift_right_logical %dynamic_arg, %static_arg : (tensor<?xi64>, tensor<2xi64>) -> tensor<2xi64>
  %not_speculatable_3 = stablehlo.shift_right_logical %dynamic_arg, %static_arg : (tensor<?xi64>, tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_4 = stablehlo.shift_right_logical %dynamic_arg, %dynamic_arg : (tensor<?xi64>, tensor<?xi64>) -> tensor<2xi64>
  %not_speculatable_5 = stablehlo.shift_right_logical %dynamic_arg, %dynamic_arg : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @subtract
// CHECK-NEXT: return
func.func @subtract(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.subtract %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.subtract %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.subtract %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_1 = stablehlo.subtract %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<?xf64>
  %not_speculatable_2 = stablehlo.subtract %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<2xf64>
  %not_speculatable_3 = stablehlo.subtract %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_4 = stablehlo.subtract %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<2xf64>
  %not_speculatable_5 = stablehlo.subtract %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @and
// CHECK-NEXT: return
func.func @and(%static_arg: tensor<2xi64>, %dynamic_arg: tensor<?xi64>) {
  %speculatable_0 = stablehlo.and %static_arg, %static_arg : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
  %speculatable_1 = stablehlo.and %static_arg, %static_arg : (tensor<2xi64>, tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_0 = stablehlo.and %static_arg, %dynamic_arg : (tensor<2xi64>, tensor<?xi64>) -> tensor<2xi64>
  %not_speculatable_1 = stablehlo.and %static_arg, %dynamic_arg : (tensor<2xi64>, tensor<?xi64>) -> tensor<?xi64>
  %not_speculatable_2 = stablehlo.and %dynamic_arg, %static_arg : (tensor<?xi64>, tensor<2xi64>) -> tensor<2xi64>
  %not_speculatable_3 = stablehlo.and %dynamic_arg, %static_arg : (tensor<?xi64>, tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_4 = stablehlo.and %dynamic_arg, %dynamic_arg : (tensor<?xi64>, tensor<?xi64>) -> tensor<2xi64>
  %not_speculatable_5 = stablehlo.and %dynamic_arg, %dynamic_arg : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @or
// CHECK-NEXT: return
func.func @or(%static_arg: tensor<2xi64>, %dynamic_arg: tensor<?xi64>) {
  %speculatable_0 = stablehlo.or %static_arg, %static_arg : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
  %speculatable_1 = stablehlo.or %static_arg, %static_arg : (tensor<2xi64>, tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_0 = stablehlo.or %static_arg, %dynamic_arg : (tensor<2xi64>, tensor<?xi64>) -> tensor<2xi64>
  %not_speculatable_1 = stablehlo.or %static_arg, %dynamic_arg : (tensor<2xi64>, tensor<?xi64>) -> tensor<?xi64>
  %not_speculatable_2 = stablehlo.or %dynamic_arg, %static_arg : (tensor<?xi64>, tensor<2xi64>) -> tensor<2xi64>
  %not_speculatable_3 = stablehlo.or %dynamic_arg, %static_arg : (tensor<?xi64>, tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_4 = stablehlo.or %dynamic_arg, %dynamic_arg : (tensor<?xi64>, tensor<?xi64>) -> tensor<2xi64>
  %not_speculatable_5 = stablehlo.or %dynamic_arg, %dynamic_arg : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @xor
// CHECK-NEXT: return
func.func @xor(%static_arg: tensor<2xi64>, %dynamic_arg: tensor<?xi64>) {
  %speculatable_0 = stablehlo.xor %static_arg, %static_arg : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
  %speculatable_1 = stablehlo.xor %static_arg, %static_arg : (tensor<2xi64>, tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_0 = stablehlo.xor %static_arg, %dynamic_arg : (tensor<2xi64>, tensor<?xi64>) -> tensor<2xi64>
  %not_speculatable_1 = stablehlo.xor %static_arg, %dynamic_arg : (tensor<2xi64>, tensor<?xi64>) -> tensor<?xi64>
  %not_speculatable_2 = stablehlo.xor %dynamic_arg, %static_arg : (tensor<?xi64>, tensor<2xi64>) -> tensor<2xi64>
  %not_speculatable_3 = stablehlo.xor %dynamic_arg, %static_arg : (tensor<?xi64>, tensor<2xi64>) -> tensor<?xi64>
  %not_speculatable_4 = stablehlo.xor %dynamic_arg, %dynamic_arg : (tensor<?xi64>, tensor<?xi64>) -> tensor<2xi64>
  %not_speculatable_5 = stablehlo.xor %dynamic_arg, %dynamic_arg : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?xi64>) -> ()
  return
}

// -----

// Other ops that take 2 or more operands

// -----

// CHECK-LABEL: func @clamp
// CHECK-NEXT: return
func.func @clamp(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.clamp %static_arg, %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.clamp %dynamic_arg, %static_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_1 = stablehlo.clamp %static_arg, %dynamic_arg, %static_arg : (tensor<2xf64>, tensor<?xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_2 = stablehlo.clamp %static_arg, %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<2xf64>, tensor<?xf64>) -> tensor<?xf64>
  %not_speculatable_3 = stablehlo.clamp %dynamic_arg, %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @compare
// CHECK-NEXT: return
func.func @compare(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.compare EQ, %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<?xi1>
  %not_speculatable_0 = stablehlo.compare EQ, %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<?xi1>
  %not_speculatable_1 = stablehlo.compare EQ, %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<?xi1>
  %not_speculatable_2 = stablehlo.compare EQ, %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<?xi1>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<?xi1>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xi1>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<?xi1>) -> ()
  return
}

// -----

// CHECK-LABEL: func @complex
// CHECK-NEXT: return
func.func @complex(%static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.complex %static_arg, %static_arg : (tensor<2xf64>, tensor<2xf64>) -> tensor<?xcomplex<f64>>
  %not_speculatable_0 = stablehlo.complex %dynamic_arg, %static_arg : (tensor<?xf64>, tensor<2xf64>) -> tensor<?xcomplex<f64>>
  %not_speculatable_1 = stablehlo.complex %static_arg, %dynamic_arg : (tensor<2xf64>, tensor<?xf64>) -> tensor<?xcomplex<f64>>
  %not_speculatable_2 = stablehlo.complex %dynamic_arg, %dynamic_arg : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xcomplex<f64>>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<?xcomplex<f64>>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<?xcomplex<f64>>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xcomplex<f64>>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<?xcomplex<f64>>) -> ()
  return
}

// -----

// CHECK-LABEL: func @select
// CHECK-NEXT: return
func.func @select(%static_pred: tensor<2xi1>, %dynamic_pred: tensor<?xi1>, %static_arg: tensor<2xf64>, %dynamic_arg: tensor<?xf64>) {
  %speculatable_0 = stablehlo.select %static_pred, %static_arg, %static_arg : (tensor<2xi1>, tensor<2xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.select %dynamic_pred, %static_arg, %static_arg : (tensor<?xi1>, tensor<2xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_1 = stablehlo.select %static_pred, %dynamic_arg, %static_arg : (tensor<2xi1>, tensor<?xf64>, tensor<2xf64>) -> tensor<?xf64>
  %not_speculatable_2 = stablehlo.select %static_pred, %static_arg, %dynamic_arg : (tensor<2xi1>, tensor<2xf64>, tensor<?xf64>) -> tensor<?xf64>
  %not_speculatable_3 = stablehlo.select %dynamic_pred, %dynamic_arg, %dynamic_arg : (tensor<?xi1>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?xf64>) -> ()
  return
}

// -----

// Other ops that take 2 or more operands and have additional conditions

// -----

// CHECK-LABEL: func @concatenate
// CHECK-NEXT: return
func.func @concatenate(%static_arg: tensor<2x2xi64>, %first_dim_dynamic: tensor<?x2xi64>, %second_dim_dynamic: tensor<2x?xi64>, %dynamic_arg: tensor<?x?xi64>) {
  %speculatable_0 = stablehlo.concatenate %static_arg, %static_arg, dim = 0 : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<?x?xi64>
  %speculatable_1 = stablehlo.concatenate %static_arg, %static_arg, dim = 1 : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<?x?xi64>
  %speculatable_2 = stablehlo.concatenate %static_arg, %first_dim_dynamic, dim = 0 : (tensor<2x2xi64>, tensor<?x2xi64>) -> tensor<?x?xi64>
  %speculatable_3 = stablehlo.concatenate %second_dim_dynamic, %static_arg, dim = 1 : (tensor<2x?xi64>, tensor<2x2xi64>) -> tensor<?x?xi64>

  %speculatable_4 = stablehlo.concatenate %static_arg, %static_arg, dim = 0 : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<4x2xi64>
  %speculatable_5 = stablehlo.concatenate %static_arg, %static_arg, dim = 1 : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x4xi64>
  %not_speculatable_0 = stablehlo.concatenate %static_arg, %first_dim_dynamic, dim = 0 : (tensor<2x2xi64>, tensor<?x2xi64>) -> tensor<4x2xi64>
  %not_speculatable_1 = stablehlo.concatenate %second_dim_dynamic, %static_arg, dim = 1 : (tensor<2x?xi64>, tensor<2x2xi64>) -> tensor<2x4xi64>

  %speculatable_6 = stablehlo.concatenate %first_dim_dynamic, %first_dim_dynamic, dim = 0 : (tensor<?x2xi64>, tensor<?x2xi64>) -> tensor<?x?xi64>
  %not_speculatable_2 = stablehlo.concatenate %first_dim_dynamic, %first_dim_dynamic, dim = 0 : (tensor<?x2xi64>, tensor<?x2xi64>) -> tensor<4x?xi64>
  %not_speculatable_3 = stablehlo.concatenate %first_dim_dynamic, %first_dim_dynamic, dim = 1 : (tensor<?x2xi64>, tensor<?x2xi64>) -> tensor<?x?xi64>

  %not_speculatable_4 = stablehlo.concatenate %second_dim_dynamic, %second_dim_dynamic, dim = 0 : (tensor<2x?xi64>, tensor<2x?xi64>) -> tensor<?x?xi64>
  %not_speculatable_5 = stablehlo.concatenate %second_dim_dynamic, %second_dim_dynamic, dim = 1 : (tensor<2x?xi64>, tensor<2x?xi64>) -> tensor<?x4xi64>
  %speculatable_7 = stablehlo.concatenate %second_dim_dynamic, %second_dim_dynamic, dim = 1 : (tensor<2x?xi64>, tensor<2x?xi64>) -> tensor<?x?xi64>

  %not_speculatable_6 = stablehlo.concatenate %dynamic_arg, %dynamic_arg, dim = 0 : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
  %not_speculatable_7 = stablehlo.concatenate %dynamic_arg, %dynamic_arg, dim = 1 : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>

  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<?x?xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?x?xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?x?xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_3) : (tensor<?x?xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_4) : (tensor<4x2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_5) : (tensor<2x4xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_6) : (tensor<?x?xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_7) : (tensor<?x?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<4x2xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<2x4xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<4x?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?x?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<?x?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_5) : (tensor<?x4xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_6) : (tensor<?x?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_7) : (tensor<?x?xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @gather
// CHECK-NEXT: return
func.func @gather(
  %static_arg: tensor<3x4x2xi32>, %static_indices: tensor<2x3x2xi64>,
  %dynamic_arg: tensor<?x?x?xi32>, %dynamic_indices: tensor<?x?x?xi64>
) {
  %speculatable_0 = "stablehlo.gather"(%static_arg, %static_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>) -> tensor<?x?x?x?xi32>
  %not_speculatable_0 = "stablehlo.gather"(%static_arg, %static_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2>,
    indices_are_sorted = true
  } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>) -> tensor<?x?x?x?xi32>
  %not_speculatable_1 = "stablehlo.gather"(%dynamic_arg, %static_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2>,
    indices_are_sorted = true
  } : (tensor<?x?x?xi32>, tensor<2x3x2xi64>) -> tensor<?x?x?x?xi32>
  %not_speculatable_2 = "stablehlo.gather"(%static_arg, %dynamic_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2>,
    indices_are_sorted = true
  } : (tensor<3x4x2xi32>, tensor<?x?x?xi64>) -> tensor<?x?x?x?xi32>
  %not_speculatable_3 = "stablehlo.gather"(%dynamic_arg, %dynamic_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2>,
    indices_are_sorted = true
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi64>) -> tensor<?x?x?x?xi32>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<?x?x?x?xi32>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<?x?x?x?xi32>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?x?x?x?xi32>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<?x?x?x?xi32>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?x?x?x?xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @scatter
// CHECK-NEXT: return
func.func @scatter(
  %static_inputs: tensor<3x4x2xf64>, %static_indices: tensor<2x3x2xi64>, %static_updates: tensor<2x3x2x2xf64>,
  %dynamic_inputs: tensor<?x?x?xf64>, %dynamic_indices: tensor<?x?x?xi64>, %dynamic_updates: tensor<?x?x?x?xf64>
) {
  %recursively_speculatable_0 = "stablehlo.scatter"(%static_inputs, %static_indices, %static_updates) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg0 : tensor<f64>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [1, 0],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<3x4x2xf64>, tensor<2x3x2xi64>, tensor<2x3x2x2xf64>) -> tensor<?x?x?xf64>
  %not_speculatable_0 = "stablehlo.scatter"(%static_inputs, %static_indices, %static_updates) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg0 : tensor<f64>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [1, 0],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    unique_indices = true
  } : (tensor<3x4x2xf64>, tensor<2x3x2xi64>, tensor<2x3x2x2xf64>) -> tensor<?x?x?xf64>
  %not_speculatable_1 = "stablehlo.scatter"(%static_inputs, %static_indices, %static_updates) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg0 : tensor<f64>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [1, 0],
      index_vector_dim = 2>,
    indices_are_sorted = true,
    unique_indices = false
  } : (tensor<3x4x2xf64>, tensor<2x3x2xi64>, tensor<2x3x2x2xf64>) -> tensor<?x?x?xf64>
  %not_speculatable_2 = "stablehlo.scatter"(%dynamic_inputs, %static_indices, %static_updates) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg0 : tensor<f64>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [1, 0],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<?x?x?xf64>, tensor<2x3x2xi64>, tensor<2x3x2x2xf64>) -> tensor<?x?x?xf64>
  %not_speculatable_3 = "stablehlo.scatter"(%static_inputs, %dynamic_indices, %static_updates) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg0 : tensor<f64>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [1, 0],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<3x4x2xf64>, tensor<?x?x?xi64>, tensor<2x3x2x2xf64>) -> tensor<?x?x?xf64>
  %not_speculatable_4 = "stablehlo.scatter"(%static_inputs, %static_indices, %dynamic_updates) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg0 : tensor<f64>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [1, 0],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<3x4x2xf64>, tensor<2x3x2xi64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?xf64>
  "hlo_test_speculatability.is_recursively_speculatable"(%recursively_speculatable_0) : (tensor<?x?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<?x?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?x?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<?x?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_3) : (tensor<?x?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_4) : (tensor<?x?x?xf64>) -> ()
  return
}

// -----

// Ops that take an output shape as operand

// -----

// CHECK-LABEL: func @dynamic_iota
// CHECK-NEXT:  return
func.func @dynamic_iota(%unknown_shape: tensor<2xi32>) {
  %constant_shape = stablehlo.constant dense<[3, 4]> : tensor<2xi32>
  %speculatable_0 = stablehlo.dynamic_iota %constant_shape, dim = 0 : (tensor<2xi32>) -> tensor<3x4xi64>
  %speculatable_1 = stablehlo.dynamic_iota %constant_shape, dim = 0 : (tensor<2xi32>) -> tensor<3x?xi64>
  %speculatable_2 = stablehlo.dynamic_iota %constant_shape, dim = 0 : (tensor<2xi32>) -> tensor<?x4xi64>
  %speculatable_3 = stablehlo.dynamic_iota %constant_shape, dim = 0 : (tensor<2xi32>) -> tensor<?x?xi64>
  %not_speculatable_0 = stablehlo.dynamic_iota %unknown_shape, dim = 0 : (tensor<2xi32>) -> tensor<3x4xi64>
  %not_speculatable_1 = stablehlo.dynamic_iota %unknown_shape, dim = 0 : (tensor<2xi32>) -> tensor<3x?xi64>
  %not_speculatable_2 = stablehlo.dynamic_iota %unknown_shape, dim = 0 : (tensor<2xi32>) -> tensor<?x4xi64>
  %speculatable_4 = stablehlo.dynamic_iota %unknown_shape, dim = 0 : (tensor<2xi32>) -> tensor<?x?xi64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<3x4xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<3x?xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?x4xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_3) : (tensor<?x?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<3x4xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<3x?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<?x4xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_4) : (tensor<?x?xi64>) -> ()
  return
}

// -----

// Recursively speculatable ops

// -----

// CHECK-LABEL: func @map
// CHECK-NEXT: return
func.func @map(%static_arg: tensor<2x4xf64>, %dynamic_arg: tensor<?x?xf64>, %arg: tensor<f64>) {
  %recursively_speculatable_0 = "stablehlo.map"(%static_arg, %static_arg) ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg : tensor<f64>
  }) {dimensions = array<i64: 0, 1>} : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<?x?xf64>
  %not_speculatable_0 = "stablehlo.map"(%static_arg, %dynamic_arg) ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg : tensor<f64>
  }) {dimensions = array<i64: 0, 1>} : (tensor<2x4xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
  %not_speculatable_1 = "stablehlo.map"(%dynamic_arg, %static_arg) ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg : tensor<f64>
  }) {dimensions = array<i64: 0, 1>} : (tensor<?x?xf64>, tensor<2x4xf64>) -> tensor<?x?xf64>
  %not_speculatable_2 = "stablehlo.map"(%dynamic_arg, %dynamic_arg) ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg : tensor<f64>
  }) {dimensions = array<i64: 0, 1>} : (tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
  "hlo_test_speculatability.is_recursively_speculatable"(%recursively_speculatable_0) : (tensor<?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<?x?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @reduce
// CHECK-NEXT: return
func.func @reduce(%static_arg: tensor<2x4xf64>, %dynamic_arg: tensor<?x?xf64>, %init_arg: tensor<f64>) {
  %recursively_speculatable_0:2 = "stablehlo.reduce"(%static_arg, %static_arg, %init_arg, %init_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg0, %arg0 : tensor<f64>, tensor<f64>
  }) {dimensions = array<i64: 0>} : (tensor<2x4xf64>, tensor<2x4xf64>, tensor<f64>, tensor<f64>) -> (tensor<?xf64>, tensor<?xf64>)
  %not_speculatable_0:2 = "stablehlo.reduce"(%dynamic_arg, %static_arg, %init_arg, %init_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg0, %arg0 : tensor<f64>, tensor<f64>
  }) {dimensions = array<i64: 0>} : (tensor<?x?xf64>, tensor<2x4xf64>, tensor<f64>, tensor<f64>) -> (tensor<?xf64>, tensor<?xf64>)
  %not_speculatable_1:2 = "stablehlo.reduce"(%static_arg, %dynamic_arg, %init_arg, %init_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg0, %arg0 : tensor<f64>, tensor<f64>
  }) {dimensions = array<i64: 0>} : (tensor<2x4xf64>, tensor<?x?xf64>, tensor<f64>, tensor<f64>) -> (tensor<?xf64>, tensor<?xf64>)
  %not_speculatable_2:2 = "stablehlo.reduce"(%dynamic_arg, %dynamic_arg, %init_arg, %init_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg0, %arg0 : tensor<f64>, tensor<f64>
  }) {dimensions = array<i64: 0>} : (tensor<?x?xf64>, tensor<?x?xf64>, tensor<f64>, tensor<f64>) -> (tensor<?xf64>, tensor<?xf64>)
  "hlo_test_speculatability.is_recursively_speculatable"(%recursively_speculatable_0) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @reduce_window
// CHECK-NEXT: return
func.func @reduce_window(%static_arg: tensor<2x4xf64>, %dynamic_arg: tensor<?x?xf64>, %init_arg: tensor<f64>) {
  %recursively_speculatable_0:2 = "stablehlo.reduce_window"(%static_arg, %static_arg, %init_arg, %init_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg0, %arg0 : tensor<f64>, tensor<f64>
    }) {
    window_dimensions = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>,
    base_dilations = array<i64: 1, 1>,
    window_dilations = array<i64: 1, 1>,
    padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>
  } : (tensor<2x4xf64>,  tensor<2x4xf64>, tensor<f64>, tensor<f64>) -> (tensor<?x?xf64>, tensor<?x?xf64>)
  %not_speculatable_0:2 = "stablehlo.reduce_window"(%dynamic_arg, %static_arg, %init_arg, %init_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg0, %arg0 : tensor<f64>, tensor<f64>
    }) {
    window_dimensions = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>,
    base_dilations = array<i64: 1, 1>,
    window_dilations = array<i64: 1, 1>,
    padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>
  } : (tensor<?x?xf64>,  tensor<2x4xf64>, tensor<f64>, tensor<f64>) -> (tensor<?x?xf64>, tensor<?x?xf64>)
  %not_speculatable_1:2 = "stablehlo.reduce_window"(%static_arg, %dynamic_arg, %init_arg, %init_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg0, %arg0 : tensor<f64>, tensor<f64>
    }) {
    window_dimensions = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>,
    base_dilations = array<i64: 1, 1>,
    window_dilations = array<i64: 1, 1>,
    padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>
  } : (tensor<2x4xf64>,  tensor<?x?xf64>, tensor<f64>, tensor<f64>) -> (tensor<?x?xf64>, tensor<?x?xf64>)
  %not_speculatable_2:2 = "stablehlo.reduce_window"(%dynamic_arg, %dynamic_arg, %init_arg, %init_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg0, %arg0 : tensor<f64>, tensor<f64>
    }) {
    window_dimensions = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>,
    base_dilations = array<i64: 1, 1>,
    window_dilations = array<i64: 1, 1>,
    padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>
  } : (tensor<?x?xf64>,  tensor<?x?xf64>, tensor<f64>, tensor<f64>) -> (tensor<?x?xf64>, tensor<?x?xf64>)
  "hlo_test_speculatability.is_recursively_speculatable"(%recursively_speculatable_0) : (tensor<?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<?x?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @sort
// CHECK-NEXT: return
func.func @sort(%static_arg: tensor<2x4xf64>, %dynamic_arg: tensor<?x?xf64>) {
  %recursively_speculatable_0:2 = "stablehlo.sort"(%static_arg, %static_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>):
      %predicate = "stablehlo.compare"(%arg0, %arg1) {
        comparison_direction = #stablehlo<comparison_direction GT>
      } : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %predicate : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<2x4xf64>, tensor<2x4xf64>) -> (tensor<?x?xf64>, tensor<?x?xf64>)
  %not_recursively_speculatable_0:2 = "stablehlo.sort"(%dynamic_arg, %static_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>):
      %predicate = "stablehlo.compare"(%arg0, %arg1) {
        comparison_direction = #stablehlo<comparison_direction GT>
      } : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %predicate : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<?x?xf64>, tensor<2x4xf64>) -> (tensor<?x?xf64>, tensor<?x?xf64>)
  %not_recursively_speculatable_1:2 = "stablehlo.sort"(%static_arg, %dynamic_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>):
      %predicate = "stablehlo.compare"(%arg0, %arg1) {
        comparison_direction = #stablehlo<comparison_direction GT>
      } : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %predicate : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<2x4xf64>, tensor<?x?xf64>) -> (tensor<?x?xf64>, tensor<?x?xf64>)
  %not_recursively_speculatable_2:2 = "stablehlo.sort"(%dynamic_arg, %dynamic_arg) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>):
      %predicate = "stablehlo.compare"(%arg0, %arg1) {
        comparison_direction = #stablehlo<comparison_direction GT>
      } : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %predicate : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<?x?xf64>, tensor<?x?xf64>) -> (tensor<?x?xf64>, tensor<?x?xf64>)
  "hlo_test_speculatability.is_recursively_speculatable"(%recursively_speculatable_0) : (tensor<?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_recursively_speculatable_0) : (tensor<?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_recursively_speculatable_1) : (tensor<?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_recursively_speculatable_2) : (tensor<?x?xf64>) -> ()
  return
}

// -----

// Miscellaneous ops

// -----

// CHECK-LABEL: func @bitcast_convert
// CHECK-NEXT:  return
func.func @bitcast_convert(
  %static_arg_64: tensor<2xi64>, %dynamic_arg_64: tensor<?xi64>,
  %static_arg_32: tensor<2x?xi32>, %dynamic_arg_32: tensor<?x?xi32>
) {
  %speculatable_0 = stablehlo.bitcast_convert %static_arg_64 : (tensor<2xi64>) -> tensor<2xf64>
  %speculatable_1 = stablehlo.bitcast_convert %static_arg_64 : (tensor<2xi64>) -> tensor<?xf64>
  %not_speculatable_0 = stablehlo.bitcast_convert %dynamic_arg_64 : (tensor<?xi64>) -> tensor<2xf64>
  %speculatable_2 = stablehlo.bitcast_convert %dynamic_arg_64 : (tensor<?xi64>) -> tensor<?xf64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_1) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<2xf64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_2) : (tensor<?xf64>) -> ()

  %speculatable_3 = stablehlo.bitcast_convert %static_arg_64 : (tensor<2xi64>) -> tensor<2x?xi32>
  %speculatable_4 = stablehlo.bitcast_convert %static_arg_64 : (tensor<2xi64>) -> tensor<?x?xi32>
  %not_speculatable_1 = stablehlo.bitcast_convert %dynamic_arg_64 : (tensor<?xi64>) -> tensor<2x?xi32>
  %speculatable_5 = stablehlo.bitcast_convert %dynamic_arg_64 : (tensor<?xi64>) -> tensor<?x?xi32>
  "hlo_test_speculatability.is_speculatable"(%speculatable_3) : (tensor<2x?xi32>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_4) : (tensor<?x?xi32>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<2x?xi32>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_5) : (tensor<?x?xi32>) -> ()

  %speculatable_6 = stablehlo.bitcast_convert %static_arg_32 : (tensor<2x?xi32>) -> tensor<2xi64>
  %speculatable_7 = stablehlo.bitcast_convert %static_arg_32 : (tensor<2x?xi32>) -> tensor<?xi64>
  %not_speculatable_2 = stablehlo.bitcast_convert %dynamic_arg_32 : (tensor<?x?xi32>) -> tensor<2xi64>
  %speculatable_8 = stablehlo.bitcast_convert %dynamic_arg_32 : (tensor<?x?xi32>) -> tensor<?xi64>
  "hlo_test_speculatability.is_speculatable"(%speculatable_6) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_7) : (tensor<?xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<2xi64>) -> ()
  "hlo_test_speculatability.is_speculatable"(%speculatable_8) : (tensor<?xi64>) -> ()

  %speculatable_9 = stablehlo.bitcast_convert %static_arg_64 : (tensor<2xi64>) -> tensor<2x2xi32>
  "hlo_test_speculatability.is_speculatable"(%speculatable_9) : (tensor<2x2xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @reshape
// CHECK-NEXT:  return
func.func @reshape(
  %static: tensor<3x4xi64>,
  %dynamic_0: tensor<?x4xi64>,
  %dynamic_1: tensor<3x?xi64>,
  %dynamic_2: tensor<?x?xi64>
) {
  %speculatable = stablehlo.reshape %static : (tensor<3x4xi64>) -> tensor<12xi64>
  %not_speculatable_0 = stablehlo.reshape %dynamic_0 : (tensor<?x4xi64>) -> tensor<12xi64>
  %not_speculatable_1 = stablehlo.reshape %dynamic_1 : (tensor<3x?xi64>) -> tensor<12xi64>
  %not_speculatable_2 = stablehlo.reshape %dynamic_2 : (tensor<?x?xi64>) -> tensor<12xi64>
  "hlo_test_speculatability.is_speculatable"(%speculatable) : (tensor<12xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_0) : (tensor<12xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_1) : (tensor<12xi64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%not_speculatable_2) : (tensor<12xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @triangular_solve
// CHECK-NEXT:  return
func.func @triangular_solve(
  %static: tensor<2x2xf64>,
  %dynamic: tensor<?x?xf64>
) {
  %static_inputs = "stablehlo.triangular_solve"(%static, %static) {
    left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false
  } : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<?x?xf64>
  %dynamic_inputs_0 = "stablehlo.triangular_solve"(%dynamic, %static) {
    left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false
  } : (tensor<?x?xf64>, tensor<2x2xf64>) -> tensor<?x?xf64>
  %dynamic_inputs_1 = "stablehlo.triangular_solve"(%static, %dynamic) {
    left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false
  } : (tensor<2x2xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
  %dynamic_inputs_2 = "stablehlo.triangular_solve"(%dynamic, %dynamic) {
    left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false
  } : (tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
  %unit_diagonal = "stablehlo.triangular_solve"(%static, %static) {
    left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true
  } : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<?x?xf64>
  "hlo_test_speculatability.is_speculatable"(%static_inputs) : (tensor<?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%dynamic_inputs_0) : (tensor<?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%dynamic_inputs_1) : (tensor<?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%dynamic_inputs_2) : (tensor<?x?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%unit_diagonal) : (tensor<?x?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @batch_norm_grad
// CHECK-NEXT:  return
func.func @batch_norm_grad(%static: tensor<2xf64>, %dynamic: tensor<?xf64>) {
  %all_inputs_static:3 = "stablehlo.batch_norm_grad" (%static, %static, %static, %static, %static) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
  %first_input_dynamic:3 = "stablehlo.batch_norm_grad" (%dynamic, %static, %static, %static, %static) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<?xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
  %last_input_dynamic:3 = "stablehlo.batch_norm_grad" (%static, %static, %static, %static, %dynamic) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<?xf64>) -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
  %all_inputs_dynamic:3 = "stablehlo.batch_norm_grad" (%dynamic, %dynamic, %dynamic, %dynamic, %dynamic) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
  "hlo_test_speculatability.is_speculatable"(%all_inputs_static#0) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%first_input_dynamic#0) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%last_input_dynamic#0) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%all_inputs_dynamic#0) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @batch_norm_inference
// CHECK-NEXT:  return
func.func @batch_norm_inference(%static: tensor<2xf64>, %dynamic: tensor<?xf64>) {
  %all_inputs_static = "stablehlo.batch_norm_inference" (%static, %static, %static, %static, %static) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> (tensor<?xf64>)
  %first_input_dynamic = "stablehlo.batch_norm_inference" (%dynamic, %static, %static, %static, %static) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<?xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> (tensor<?xf64>)
  %last_input_dynamic = "stablehlo.batch_norm_inference" (%static, %static, %static, %static, %dynamic) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<?xf64>) -> (tensor<?xf64>)
  %all_inputs_dynamic = "stablehlo.batch_norm_inference" (%dynamic, %dynamic, %dynamic, %dynamic, %dynamic) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) -> (tensor<?xf64>)
  "hlo_test_speculatability.is_speculatable"(%all_inputs_static) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%first_input_dynamic) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%last_input_dynamic) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%all_inputs_dynamic) : (tensor<?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @batch_norm_training
// CHECK-NEXT:  return
func.func @batch_norm_training(%static: tensor<2xf64>, %dynamic: tensor<?xf64>) {
  %all_inputs_static:3 = "stablehlo.batch_norm_training" (%static, %static, %static) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
  %first_input_dynamic:3 = "stablehlo.batch_norm_training" (%dynamic, %static, %static) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<?xf64>, tensor<2xf64>, tensor<2xf64>) -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
  %last_input_dynamic:3 = "stablehlo.batch_norm_training" (%static, %static, %dynamic) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2xf64>, tensor<2xf64>, tensor<?xf64>) -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
  %all_inputs_dynamic:3 = "stablehlo.batch_norm_training" (%dynamic, %dynamic, %dynamic) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
  "hlo_test_speculatability.is_speculatable"(%all_inputs_static#0) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%first_input_dynamic#0) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%last_input_dynamic#0) : (tensor<?xf64>) -> ()
  "hlo_test_speculatability.is_not_speculatable"(%all_inputs_dynamic#0) : (tensor<?xf64>) -> ()
  return
}
