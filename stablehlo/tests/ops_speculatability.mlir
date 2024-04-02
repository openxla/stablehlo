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
