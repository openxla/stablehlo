// RUN: stablehlo-opt --stablehlo-refine-arguments='types=tensor<f32>,tensor<1xf32>,tensor<?xf32>,tensor<1x?x?xf32>,tensor<*xf32>,!stablehlo.token' --split-input-file --verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK,NO-REFINE
// RUN: stablehlo-opt --stablehlo-refine-arguments='types=tensor<f32>,tensor<1xf32>,tensor<2xf32>,tensor<1x4x6xf32>,tensor<8x10xf32>,!stablehlo.token' --split-input-file --verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK,FULL-REFINE
// RUN: stablehlo-opt --stablehlo-refine-arguments='types=tensor<f32>,tensor<1xf32>,tensor<?xf32>,tensor<1x4x?xf32>,tensor<8x?xf32>,!stablehlo.token' --split-input-file --verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK,PARTIAL-REFINE

// RUN: not stablehlo-opt --stablehlo-refine-arguments='types=!not.a_type' %s 2>&1 | FileCheck %s --check-prefixes=UNKNOWN-TYPE
// UNKNOWN-TYPE: Invalid type string: !not.a_type

// RUN: not stablehlo-opt --stablehlo-refine-arguments='types=tensor<f32>,tensor<1xf32>,tensor<?xf32>,tensor<*xf32>,tensor<*xf32>,!stablehlo.token' %s 2>&1 | FileCheck %s --check-prefixes=UNRANKED-ERROR
func.func @main(%arg0: tensor<f32>, %arg1: tensor<1xf32>, %arg2: tensor<?xf32>, %arg3: tensor<1x?x?xf32>, %arg4: tensor<*xf32>, %arg5: !stablehlo.token) {
  // UNRANKED-ERROR: invalid refinement for argument 3, refinement must be ranked in 'tensor<1x?x?xf32>'->'tensor<*xf32>'
  return
}

// -----

// CHECK-LABEL: refine_params
func.func @refine_params(%arg0: tensor<f32>, %arg1: tensor<1xf32>, %arg2: tensor<?xf32>, %arg3: tensor<1x?x?xf32>, %arg4: tensor<*xf32>, %arg5: !stablehlo.token)
    -> (tensor<f32>, tensor<1xf32>, tensor<?xf32>, tensor<1x?x?xf32>, tensor<*xf32>, !stablehlo.token) {
  // NO-REFINE-NOT: custom_call
  // NO-REFINE: %[[ARG0:.+]]: tensor<f32>, %[[ARG1:.+]]: tensor<1xf32>, %[[ARG2:.+]]: tensor<?xf32>, %[[ARG3:.+]]: tensor<1x?x?xf32>, %[[ARG4:.+]]: tensor<*xf32>, %[[ARG5:.+]]: !stablehlo.token
  // NO-REFINE: return %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]] : tensor<f32>, tensor<1xf32>, tensor<?xf32>, tensor<1x?x?xf32>, tensor<*xf32>, !stablehlo.token


  // FULL-REFINE: %[[ARG0:.+]]: tensor<f32>, %[[ARG1:.+]]: tensor<1xf32>, %[[ARG2:.+]]: tensor<2xf32>, %[[ARG3:.+]]: tensor<1x4x6xf32>, %[[ARG4:.+]]: tensor<8x10xf32>, %[[ARG5:.+]]: !stablehlo.token
  // FULL-REFINE: %[[ARG2_CC:.+]] = stablehlo.custom_call @stablehlo.shape_refinement_operand_wrapper(%[[ARG2]], {{.*}}) {indices_of_shape_operands = dense<1> : tensor<1xi64>} : (tensor<2xf32>, tensor<1xi64>) -> tensor<?xf32>
  // FULL-REFINE: %[[ARG3_CC:.+]] = stablehlo.custom_call @stablehlo.shape_refinement_operand_wrapper(%[[ARG3]], {{.*}}) {indices_of_shape_operands = dense<1> : tensor<1xi64>} : (tensor<1x4x6xf32>, tensor<3xi64>) -> tensor<1x?x?xf32>
  // FULL-REFINE: %[[ARG4_CC:.+]] = stablehlo.custom_call @stablehlo.shape_refinement_operand_wrapper(%[[ARG4]], {{.*}}) {indices_of_shape_operands = dense<1> : tensor<1xi64>} : (tensor<8x10xf32>, tensor<2xi64>) -> tensor<*xf32>
  // FULL-REFINE: return %[[ARG0]], %[[ARG1]], %[[ARG2_CC]], %[[ARG3_CC]], %[[ARG4_CC]], %[[ARG5]] : tensor<f32>, tensor<1xf32>, tensor<?xf32>, tensor<1x?x?xf32>, tensor<*xf32>, !stablehlo.token

  // PARTIAL-REFINE: %[[ARG0:.+]]: tensor<f32>, %[[ARG1:.+]]: tensor<1xf32>, %[[ARG2:.+]]: tensor<?xf32>, %[[ARG3:.+]]: tensor<1x4x?xf32>, %[[ARG4:.+]]: tensor<8x?xf32>, %[[ARG5:.+]]: !stablehlo.token
  // PARTIAL-REFINE: %[[ARG3_CC:.+]] = stablehlo.custom_call @stablehlo.shape_refinement_operand_wrapper(%[[ARG3]], {{.*}}) {indices_of_shape_operands = dense<1> : tensor<1xi64>} : (tensor<1x4x?xf32>, tensor<3xi64>) -> tensor<1x?x?xf32>
  // PARTIAL-REFINE: %[[ARG4_CC:.+]] = stablehlo.custom_call @stablehlo.shape_refinement_operand_wrapper(%[[ARG4]], {{.*}}) {indices_of_shape_operands = dense<1> : tensor<1xi64>} : (tensor<8x?xf32>, tensor<2xi64>) -> tensor<*xf32>
  // PARTIAL-REFINE: return %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3_CC]], %[[ARG4_CC]], %[[ARG5]] : tensor<f32>, tensor<1xf32>, tensor<?xf32>, tensor<1x?x?xf32>, tensor<*xf32>, !stablehlo.token
  return %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : tensor<f32>, tensor<1xf32>, tensor<?xf32>, tensor<1x?x?xf32>, tensor<*xf32>, !stablehlo.token
}

// -----

// expected-error @+1 {{number of refinements must match number of function operands 6 vs 1}}
func.func @refine_arguments_invalid_arg_num_mismatch(%arg0: tensor<f32>) {
  return
}

// -----

// expected-error @+1 {{invalid refinement for argument 5, refinement must be a tensor in 'tensor<f32>'->'!stablehlo.token'}}
func.func @refine_arguments_invalid_type_mismatch(%arg0: tensor<f32>, %arg1: tensor<1xf32>, %arg2: tensor<?xf32>, %arg3: tensor<1x?x?xf32>, %arg4: tensor<*xf32>, %arg5: tensor<f32>) {
  return
}

// -----

// expected-error @+1 {{invalid refinement for argument 1, refinement rank must match operand rank in 'tensor<f32>'->'tensor<1xf32>'}}
func.func @refine_arguments_invalid_refine_rank_mismatch(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<?xf32>, %arg3: tensor<1x?x?xf32>, %arg4: tensor<*xf32>, %arg5: !stablehlo.token) {
  return
}

// -----

// expected-error @+1 {{invalid refinement for argument 1, refinement dimension sizes must match for static dimensions in 'tensor<2xf32>'->'tensor<1xf32>'}}
func.func @refine_arguments_invalid_static_dim_mismatch(%arg0: tensor<f32>, %arg1: tensor<2xf32>, %arg2: tensor<?xf32>, %arg3: tensor<1x?x?xf32>, %arg4: tensor<*xf32>, %arg5: !stablehlo.token) {
  return
}
