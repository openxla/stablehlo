// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file  | FileCheck %s

// -----

// CHECK-LABEL: func @reduce
func.func @reduce(%arg0: tensor<4x4xf32>, %arg1 : tensor<4xf32>)
    -> (tensor<4xf32>) {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<4xf32>, %arg3: tensor<4xf32> ):
    %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "stablehlo.return"(%1) : (tensor<4xf32>) -> ()

  }) {dimensions = array<i64: 0>} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %0: tensor<4xf32>
}

// -----

// CHECK-LABEL: func @reduce_complex_type
func.func @reduce_complex_type(%arg0: tensor<1x2xcomplex<f32>>, %arg1 : tensor<complex<f32>>)
    -> (tensor<1xcomplex<f32>>) {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<complex<f32>> loc("foo"), %arg3: tensor<complex<f32>> loc("foo")):
    %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>> loc("foo")
    "stablehlo.return"(%1) : (tensor<complex<f32>>) -> () loc("foo")
  }) {dimensions = array<i64: 1>} : (tensor<1x2xcomplex<f32>>, tensor<complex<f32>>) -> tensor<1xcomplex<f32>> loc("foo")

  func.return %0: tensor<1xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @reduce_mixed_dynamism
func.func @reduce_mixed_dynamism(%arg0: tensor<4x4xf32>, %arg1 : tensor<f32>)
    -> (tensor<?xf32>) {
  %0 = stablehlo.reduce(%arg0 init: %arg1)
    applies stablehlo.multiply across dimensions = [1]
    : (tensor<4x4xf32>, tensor<f32>) -> tensor<?xf32>
  func.return %0: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @reduce_unranked
func.func @reduce_unranked(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>,
    %arg2: tensor<*xf32>, %arg3: tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0:2 = "stablehlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<*xf32>, %arg5: tensor<*xf32>, %arg6: tensor<*xf32>, %arg7: tensor<*xf32>):
    %1 = "stablehlo.add"(%arg4, %arg6) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %2 = "stablehlo.add"(%arg5, %arg7) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "stablehlo.return"(%1, %2) : (tensor<*xf32>, tensor<*xf32>) -> ()

  }) {dimensions = array<i64: 1>} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)

  func.return %0#0, %0#1 : tensor<*xf32>, tensor<*xf32>
}

// -----

// Verifies that dynamic input type is allowed with reducer function with static shapes.
// CHECK-LABEL: func @reduce_verify_dynamic_operand
func.func @reduce_verify_dynamic_operand(%arg0: tensor<8x?xf32>, %arg1 : tensor<4xf32>)
    -> (tensor<?xf32>) {

  %0 = "stablehlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<4xf32>, %arg3: tensor<4xf32> ):
    %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "stablehlo.return"(%1) : (tensor<4xf32>) -> ()

  }) {dimensions = array<i64: 0>} : (tensor<8x?xf32>, tensor<4xf32>) -> tensor<?xf32>

  func.return %0: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @reduce_mix_rank_and_unranked
func.func @reduce_mix_rank_and_unranked(%arg0: tensor<4x4xf32>, %arg1: tensor<*xf32>,
    %arg2: tensor<4xf32>, %arg3: tensor<*xf32>) -> (tensor<4xf32>, tensor<*xf32>) {
  %0:2 = "stablehlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<4xf32>, %arg5: tensor<*xf32>, %arg6: tensor<4xf32>, %arg7: tensor<*xf32>):
    %1 = "stablehlo.add"(%arg4, %arg6) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %2 = "stablehlo.add"(%arg5, %arg7) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "stablehlo.return"(%1, %2) : (tensor<4xf32>, tensor<*xf32>) -> ()

  }) {dimensions = array<i64: 1>} : (tensor<4x4xf32>, tensor<*xf32>, tensor<4xf32>, tensor<*xf32>) -> (tensor<4xf32>, tensor<*xf32>)

  func.return %0#0, %0#1 : tensor<4xf32>, tensor<*xf32>
}

// -----

// CHECK-LABEL: func @reduce_with_promotable_types
func.func @reduce_with_promotable_types(%arg0: tensor<4x4xf32>, %arg1 : tensor<f32>)
    -> (tensor<4xf64>) {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64> ):
    %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    "stablehlo.return"(%1) : (tensor<f64>) -> ()

  }) {dimensions = array<i64: 0>} : (tensor<4x4xf32>, tensor<f32>) -> tensor<4xf64>

  func.return %0: tensor<4xf64>
}

// -----

// CHECK-LABEL: func @reduce_with_promotable_quantized_types
func.func @reduce_with_promotable_quantized_types(%arg0: tensor<4x4x!quant.uniform<i8:f32, 2.000000e+00:15>>,
  %arg1: tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>) -> tensor<4x!quant.uniform<i32:f32, 2.000000e+00:15>> {
  %0 = stablehlo.reduce(%arg0 init: %arg1) across dimensions = [0] : (tensor<4x4x!quant.uniform<i8:f32, 2.000000e+00:15>>, tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>) -> tensor<4x!quant.uniform<i32:f32, 2.000000e+00:15>>
  reducer(%arg2: tensor<!quant.uniform<i32:f32, 2.000000e+00:15>>, %arg3: tensor<!quant.uniform<i32:f32, 2.000000e+00:15>>)  {
    %1 = stablehlo.add %arg2, %arg3 : tensor<!quant.uniform<i32:f32, 2.000000e+00:15>>
    stablehlo.return %1 : tensor<!quant.uniform<i32:f32, 2.000000e+00:15>>
  }
  return %0 : tensor<4x!quant.uniform<i32:f32, 2.000000e+00:15>>
}

// -----

func.func @reduce_c1(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>,
    %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<2xf32>, tensor<2xf32>) {

  // expected-error@+1 {{expects all inputs to have compatible shapes. Shape at input-index 1 is not compatible with shape at input-index 0}}
  %0:2 = "stablehlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
    %1 = "stablehlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.add"(%arg5, %arg7) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()

  }) {dimensions = array<i64: 1>} : (tensor<2x3xf32>, tensor<3x2xf32>, tensor<f32>, tensor<f32>) -> (tensor<2xf32>, tensor<2xf32>)

  func.return %0#0, %0#1 : tensor<2xf32>, tensor<2xf32>
}

// -----

func.func @reduce_c1(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>,
    %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>) {

  // expected-error@+1 {{expects all inputs to have compatible shapes. Shape at input-index 1 is not compatible with shape at input-index 0}}
  %0:2 = "stablehlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
    %1 = "stablehlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.add"(%arg5, %arg7) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()

  }) {dimensions = array<i64: 1>} : (tensor<?x?xf32>, tensor<?xf32>, tensor<f32>, tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

func.func @reduce_c2(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>) {

  // expected-error@+1 {{The type of reduction-region's parameter at index 0 is different than the corresponding result type: 'tensor<f32>' vs 'tensor<i32>'}}
  %0:2 = "stablehlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<i32>, %arg6: tensor<f32>, %arg7: tensor<i32>):
    %1 = "stablehlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.add"(%arg5, %arg7) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%2, %1) : (tensor<i32>, tensor<f32>) -> ()

  }) {dimensions = array<i64: 1>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xi32>
}

// -----

func.func @reduce_c2(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>) {

  // expected-error@+1 {{The type of reduction-region's parameter at index 3 is different than the corresponding result type: 'tensor<i32>' vs 'tensor<f32>'}}
  %0:2 = "stablehlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<i32>):
    %1 = "stablehlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.maximum"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()

  }) {dimensions = array<i64: 1>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<f32>) -> (tensor<?xf32>, tensor<?xi32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xi32>
}

// -----

func.func @reduce_c4(%arg0: tensor<?x?xf32>, %arg1 : tensor<f32>)
    -> (tensor<?xf32>) {

  // expected-error@+1 {{Out-of-bounds dimension -1, expected to be less than the input-tensor rank 2}}
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32> ):
    %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: -1>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>

  func.return %0: tensor<?xf32>
}

// -----

func.func @reduce_c4(%arg0: tensor<?x?xf32>, %arg1 : tensor<f32>)
    -> (tensor<?xf32>) {

  // expected-error@+1 {{Out-of-bounds dimension 2, expected to be less than the input-tensor rank 2}}
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32> ):
    %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 2>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>

  func.return %0: tensor<?xf32>
}

// -----

func.func @reduce_c4(%arg0: tensor<*xf32>, %arg1 : tensor<*xf32>)
    -> (tensor<*xf32>) {

  // expected-error@+1 {{Out-of-bounds dimension -1, expected to be > 0}}
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32> ):
    %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "stablehlo.return"(%1) : (tensor<*xf32>) -> ()
  }) {dimensions = array<i64: -1>} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0: tensor<*xf32>
}

// -----

func.func @reduce_c5(%arg0: tensor<?x?xf32>, %arg1 : tensor<f32>)
    -> (tensor<?xf32>) {

  // expected-error@+1 {{Duplicate reduction dimension: 1}}
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32> ):
    %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()

  }) {dimensions = array<i64: 1, 1>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>

  func.return %0: tensor<?xf32>
}

// -----

func.func @reduce_c6(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>) {

  // As there are only 2 parameters of op, ODS takes in 1 for inputs and 1 for init_values
  // by SameVariadicOperandSize, so the Reduction-region expect 1*2=2 parameters.
  // expected-error@+1 {{Reduction-region must take 2 parameters, but takes 4 parameter(s)}}
  %0:2 = "stablehlo.reduce"(%arg0, %arg1, %arg2) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
    %1 = "stablehlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.add"(%arg5, %arg7) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%arg5, %arg7) : (tensor<f32>, tensor<f32>) -> ()

  }) {dimensions = array<i64: 1>} : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

func.func @reduce_c6(%arg0: tensor<?x?xf32>, %arg1 : tensor<f32>)
    -> (tensor<f32>) {

  // expected-error@+1 {{The reduction-region expected to return some value(s)}}
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32> ):
    %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"() : () -> ()

  }) {dimensions = array<i64: 1>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<f32>

    func.return %0: tensor<f32>
}

// -----

func.func @reduce_c6(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>) {

  // expected-error@+1 {{Reduction-region here must produce 2 tensors, but produces 1 instead}}
  %0:2 = "stablehlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
    %1 = "stablehlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()

  }) {dimensions = array<i64: 1>} : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<f32>, tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

func.func @reduce_c6(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>) {

  // expected-error@+1 {{Reduction-region here must produce tensor-typed result(s), but produces 'tuple<tensor<f32>, tensor<i32>>' instead}}
  %0:2 = "stablehlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<i32>, %arg6: tensor<f32>, %arg7: tensor<i32>):
    %1 = "stablehlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.add"(%arg5, %arg7) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = "stablehlo.tuple"(%1, %2) : (tensor<f32>, tensor<i32>) -> tuple<tensor<f32>, tensor<i32>>
    "stablehlo.return"(%3, %1) : (tuple<tensor<f32>, tensor<i32>>, tensor<f32>) -> ()

  }) {dimensions = array<i64: 1>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xi32>
}

// -----

func.func @reduce_c6(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>) {

  // expected-error@+1 {{The element-type of reduction-region's result type at index 0 is expected to be promotable from the op's corresponding init-value element-type: 'tensor<i32>' vs 'tensor<f32>'}}
  %0:2 = "stablehlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>, %arg7: tensor<i32>):
    %1 = "stablehlo.add"(%arg4, %arg6) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = "stablehlo.add"(%arg5, %arg7) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%1, %2) : (tensor<i32>, tensor<i32>) -> ()

  }) {dimensions = array<i64: 1>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xi32>
}

// -----

func.func @reduce_c6(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>) {

  // expected-error@+1 {{The element-type of reduction-region's result type at index 1 is expected to be promotable from the op's corresponding init-value element-type: 'tensor<f32>' vs 'tensor<i32>'}}
  %0:2 = "stablehlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
    %1 = "stablehlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.add"(%arg5, %arg7) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()

  }) {dimensions = array<i64: 1>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xi32>
}

// -----

func.func @reduce_c6(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>) {

  // expected-error@+1 {{The element-type of reduction-region's argument at index 3 is expected to be promotable from 'i32', but got 'f32'}}
  %0:2 = "stablehlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
    %1 = "stablehlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.maximum"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()

  }) {dimensions = array<i64: 1>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

func.func @reduce_c6(%arg0: tensor<?xf32>, %arg1 : tensor<?xf32>)
    -> (tensor<f32>) {

  // expected-error@+1 {{The rank of reduction-region's argument at index 1 is expected to be <= 0, got 1}}
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<?xf32>, %arg3: tensor<?xf32> ):
    %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "stablehlo.return"(%1) : (tensor<?xf32>) -> ()

  }) {dimensions = array<i64: 0>} : (tensor<?xf32>, tensor<?xf32>) -> tensor<f32>

  func.return %0: tensor<f32>
}

// -----

func.func @reduce_c6(%arg0: tensor<8x5xf32>, %arg1 : tensor<4xf32>)
    -> (tensor<5xf32>) {

  // expected-error@+1 {{The shape of reduction-region's argument at index 1 is not compatible with that of reduce-op's input-parameter at index 0}}
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<4xf32>, %arg3: tensor<4xf32> ):
    %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "stablehlo.return"(%1) : (tensor<4xf32>) -> ()

  }) {dimensions = array<i64: 0>} : (tensor<8x5xf32>, tensor<4xf32>) -> tensor<5xf32>

  func.return %0: tensor<5xf32>
}

// -----


func.func @reduce_c6(%arg0: tensor<4x4xi32>, %arg1 : tensor<i32>)
    -> (tensor<4xi8>) {

  // expected-error@+1 {{The element-type of reduction-region's result type at index 0 is expected to be promotable from the op's corresponding init-value element-type: 'tensor<i8>' vs 'tensor<i32>'}}
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<i8>, %arg3: tensor<i8> ):
    %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<i8>, tensor<i8>) -> tensor<i8>
    "stablehlo.return"(%1) : (tensor<i8>) -> ()

  }) {dimensions = array<i64: 0>} : (tensor<4x4xi32>, tensor<i32>) -> tensor<4xi8>

  func.return %0: tensor<4xi8>
}

// -----

func.func @reduce_c6(%arg0: tensor<4x4xi32>, %arg1 : tensor<i8>)
    -> (tensor<4xi8>) {

  // expected-error@+1 {{The element-type of reduction-region's argument at index 1 is expected to be promotable from 'i32', but got 'i8'}}
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<i8>, %arg3: tensor<i8> ):
    %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<i8>, tensor<i8>) -> tensor<i8>
    "stablehlo.return"(%1) : (tensor<i8>) -> ()

  }) {dimensions = array<i64: 0>} : (tensor<4x4xi32>, tensor<i8>) -> tensor<4xi8>

  func.return %0: tensor<4xi8>
}

// -----

func.func @reduce_c6(%arg0: tensor<4x4x!quant.uniform<i8:f32, 2.000000e+00:15>>,
  %arg1: tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>) -> tensor<4x!quant.uniform<i32:f64, 2.000000e+00:15>> {

  // expected-error@+1 {{The element-type of reduction-region's result type at index 0 is expected to be promotable from the op's corresponding init-value element-type: 'tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>' vs 'tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>'}}
  %0 = stablehlo.reduce(%arg0 init: %arg1) across dimensions = [0] : (tensor<4x4x!quant.uniform<i8:f32, 2.000000e+00:15>>,
      tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>) -> tensor<4x!quant.uniform<i32:f64, 2.000000e+00:15>>

  reducer(%arg2: tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>, %arg3: tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>)  {
    %1 = stablehlo.add %arg2, %arg3 : tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>
    stablehlo.return %1 : tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>
  }
  return %0 : tensor<4x!quant.uniform<i32:f64, 2.000000e+00:15>>
}

// -----

func.func @reduce_c6(%arg0: tensor<4x4x!quant.uniform<i8:f64, 2.000000e+00:15>>,
  %arg1: tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>) -> tensor<4x!quant.uniform<i32:f32, 2.000000e+00:15>> {

  // expected-error@+1 {{The element-type of reduction-region's argument at index 1 is expected to be promotable from '!quant.uniform<i8:f64, 2.000000e+00:15>', but got '!quant.uniform<i32:f32, 2.000000e+00:15>'}}
  %0 = stablehlo.reduce(%arg0 init: %arg1) across dimensions = [0] : (tensor<4x4x!quant.uniform<i8:f64, 2.000000e+00:15>>,
      tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>) -> tensor<4x!quant.uniform<i32:f32, 2.000000e+00:15>>

  reducer(%arg2: tensor<!quant.uniform<i32:f32, 2.000000e+00:15>>, %arg3: tensor<!quant.uniform<i32:f32, 2.000000e+00:15>>)  {
    %1 = stablehlo.add %arg2, %arg3 : tensor<!quant.uniform<i32:f32, 2.000000e+00:15>>
    stablehlo.return %1 : tensor<!quant.uniform<i32:f32, 2.000000e+00:15>>
  }
  return %0 : tensor<4x!quant.uniform<i32:f32, 2.000000e+00:15>>
}

// -----

func.func @reduce_i3(%input: tensor<1x6xi64>, %init_value: tensor<i64>) -> tensor<1xi64> {
  // expected-error@+1 {{attribute 'dimensions' failed to satisfy constraint: either a DenseI64ArrayAttr or a 1-dimensional I64ElementsAttr.}}
  %0 = "stablehlo.reduce"(%input, %init_value) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      stablehlo.return %arg0 : tensor<i64>
  }) {
    dimensions = dense<1> : tensor<1x1xi64>
  } : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
  func.return %0 : tensor<1xi64>
}

// The following invalid cases arises while parsing a pretty-printed version of reduce-op will "non-eligible" inner-op.
// -----

func.func @reduce_parsing_pretty_reduce_non_commutative(%arg0: tensor<?x?xf32> , %arg1: tensor<f32> ) -> tensor<?xf32> {
  // expected-error@+1 {{expected the inner-op to be a commutative binary-op from stablehlo dialect, zero region, producing single result}}
 %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.divide across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32> loc("foo")
 func.return %0 : tensor<?xf32>
}

// -----

func.func @reduce_parsing_pretty_reduce_wrong_dialect(%arg0: tensor<?x?xf32> , %arg1: tensor<f32> ) -> tensor<?xf32> {
  // expected-error@+1 {{expected the inner-op to be a commutative binary-op from stablehlo dialect, zero region, producing single result}}
 %0 = stablehlo.reduce(%arg0 init: %arg1) applies std.add across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32> loc("foo")
 func.return %0 : tensor<?xf32>
}

// -----

func.func @reduce_parsing_pretty_reduce_non_binary(%arg0: tensor<?x?xf32> , %arg1: tensor<f32> ) -> tensor<?xf32> {
  // expected-error@+1 {{expected the inner-op to be a commutative binary-op from stablehlo dialect, zero region, producing single result}}
 %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.reshape across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32> loc("foo")
 func.return %0 : tensor<?xf32>
}
