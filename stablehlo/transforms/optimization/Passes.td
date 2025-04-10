/* Copyright 2025 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

include "mlir/Pass/PassBase.td"

defvar stablehlo_aggressive_folder_options = [
  Option<"foldFloat", "fold-float", "bool", /*default=*/"true",
          "Allow for potentially lossy computations using float type.">,
  Option<"foldOpElementLimit", "fold-op-element-limit", "int64_t",
          /*default=*/"65536",
          "Upper limit on how many elements may be folded by an op folder. "
          "This limit doesn't apply in certain special cases such as when "
          "adding 0, multiplying by 1, or operating on splats in some ways.">,
];

def StablehloAggressiveFolderPass
    : Pass<"stablehlo-aggressive-folder", "func::FuncOp"> {
  let summary = "Folds StableHLO operations";
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect",
  ];
  let options = stablehlo_aggressive_folder_options;
}

def StablehloAggressiveSimplificationPass
    : Pass<"stablehlo-aggressive-simplification", "func::FuncOp"> {
  let summary = "Canonicalizes StableHLO operations";
  let description = [{

    <!--
      The following is generated list of patterns from code comments and can be
      reconstructed using the following command:
      $ cd stablehlo/transforms/optimization
      $ grep "// Pattern:" *Simplification* | sed 's/.*Pattern: /- /' | sort
    -->

    Note: Prefer StablehloTargetIndependentOptimizationPass to get best results.

    Performs graph simplifications, including:

    ```
    - add(cst, X) -> add(X, cst)
    - add(X, 0) -> X
    - and(cst, X) -> and(X, cst)
    - and(X, 0) -> 0
    - and(X, 1) -> X
    - broadcast_in_dim(broadcast_in_dim(X, [dimsA...]), [dimsB...]) -> broadcast_in_dim(X, merge(dimsA, dimsB))
    - broadcast_in_dim(X, [dims...]) -> transpose(X, [dims...]) [if same numel & rank]
    - broadcast_in_dim(X, [iota...]) -> X
    - broadcast_in_dim(X, [sorted...]) -> reshape(X, [sorted...]) [if same numel]
    - compare(cst, X, comparator) -> compare(X, cst, inv(comparator))
    - compare(X, X, [EQ,GE,LE]) -> true
    - compare(X, X, [NE,GT,LT]) -> false
    - complex(real(X), imag(X))) -> X
    - concatenate(concatenate(X, Y), Z) -> concatenate(X, Y, Z)
    - concatenate(X) -> X
    - concatenate(X, Y, []) -> concatenate(X, Y)
    - convert(X, [X.type]) -> X
    - dynamic_broadcast_in_dim(dynamic_broadcast_in_dim(X, _, [dimsA...]), shape, [dimsB...]) -> dynamic_broadcast_in_dim(X, shape, merge(dimsA, dimsB))
    - dynamic_broadcast_in_dim(dynamic_reshape(X, shape), shape) -> dynamic_reshape(X, shape)
    - dynamic_broadcast_in_dim(X, _, _, [all_nonexpanding...]) -> convert(X)
    - dynamic_broadcast_in_dim(X, shape_of(X)) -> X
    - dynamic_gather(x, constant(slice_sizes)) -> gather(x, slice_sizes)
    - dynamic_iota(shape, dim) ->
    - dynamic_pad(X, low, high, interior) -> pad(X, low, high, interior)
    - dynamic_reshape(dynamic_reshape(X, _), shape)) -> dynamic_reshape(X, shape)
    - dynamic_reshape(op(dynamic_reshape(X, shape)), shape)
    - dynamic_slice(X, begin, slice_sizes) -> slice(X, begin, slice_sizes)
    - dynamic_update_slice(X, update, start_indices : zero)) -> update
    - dynamic_update_slice(X, update : zero_extent)) -> X
    - gather(X, cst_start_indices) -> slice(X, slice_start, slice_end)
    - get_dimension_size(X, i) -> X.shape[i]
    - get_tuple_element(tuple(X_0, X_1, ...), i) -> X_i
    - imag(complex(R,I)) -> I
    - iota(dim) : multi_rank
    - iota(dim) : type -> constant(0) : type [if type[dim] == 1]
    - max(cst, X) -> max(X, cst)
    - minimum(cst, X) -> minimum(X, cst)
    - multiply(cst, X) -> multiply(X, cst)
    - multiply(X, 0i) -> 0i
    - multiply(X, 1i) -> X
    - op(X : zero_extent_tensor) -> constant([])
    - or(cst, X) -> or(X, cst)
    - or(X, 0) -> X
    - or(X, 1) -> 1
    - pad(empty_tensor, _) -> broadcast_in_dim(empty_tensor, _)
    - real(complex(R,I)) -> X
    - real_dynamic_slice(X, start, limit, strides)
    - real_dynamic_slice(X, start, limit, strides)
    - reduce[A](_, _, fn:return A) -> A...
    - reduce(empty_0, empty_1, ...) -> [broadcast_in_dim(empty_i)...]
    - reduce(in_1, in_2, _, _) -> reduce(in_1, _, _) [if unused(in_2)]
    - reduce(X..., dims=[], add) -> X...
    - reshape(reshape(X, _), [shape]) -> reshape(X, [shape])
    - reshape(X, [X.shape]) -> X
    - select(broadcast(not(p)), t, f) => select(broadcast(p), f, t)
    - select(not(p), t, f) => select(p, f, t)
    - shape_of(dynamic_reshape(X, shape)) -> shape
    - slice(concat(X,Y,Z,...),...) -> concat(slice(X),slice(Y),slice(Z))
    - slice(X, [A:A], [B:B], ...) -> X
    - sort(X) -> sort(X, dim = N) [when dim can be inferred]
    - sort(X,Y) -> sort(X) [if Y unused and unused in comparator]
    - subtract(X, 0) -> X
    - subtract(X, X) -> 0
    - transpose(X, [iota...]) -> X
    - transpose(X, [no_mem_layout_change...]) -> reshape(X)
    - tuple(get_tuple_element(X, 0), get_tuple_element(X, 1), ...) -> X
    - while -> while (loop invariants as implicit captures)
    - xor(cst, X) -> xor(X, cst)
    - (+more)
    ```

    This list is pulled from code comments so is not fully exhaustive, but has
    high coverage of the pass today.
  }];
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect",
    "mlir::arith::ArithDialect",
  ];
}

def StablehloTargetIndependentOptimizationPass
    : Pass<"stablehlo-target-independent-optimization", "func::FuncOp"> {
  let summary = "Runs canonicalizers, folders, and other target-independent optimizations.";
  let description = [{
    Uses patterns from StablehloAggressiveSimplificationPass and
    StablehloAggressiveFolderPass together, allowing canonicalization and
    folding to be performed in the same pattern set, often leading to better
    results.

    Users should prefer this pass to calling the others directly.
  }];
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect",
  ];
  let options = stablehlo_aggressive_folder_options;
}
