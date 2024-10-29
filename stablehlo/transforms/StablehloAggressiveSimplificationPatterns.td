// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This is the legalization pattern definition file for CHLO to StableHLO.
// These are included in the populateDecompositionPatterns factory
// and should only include canonical expansions which are not actually
// ambiguous/different for various backends. Avoid patterns that are actually
// lowering to non-canonical forms.

include "mlir/IR/OpBase.td"
include "stablehlo/dialect/StablehloOps.td"
include "mlir/Dialect/Shape/IR/ShapeOps.td"

///////////
//// Op & Type Constraints

class DimSizeEquals<int dimSize> : Constraint<
    CPred<"llvm::cast<ShapedType>($0.getType()).getDimSize($1.getInt()) == " # dimSize>,
    "dim size is " # dimSize>;

def AllDimsNonExpanding : Constraint<
    CPred<"$0 && cast<DenseI64ArrayAttr>($0).size() == llvm::cast<ShapedType>($1.getType()).getRank()">,
    "all dims are non-expanding">;

def AllZero : Constraint<
    CPred<"llvm::all_of($0, [](Value operand) {return matchPattern(operand, m_Zero()); })">,
    "is all zero">;

def CommutativeOp : Constraint<
    CPred<"$0.getDefiningOp()->hasTrait<hlo::OpTrait::IsCommutative>()">,
    "op is commutative">;

def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

def NotConstantOp : Constraint<
    CPred<"llvm::isa<BlockArgument>($0) || !llvm::isa<stablehlo::ConstantOp>($0.getDefiningOp())">,
    "is not a constant.">;

def NumberOfElementsEqual : Constraint<
    CPred<"llvm::cast<ShapedType>($0.getType()).getNumElements() == llvm::cast<ShapedType>($1.getType()).getNumElements()">,
    "same number of elements">;

def OperandsEqual : Constraint<CPred<"$0 == $1">, "operands are equal">;

def RankEqual : Constraint<
    CPred<"llvm::cast<ShapedType>($0.getType()).getRank() == llvm::cast<ShapedType>($1.getType()).getRank()">,
    "same rank">;

def TypesEqual : Constraint<CPred<"$0.getType() == $1.getType()">, "operands are equal">;

///////////
//// Attribute Constraints

def AnySplat : AttrConstraint<CPred<"$_self.isSplat()">, "is any splat">;

def AnyZero : AttrConstraint<
    CPred<"::mlir::matchPattern($_self, m_AnyAttrOf(m_Zero(), m_AnyZeroFloat()))">,
    "is int or float zero">;

def DenseIntElementsAttr : AttrConstraint<
    CPred<"llvm::isa<DenseIntElementsAttr>($_self)">,
    "is dense int elements attr">;

def EmptyI64Array : AttrConstraint<
    CPred<"cast<DenseI64ArrayAttr>($_self).empty()">,
    "is empty i64 array">;

def IntOne : AttrConstraint<
    CPred<"::mlir::matchPattern($_self, m_One())">,
    "is integer one">;

def IntZero : AttrConstraint<
    CPred<"::mlir::matchPattern($_self, m_Zero())">,"is integer zero">;

def IotaDims : AttrConstraint<
    CPred<"isIotaRange(cast<DenseI64ArrayAttr>($_self).asArrayRef())">,
    "is iota dimensions">;

def SortedDims : AttrConstraint<
    CPred<"llvm::is_sorted(cast<DenseI64ArrayAttr>($_self).asArrayRef())">,
    "is sorted dimensions">;

def ZeroExtent : AttrConstraint<
    CPred<"cast<DenseIntElementsAttr>($_self).getNumElements() == 0">,
    "is zero extent">;

///////////
//// Native Code Call Utilities

def CastIntElementsAttr : NativeCodeCall<"cast<DenseIntElementsAttr>($0)">;

def ConvertToI64Array : NativeCodeCall<"convertToI64Array($_builder, $0)">;

def GetOperandN : NativeCodeCall<"$0.getDefiningOp()->getOperand($1.getInt())">;

def GetEmptyI64Array : NativeCodeCall<"$_builder.getDenseI64ArrayAttr({})">;

def MergeBroadcastDims : NativeCodeCall<"getMergedBroadcastDimensions($_builder, $0, $1)">;

def StableHLO_ConvertOpWithShape : NativeCodeCall<
    "$_builder.create<stablehlo::ConvertOp>($_loc, $0.getType(), $1)">;

def StableHLO_ReshapeOpWithShape : NativeCodeCall<
    "$_builder.create<stablehlo::ReshapeOp>($_loc, $0.getType(), $1)">;

class StableHLO_ConstantLike<string value> : NativeCodeCall<
    "::mlir::stablehlo::getConstantLike($_builder, $_loc, " # value # ", $0)">;

////////////////////////////
// Generic BinaryOp Patterns

// op(cst, X) -> op(X, cst)
class CanonicalizeConstantToRhs<Op StableHLO_OpType>
  : Pat<(StableHLO_OpType:$op (StableHLO_ConstantOp:$lhs $value), $rhs),
        (StableHLO_OpType $rhs, $lhs),
        [(NotConstantOp $rhs), (CommutativeOp $op)]>;

////////
// AddOp

// Pattern: add(cst, X) -> add(X, cst)
def : CanonicalizeConstantToRhs<StableHLO_AddOp>;

// Pattern: add(X, 0) -> X
def : Pat<(StableHLO_AddOp $lhs, (ConstantLikeMatcher AnyZero:$value)),
          (replaceWithValue $lhs)>;

////////
// AndOp

// Pattern: and(cst, X) -> and(X, cst)
def : CanonicalizeConstantToRhs<StableHLO_AndOp>;

// Pattern: and(X, 0) -> 0
def : Pat<(StableHLO_AndOp $lhs, (StableHLO_ConstantOp:$zero IntZero:$value)),
          (replaceWithValue $zero)>;

// Pattern: and(X, 1) -> X
def : Pat<(StableHLO_AndOp $lhs, (StableHLO_ConstantOp:$one IntOne:$value)),
          (replaceWithValue $lhs)>;

////////
// BroadcastInDimOp

// Pattern: broadcast_in_dim(X, [iota...]) -> X
def : Pat<(StableHLO_BroadcastInDimOp:$op $operand, IotaDims:$dims),
          (replaceWithValue $operand),
          [(TypesEqual $op, $operand)]>;

// Pattern: broadcast_in_dim(broadcast_in_dim(X, [dimsA...]), [dimsB...]) -> broadcast_in_dim(X, merge(dimsA, dimsB))
def : Pat<(StableHLO_BroadcastInDimOp
            (StableHLO_BroadcastInDimOp $operand, $dims_parent), $dims),
          (StableHLO_BroadcastInDimOp $operand, (MergeBroadcastDims $dims, $dims_parent))>;

// Pattern: broadcast_in_dim(X, [sorted...]) -> reshape(X, [sorted...]) [if same numel]
def : Pat<(StableHLO_BroadcastInDimOp:$op $operand, SortedDims:$dims),
          (StableHLO_ReshapeOpWithShape $op, $operand),
          [(NumberOfElementsEqual $op, $operand)]>;

// Pattern: broadcast_in_dim(X, [dims...]) -> transpose(X, [dims...]) [if same numel & rank]
def : Pat<(StableHLO_BroadcastInDimOp:$op $operand, $dims),
          (StableHLO_TransposeOp $operand, $dims),
          [(NumberOfElementsEqual $op, $operand), (RankEqual $op, $operand)]>;

////////
// ConvertOp

// Pattern: convert(X, [X.type]) -> X
def : Pat<(StableHLO_ConvertOp:$convert $operand),
          (replaceWithValue $operand),
          [(TypesEqual $convert, $operand)]>;

////////
// DynamicBroadcastInDimOp

// Pattern: dynamic_broadcast_in_dim(dynamic_broadcast_in_dim(X, _, [dimsA...]), shape, [dimsB...]) -> dynamic_broadcast_in_dim(X, shape, merge(dimsA, dimsB))
// TODO: Think more if the values of known_expanding_dimensions and known_non_expanding_dimensions can be preserved.
def : Pat<(StableHLO_DynamicBroadcastInDimOp
            (StableHLO_DynamicBroadcastInDimOp $operand, $shape_p, $dims_p, $expanding_p, $nonexpanding_p),
             $shape, $dims, $expanding, $nonexpanding),
          (StableHLO_DynamicBroadcastInDimOp $operand, $shape, (MergeBroadcastDims $dims, $dims_p), (GetEmptyI64Array), (GetEmptyI64Array))>;

// Pattern: dynamic_broadcast_in_dim(X, _, _, [all_nonexpanding...]) -> convert(X)
// No-op, but wrap in ConvertOp to preserve dynamic output shape, can be
// important if this result is returned, where refining type would require
// also updating the funciton signature.
def : Pat<(StableHLO_DynamicBroadcastInDimOp:$op $operand, $shape, IotaDims:$dims, $expanding, $nonexpanding),
          (StableHLO_ConvertOpWithShape $op, $operand),
          [(AllDimsNonExpanding $nonexpanding, $op)]>;

// Pattern: dynamic_broadcast_in_dim(dynamic_reshape(X, shape), shape) -> dynamic_reshape(X, shape)
// If sharing same shape operand, is dynamic reshape.
def : Pat<(StableHLO_DynamicBroadcastInDimOp
            (StableHLO_DynamicReshapeOp $operand, $shape), $shape, IotaDims:$dims, $expanding, $nonexpanding),
          (StableHLO_DynamicReshapeOp $operand, $shape)>;

// Pattern: dynamic_broadcast_in_dim(X, shape_of(X)) -> X
def : Pat<(StableHLO_DynamicBroadcastInDimOp 
            $operand, (Shape_ShapeOfOp $operand), IotaDims:$dims, $expanding, $nonexpanding),
          (replaceWithValue $operand)>;

////////
// DynamicGatherOp

// Pattern: dynamic_gather(x, constant(slice_sizes)) -> gather(x, slice_sizes)
def : Pat<(StableHLO_DynamicGatherOp $operand, $start_indices, (StableHLO_ConstantOp DenseIntElementsAttr:$slice_sizes), $dimension_numbers, $indices_are_sorted),
          (StableHLO_GatherOp $operand, $start_indices, $dimension_numbers, (ConvertToI64Array $slice_sizes), $indices_are_sorted)>;

////////
// DynamicPadOp

// Pattern: dynamic_pad(X, low, high, interior) -> pad(X, low, high, interior)
//            [if low, high, interior are all constants]
def : Pat<(StableHLO_DynamicPadOp $input,
            $padding_value,
            (ConstantLikeMatcher AnyIntElementsAttr:$edge_padding_low),
            (ConstantLikeMatcher AnyIntElementsAttr:$edge_padding_high),
            (ConstantLikeMatcher AnyIntElementsAttr:$interior_padding)),
          (StableHLO_PadOp $input, $padding_value,
            (ConvertToI64Array $edge_padding_low),
            (ConvertToI64Array $edge_padding_high),
            (ConvertToI64Array $interior_padding))>;

////////
// DynamicReshapeOp

// Pattern: dynamic_reshape(dynamic_reshape(X, _), shape)) -> dynamic_reshape(X, shape)
def : Pat<(StableHLO_DynamicReshapeOp (StableHLO_DynamicReshapeOp $operand, $shape_p), $shape),
           (StableHLO_DynamicReshapeOp $operand, $shape)>;

// Pattern: shape_of(dynamic_reshape(X, shape)) -> shape
def : Pat<(Shape_ShapeOfOp:$op (StableHLO_DynamicReshapeOp $x, $shape)),
          (replaceWithValue $shape),
          [(TypesEqual $shape, $op)]>;

////////
// DynamicUpdateSliceOp

// Pattern: dynamic_update_slice(X, update : zero_extent)) -> X
def : Pat<(StableHLO_DynamicUpdateSliceOp $operand, (ConstantLikeMatcher ZeroExtent:$update), $start_indices),
           (replaceWithValue $operand)>;

// Pattern: dynamic_update_slice(X, update, start_indices : zero)) -> update
def : Pat<(StableHLO_DynamicUpdateSliceOp AnyStaticShapeTensor:$operand, AnyStaticShapeTensor:$update, $start_indices),
           (replaceWithValue $update),
           [(TypesEqual $operand, $update), (AllZero $start_indices)]>;


////////
// ComplexOp

// Pattern: complex(real(X), imag(X))) -> X
def : Pat<(StableHLO_ComplexOp (StableHLO_RealOp $operand), (StableHLO_ImagOp $operand)),
          (replaceWithValue $operand)>;


////////
// ImagOp

// Pattern: imag(complex(R,I)) -> I
def : Pat<(StableHLO_ImagOp (StableHLO_ComplexOp $lhs, $rhs)),
          (replaceWithValue $rhs)>;

////////
// IotaOp

// Pattern: iota(dim) : type -> constant(0) : type [if type[dim] == 1]
def : Pat<(StableHLO_IotaOp:$iota $dim),
          (StableHLO_ConstantLike<"0"> $iota),
          [(DimSizeEquals<1> $iota, $dim)]>;


////////
// MaxOp

// Pattern: max(cst, X) -> max(X, cst)
def : CanonicalizeConstantToRhs<StableHLO_MaxOp>;

////////
// MinOp

// Pattern: minimum(cst, X) -> minimum(X, cst)
def : CanonicalizeConstantToRhs<StableHLO_MinOp>;

////////
// MulOp

// Pattern: multiply(cst, X) -> multiply(X, cst)
def : CanonicalizeConstantToRhs<StableHLO_MulOp>;

// Pattern: multiply(X, 0i) -> 0i
// Multiplication by 0. This fold is not trivial for floats in presence of NaNs
def : Pat<(StableHLO_MulOp $lhs, (StableHLO_ConstantOp:$zero IntZero:$value)),
          (replaceWithValue $zero)>;

// Pattern: multiply(X, 1i) -> X
def : Pat<(StableHLO_MulOp $lhs, (StableHLO_ConstantOp IntOne:$value)),
          (replaceWithValue $lhs)>;

////////
// OrOp

// Pattern: or(cst, X) -> or(X, cst)
def : CanonicalizeConstantToRhs<StableHLO_OrOp>;

// Pattern: or(X, 1) -> 1
def : Pat<(StableHLO_OrOp $lhs, (StableHLO_ConstantOp:$one IntOne:$value)),
          (replaceWithValue $one)>;

// Pattern: or(X, 0) -> X
def : Pat<(StableHLO_OrOp $lhs, (StableHLO_ConstantOp:$zero IntZero:$value)),
          (replaceWithValue $lhs)>;

////////
// RealDynamicSliceOp

// Pattern: real_dynamic_slice(X, start, limit, strides) 
//           -> slice(X, start, limit, strides)
//          [if start, limit, strides are all constants]
def : Pat<(StableHLO_RealDynamicSliceOp $operand,
            (ConstantLikeMatcher DenseIntElementsAttr:$start_indices),
            (ConstantLikeMatcher DenseIntElementsAttr:$limit_indices),
            (ConstantLikeMatcher DenseIntElementsAttr:$strides)),
          (StableHLO_SliceOp $operand,
            (ConvertToI64Array $start_indices),
            (ConvertToI64Array $limit_indices),
            (ConvertToI64Array $strides))>;

////////
// RealOp

// Pattern: real(complex(R,I)) -> X
def : Pat<(StableHLO_RealOp (StableHLO_ComplexOp $lhs, $rhs)),
          (replaceWithValue $lhs)>;

////////
// ReduceOp
// Note: If modifying region is required, must write pattern in C++

// Pattern: reduce(X..., dims=[], add) -> X...
def : Pat<(StableHLO_ReduceOp $operands, $init, EmptyI64Array:$dims),
          (replaceWithValue $operands)>;

////////
// ReshapeOp

// Pattern: reshape(reshape(X, _), [shape]) -> reshape(X, [shape])
def : Pat<(StableHLO_ReshapeOp:$reshape (StableHLO_ReshapeOp $operand)),
          (StableHLO_ReshapeOpWithShape $reshape, $operand)>;

// Pattern: reshape(X, [X.shape]) -> X
def : Pat<(StableHLO_ReshapeOp:$reshape $operand),
          (replaceWithValue $operand),
          [(TypesEqual $reshape, $operand)]>;


////////
// SelectOp

// Pattern: select(not(p), t, f) => select(p, f, t)
def : Pat<
  (StableHLO_SelectOp (StableHLO_NotOp $pred), $on_true, $on_false),
  (StableHLO_SelectOp $pred, $on_false, $on_true)>;

// Pattern: select(broadcast(not(p)), t, f) => select(broadcast(p), f, t)
def : Pat<(StableHLO_SelectOp (StableHLO_BroadcastInDimOp:$b (StableHLO_NotOp $pred), $broadcast_dimensions), $on_true, $on_false),
          (StableHLO_SelectOp (StableHLO_BroadcastInDimOp $pred, $broadcast_dimensions, (returnType $b)), $on_false, $on_true),
          [(HasOneUse $b)]>;

////////
// SubtractOp

// Pattern: subtract(X, X) -> 0
// Must be static shape, otherwise would require broadcasting via CHLO_ConstantLike
def : Pat<(StableHLO_SubtractOp AnyStaticShapeTensor:$operand, $operand),
          (StableHLO_ConstantLike<"0"> $operand)>;

// Pattern: subtract(X, 0) -> X
def : Pat<(StableHLO_SubtractOp $lhs, (StableHLO_ConstantOp AnyZero:$value)),
          (replaceWithValue $lhs)>;

////////
// SliceOp

// Pattern: slice(X, [A:A], [B:B], ...) -> X
def : Pat<(StableHLO_SliceOp:$op AnyStaticShapeTensor:$operand, $start_indices, $limit_indices, $strides),
          (replaceWithValue $operand),
          [(TypesEqual $operand, $op)]>;

////////
// TransposeOp

// Pattern: transpose(X, [iota...]) -> X
def : Pat<(StableHLO_TransposeOp $lhs, IotaDims:$dims),
          (replaceWithValue $lhs)>;

////////
// GetTupleElementOp

// Pattern: get_tuple_element(tuple(X_0, X_1, ...), i) -> X_i
def : Pat<(StableHLO_GetTupleElementOp (StableHLO_TupleOp:$tuple $operands), $idx),
          (GetOperandN $tuple, $idx)>;

////////
// XorOp

// Pattern: xor(cst, X) -> xor(X, cst)
def : CanonicalizeConstantToRhs<StableHLO_XorOp>;

// To consider: xor(X, X) -> 0
// Unclear if this is beneficial on hardware vs adding another constant
//
// def : Pat<(StableHLO_XorOp AnyStaticShapeTensor:$operand, $operand),
//           (StableHLO_ConstantLike<"0"> $operand)>;
