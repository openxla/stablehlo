# `chlo` MLIR Dialect Builder API

[TOC]

## Builder Methods

### `chlo::AcosOp`

Creates a new [`chlo.acos`](https://openxla.org/stablehlo/generated/chlo#chloacos_chloacosop)
operation.

```c++
MlirOp Acos(MlirOp &operand);
```

### `chlo::AcoshOp`

Creates a new [`chlo.acosh`](https://openxla.org/stablehlo/generated/chlo#chloacosh_chloacoshop)
operation.

```c++
MlirOp Acosh(MlirOp &operand);
```

### `chlo::AsinAcosKernelOp`

Creates a new [`chlo._asin_acos_kernel`](https://openxla.org/stablehlo/generated/chlo#chlo_asin_acos_kernel_chlo_asin_acos_kernelop)
operation.

```c++
MlirOp AsinAcosKernel(MlirOp &operand);
```

### `chlo::AsinOp`

Creates a new [`chlo.asin`](https://openxla.org/stablehlo/generated/chlo#chloasin_chloasinop)
operation.

```c++
MlirOp Asin(MlirOp &operand);
```

### `chlo::AsinhOp`

Creates a new [`chlo.asinh`](https://openxla.org/stablehlo/generated/chlo#chloasinh_chloasinhop)
operation.

```c++
MlirOp Asinh(MlirOp &operand);
```

### `chlo::AtanOp`

Creates a new [`chlo.atan`](https://openxla.org/stablehlo/generated/chlo#chloatan_chloatanop)
operation.

```c++
MlirOp Atan(MlirOp &operand);
```

### `chlo::AtanhOp`

Creates a new [`chlo.atanh`](https://openxla.org/stablehlo/generated/chlo#chloatanh_chloatanhop)
operation.

```c++
MlirOp Atanh(MlirOp &operand);
```

### `chlo::BesselI1eOp`

Creates a new [`chlo.bessel_i1e`](https://openxla.org/stablehlo/generated/chlo#chlobessel_i1e_chlobessel_i1eop)
operation.

```c++
MlirOp BesselI1e(MlirOp &operand);
```

### `chlo::BroadcastAddOp`

Creates a new [`chlo.broadcast_add`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_add_chlobroadcast_addop)
operation.

```c++
MlirOp BroadcastAdd(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastAndOp`

Creates a new [`chlo.broadcast_and`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_and_chlobroadcast_andop)
operation.

```c++
MlirOp BroadcastAnd(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastAtan2Op`

Creates a new [`chlo.broadcast_atan2`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_atan2_chlobroadcast_atan2op)
operation.

```c++
MlirOp BroadcastAtan2(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastCompareOp`

Creates a new [`chlo.broadcast_compare`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_compare_chlobroadcast_compareop)
operation.

```c++
MlirOp BroadcastCompare(MlirOp &lhs, MlirOp &rhs, ::mlir::chlo::ComparisonDirection comparison_direction, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {}, /*optional*/::mlir::chlo::ComparisonTypeAttr compare_type = {});
```

### `chlo::BroadcastComplexOp`

Creates a new [`chlo.broadcast_complex`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_complex_chlobroadcast_complexop)
operation.

```c++
MlirOp BroadcastComplex(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastDivOp`

Creates a new [`chlo.broadcast_divide`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_divide_chlobroadcast_divideop)
operation.

```c++
MlirOp BroadcastDiv(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastMaxOp`

Creates a new [`chlo.broadcast_maximum`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_maximum_chlobroadcast_maximumop)
operation.

```c++
MlirOp BroadcastMax(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastMinOp`

Creates a new [`chlo.broadcast_minimum`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_minimum_chlobroadcast_minimumop)
operation.

```c++
MlirOp BroadcastMin(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastMulOp`

Creates a new [`chlo.broadcast_multiply`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_multiply_chlobroadcast_multiplyop)
operation.

```c++
MlirOp BroadcastMul(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastNextAfterOp`

Creates a new [`chlo.broadcast_next_after`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_next_after_chlobroadcast_next_afterop)
operation.

```c++
MlirOp BroadcastNextAfter(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastOrOp`

Creates a new [`chlo.broadcast_or`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_or_chlobroadcast_orop)
operation.

```c++
MlirOp BroadcastOr(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastPolygammaOp`

Creates a new [`chlo.broadcast_polygamma`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_polygamma_chlobroadcast_polygammaop)
operation.

```c++
MlirOp BroadcastPolygamma(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastPowOp`

Creates a new [`chlo.broadcast_power`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_power_chlobroadcast_powerop)
operation.

```c++
MlirOp BroadcastPow(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastRemOp`

Creates a new [`chlo.broadcast_remainder`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_remainder_chlobroadcast_remainderop)
operation.

```c++
MlirOp BroadcastRem(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastSelectOp`

Creates a new [`chlo.broadcast_select`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_select_chlobroadcast_selectop)
operation.

```c++
MlirOp BroadcastSelect(MlirOp &pred, MlirOp &on_true, MlirOp &on_false);
```

### `chlo::BroadcastShiftLeftOp`

Creates a new [`chlo.broadcast_shift_left`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_shift_left_chlobroadcast_shift_leftop)
operation.

```c++
MlirOp BroadcastShiftLeft(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastShiftRightArithmeticOp`

Creates a new [`chlo.broadcast_shift_right_arithmetic`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_shift_right_arithmetic_chlobroadcast_shift_right_arithmeticop)
operation.

```c++
MlirOp BroadcastShiftRightArithmetic(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastShiftRightLogicalOp`

Creates a new [`chlo.broadcast_shift_right_logical`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_shift_right_logical_chlobroadcast_shift_right_logicalop)
operation.

```c++
MlirOp BroadcastShiftRightLogical(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastSubOp`

Creates a new [`chlo.broadcast_subtract`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_subtract_chlobroadcast_subtractop)
operation.

```c++
MlirOp BroadcastSub(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastXorOp`

Creates a new [`chlo.broadcast_xor`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_xor_chlobroadcast_xorop)
operation.

```c++
MlirOp BroadcastXor(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::BroadcastZetaOp`

Creates a new [`chlo.broadcast_zeta`](https://openxla.org/stablehlo/generated/chlo#chlobroadcast_zeta_chlobroadcast_zetaop)
operation.

```c++
MlirOp BroadcastZeta(MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::DenseI64ArrayAttr broadcast_dimensions = {});
```

### `chlo::ConjOp`

Creates a new [`chlo.conj`](https://openxla.org/stablehlo/generated/chlo#chloconj_chloconjop)
operation.

```c++
MlirOp Conj(MlirOp &operand);
```

### `chlo::ConstantOp`

Creates a new [`chlo.constant`](https://openxla.org/stablehlo/generated/chlo#chloconstant_chloconstantop)
operation.

```c++
MlirOp Constant(MlirBuilder &builder, ::mlir::ElementsAttr value);
```

### `chlo::CoshOp`

Creates a new [`chlo.cosh`](https://openxla.org/stablehlo/generated/chlo#chlocosh_chlocoshop)
operation.

```c++
MlirOp Cosh(MlirOp &operand);
```

### `chlo::DigammaOp`

Creates a new [`chlo.digamma`](https://openxla.org/stablehlo/generated/chlo#chlodigamma_chlodigammaop)
operation.

```c++
MlirOp Digamma(MlirOp &operand);
```

### `chlo::ErfInvOp`

Creates a new [`chlo.erf_inv`](https://openxla.org/stablehlo/generated/chlo#chloerf_inv_chloerf_invop)
operation.

```c++
MlirOp ErfInv(MlirOp &operand);
```

### `chlo::ErfOp`

Creates a new [`chlo.erf`](https://openxla.org/stablehlo/generated/chlo#chloerf_chloerfop)
operation.

```c++
MlirOp Erf(MlirOp &operand);
```

### `chlo::ErfcOp`

Creates a new [`chlo.erfc`](https://openxla.org/stablehlo/generated/chlo#chloerfc_chloerfcop)
operation.

```c++
MlirOp Erfc(MlirOp &operand);
```

### `chlo::IsInfOp`

Creates a new [`chlo.is_inf`](https://openxla.org/stablehlo/generated/chlo#chlois_inf_chlois_infop)
operation.

```c++
MlirOp IsInf(MlirOp &operand);
```

### `chlo::IsNegInfOp`

Creates a new [`chlo.is_neg_inf`](https://openxla.org/stablehlo/generated/chlo#chlois_neg_inf_chlois_neg_infop)
operation.

```c++
MlirOp IsNegInf(MlirOp &operand);
```

### `chlo::IsPosInfOp`

Creates a new [`chlo.is_pos_inf`](https://openxla.org/stablehlo/generated/chlo#chlois_pos_inf_chlois_pos_infop)
operation.

```c++
MlirOp IsPosInf(MlirOp &operand);
```

### `chlo::LgammaOp`

Creates a new [`chlo.lgamma`](https://openxla.org/stablehlo/generated/chlo#chlolgamma_chlolgammaop)
operation.

```c++
MlirOp Lgamma(MlirOp &operand);
```

### `chlo::NextAfterOp`

Creates a new [`chlo.next_after`](https://openxla.org/stablehlo/generated/chlo#chlonext_after_chlonext_afterop)
operation.

```c++
MlirOp NextAfter(MlirOp &x, MlirOp &y);
```

### `chlo::PolygammaOp`

Creates a new [`chlo.polygamma`](https://openxla.org/stablehlo/generated/chlo#chlopolygamma_chlopolygammaop)
operation.

```c++
MlirOp Polygamma(MlirOp &n, MlirOp &x);
```

### `chlo::RaggedDotOp`

Creates a new [`chlo.ragged_dot`](https://openxla.org/stablehlo/generated/chlo#chloragged_dot_chloragged_dotop)
operation.

```c++
MlirOp RaggedDot(Type resultType, MlirOp &lhs, MlirOp &rhs, MlirOp &group_sizes, ::mlir::chlo::RaggedDotDimensionNumbersAttr ragged_dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config = {});
```

### `chlo::SinhOp`

Creates a new [`chlo.sinh`](https://openxla.org/stablehlo/generated/chlo#chlosinh_chlosinhop)
operation.

```c++
MlirOp Sinh(MlirOp &operand);
```

### `chlo::SquareOp`

Creates a new [`chlo.square`](https://openxla.org/stablehlo/generated/chlo#chlosquare_chlosquareop)
operation.

```c++
MlirOp Square(MlirOp &operand);
```

### `chlo::TanOp`

Creates a new [`chlo.tan`](https://openxla.org/stablehlo/generated/chlo#chlotan_chlotanop)
operation.

```c++
MlirOp Tan(MlirOp &operand);
```

### `chlo::TopKOp`

Creates a new [`chlo.top_k`](https://openxla.org/stablehlo/generated/chlo#chlotop_k_chlotop_kop)
operation.

```c++
SmallVector<MlirOp, 2> TopK(MlirOp &operand, uint64_t k);
```

### `chlo::ZetaOp`

Creates a new [`chlo.zeta`](https://openxla.org/stablehlo/generated/chlo#chlozeta_chlozetaop)
operation.

```c++
MlirOp Zeta(MlirOp &x, MlirOp &q);
```

## Skipped Operations

Unable to generate builder for the following operations:

 - [`chlo.constant_like`](https://openxla.org/stablehlo/generated/chlo#chloconstant_like_chloconstant_likeop)

 - [`chlo.scan`](https://openxla.org/stablehlo/generated/chlo#chloscan_chloscanop)

