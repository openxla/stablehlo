# `stablehlo` MLIR Dialect Builder API

[TOC]

## Builder Methods

### `stablehlo::AbsOp`

Creates a new [`stablehlo.abs`](https://openxla.org/stablehlo/spec#abs)
operation.

```c++
MlirOp Abs(MlirOp &operand);
```

### `stablehlo::AddOp`

Creates a new [`stablehlo.add`](https://openxla.org/stablehlo/spec#add)
operation.

```c++
MlirOp Add(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::AfterAllOp`

Creates a new [`stablehlo.after_all`](https://openxla.org/stablehlo/spec#after_all)
operation.

```c++
MlirOp AfterAll(MlirBuilder &builder, ArrayRef<MlirOp> inputs);
```

### `stablehlo::AllGatherOp`

Creates a new [`stablehlo.all_gather`](https://openxla.org/stablehlo/spec#all_gather)
operation.

```c++
SmallVector<MlirOp> AllGather(MlirBuilder &builder, TypeRange resultTypes, ArrayRef<MlirOp> operands, uint64_t all_gather_dim, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::stablehlo::ChannelHandleAttr channel_handle = {}, /*optional*/bool use_global_device_ids = false);
```

### `stablehlo::AllReduceOp`

Creates a new [`stablehlo.all_reduce`](https://openxla.org/stablehlo/spec#all_reduce)
operation.

This operation has a body region built via a callback function.

```c++
SmallVector<MlirOp> AllReduce(MlirBuilder &builder, ArrayRef<MlirOp> operands, const RegionBuilderCallback &computation, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::stablehlo::ChannelHandleAttr channel_handle = {}, /*optional*/bool use_global_device_ids = false);
```

### `stablehlo::AllToAllOp`

Creates a new [`stablehlo.all_to_all`](https://openxla.org/stablehlo/spec#all_to_all)
operation.

```c++
SmallVector<MlirOp> AllToAll(MlirBuilder &builder, ArrayRef<MlirOp> operands, uint64_t split_dimension, uint64_t concat_dimension, uint64_t split_count, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::stablehlo::ChannelHandleAttr channel_handle = {});
```

### `stablehlo::AndOp`

Creates a new [`stablehlo.and`](https://openxla.org/stablehlo/spec#and)
operation.

```c++
MlirOp And(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::Atan2Op`

Creates a new [`stablehlo.atan2`](https://openxla.org/stablehlo/spec#atan2)
operation.

```c++
MlirOp Atan2(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::BatchNormGradOp`

Creates a new [`stablehlo.batch_norm_grad`](https://openxla.org/stablehlo/spec#batch_norm_grad)
operation.

```c++
SmallVector<MlirOp, 3> BatchNormGrad(MlirOp &operand, MlirOp &scale, MlirOp &mean, MlirOp &variance, MlirOp &grad_output, ::llvm::APFloat epsilon, uint64_t feature_index);
```

### `stablehlo::BatchNormInferenceOp`

Creates a new [`stablehlo.batch_norm_inference`](https://openxla.org/stablehlo/spec#batch_norm_inference)
operation.

```c++
MlirOp BatchNormInference(MlirOp &operand, MlirOp &scale, MlirOp &offset, MlirOp &mean, MlirOp &variance, ::llvm::APFloat epsilon, uint64_t feature_index);
```

### `stablehlo::BatchNormTrainingOp`

Creates a new [`stablehlo.batch_norm_training`](https://openxla.org/stablehlo/spec#batch_norm_training)
operation.

```c++
SmallVector<MlirOp, 3> BatchNormTraining(MlirOp &operand, MlirOp &scale, MlirOp &offset, ::llvm::APFloat epsilon, uint64_t feature_index);
```

### `stablehlo::BitcastConvertOp`

Creates a new [`stablehlo.bitcast_convert`](https://openxla.org/stablehlo/spec#bitcast_convert)
operation.

```c++
MlirOp BitcastConvert(Type resultType, MlirOp &operand);
```

### `stablehlo::BroadcastInDimOp`

Creates a new [`stablehlo.broadcast_in_dim`](https://openxla.org/stablehlo/spec#broadcast_in_dim)
operation.

```c++
MlirOp BroadcastInDim(Type resultType, MlirOp &operand, ::llvm::ArrayRef<int64_t> broadcast_dimensions);
```

### `stablehlo::BroadcastOp`

Creates a new [`stablehlo.broadcast`](https://openxla.org/stablehlo/spec#broadcast)
operation.

```c++
MlirOp Broadcast(MlirOp &operand, ::llvm::ArrayRef<int64_t> broadcast_sizes);
```

### `stablehlo::CbrtOp`

Creates a new [`stablehlo.cbrt`](https://openxla.org/stablehlo/spec#cbrt)
operation.

```c++
MlirOp Cbrt(MlirOp &operand, /*optional*/::mlir::stablehlo::ResultAccuracyAttr result_accuracy = {});
```

### `stablehlo::CeilOp`

Creates a new [`stablehlo.ceil`](https://openxla.org/stablehlo/spec#ceil)
operation.

```c++
MlirOp Ceil(MlirOp &operand);
```

### `stablehlo::CholeskyOp`

Creates a new [`stablehlo.cholesky`](https://openxla.org/stablehlo/spec#cholesky)
operation.

```c++
MlirOp Cholesky(MlirOp &a, /*optional*/bool lower = false);
```

### `stablehlo::ClampOp`

Creates a new [`stablehlo.clamp`](https://openxla.org/stablehlo/spec#clamp)
operation.

```c++
MlirOp Clamp(MlirOp &min, MlirOp &operand, MlirOp &max);
```

### `stablehlo::ClzOp`

Creates a new [`stablehlo.count_leading_zeros`](https://openxla.org/stablehlo/spec#count_leading_zeros)
operation.

```c++
MlirOp Clz(MlirOp &operand);
```

### `stablehlo::CollectiveBroadcastOp`

Creates a new [`stablehlo.collective_broadcast`](https://openxla.org/stablehlo/spec#collective_broadcast)
operation.

```c++
MlirOp CollectiveBroadcast(MlirOp &operand, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::stablehlo::ChannelHandleAttr channel_handle = {});
```

### `stablehlo::CollectivePermuteOp`

Creates a new [`stablehlo.collective_permute`](https://openxla.org/stablehlo/spec#collective_permute)
operation.

```c++
MlirOp CollectivePermute(MlirOp &operand, ::mlir::DenseIntElementsAttr source_target_pairs, /*optional*/::mlir::stablehlo::ChannelHandleAttr channel_handle = {});
```

### `stablehlo::CompareOp`

Creates a new [`stablehlo.compare`](https://openxla.org/stablehlo/spec#compare)
operation.

```c++
MlirOp Compare(MlirOp &lhs, MlirOp &rhs, ::mlir::stablehlo::ComparisonDirection comparison_direction, /*optional*/::mlir::stablehlo::ComparisonTypeAttr compare_type = {});
```

### `stablehlo::ComplexOp`

Creates a new [`stablehlo.complex`](https://openxla.org/stablehlo/spec#complex)
operation.

```c++
MlirOp Complex(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::CompositeOp`

Creates a new [`stablehlo.composite`](https://openxla.org/stablehlo/spec#composite)
operation.

```c++
SmallVector<MlirOp> Composite(MlirBuilder &builder, TypeRange resultTypes, ArrayRef<MlirOp> inputs, ::llvm::StringRef name, ::llvm::StringRef decomposition, /*optional*/::mlir::DictionaryAttr composite_attributes = {}, /*optional*/uint32_t version = 0);
```

### `stablehlo::ConcatenateOp`

Creates a new [`stablehlo.concatenate`](https://openxla.org/stablehlo/spec#concatenate)
operation.

```c++
MlirOp Concatenate(MlirBuilder &builder, ArrayRef<MlirOp> inputs, uint64_t dimension);
```

### `stablehlo::ConstantOp`

Creates a new [`stablehlo.constant`](https://openxla.org/stablehlo/spec#constant)
operation.

```c++
MlirOp Constant(MlirBuilder &builder, ::mlir::ElementsAttr value);
```

### `stablehlo::ConvertOp`

Creates a new [`stablehlo.convert`](https://openxla.org/stablehlo/spec#convert)
operation.

```c++
MlirOp Convert(Type resultType, MlirOp &operand);
```

### `stablehlo::ConvolutionOp`

Creates a new [`stablehlo.convolution`](https://openxla.org/stablehlo/spec#convolution)
operation.

```c++
MlirOp Convolution(Type resultType, MlirOp &lhs, MlirOp &rhs, ::mlir::stablehlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::DenseI64ArrayAttr window_strides = {}, /*optional*/::mlir::DenseIntElementsAttr padding = {}, /*optional*/::mlir::DenseI64ArrayAttr lhs_dilation = {}, /*optional*/::mlir::DenseI64ArrayAttr rhs_dilation = {}, /*optional*/::mlir::DenseBoolArrayAttr window_reversal = {}, /*optional*/::mlir::ArrayAttr precision_config = {});
```

### `stablehlo::CosineOp`

Creates a new [`stablehlo.cosine`](https://openxla.org/stablehlo/spec#cosine)
operation.

```c++
MlirOp Cosine(MlirOp &operand, /*optional*/::mlir::stablehlo::ResultAccuracyAttr result_accuracy = {});
```

### `stablehlo::CreateTokenOp`

Creates a new [`stablehlo.create_token`](https://openxla.org/stablehlo/spec#create_token)
operation.

```c++
MlirOp CreateToken(MlirBuilder &builder);
```

### `stablehlo::CrossReplicaSumOp`

Creates a new [`stablehlo.cross-replica-sum`](https://openxla.org/stablehlo/spec#cross-replica-sum)
operation.

```c++
MlirOp CrossReplicaSum(MlirOp &operand, ::mlir::DenseIntElementsAttr replica_groups);
```

### `stablehlo::CustomCallOp`

Creates a new [`stablehlo.custom_call`](https://openxla.org/stablehlo/spec#custom_call)
operation.

```c++
SmallVector<MlirOp> CustomCall(MlirBuilder &builder, TypeRange resultTypes, ArrayRef<MlirOp> inputs, ::llvm::StringRef call_target_name, /*optional*/bool has_side_effect = false, /*optional*/::mlir::Attribute backend_config = {}, /*optional*/::mlir::stablehlo::CustomCallApiVersion api_version = ::mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL, /*optional*/::mlir::ArrayAttr called_computations = {}, /*optional*/::mlir::ArrayAttr operand_layouts = {}, /*optional*/::mlir::ArrayAttr result_layouts = {}, /*optional*/::mlir::ArrayAttr output_operand_aliases = {});
```

### `stablehlo::DivOp`

Creates a new [`stablehlo.divide`](https://openxla.org/stablehlo/spec#divide)
operation.

```c++
MlirOp Div(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::DotGeneralOp`

Creates a new [`stablehlo.dot_general`](https://openxla.org/stablehlo/spec#dot_general)
operation.

```c++
MlirOp DotGeneral(Type resultType, MlirOp &lhs, MlirOp &rhs, ::mlir::stablehlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config = {}, /*optional*/::mlir::stablehlo::DotAlgorithmAttr algorithm = {});
```

### `stablehlo::DotOp`

Creates a new [`stablehlo.dot`](https://openxla.org/stablehlo/spec#dot)
operation.

```c++
MlirOp Dot(Type resultType, MlirOp &lhs, MlirOp &rhs, /*optional*/::mlir::ArrayAttr precision_config = {});
```

### `stablehlo::DynamicBroadcastInDimOp`

Creates a new [`stablehlo.dynamic_broadcast_in_dim`](https://openxla.org/stablehlo/spec#dynamic_broadcast_in_dim)
operation.

```c++
MlirOp DynamicBroadcastInDim(Type resultType, MlirOp &operand, MlirOp &output_dimensions, ::llvm::ArrayRef<int64_t> broadcast_dimensions, /*optional*/::mlir::DenseI64ArrayAttr known_expanding_dimensions = {}, /*optional*/::mlir::DenseI64ArrayAttr known_nonexpanding_dimensions = {});
```

### `stablehlo::DynamicConvOp`

Creates a new [`stablehlo.dynamic_conv`](https://openxla.org/stablehlo/spec#dynamic_conv)
operation.

```c++
MlirOp DynamicConv(Type resultType, MlirOp &lhs, MlirOp &rhs, MlirOp &padding, ::mlir::stablehlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::DenseI64ArrayAttr window_strides = {}, /*optional*/::mlir::DenseI64ArrayAttr lhs_dilation = {}, /*optional*/::mlir::DenseI64ArrayAttr rhs_dilation = {}, /*optional*/::mlir::DenseBoolArrayAttr window_reversal = {}, /*optional*/::mlir::ArrayAttr precision_config = {});
```

### `stablehlo::DynamicGatherOp`

Creates a new [`stablehlo.dynamic_gather`](https://openxla.org/stablehlo/spec#dynamic_gather)
operation.

```c++
MlirOp DynamicGather(MlirOp &operand, MlirOp &start_indices, MlirOp &slice_sizes, ::mlir::stablehlo::GatherDimensionNumbersAttr dimension_numbers, /*optional*/bool indices_are_sorted = false);
```

### `stablehlo::DynamicIotaOp`

Creates a new [`stablehlo.dynamic_iota`](https://openxla.org/stablehlo/spec#dynamic_iota)
operation.

```c++
MlirOp DynamicIota(Type resultType, MlirOp &output_shape, uint64_t iota_dimension);
```

### `stablehlo::DynamicPadOp`

Creates a new [`stablehlo.dynamic_pad`](https://openxla.org/stablehlo/spec#dynamic_pad)
operation.

```c++
MlirOp DynamicPad(Type resultType, MlirOp &operand, MlirOp &padding_value, MlirOp &edge_padding_low, MlirOp &edge_padding_high, MlirOp &interior_padding);
```

### `stablehlo::DynamicReshapeOp`

Creates a new [`stablehlo.dynamic_reshape`](https://openxla.org/stablehlo/spec#dynamic_reshape)
operation.

```c++
MlirOp DynamicReshape(Type resultType, MlirOp &operand, MlirOp &output_shape);
```

### `stablehlo::DynamicSliceOp`

Creates a new [`stablehlo.dynamic_slice`](https://openxla.org/stablehlo/spec#dynamic_slice)
operation.

```c++
MlirOp DynamicSlice(MlirOp &operand, ArrayRef<MlirOp> start_indices, ::llvm::ArrayRef<int64_t> slice_sizes);
```

### `stablehlo::DynamicUpdateSliceOp`

Creates a new [`stablehlo.dynamic_update_slice`](https://openxla.org/stablehlo/spec#dynamic_update_slice)
operation.

```c++
MlirOp DynamicUpdateSlice(MlirOp &operand, MlirOp &update, ArrayRef<MlirOp> start_indices);
```

### `stablehlo::EinsumOp`

Creates a new [`stablehlo.einsum`](https://openxla.org/stablehlo/spec#einsum)
operation.

```c++
MlirOp Einsum(Type resultType, MlirOp &lhs, MlirOp &rhs, ::llvm::StringRef einsum_config);
```

### `stablehlo::ExpOp`

Creates a new [`stablehlo.exponential`](https://openxla.org/stablehlo/spec#exponential)
operation.

```c++
MlirOp Exp(MlirOp &operand, /*optional*/::mlir::stablehlo::ResultAccuracyAttr result_accuracy = {});
```

### `stablehlo::Expm1Op`

Creates a new [`stablehlo.exponential_minus_one`](https://openxla.org/stablehlo/spec#exponential_minus_one)
operation.

```c++
MlirOp Expm1(MlirOp &operand, /*optional*/::mlir::stablehlo::ResultAccuracyAttr result_accuracy = {});
```

### `stablehlo::FftOp`

Creates a new [`stablehlo.fft`](https://openxla.org/stablehlo/spec#fft)
operation.

```c++
MlirOp Fft(MlirOp &operand, ::mlir::stablehlo::FftType fft_type, ::llvm::ArrayRef<int64_t> fft_length);
```

### `stablehlo::FloorOp`

Creates a new [`stablehlo.floor`](https://openxla.org/stablehlo/spec#floor)
operation.

```c++
MlirOp Floor(MlirOp &operand);
```

### `stablehlo::GatherOp`

Creates a new [`stablehlo.gather`](https://openxla.org/stablehlo/spec#gather)
operation.

```c++
MlirOp Gather(MlirOp &operand, MlirOp &start_indices, ::mlir::stablehlo::GatherDimensionNumbersAttr dimension_numbers, ::llvm::ArrayRef<int64_t> slice_sizes, /*optional*/bool indices_are_sorted = false);
```

### `stablehlo::GetDimensionSizeOp`

Creates a new [`stablehlo.get_dimension_size`](https://openxla.org/stablehlo/spec#get_dimension_size)
operation.

```c++
MlirOp GetDimensionSize(MlirOp &operand, uint64_t dimension);
```

### `stablehlo::GetTupleElementOp`

Creates a new [`stablehlo.get_tuple_element`](https://openxla.org/stablehlo/spec#get_tuple_element)
operation.

```c++
MlirOp GetTupleElement(MlirOp &operand, uint32_t index);
```

### `stablehlo::IfOp`

Creates a new [`stablehlo.if`](https://openxla.org/stablehlo/spec#if)
operation.

This operation has a body region built via a callback function.

```c++
SmallVector<MlirOp> If(MlirOp &pred, const RegionBuilderCallback &true_branch, const RegionBuilderCallback &false_branch);
```

### `stablehlo::ImagOp`

Creates a new [`stablehlo.imag`](https://openxla.org/stablehlo/spec#imag)
operation.

```c++
MlirOp Imag(MlirOp &operand);
```

### `stablehlo::InfeedOp`

Creates a new [`stablehlo.infeed`](https://openxla.org/stablehlo/spec#infeed)
operation.

```c++
SmallVector<MlirOp> Infeed(TypeRange resultTypes, MlirOp &token, ::llvm::StringRef infeed_config = "", /*optional*/::mlir::ArrayAttr layout = {});
```

### `stablehlo::IotaOp`

Creates a new [`stablehlo.iota`](https://openxla.org/stablehlo/spec#iota)
operation.

```c++
MlirOp Iota(MlirBuilder &builder, Type resultType, uint64_t iota_dimension);
```

### `stablehlo::IsFiniteOp`

Creates a new [`stablehlo.is_finite`](https://openxla.org/stablehlo/spec#is_finite)
operation.

```c++
MlirOp IsFinite(MlirOp &x);
```

### `stablehlo::Log1pOp`

Creates a new [`stablehlo.log_plus_one`](https://openxla.org/stablehlo/spec#log_plus_one)
operation.

```c++
MlirOp Log1p(MlirOp &operand, /*optional*/::mlir::stablehlo::ResultAccuracyAttr result_accuracy = {});
```

### `stablehlo::LogOp`

Creates a new [`stablehlo.log`](https://openxla.org/stablehlo/spec#log)
operation.

```c++
MlirOp Log(MlirOp &operand, /*optional*/::mlir::stablehlo::ResultAccuracyAttr result_accuracy = {});
```

### `stablehlo::LogisticOp`

Creates a new [`stablehlo.logistic`](https://openxla.org/stablehlo/spec#logistic)
operation.

```c++
MlirOp Logistic(MlirOp &operand, /*optional*/::mlir::stablehlo::ResultAccuracyAttr result_accuracy = {});
```

### `stablehlo::MapOp`

Creates a new [`stablehlo.map`](https://openxla.org/stablehlo/spec#map)
operation.

This operation has a body region built via a callback function.

```c++
MlirOp Map(MlirBuilder &builder, ArrayRef<MlirOp> inputs, const RegionBuilderCallback &computation, ::llvm::ArrayRef<int64_t> dimensions);
```

### `stablehlo::MaxOp`

Creates a new [`stablehlo.maximum`](https://openxla.org/stablehlo/spec#maximum)
operation.

```c++
MlirOp Max(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::MinOp`

Creates a new [`stablehlo.minimum`](https://openxla.org/stablehlo/spec#minimum)
operation.

```c++
MlirOp Min(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::MulOp`

Creates a new [`stablehlo.multiply`](https://openxla.org/stablehlo/spec#multiply)
operation.

```c++
MlirOp Mul(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::NegOp`

Creates a new [`stablehlo.negate`](https://openxla.org/stablehlo/spec#negate)
operation.

```c++
MlirOp Neg(MlirOp &operand);
```

### `stablehlo::NotOp`

Creates a new [`stablehlo.not`](https://openxla.org/stablehlo/spec#not)
operation.

```c++
MlirOp Not(MlirOp &operand);
```

### `stablehlo::OptimizationBarrierOp`

Creates a new [`stablehlo.optimization_barrier`](https://openxla.org/stablehlo/spec#optimization_barrier)
operation.

```c++
SmallVector<MlirOp> OptimizationBarrier(MlirBuilder &builder, ArrayRef<MlirOp> operand);
```

### `stablehlo::OrOp`

Creates a new [`stablehlo.or`](https://openxla.org/stablehlo/spec#or)
operation.

```c++
MlirOp Or(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::OutfeedOp`

Creates a new [`stablehlo.outfeed`](https://openxla.org/stablehlo/spec#outfeed)
operation.

```c++
MlirOp Outfeed(ArrayRef<MlirOp> inputs, MlirOp &token, ::llvm::StringRef outfeed_config = "");
```

### `stablehlo::PadOp`

Creates a new [`stablehlo.pad`](https://openxla.org/stablehlo/spec#pad)
operation.

```c++
MlirOp Pad(MlirOp &operand, MlirOp &padding_value, ::llvm::ArrayRef<int64_t> edge_padding_low, ::llvm::ArrayRef<int64_t> edge_padding_high, ::llvm::ArrayRef<int64_t> interior_padding);
```

### `stablehlo::PartitionIdOp`

Creates a new [`stablehlo.partition_id`](https://openxla.org/stablehlo/spec#partition_id)
operation.

```c++
MlirOp PartitionId(MlirBuilder &builder);
```

### `stablehlo::PopulationCountOp`

Creates a new [`stablehlo.popcnt`](https://openxla.org/stablehlo/spec#popcnt)
operation.

```c++
MlirOp PopulationCount(MlirOp &operand);
```

### `stablehlo::PowOp`

Creates a new [`stablehlo.power`](https://openxla.org/stablehlo/spec#power)
operation.

```c++
MlirOp Pow(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::RealDynamicSliceOp`

Creates a new [`stablehlo.real_dynamic_slice`](https://openxla.org/stablehlo/spec#real_dynamic_slice)
operation.

```c++
MlirOp RealDynamicSlice(Type resultType, MlirOp &operand, MlirOp &start_indices, MlirOp &limit_indices, MlirOp &strides);
```

### `stablehlo::RealOp`

Creates a new [`stablehlo.real`](https://openxla.org/stablehlo/spec#real)
operation.

```c++
MlirOp Real(MlirOp &operand);
```

### `stablehlo::RecvOp`

Creates a new [`stablehlo.recv`](https://openxla.org/stablehlo/spec#recv)
operation.

```c++
SmallVector<MlirOp> Recv(TypeRange resultTypes, MlirOp &token, ::mlir::stablehlo::ChannelHandleAttr channel_handle, /*optional*/bool is_host_transfer = false, /*optional*/::mlir::DenseIntElementsAttr source_target_pairs = {});
```

### `stablehlo::ReduceOp`

Creates a new [`stablehlo.reduce`](https://openxla.org/stablehlo/spec#reduce)
operation.

This operation has a body region built via a callback function.

```c++
SmallVector<MlirOp> Reduce(MlirBuilder &builder, ArrayRef<MlirOp> inputs, ArrayRef<MlirOp> init_values, const RegionBuilderCallback &body, ::llvm::ArrayRef<int64_t> dimensions);
```

### `stablehlo::ReducePrecisionOp`

Creates a new [`stablehlo.reduce_precision`](https://openxla.org/stablehlo/spec#reduce_precision)
operation.

```c++
MlirOp ReducePrecision(MlirOp &operand, uint32_t exponent_bits, uint32_t mantissa_bits);
```

### `stablehlo::ReduceScatterOp`

Creates a new [`stablehlo.reduce_scatter`](https://openxla.org/stablehlo/spec#reduce_scatter)
operation.

This operation has a body region built via a callback function.

```c++
MlirOp ReduceScatter(Type resultType, MlirOp &operand, const RegionBuilderCallback &computation, uint64_t scatter_dimension, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::stablehlo::ChannelHandleAttr channel_handle = {}, /*optional*/bool use_global_device_ids = false);
```

### `stablehlo::ReduceWindowOp`

Creates a new [`stablehlo.reduce_window`](https://openxla.org/stablehlo/spec#reduce_window)
operation.

This operation has a body region built via a callback function.

```c++
SmallVector<MlirOp> ReduceWindow(MlirBuilder &builder, ArrayRef<MlirOp> inputs, ArrayRef<MlirOp> init_values, const RegionBuilderCallback &body, ::llvm::ArrayRef<int64_t> window_dimensions, /*optional*/::mlir::DenseI64ArrayAttr window_strides = {}, /*optional*/::mlir::DenseI64ArrayAttr base_dilations = {}, /*optional*/::mlir::DenseI64ArrayAttr window_dilations = {}, /*optional*/::mlir::DenseIntElementsAttr padding = {});
```

### `stablehlo::RemOp`

Creates a new [`stablehlo.remainder`](https://openxla.org/stablehlo/spec#remainder)
operation.

```c++
MlirOp Rem(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::ReplicaIdOp`

Creates a new [`stablehlo.replica_id`](https://openxla.org/stablehlo/spec#replica_id)
operation.

```c++
MlirOp ReplicaId(MlirBuilder &builder);
```

### `stablehlo::ReshapeOp`

Creates a new [`stablehlo.reshape`](https://openxla.org/stablehlo/spec#reshape)
operation.

```c++
MlirOp Reshape(Type resultType, MlirOp &operand);
```

### `stablehlo::ReturnOp`

Creates a new [`stablehlo.return`](https://openxla.org/stablehlo/spec#return)
operation.

This operation is a Region's Terminator. It can only be called in a RegionBuilder
function callback when constructing the body of an op.

```c++
void Return(RegionBuilder &builder, ArrayRef<MlirOp> results);
```

### `stablehlo::ReverseOp`

Creates a new [`stablehlo.reverse`](https://openxla.org/stablehlo/spec#reverse)
operation.

```c++
MlirOp Reverse(MlirOp &operand, ::llvm::ArrayRef<int64_t> dimensions);
```

### `stablehlo::RngOp`

Creates a new [`stablehlo.rng`](https://openxla.org/stablehlo/spec#rng)
operation.

```c++
MlirOp Rng(MlirOp &a, MlirOp &b, MlirOp &shape, ::mlir::stablehlo::RngDistribution rng_distribution);
```

### `stablehlo::RoundNearestEvenOp`

Creates a new [`stablehlo.round_nearest_even`](https://openxla.org/stablehlo/spec#round_nearest_even)
operation.

```c++
MlirOp RoundNearestEven(MlirOp &operand);
```

### `stablehlo::RoundOp`

Creates a new [`stablehlo.round_nearest_afz`](https://openxla.org/stablehlo/spec#round_nearest_afz)
operation.

```c++
MlirOp Round(MlirOp &operand);
```

### `stablehlo::RsqrtOp`

Creates a new [`stablehlo.rsqrt`](https://openxla.org/stablehlo/spec#rsqrt)
operation.

```c++
MlirOp Rsqrt(MlirOp &operand, /*optional*/::mlir::stablehlo::ResultAccuracyAttr result_accuracy = {});
```

### `stablehlo::ScatterOp`

Creates a new [`stablehlo.scatter`](https://openxla.org/stablehlo/spec#scatter)
operation.

This operation has a body region built via a callback function.

```c++
SmallVector<MlirOp> Scatter(ArrayRef<MlirOp> inputs, MlirOp &scatter_indices, ArrayRef<MlirOp> updates, const RegionBuilderCallback &update_computation, ::mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers, /*optional*/bool indices_are_sorted = false, /*optional*/bool unique_indices = false);
```

### `stablehlo::SelectAndScatterOp`

Creates a new [`stablehlo.select_and_scatter`](https://openxla.org/stablehlo/spec#select_and_scatter)
operation.

This operation has a body region built via a callback function.

```c++
MlirOp SelectAndScatter(MlirOp &operand, MlirOp &source, MlirOp &init_value, const RegionBuilderCallback &select, const RegionBuilderCallback &scatter, /*optional*/::mlir::DenseI64ArrayAttr window_dimensions = {}, /*optional*/::mlir::DenseI64ArrayAttr window_strides = {}, /*optional*/::mlir::DenseIntElementsAttr padding = {});
```

### `stablehlo::SelectOp`

Creates a new [`stablehlo.select`](https://openxla.org/stablehlo/spec#select)
operation.

```c++
MlirOp Select(MlirOp &pred, MlirOp &on_true, MlirOp &on_false);
```

### `stablehlo::SendOp`

Creates a new [`stablehlo.send`](https://openxla.org/stablehlo/spec#send)
operation.

```c++
MlirOp Send(ArrayRef<MlirOp> inputs, MlirOp &token, ::mlir::stablehlo::ChannelHandleAttr channel_handle, /*optional*/bool is_host_transfer = false, /*optional*/::mlir::DenseIntElementsAttr source_target_pairs = {});
```

### `stablehlo::SetDimensionSizeOp`

Creates a new [`stablehlo.set_dimension_size`](https://openxla.org/stablehlo/spec#set_dimension_size)
operation.

```c++
MlirOp SetDimensionSize(MlirOp &operand, MlirOp &size, uint64_t dimension);
```

### `stablehlo::ShiftLeftOp`

Creates a new [`stablehlo.shift_left`](https://openxla.org/stablehlo/spec#shift_left)
operation.

```c++
MlirOp ShiftLeft(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::ShiftRightArithmeticOp`

Creates a new [`stablehlo.shift_right_arithmetic`](https://openxla.org/stablehlo/spec#shift_right_arithmetic)
operation.

```c++
MlirOp ShiftRightArithmetic(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::ShiftRightLogicalOp`

Creates a new [`stablehlo.shift_right_logical`](https://openxla.org/stablehlo/spec#shift_right_logical)
operation.

```c++
MlirOp ShiftRightLogical(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::SignOp`

Creates a new [`stablehlo.sign`](https://openxla.org/stablehlo/spec#sign)
operation.

```c++
MlirOp Sign(MlirOp &operand);
```

### `stablehlo::SineOp`

Creates a new [`stablehlo.sine`](https://openxla.org/stablehlo/spec#sine)
operation.

```c++
MlirOp Sine(MlirOp &operand, /*optional*/::mlir::stablehlo::ResultAccuracyAttr result_accuracy = {});
```

### `stablehlo::SliceOp`

Creates a new [`stablehlo.slice`](https://openxla.org/stablehlo/spec#slice)
operation.

```c++
MlirOp Slice(MlirOp &operand, ::llvm::ArrayRef<int64_t> start_indices, ::llvm::ArrayRef<int64_t> limit_indices, ::llvm::ArrayRef<int64_t> strides);
```

### `stablehlo::SortOp`

Creates a new [`stablehlo.sort`](https://openxla.org/stablehlo/spec#sort)
operation.

This operation has a body region built via a callback function.

```c++
SmallVector<MlirOp> Sort(MlirBuilder &builder, ArrayRef<MlirOp> inputs, const RegionBuilderCallback &comparator, /*optional*/uint64_t dimension = -1, /*optional*/bool is_stable = false);
```

### `stablehlo::SqrtOp`

Creates a new [`stablehlo.sqrt`](https://openxla.org/stablehlo/spec#sqrt)
operation.

```c++
MlirOp Sqrt(MlirOp &operand, /*optional*/::mlir::stablehlo::ResultAccuracyAttr result_accuracy = {});
```

### `stablehlo::SubtractOp`

Creates a new [`stablehlo.subtract`](https://openxla.org/stablehlo/spec#subtract)
operation.

```c++
MlirOp Subtract(MlirOp &lhs, MlirOp &rhs);
```

### `stablehlo::TanOp`

Creates a new [`stablehlo.tan`](https://openxla.org/stablehlo/spec#tan)
operation.

```c++
MlirOp Tan(MlirOp &operand, /*optional*/::mlir::stablehlo::ResultAccuracyAttr result_accuracy = {});
```

### `stablehlo::TanhOp`

Creates a new [`stablehlo.tanh`](https://openxla.org/stablehlo/spec#tanh)
operation.

```c++
MlirOp Tanh(MlirOp &operand, /*optional*/::mlir::stablehlo::ResultAccuracyAttr result_accuracy = {});
```

### `stablehlo::TorchIndexSelectOp`

Creates a new [`stablehlo.torch_index_select`](https://openxla.org/stablehlo/spec#torch_index_select)
operation.

```c++
MlirOp TorchIndexSelect(Type resultType, MlirOp &operand, MlirOp &index, uint64_t dim, uint64_t batch_dims);
```

### `stablehlo::TransposeOp`

Creates a new [`stablehlo.transpose`](https://openxla.org/stablehlo/spec#transpose)
operation.

```c++
MlirOp Transpose(MlirOp &operand, ::llvm::ArrayRef<int64_t> permutation);
```

### `stablehlo::TupleOp`

Creates a new [`stablehlo.tuple`](https://openxla.org/stablehlo/spec#tuple)
operation.

```c++
MlirOp Tuple(MlirBuilder &builder, ArrayRef<MlirOp> val);
```

### `stablehlo::UnaryEinsumOp`

Creates a new [`stablehlo.unary_einsum`](https://openxla.org/stablehlo/spec#unary_einsum)
operation.

```c++
MlirOp UnaryEinsum(Type resultType, MlirOp &operand, ::llvm::StringRef einsum_config);
```

### `stablehlo::UniformDequantizeOp`

Creates a new [`stablehlo.uniform_dequantize`](https://openxla.org/stablehlo/spec#uniform_dequantize)
operation.

```c++
MlirOp UniformDequantize(MlirOp &operand);
```

### `stablehlo::UniformQuantizeOp`

Creates a new [`stablehlo.uniform_quantize`](https://openxla.org/stablehlo/spec#uniform_quantize)
operation.

```c++
MlirOp UniformQuantize(Type resultType, MlirOp &operand);
```

### `stablehlo::WhileOp`

Creates a new [`stablehlo.while`](https://openxla.org/stablehlo/spec#while)
operation.

This operation has a body region built via a callback function.

```c++
SmallVector<MlirOp> While(MlirBuilder &builder, ArrayRef<MlirOp> operand, const RegionBuilderCallback &cond, const RegionBuilderCallback &body);
```

### `stablehlo::XorOp`

Creates a new [`stablehlo.xor`](https://openxla.org/stablehlo/spec#xor)
operation.

```c++
MlirOp Xor(MlirOp &lhs, MlirOp &rhs);
```

## Skipped Operations

Unable to generate builder for the following operations:

 - [`stablehlo.case`](https://openxla.org/stablehlo/spec#case)

 - [`stablehlo.rng_bit_generator`](https://openxla.org/stablehlo/spec#rng_bit_generator)

 - [`stablehlo.triangular_solve`](https://openxla.org/stablehlo/spec#triangular_solve)

