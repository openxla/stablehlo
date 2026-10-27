# `func` MLIR Dialect Builder API

[TOC]

## Builder Methods

### `func::CallIndirectOp`

Creates a new [`func.call_indirect`](https://mlir.llvm.org/docs/Dialects/Func/#funccall_indirect-funccall_indirectop)
operation.

```c++
SmallVector<MlirOp> CallIndirect(TypeRange resultTypes, MlirOp &callee, ArrayRef<MlirOp> callee_operands, /*optional*/::mlir::ArrayAttr arg_attrs = {}, /*optional*/::mlir::ArrayAttr res_attrs = {});
```

### `func::ConstantOp`

Creates a new [`func.constant`](https://mlir.llvm.org/docs/Dialects/Func/#funcconstant-funcconstantop)
operation.

```c++
MlirOp Constant(MlirBuilder &builder, Type resultType, ::llvm::StringRef value);
```

### `func::ReturnOp`

Creates a new [`func.return`](https://mlir.llvm.org/docs/Dialects/Func/#funcreturn-funcreturnop)
operation.

This operation is a Region's Terminator. It can only be called in a RegionBuilder
function callback when constructing the body of an op.

```c++
void Return(RegionBuilder &builder, ArrayRef<MlirOp> operands);
```

## Skipped Operations

Unable to generate builder for the following operations:

 - [`func.call`](https://mlir.llvm.org/docs/Dialects/Func/#funccall-funccallop)

 - [`func.func`](https://mlir.llvm.org/docs/Dialects/Func/#funcfunc-funcfuncop)

