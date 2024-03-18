# Examples

This directory should contain a series of examples as a starting point
for how to use StableHLO.

Note: If you have a great example to highlight, we welcome contributions!

* [example-add](./ExampleAdd.cpp): A simple example that demonstrates how to
  * Use the StableHLO library to add two numbers.
  * Interpret a StableHLO program using concrete inputs.

```c++
  // Assume 'module' is an MLIR module with function "main" containing StableHLO
  // operations.
  llvm::outs() << "Program:\n " << module << "\n";

  // Create concrete inputs to be used for interpreting "main".
  auto inputValue1 = mlir::DenseElementsAttr::get(
      tensorType, block_builder.getFloatAttr(tensorType.getElementType(),
                                         static_cast<double>(10)));
  auto inputValue2 = mlir::DenseElementsAttr::get(
      tensorType, block_builder.getFloatAttr(tensorType.getElementType(),
                                       static_cast<double>(20)));
  llvm::outs() << "Inputs: " << inputValue1 << ", " << inputValue2 << "\n";


  mlir::stablehlo::InterpreterConfiguration config;
  auto results = evalModule(module, {inputValue1, inputValue2}, config);
  llvm::outs() << "Output: " << (*results)[0];
```

Output:

```mlir
Program:
module @test_module {
  func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<3x4xf32>
    %1 = stablehlo.add %arg0, %arg1 : tensor<3x4xf32>
    return %1 : tensor<3x4xf32>
  }
}
Inputs: dense<1.000000e+01> : tensor<3x4xf32>, dense<2.000000e+01> : tensor<3x4xf32>
Output: dense<3.000000e+01> : tensor<3x4xf32>
```
