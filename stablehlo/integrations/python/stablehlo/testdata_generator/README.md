# Test Data Generation

This module provides utilities for generating test data for StableHLO modules.
The primary API is the `testdata_generator` function, which automates the
process of creating test cases from existing StableHLO code.

## Usage

```python
def testdata_generator(
    module: ir.Module, args: Sequence[np.ndarray] = []
) -> ir.Module:
```

* `module`: The StableHLO module to generate test data for.
* `args`: (Optional) A sequence of NumPy arrays representing input values for
    the module. If not provided, the function will attempt to extract input
    values from the module itself.

## Example

```python
# Input (module_str)
module_str = """
module {
  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
"""

# Input (args)
args = [
    np.array([1.0, 2.0], dtype=np.float32),
    np.array([3.0, 4.0], dtype=np.float32)
]

# Generate test data
module_output = testdata_generator(module, args)

# Output (module_output)
module_output_str = """
module {
  func.func @main() -> tensor<2xf32> {
    %cst = stablehlo.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
    %cst_0 = stablehlo.constant dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf32>
    %cst_1 = stablehlo.constant dense<[4.000000e+00, 6.000000e+00]> : tensor<2xf32>
    %0 = stablehlo.add %cst, %cst_0 : tensor<2xf32>
    %1 = stablehlo.custom_call @check.eq(%cst_1, %0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    return %0 : tensor<2xf32>
  }
}
"""
```
