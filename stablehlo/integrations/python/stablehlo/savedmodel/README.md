# StableHLO to Tensorflow SavedModel

`stablehlo_to_tf_saved_model.py` provides the following API to convert a
stablehlo program to TensorFlow SavedModel.

```python
stablehlo_to_tf_saved_model(
        module: mlir.ir.Module,
        saved_model_dir: os.PathLike,
        input_locations: list = [],
        state_dict: dict = {},
)
```

where

* `module`: An StableHLO module.
* `saved_model_dir`: Path to save TF saved-model artifacts.
* `target_version`: Serialization version of StableHLO. Default: current
  stablehlo version.
* `input_locations`: List of input argument types: either it could be a
  parameter with a name associated with it or a positional argument. The
  parameters are generally the weights or biases of a model with pre-trained
  constant values. Default: empty list.
* `state_dict`: Mapping of named input parameters with constants. Default:
  empty list.

For example, to export a simple
[torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
model to TensorFlow SavedModel using the above API, we need the following
arguments to the API.

* `module`

```mlir
 module @linearmodule attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {

  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2,2]{0,1}"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = stablehlo.dot_general %arg2, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
    %3 = stablehlo.add %1, %2 : tensor<2x2xf32>
    return %3 : tensor<2x2xf32>
  }
}
```

* `input_locations`

```python
input_locations = [
    InputLocation.parameter(name='linear_layer.bias'), # bias parameter
    InputLocation.parameter(name='linear_layer.weight'), # weight parameter
    InputLocation.input_arg(position=0), # positional input argument
]
```

* `state_dict`

```python
state_dict = {
    'linear_layer.weight': np.array(
        [[0.19075723, -0.13815854], [0.46516803, 0.12362058]], dtype='float32'
    ),
    'linear_layer.bias': np.array([-0.37076423, 0.03301], dtype='float32'),
}
```

## Python package dependencies

The above API depends on

* MLIR Python bindings: To express an MLIR module.
* TensorFlow: Only used to work with TF saved model artifacts.

## Testing

The repository provides [stablehlo_to_tf_saved_model_test.py](https://github.com/openxla/stablehlo/blob/main/stablehlo/integrations/python/tests/[stablehlo_to_tf_saved_model_test.py)
to test the API and here is how to run it.

```sh
pip install tensorflow-cpu
PYTHONPATH="./build/python_packages/stablehlo" python3 ../../tests/stablehlo_to_tf_saved_model_test.py
```
