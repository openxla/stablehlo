# Copyright 2024 The StableHLO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import dataclasses
from dataclasses import dataclass
import enum
import itertools
import logging
import os
from typing import Any, Dict, Iterator, List
import mlir.dialects.stablehlo as stablehlo
import mlir.ir as ir

try:
  import tensorflow as tf
  from tensorflow.compiler.tf2xla.python import xla as tfxla
except ImportError:
  logging.error(
      'This module is need tensorflow with xla support.\n'
      'Please install tensorflow with `pip install tf-nightly`.\n'
  )
  raise


# Class to specifiy the input or output signature of a stablehlo function.
@dataclass
class VariableSignature:  # either argument or parameters
  shape: List[int]
  dtype: str
  dynamic_dims: List[int] = dataclasses.field(default_factory=list)


# Classes to specify the input type (parameter, argument) of a function.
class VariableType(enum.Enum):
  INPUT_ARG = 'input_arg'
  PARAMETER = 'parameter'


@dataclass
class InputLocation:
  type_: VariableType
  position: int = -1
  name: str = ''

  @classmethod
  def parameter(cls, name: str):
    return cls(type_=VariableType.PARAMETER, name=name)

  @classmethod
  def input_arg(cls, position: int):
    return cls(type_=VariableType.INPUT_ARG, position=position)


# Class to specify stablehlo input specification.
@dataclass
class StableHLOFuncSpec:
  # stablehlo input signature
  input_signature: List[VariableSignature]
  # stablehlo output signature
  output_signature: List[VariableSignature]
  # annotations on stablehlo arguments as constants or variables
  input_locations: List[InputLocation]
  # serialized stablehlo format
  bytecode: bytes
  # map from constant arguments to constant values
  state_dict: Dict[str, Any]


class StableHLOToTFSavedModel:

  def __init__(self, spec: StableHLOFuncSpec):
    self.stablehlo_type_to_tf_type = {
        'i1': 'bool',
        'i8': 'int8',
        'i16': 'i32',
        'i32': 'int32',
        'i64': 'int64',
        'f16': 'float16',
        'f32': 'float32',
        'f64': 'float64',
        'bf16': 'bfloat16',
    }
    self.stablehlo_program = spec

  # Logic to convert stablehlo program to tf saved model

  def _get_shape_with_dynamic(self, signature: VariableSignature):
    shape = copy.copy(signature.shape)
    for i in signature.dynamic_dims:
      shape[i] = None
    return shape

  def _extract_call_parameters(self, args):
    call_args = []
    for loc in self.stablehlo_program.input_locations:
      if str(loc.type_) == str(VariableType.PARAMETER):
        call_args.append(self.stablehlo_program.state_dict[loc.name])
      else:
        call_args.append(args[loc.position])
    return call_args

  def _wrap_as_tf_func(self):
    def inner(*args):
      try:
        Touts = [
            self.stablehlo_type_to_tf_type[sig.dtype]
            for sig in self.stablehlo_program.output_signature
        ]
      except KeyError as e:
        raise KeyError(f'TensorFlow type mapping not found: {e}') from None

      Souts = [
          self._get_shape_with_dynamic(sig)
          for sig in self.stablehlo_program.output_signature
      ]
      call_args = self._extract_call_parameters(args)
      m = tfxla.call_module(
          tuple(call_args),
          version=5,
          Tout=Touts,  # dtype information
          Sout=Souts,  # Shape information
          function_list=[],
          module=self.stablehlo_program.bytecode,
      )
      return m

    return inner

  def _make_tf_function(self):
    return self._wrap_as_tf_func()

  def _make_input_signatures(self) -> Iterator[tf.TensorSpec]:
    input_pos_to_spec = {
        loc.position: spec
        for loc, spec in itertools.chain(
            zip(
                self.stablehlo_program.input_locations,
                self.stablehlo_program.input_signature,
            ),
            [],
        )
        if str(loc.type_) == str(VariableType.INPUT_ARG)
    }
    for i in range(len(input_pos_to_spec)):
      spec = input_pos_to_spec[i]
      shape = self._get_shape_with_dynamic(spec)
      try:
        dtype = getattr(tf, self.stablehlo_type_to_tf_type[spec.dtype])
      except KeyError as e:
        raise KeyError(
            f'TensorFlow type mapping not found for {spec.dtype}: {e}'
        ) from None

      yield tf.TensorSpec(
          shape=shape,
          dtype=dtype,
          name=f'args_{i}',
      )

  def to_tf_saved_model(
      self,
      path: os.PathLike,
      serving_key: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
      function_alias: str = '',
  ) -> None:
    tfm = tf.Module()

    self.stablehlo_program.state_dict = {
        k: tf.Variable(v, trainable=False, name=k)
        for k, v in self.stablehlo_program.state_dict.items()
    }

    input_signatures = list(self._make_input_signatures())

    tfm.f = tf.function(
        self._make_tf_function(), input_signature=input_signatures
    )
    tfm._variables = list(self.stablehlo_program.state_dict.values())
    signatures = {serving_key: tfm.f.get_concrete_function(*input_signatures)}
    save_options = tf.saved_model.SaveOptions(
        function_aliases={
            function_alias: tfm.f,
        }
    )
    tf.saved_model.save(
        tfm,
        path,
        signatures=signatures,
        options=save_options,
    )


# Top level API for stablehlo to tf saved model


def stablehlo_to_tf_saved_model(
    module: ir.Module,
    saved_model_dir: os.PathLike,
    target_version: str,
    input_locations: list = [],
    state_dict: dict = {},
):
  input_signatures = [
      VariableSignature(
          shape=input.shape,
          dtype=str(input.element_type),
          dynamic_dims=[],
      )
      for input in module.body.operations[0].type.inputs
  ]
  output_signature = [
      VariableSignature(
          shape=result.shape,
          dtype=str(result.element_type),
          dynamic_dims=[],
      )
      for result in module.body.operations[0].type.results
  ]

  if input_locations == []:
    for i in range(len(module.body.operations[0].type.inputs)):
      input_locations.append(InputLocation.input_arg(position=i))

  shlo_spec = StableHLOFuncSpec(
      input_signature=input_signatures,
      output_signature=output_signature,
      input_locations=input_locations,
      state_dict=state_dict,
      bytecode=stablehlo.serialize_portable_artifact(module, target_version),
  )

  StableHLOToTFSavedModel(shlo_spec).to_tf_saved_model(saved_model_dir)
