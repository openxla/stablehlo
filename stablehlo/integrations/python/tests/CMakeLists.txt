# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 The StableHLO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
add_custom_target(check-stablehlo-python)

function(add_stablehlo_python_test test_name file_name)
add_custom_target(${test_name}
  ${CMAKE_COMMAND} -E env PYTHONPATH=${STABLEHLO_BINARY_DIR}/python_packages/stablehlo ${Python3_EXECUTABLE} ${file_name}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS
    StablehloUnifiedPythonModules
)
add_dependencies(check-stablehlo-python ${test_name})
endfunction()

add_stablehlo_python_test(stablehlo-python-check check.py)
add_stablehlo_python_test(stablehlo-python-chlo chlo.py)
add_stablehlo_python_test(stablehlo-python-smoketest smoketest.py)
add_stablehlo_python_test(stablehlo-python-stablehlo stablehlo.py)
add_stablehlo_python_test(stablehlo-python-vhlo vhlo.py)
add_stablehlo_python_test(stablehlo-python-testdata-generator testdata_generator_test.py)

if(STABLEHLO_ENABLE_PYTHON_TF_TESTS)
  add_stablehlo_python_test(stablehlo-python-stablehlo-to-saved-model stablehlo_to_tf_saved_model_test.py)
endif()


add_dependencies(check-stablehlo-quick check-stablehlo-python)
