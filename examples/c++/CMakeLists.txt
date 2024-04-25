# Copyright 2024 The StableHLO Authors.
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

# TODO(fzakaria): As we think about adding more tests we should consider
# making this a function to be DRY.
add_executable(example-add ExampleAdd.cpp)
llvm_update_compile_flags(example-add)
target_link_libraries(example-add PRIVATE StablehloOps StablehloReferenceApi)

mlir_check_all_link_libraries(example-add)
add_custom_target(example-add-test
  COMMAND example-add
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Executing example-add to validate it works."
  DEPENDS
    example-add
)
add_dependencies(check-stablehlo-quick example-add-test)
