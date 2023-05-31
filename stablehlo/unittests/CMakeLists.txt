# Copyright 2023 The StableHLO Authors.
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

# StableHLO unit tests are run in a non-obvious way.
# We cannot just say `ninja StablehloUnitTests` - that would simply build
# the corresponding binaries, but not run them.
# In order to actually run the unit tests, we use add_lit_testsuite (it is
# usually used to run FileCheck tests, but it can be used for this purpose too).
# See lit.cfg.py for details of how this test suite works.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
add_lit_testsuite(check-stablehlo-unit "Running the unittests/ suite"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS StablehloUnitTests
  )
add_dependencies(check-stablehlo-quick check-stablehlo-unit)
