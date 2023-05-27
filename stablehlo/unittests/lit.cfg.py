"""Lit configuration to drive test in this repo."""
# Copyright 2023 The StableHLO Authors.
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

# -*- Python -*-
# pylint: disable=undefined-variable

import os

import lit.formats

# Populate Lit configuration with the minimal required metadata.
# Some metadata is populated in lit.site.cfg.py.in.
config.name = 'STABLEHLO_UNITTESTS_SUITE'

# This Lit test suite is used for running StableHLO unit tests.
# It searches config.stablehlo_unittests_dir for all files which end in "Test"
# and then runs them, expecting their output to follow the style of GoogleTest.
# In order to make your unit tests findable by this test suite:
#   1) Make sure to use `add_stablehlo_unittest`, so that they end up in the
#      directory expected by this suite.
#   2) Make sure that the corresponding CMake target name ends with "Test",
#      so that they match the filter of this suite.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "Test")
config.test_source_root = config.stablehlo_unittests_dir
