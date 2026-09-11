# Copyright 2026 The StableHLO Authors. All Rights Reserved.
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

"""Defines the `lit_test_suite` macro."""

load("@llvm-project//llvm:lit_test.bzl", "lit_test")

def lit_test_suite(
        name,
        srcs,
        size_per_test = "small",
        data = None,
        tags = None,
        args = None):
    """Defines a `lit_test` for each file in `srcs`; wraps them in a `test_suite`.

    Args:
      name: The name of the test suite.
      srcs: The list/glob of `.mlir` LIT test files included in the test suite.
      size_per_test: The size of the individual tests comprising the suite.
      data: List of labels for the test's data dependencies (i.e. runtime deps).
      tags: List of tags to attach to the tests.
      args: List of extra arguments to pass to the test driver.
    """

    if data == None:
        data = []
    if tags == None:
        tags = []
    if args == None:
        args = []

    tests_in_suite = []

    for src in srcs:
        test_name = ("%s.test" % src).replace("/", "_")
        lit_test(
            name = test_name,  # copybara:comment
            # copybara:uncomment name = src,
            srcs = [src],
            # copybara:uncomment src = src,
            args = args,
            data = data + native.glob(
                ["%s.bc" % src],
                # Allow empty since not all `.mlir` tests have `.mlir.bc` files.
                allow_empty = True,
            ),
            deps = ["@rules_python//python/runfiles"],  # copybara:comment
            # copybara:uncomment driver = "//third_party/llvm/llvm-project/mlir:run_lit.sh",
            tags = tags,
            size = size_per_test,
        )
        tests_in_suite.append(test_name)

    native.test_suite(
        name = name,
        tests = tests_in_suite,
    )
