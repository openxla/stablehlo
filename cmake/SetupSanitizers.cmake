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


# setup_sanitizers: sets up the compile and link options necessary to use
#   various sanitizers.
# At the moment we disallow setting up the sanitizers for Python bindings
# since this seems especially broken when building with Clang.
# As of 1/29/2024 here are some of the issues encountered with Python:
# * Clang only links in the sanitizer runtime for the main executable, not
#    for the shared libraries. Since Python is un-sanitized, this is
#    problematic.
# * Linking to the Clang Sanitizer is in the directory
#    /usr/lib/llvm-16/lib/clang/16/lib/linux which needs to be added to RPATH
# * We are using LLVM with RTTI enabled, but installed LLVM has RTTI disabled.
#    This causes a linker error when linking to the sanitizer runtime.
#
# Other MLIR based projects (IREE) seem to have similar issues and just skip
# the Python bindings for sanitization builds.
# TODO(fzakaria): Revisit a better way or if this can be solved.
# @see: https://github.com/google/sanitizers/wiki
function(setup_sanitizers)
    if (NOT STABLEHLO_ENABLE_SANITIZER)
        return()
    endif ()

    string(TOLOWER "${STABLEHLO_ENABLE_SANITIZER}" STABLEHLO_ENABLE_SANITIZER_LOWERCASE)
    if (STABLEHLO_ENABLE_SANITIZER_LOWERCASE STREQUAL "off")
        return()
    endif ()

    if (STABLEHLO_ENABLE_BINDINGS_PYTHON)
        message(FATAL_ERROR "STABLEHLO_ENABLE_SANITIZER must be set to OFF when building Python bindings")
        return()
    endif ()

    if (STABLEHLO_ENABLE_SANITIZER_LOWERCASE STREQUAL "address")
        add_compile_options(-fsanitize=address -fsanitize=undefined -fsanitize=leak -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address -fsanitize=undefined -fsanitize=leak)
    else ()
        message(FATAL_ERROR "Unknown sanitizer type: ${STABLEHLO_ENABLE_SANITIZER}")
    endif ()
endfunction()


