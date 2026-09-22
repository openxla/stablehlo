# Copyright 2026 The StableHLO Authors.
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

# CMake script that writes the StableHLO source revision to a header, so a
# release-tag build (e.g. "v1.19.0") is distinguishable from an off-tag or
# modified build (a "-<n>-g<hash>" suffix, or "-dirty"). Defines
# STABLEHLO_REVISION when a revision is available and undefines it otherwise,
# so the include always resolves.
#
# Input variables:
#   STABLEHLO_APPEND_VC_REV     - when false, no revision is embedded
#   STABLEHLO_SOURCE_DIR        - path to the StableHLO source tree
#   HEADER_FILE                 - the header file to write
#   STABLEHLO_FORCE_VC_REVISION - optional override (e.g. for tarball/CI builds)

# Handle strange terminals, as GenerateVersionFromVCS.cmake does.
set(ENV{TERM} "dumb")

set(revision "")

if(NOT DEFINED STABLEHLO_APPEND_VC_REV OR STABLEHLO_APPEND_VC_REV)
  if(STABLEHLO_FORCE_VC_REVISION)
    set(revision "${STABLEHLO_FORCE_VC_REVISION}")
  else()
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${STABLEHLO_SOURCE_DIR}/.git")
      execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty --long
        WORKING_DIRECTORY "${STABLEHLO_SOURCE_DIR}"
        RESULT_VARIABLE git_result
        OUTPUT_VARIABLE git_output
        ERROR_QUIET)
      if(git_result EQUAL 0)
        string(STRIP "${git_output}" revision)
      endif()
    endif()
  endif()
endif()

if(revision)
  set(content "#define STABLEHLO_REVISION R\"(${revision})\"\n")
else()
  set(content "#undef STABLEHLO_REVISION\n")
endif()

# Write the header only if it changed, so an unchanged revision does not force
# a rebuild of everything that includes it.
file(WRITE "${HEADER_FILE}.tmp" "${content}")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${HEADER_FILE}.tmp" "${HEADER_FILE}")
file(REMOVE "${HEADER_FILE}.tmp")
