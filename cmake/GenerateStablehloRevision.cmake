# Writes the StableHLO source revision to a header via `git describe`, so a
# release-tag build (e.g. "v1.19.0") is distinguishable from an off-tag or
# modified build (a "-<n>-g<hash>" suffix, or "-dirty"). Writes an undefined
# revision when git information is unavailable.
#
# Input variables:
#   STABLEHLO_SOURCE_DIR        - path to the StableHLO source tree
#   HEADER_FILE                 - the header file to write
#   STABLEHLO_FORCE_VC_REVISION - optional override (e.g. for tarball/CI builds)

set(revision "")

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

if(revision)
  set(content "#define STABLEHLO_REVISION R\"(${revision})\"\n")
else()
  set(content "#undef STABLEHLO_REVISION\n")
endif()

file(WRITE "${HEADER_FILE}.tmp" "${content}")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${HEADER_FILE}.tmp" "${HEADER_FILE}")
file(REMOVE "${HEADER_FILE}.tmp")
