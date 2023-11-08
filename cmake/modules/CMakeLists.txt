# Since we just use MLIR's cmake functions (add_mlir_library, etc.),
# MLIR_EXPORTS would have the necessary targets to export.
get_property(STABLEHLO_EXPORTS GLOBAL PROPERTY MLIR_EXPORTS)

set(STABLEHLO_CONFIG_CMAKE_DIR "${CMAKE_BINARY_DIR}/lib/cmake/stablehlo")

export(TARGETS ${STABLEHLO_EXPORTS}
  NAMESPACE Stablehlo::
  FILE "${STABLEHLO_CONFIG_CMAKE_DIR}/StablehloTargets.cmake"
)

# Set variables and configure StablehloConfig.cmake for the _build_ tree.
set(STABLEHLO_CONFIG_INCLUDE_DIRS
  "${STABLEHLO_SOURCE_DIR}"
  "${STABLEHLO_BINARY_DIR}"
)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/StablehloConfig.cmake.in"
  "${STABLEHLO_CONFIG_CMAKE_DIR}/StablehloConfig.cmake"
  @ONLY
)

export(EXPORT MLIRTargets
  NAMESPACE Stablehlo::
  FILE "${PROJECT_BINARY_DIR}/StablehloTargets.cmake"
)

# Set variables and configure StablehloConfig.cmake for the _install_ tree.
set(STABLEHLO_CONFIG_INCLUDE_DIRS
  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}"
)

set(STABLEHLO_CONFIG_CMAKE_DIR
  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/stablehlo"
)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/StablehloConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/StablehloConfig.cmake"
  @ONLY
)

# Specify sources and destinations for install files.
install(EXPORT MLIRTargets
  NAMESPACE Stablehlo::
  FILE StablehloTargets.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/stablehlo"
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/StablehloConfig.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/stablehlo"
)

# Copy header files from the source path to the install location.
install(DIRECTORY "${STABLEHLO_SOURCE_DIR}/stablehlo"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING PATTERN "*.h"
)

# Copy (generated) header files from the build path to the install location.
install(DIRECTORY "${STABLEHLO_BINARY_DIR}/stablehlo"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "CMakeFiles" EXCLUDE
)
