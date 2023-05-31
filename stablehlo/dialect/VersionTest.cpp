#include "gtest/gtest.h"
#include "mlir/Bytecode/Encoding.h"
#include "stablehlo/dialect/Version.h"

namespace mlir {
namespace stablehlo {
namespace {

TEST(VersionTest, CurrentBytecodeVersionIsLatestUpstream) {
  auto currentVersion = vhlo::Version::getCurrentVersion();
  auto currentBytecodeVersion = currentVersion.getBytecodeVersion();
  ASSERT_EQ(currentBytecodeVersion.value(), bytecode::kVersion);
}

}  // namespace
}  // namespace stablehlo
}  // namespace mlir
