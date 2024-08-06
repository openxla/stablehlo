/* Copyright 2023 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "stablehlo/tests/CheckOps.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/reference/Element.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/NumPy.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/reference/Types.h"

#define GET_OP_CLASSES
#include "stablehlo/tests/CheckOps.cpp.inc"

namespace mlir {
namespace stablehlo {
namespace check {
namespace {

using SerializedTensorMetadata =
    std::pair</*type=*/std::string, /*path=*/std::string>;

llvm::ErrorOr<SerializedTensorMetadata> extractMetadata(StringRef line) {
  // Parse a CSV record in the form of: probe_id,mlir_type,serialized_path
  constexpr int kNumFields = 3;
  SmallVector<StringRef, kNumFields> fields;
  line.split(fields, ',', kNumFields);

  if (fields.size() != 3) return llvm::errc::invalid_argument;

  return std::make_pair(/*type=*/fields[1].str(), /*path=*/fields[2].str());
}
}  // namespace

//===----------------------------------------------------------------------===//
// Check Dialect Constructor
//===----------------------------------------------------------------------===//

CheckDialect::CheckDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<CheckDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "stablehlo/tests/CheckOps.cpp.inc"
      >();
}

llvm::Error evalExpectAlmostEqConstOp(const Tensor &lhs, ElementsAttr value,
                                      APFloat tolerance) {
  auto rhs = makeTensor(cast<DenseElementsAttr>(value));
  return evalExpectAlmostEqOp(lhs, rhs, tolerance);
}

llvm::Error evalExpectAlmostEqOp(const Tensor &lhs, const Tensor &rhs,
                                 APFloat tolerance) {
  for (auto lhsIt = lhs.index_begin(), rhsIt = rhs.index_begin();
       lhsIt != lhs.index_end(); ++lhsIt, ++rhsIt)
    if (!areApproximatelyEqual(lhs.get(*lhsIt), rhs.get(*rhsIt), tolerance)
             .getBooleanValue())
      return invalidArgument(
          "Element values don't match with tolerance %f: %s (actual) vs %s "
          "(expected) at index %s\n",
          tolerance.convertToDouble(), debugString(lhs.get(*lhsIt)).c_str(),
          debugString(rhs.get(*rhsIt)).c_str(), debugString((*lhsIt)).c_str());

  return llvm::Error::success();
}

llvm::Error evalExpectEqConstOp(const Tensor &lhs, ElementsAttr value) {
  auto rhs = makeTensor(cast<DenseElementsAttr>(value));
  return evalExpectEqOp(lhs, rhs);
}

llvm::Error evalExpectEqOp(const Tensor &lhs, const Tensor &rhs) {
  for (auto lhsIt = lhs.index_begin(), rhsIt = rhs.index_begin();
       lhsIt != lhs.index_end(); ++lhsIt, ++rhsIt)
    if ((lhs.get(*lhsIt) != rhs.get(*rhsIt)).getBooleanValue())
      return invalidArgument(
          "Element values don't match: %s (actual) vs %s (expected) at index "
          "%s\n",
          debugString(lhs.get(*lhsIt)).c_str(),
          debugString(rhs.get(*rhsIt)).c_str(), debugString((*lhsIt)).c_str());

  return llvm::Error::success();
}

// Fetch a previously serialized MLIR type and data filepath given a `probeId`
// and a `probeDir` for a specified `iteration` value from an `index.csv`
// metadata file. If no data is found, returns an error.
static llvm::ErrorOr<SerializedTensorMetadata> getSerializedTensorMetadata(
    StringRef probeId, StringRef probeDir, uint32_t iteration) {
  if (probeDir.empty()) return llvm::errc::invalid_argument;

  llvm::SmallString<128> instrumentationMetadataFile(probeDir);
  llvm::sys::path::append(instrumentationMetadataFile,
                          numpy::kInstrumentationMetadataFilename);
  std::ifstream metadataFile(instrumentationMetadataFile.str().str());

  if (!metadataFile.is_open()) return llvm::errc::io_error;

  std::string line;

  for (uint32_t match = 0; metadataFile >> line && match <= iteration;
       ++match) {
    auto pos = line.find(probeId);

    if (pos != std::string::npos && match == iteration)
      return extractMetadata(line);
  }

  return llvm::errc::bad_address;
}

llvm::Error evalExpectSerializedEqOp(const Tensor &expected, StringRef probeId,
                                     StringRef probeDir, uint32_t iteration) {
  auto serializedMetadata =
      getSerializedTensorMetadata(probeId, probeDir, iteration);

  if (!serializedMetadata)
    return llvm::createStringError(
        serializedMetadata.getError(),
        "Failed to find serialized data for probe %s.", probeId.str().c_str());

  const std::string type = serializedMetadata->first;
  const std::string serializedPath = serializedMetadata->second;

  auto tensor = numpy::deserializeTensor(serializedPath, expected.getType());

  if (!tensor)
    return llvm::createStringError(tensor.getError(),
                                   "Failed to verify serialized tensor %s.",
                                   probeId.str().c_str());

  const std::string expectedType = debugString(expected.getType());
  if (type != expectedType)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Serialized types don't match: %s (actual) "
                                   "vs %s (expected) for probe %s.",
                                   expectedType.c_str(), type.c_str(),
                                   probeId.str().c_str());

  return evalExpectEqOp(expected, *tensor);
}

static uint64_t ULPDifference(APFloat f, APFloat g) {
  if (f.bitwiseIsEqual(g))
    return 0;  // f, g are identical finite or non-finite floats
  if (f.isFinite() && g.isFinite()) {
    auto af = (f.isNegative() ? -f : f).bitcastToAPInt();
    auto ag = (g.isNegative() ? -g : g).bitcastToAPInt();
    assert(af.getBitWidth() <= 64 && ag.getBitWidth() <= 64);
    // a is ULP-distance between exact 0 and abs(f):
    uint64_t a = af.getLimitedValue();
    // b is ULP-distance between exact 0 and abs(g):
    uint64_t b = ag.getLimitedValue();
    if (f.isNegative() != g.isNegative()) {
      return a + b;
    }
    return (a > b ? a - b : b - a);
  }
  // We do not distinguish signaling NaN and quiet NaN values because
  // expected NaN values are typically quiet while functions NaN
  // values depend on implementations and these can be signaling NaN
  // values:
  if (f.isNaN() && g.isNaN()) return 0;
  // Here, one or both operands are non-finite values that are not
  // bitwise-equal. For such cases, we defined the ULP-difference as a
  // maximal possible value because ULP-distance between finite and
  // non-finite values is meaningless in the context of closeness
  // tests.
  return std::numeric_limits<uint64_t>::max();
}

static uint64_t ULPDifference(const Element &e1, const Element &e2) {
  // caller is responsible for ensuring that e1, e2 have both the same
  // float or complex types
  if (isSupportedComplexType(e1.getType())) {
    auto complexLhs = e1.getComplexValue();
    auto complexRhs = e2.getComplexValue();
    return std::max(ULPDifference(complexLhs.real(), complexRhs.real()),
                    ULPDifference(complexLhs.imag(), complexRhs.imag()));
  }
  return ULPDifference(e1.getFloatValue(), e2.getFloatValue());
}

llvm::Error evalExpectCloseOp(const Tensor &actual, const Tensor &expected,
                              uint64_t min_ulp_difference,
                              uint64_t max_ulp_difference) {
  auto type = actual.getElementType();
  if (!isSupportedFloatType(type) && !isSupportedComplexType(type))
    report_fatal_error(invalidArgument("Unsupported element type: %s",
                                       debugString(type).c_str()));
  std::string mismatches;
  llvm::raw_string_ostream output(mismatches);
  constexpr size_t ulp_diff_counter_size = 5;
  int ulp_diff_counter[ulp_diff_counter_size] = {};
  for (auto lhsIt = actual.index_begin(), rhsIt = expected.index_begin();
       lhsIt != actual.index_end(); ++lhsIt, ++rhsIt) {
    auto e1 = actual.get(*lhsIt);
    auto e2 = expected.get(*rhsIt);
    size_t ulp_diff = ULPDifference(e1, e2);
    if (ulp_diff > max_ulp_difference || ulp_diff < min_ulp_difference) {
      output << "\n  index=" << (*lhsIt) << ", actual=" << e1
             << ", expected=" << e2 << ", ULP difference=" << ulp_diff;
    }
    // Gather ULP difference statistics:
    ulp_diff_counter[std::min(ulp_diff, ulp_diff_counter_size - 1)] += 1;
  }
  if (!mismatches.empty()) {
    // Append ULP difference statistics in exception message:
    for (size_t i = 0; i < ulp_diff_counter_size; i++) {
      output << "\nULP difference";
      if (i + 1 == ulp_diff_counter_size)
        output << " >= ";
      else
        output << " == ";
      output << i << " count is " << ulp_diff_counter[i];
    }
    return invalidArgument(
        "Elements values don't match with respect to maximal ULP "
        "difference=%" PRIu64 " limit:%s",
        max_ulp_difference, mismatches.c_str());
  }
  return llvm::Error::success();
}

}  // namespace check
}  // namespace stablehlo
}  // namespace mlir
