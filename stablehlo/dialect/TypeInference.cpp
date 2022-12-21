/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#include "stablehlo/dialect/TypeInference.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/AssemblyFormat.h"
#include "stablehlo/dialect/Base.h"

namespace mlir {
namespace hlo {

//===----------------------------------------------------------------------===//
// Utils for shape functions.
//===----------------------------------------------------------------------===//

// Checks if the vector `nums` has duplicates.
const auto hasDuplicates = [](const ArrayRef<int64_t> nums) {
  llvm::SmallDenseSet<int64_t> set(nums.begin(), nums.end());
  return set.size() != nums.size();
};

// Return true if type1 and type2 are tensors and have the same
// element-type, else return false. With float element-types, ignore comparing
// floating-point precision if ignoreFpPrecision is True.
bool tensorsHaveSameElType(Type type1, Type type2, bool ignoreFpPrecision) {
  auto tensorTy1 = type1.dyn_cast<TensorType>();
  auto tensorTy2 = type2.dyn_cast<TensorType>();

  if (!tensorTy1 || !tensorTy2) return false;

  if (ignoreFpPrecision && tensorTy1.getElementType().isa<FloatType>() &&
      tensorTy2.getElementType().isa<FloatType>())
    return true;

  return tensorTy1.getElementType() == tensorTy2.getElementType();
}

// Return true if type1 and type2 are shape-compatible and have same element
// type. If 'ignoreFpPrecision' is True, then allow floats with different
// precisions while checking element-types.
bool compatibleShapeAndElementType(Type type1, Type type2,
                                   bool ignoreFpPrecision) {
  if (failed(verifyCompatibleShape(type1, type2))) return false;
  return tensorsHaveSameElType(type1.cast<ShapedType>(),
                               type2.cast<ShapedType>(), ignoreFpPrecision);
}

bool verifyCompatibleDims(int64_t dimSize1, int64_t dimSize2) {
  return hlo::isDynamicDimSize(dimSize1) || hlo::isDynamicDimSize(dimSize2) ||
         dimSize1 == dimSize2;
}

// Convert a 1D dense int64 attribute to a list of values.
FailureOr<SmallVector<int64_t>> convert1DAttribute(
    Optional<DenseIntElementsAttr> optionalAttr, Optional<Location> loc,
    StringRef attrName) {
  if (!optionalAttr.has_value()) return SmallVector<int64_t>{};

  DenseIntElementsAttr attr = *optionalAttr;
  auto attrType = attr.getType().cast<RankedTensorType>();
  if (attrType.getRank() != 1)
    return emitOptionalError(loc, "expects the shape of ", attrName,
                             " attribute to be 1-D, but got {",
                             attrType.getShape(), "}.");
  auto values = attr.getValues<int64_t>();
  return SmallVector<int64_t>{values.begin(), values.end()};
}

// Convert a Nx2 dense int64 padding attribute to a list of tuples.
FailureOr<SmallVector<std::pair<int64_t, int64_t>>> convertPaddingAttribute(
    Optional<DenseIntElementsAttr> optionalAttr, Optional<Location> loc) {
  if (!optionalAttr.has_value())
    return SmallVector<std::pair<int64_t, int64_t>>{};

  DenseIntElementsAttr attr = *optionalAttr;
  auto attrType = attr.getType().cast<RankedTensorType>();
  if (attrType.getRank() != 2 || attrType.getShape()[1] != 2)
    return emitOptionalError(
        loc, "expects the shape of padding-attribute to be {N, 2}, but got {",
        attrType.getShape(), "}.");

  auto it = attr.getValues<int64_t>().begin();
  SmallVector<std::pair<int64_t, int64_t>> out(attr.getNumElements() / 2);
  for (auto& item : out) {
    int64_t first = *it;
    ++it;
    int64_t second = *it;
    ++it;
    item = {first, second};
  }
  return out;
}

// Convert a 1D dense bool attribute to a list of values.
FailureOr<SmallVector<bool>> convertWindowReversalAttribute(
    Optional<DenseElementsAttr> optionalAttr, Optional<Location> loc,
    StringRef attrName) {
  if (!optionalAttr.has_value()) return SmallVector<bool>{};

  DenseElementsAttr attr = *optionalAttr;
  auto attrType = attr.getType().cast<RankedTensorType>();
  if (attrType.getRank() != 1)
    return emitOptionalError(loc, "expects the shape of ", attrName,
                             " attribute to be 1-D, but got {",
                             attrType.getShape(), "}.");
  auto values = attr.getValues<bool>();
  return SmallVector<bool>{values.begin(), values.end()};
}

// If a window with the given bound in some dimension is dilated with the given
// dilation factor in that dimension, then the value returned is the bound for
// the array in that dimension after dilation.
//
// For a 1D array with 3 entries 1, 2, 3, a dilation factor of 2 yields a new
// window with values 1, x, 2, x, 3, where x indicates holes left by the
// dilation. So DilatedBound(3, 2) == 5.
int64_t dilatedBound(int64_t bound, int64_t dilation) {
  assert(bound >= 0 && "The dimension to dialate must be >= 0");
  if (bound == 0) return 0;

  // Suppose the array has three entries 123 and the dilation factor is 4. Then
  // the dilated array has 9 entries 1xxx2xxx3. Here, each original entry except
  // the last expands into 4 entries, so that is (bound - 1) * dilation. Then we
  // add 1 to account for the final input element.
  return (bound - 1) * dilation + 1;
}

// Returns the number of valid positions of a window with the given size and
// stride within an array with the given bound. This is the bound of an output
// array with one element per valid position of the window.
//
// For example, for arguments of (bound=5, window_size=2, stride=2), the
// returned value is 2. There are valid positions at offset 0 and offset 2,
// while offset 4 is not valid since the window's last entry would be at 5,
// which is beyond the bound of 5.
int64_t stridedBound(int64_t bound, int64_t windowSize, int64_t stride) {
  assert(windowSize >= 0 && "Expected window size to be >= 0");
  assert(bound >= 0 && "Expected bound to be >= 0");

  if (bound == 0 || windowSize > bound) return 0;

  // Without considering stride, the maximum valid offset is bound -
  // window_size. Taking stride into account, the valid offsets then have the
  // form q * stride for q = 0, ..., Q such that q * stride <= bound -
  // window_size. This implies that Q equals floor(bound - window_size /
  // stride). There are Q + 1 valid values of q, yielding the formula below.
  return (bound - windowSize) / stride + 1;
}

LogicalResult verifyBatchNorm(Optional<Location> location, Value operand,
                              Value scale, int64_t feature_index) {
  auto operandType = operand.getType().cast<RankedTensorType>();
  if (feature_index >= operandType.getRank())
    return emitOptionalError(
        location,
        "expects feature_index to be smaller than the rank of "
        "operand type; got feature_index ",
        feature_index, ", and rank ", operandType.getRank(), ".");

  if (feature_index < 0)
    return emitOptionalError(location, "expects feature_index to be a ",
                             "non-negative number, got ", feature_index, ".");

  // Note: the above checks '0 <= feature-index < operandType.getRank()'
  // imply 'operand_type.getRank() >= 1'.

  const int64_t featureCount = operandType.getDimSize(feature_index);
  const int64_t scaleShape =
      scale.getType().cast<RankedTensorType>().getDimSize(0);
  // As ODS enforces `scale`, `mean`, `variance`, `offset` are AllShapesMatch,
  // this also infers that featureCount is aligned with them.
  if (scaleShape != featureCount)
    return emitOptionalError(
        location,
        "expects the size of scale factor to be same as the "
        "feature count, but the size of scale factor is ",
        dimSizeToString(scaleShape), " and the feature count is ",
        dimSizeToString(featureCount), ".");

  return success();
}

// Verifies various properties of window-attributes (viz., stride, padding,
// lhs_dilation and rhs_dilation) and collects all the window-attributes for
// each kernel spatial dimensions.
FailureOr<SmallVector<WindowDimension>>
verifyWindowAttributesAndInferWindowDimensions(
    ArrayRef<int64_t> windowDimensions, ArrayRef<int64_t> windowStrides,
    ArrayRef<std::pair<int64_t, int64_t>> padding,
    ArrayRef<int64_t> lhsDilation, ArrayRef<int64_t> rhsDilation,
    ArrayRef<bool> windowReversal, Optional<Location> loc) {
  const auto verifySize = [&](const size_t attrSize,
                              StringRef attrName) -> LogicalResult {
    if (attrSize == 0 || attrSize == windowDimensions.size()) return success();
    return emitOptionalError(
        loc, "expects ", attrName,
        " to have same dimension-size as size of window dimensions (",
        windowDimensions.size(), "), but got: ", attrSize, ".");
  };

  if (failed(verifySize(windowStrides.size(), "window-strides")))
    return failure();
  if (failed(verifySize(lhsDilation.size(), "base-dilation factors")))
    return failure();
  if (failed(verifySize(rhsDilation.size(), "window-dilation factors")))
    return failure();
  if (failed(verifySize(padding.size(), "padding-entries"))) return failure();
  if (failed(verifySize(windowReversal.size(), "window-reversal")))
    return failure();

  SmallVector<WindowDimension> window(windowDimensions.size());
  for (size_t i = 0; i < windowDimensions.size(); i++) {
    WindowDimension& dim = window[i];

    dim.size = windowDimensions[i];
    if (!isDynamicDimSize(dim.size) && dim.size <= 0)
      return emitOptionalError(loc,
                               "expects window to have positive value for ", i,
                               "-th window dimension, but got ", dim.size, ".");

    if (!windowStrides.empty()) dim.stride = windowStrides[i];
    if (dim.stride <= 0)
      return emitOptionalError(
          loc, "expects window to have positive stride for ", i,
          "-th window dimension, but got ", dim.stride, ".");

    if (!lhsDilation.empty()) dim.baseDilation = lhsDilation[i];
    if (dim.baseDilation <= 0)
      return emitOptionalError(
          loc, "expects window to have positive base dilation factor for ", i,
          "-th window dimension, but got ", dim.baseDilation, ".");

    if (!rhsDilation.empty()) dim.windowDilation = rhsDilation[i];
    if (dim.windowDilation <= 0)
      return emitOptionalError(
          loc, "expects window to have positive window dilation factor for ", i,
          "-th window dimension, but got ", dim.windowDilation, ".");

    if (!padding.empty()) {
      dim.paddingLow = padding[i].first;
      dim.paddingHigh = padding[i].second;
    }
  }

  return window;
}

// Infer the shape of the output window.
//  Foreach dimension d,
//    output-window-shape[d] =
//            stridedBound(padding_low + dilatedBound(base_shape[d]) +
//            padding_high,
//                         dilatedBound(window_shape[d]))
//      where (padding_low, padding_high) is the padding-pair for d.
SmallVector<int64_t> inferWindowOutputShape(
    const ArrayRef<int64_t> baseShape, const ArrayRef<WindowDimension> window) {
  assert(baseShape.size() == window.size() &&
         "Size of window dimensions must match the size of base shape.");

  SmallVector<int64_t> outputDimensions(window.size());
  for (int64_t i = 0; i < static_cast<int64_t>(window.size()); ++i) {
    if (isDynamicDimSize(baseShape[i]) || isDynamicDimSize(window[i].size)) {
      outputDimensions[i] = ShapedType::kDynamic;
    } else {
      const auto& dim = window[i];

      const int64_t dilatedBase = dilatedBound(baseShape[i], dim.baseDilation);
      const int64_t paddedDilatedBase =
          dim.paddingLow + dilatedBase + dim.paddingHigh;
      const int64_t dilatedWindow = dilatedBound(dim.size, dim.windowDilation);

      outputDimensions[i] =
          stridedBound(paddedDilatedBase, dilatedWindow, dim.stride);
    }
  }

  return outputDimensions;
}

unsigned potentiallyComplexBitwidth(Type type) {
  auto complexTy = type.dyn_cast<ComplexType>();
  return complexTy ? 2 * complexTy.getElementType().getIntOrFloatBitWidth()
                   : type.getIntOrFloatBitWidth();
}

LogicalResult verifyReplicaGroups(Optional<Location> location,
                                  DenseIntElementsAttr replicaGroups,
                                  bool allGroupsMustHaveSameSize,
                                  bool useGlobalDeviceIds,
                                  Optional<size_t> expectedGroupSize) {
  auto replicaGroupType = replicaGroups.getType().cast<RankedTensorType>();

  if (replicaGroupType.getRank() != 2)
    return emitOptionalError(location,
                             "replica groups should be a rank 2 tensor");

  // Revisit the following check in light of #498.
  if (useGlobalDeviceIds &&
      (replicaGroupType.getShape()[0] * replicaGroupType.getShape()[1] == 0)) {
    return emitOptionalError(location,
                             "if `use_global_device_ids` is set, the replica "
                             "groups cannot be empty");
  }

  auto replicaIds = replicaGroups.getValues<int64_t>();
  llvm::SmallSet<int64_t, 8> replicaIdsSeen;
  for (int64_t replicaId : replicaIds) {
    // Replica groups are stored in a 2D tensor. If the op supports non-uniform
    // groups, null replica IDs are stored as -1.
    if (replicaId == -1) {
      if (allGroupsMustHaveSameSize) {
        return emitOptionalError(location, "Invalid replica id -1");
      }
      continue;
    }

    if (!replicaIdsSeen.insert(replicaId).second) {
      return emitOptionalError(location, "replica id #", replicaId,
                               " seen more than once");
    }
  }

  for (size_t id = 0; id < replicaIdsSeen.size(); id++) {
    if (!replicaIdsSeen.contains(id)) {
      return emitOptionalError(location, "replica id #", id,
                               " not seen in replica groups");
    }
  }

  if (allGroupsMustHaveSameSize && expectedGroupSize &&
      (replicaIds.size() / replicaGroupType.getShape()[0] !=
       *expectedGroupSize))
    return emitOptionalError(location, "group size of replica_groups must be ",
                             *expectedGroupSize);

  return success();
}

LogicalResult verifyReduceOpInputsAndInferShape(
    Optional<Location> location, SmallVector<TensorType> inputArgTypes,
    SmallVector<TensorType> initValueTypes, DenseIntElementsAttr dimensions,
    SmallVector<int64_t>& newDimensions, Attribute& encoding) {
  // Check for unranked tensors in input operands.
  uint64_t numInputs = inputArgTypes.size();
  int64_t rankedInputIdx = -1;
  for (uint64_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    if (inputArgTypes[inputIdx].hasRank()) {
      rankedInputIdx = inputIdx;
      break;
    }
  }
  bool allInputsUnranked = (rankedInputIdx == -1);

  if (!allInputsUnranked) {
    for (uint64_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
      if (failed(mlir::verifyCompatibleShape(inputArgTypes[rankedInputIdx],
                                             inputArgTypes[inputIdx]))) {
        return emitOptionalError(
            location, "expects all inputs to have compatible shapes. Shape at",
            " input-index ", inputIdx,
            " is not compatible with shape at input-index ", rankedInputIdx);
      }
    }
  }

  DenseSet<int64_t> dimensionsToReduceSet;
  for (int64_t dimension : dimensions.getValues<int64_t>()) {
    if ((!allInputsUnranked &&
         dimension >= inputArgTypes[rankedInputIdx].getRank()) ||
        dimension < 0) {
      return emitOptionalError(
          location, "Out-of-bounds dimension ", dimension,
          " for input-tensor rank: ", inputArgTypes[rankedInputIdx].getRank());
    }

    if (!dimensionsToReduceSet.insert(dimension).second) {
      return emitOptionalError(location,
                               "Duplicate reduction dimension: ", dimension);
    }
  }

  if (!allInputsUnranked) {
    auto rankedInput = inputArgTypes[rankedInputIdx].cast<RankedTensorType>();

    ArrayRef<int64_t> inputBounds = encodingToBounds(rankedInput.getEncoding());
    SmallVector<int64_t> newBounds;
    for (int inputIdx = 0; inputIdx < rankedInput.getRank(); ++inputIdx) {
      if (!dimensionsToReduceSet.count(inputIdx)) {
        newDimensions.push_back(rankedInput.getDimSize(inputIdx));
        if (!inputBounds.empty()) {
          newBounds.push_back(inputBounds[inputIdx]);
        }
      }
    }
    if (!inputBounds.empty()) {
      encoding = boundsToEncoding(rankedInput.getEncoding(), newBounds);
    }
  }
  return success();
}

// TODO(zhouxin) remove args `allInputsUnranked` and `numInputs`
LogicalResult verifyReducerShape(Optional<Location> loc, Block& block,
                                 ArrayRef<TensorType> inputArgTypes,
                                 ArrayRef<TensorType> initValueTypes,
                                 int64_t numInputs,
                                 ArrayRef<int64_t> allowedDimensions,
                                 bool allInputsUnranked) {
  // Check that the number of reduction-region arguments matches with that of
  // reduce-op's arguments.
  if (static_cast<int64_t>(block.getArguments().size()) != numInputs * 2)
    return emitOptionalError(loc, "Reduction-region must take ", numInputs * 2,
                             " parameters, but takes ",
                             block.getArguments().size(), " parameter(s)");

  // Check if the reduction-region produces non-zero outputs.
  if (block.getTerminator()->getOperands().empty())
    return emitOptionalError(
        loc, "The reduction-region expected to return some value(s)");

  // Check that the reduction-region returns list- of tensors.
  // The number of result-tensors must match the `numInputs`.
  if (static_cast<int64_t>(block.getTerminator()->getOperands().size()) !=
      numInputs)
    return emitOptionalError(loc, "Reduction-region here must produce ",
                             numInputs, " tensors, but produces ",
                             block.getTerminator()->getOperands().size(),
                             " instead");

  SmallVector<TensorType> accumulatorSubShapes;
  for (Value retOperand : block.getTerminator()->getOperands()) {
    auto tensorTy = retOperand.getType().dyn_cast<TensorType>();
    if (!tensorTy)
      return emitOptionalError(loc,
                               "Reduction-region here must produce "
                               "tensor-typed result(s), but "
                               "produces ",
                               retOperand.getType(), " instead");

    accumulatorSubShapes.push_back(tensorTy);
  }

  // Consider typical reduce-* op syntax:
  //
  //      op(I(i), V(j)):
  //       block(BI(i), BV(j)):
  //         ... some computation ...
  //         return(R(i))
  //
  // where
  //  I(i)  : i-th input of op
  //  V(j)  : j-th init-value of op
  //  BI(i) : i-th input of reducer-function
  //  BV(j) : j-th init-value of reducer-function
  //  R(i)  : i-th return-type
  //
  //  Note that: |I(i)| == |V(j)| == |BI(i)| == |BV(j)| == |R(i)|
  //
  //  Here are the type-constraints among V(j), BI(i), BV(j), and R(i).
  //    C1 : Check that BI(i) and R(i) have same shape and element-type.
  //    C2 : Check that BV(j) and R(i) have same shape and element-type.
  //    C3 : Check that V(j) and R(i) have same shape and element-type.
  //
  //  From C1, C2, and C3, we can infer that V(j), BI(i), BV(j), and R(i) all
  //  have compatible shapes and element-types.
  //  The next check, C4, adds constraints on how the type if I(i) is related
  //  to any_of(V(j), BI(i), BV(j), and R(i)), say BV(j);
  //
  //  C4.1 : Check that I(i) and BV(j) have same element-type.
  //  C4.2 : Check that shape of BV(j) is a 'sub-sequence' of
  //         'allowedDimensions'. 'allowedDimensions' is a list of dimensions
  //         which any of BI(i), BV(j), and R(i) is allowed to have.
  for (int64_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    // Check C1.
    if (!compatibleShapeAndElementType(accumulatorSubShapes[inputIdx],
                                       block.getArgument(inputIdx).getType()))
      return emitOptionalError(
          loc, "The type of reduction-region's parameter at index ", inputIdx,
          " is different than the corresponding result type: ",
          block.getArgument(inputIdx).getType(), " vs ",
          accumulatorSubShapes[inputIdx]);

    // Check C2.
    if (!compatibleShapeAndElementType(
            accumulatorSubShapes[inputIdx],
            block.getArgument(numInputs + inputIdx).getType(),
            /*ignoreFpPrecision=*/true))
      return emitOptionalError(
          loc, "The type of reduction-region's parameter at index ",
          numInputs + inputIdx,
          " is different than the corresponding result type: ",
          block.getArgument(numInputs + inputIdx).getType(), " vs ",
          accumulatorSubShapes[inputIdx]);

    // Check C3.
    if (!compatibleShapeAndElementType(accumulatorSubShapes[inputIdx],
                                       initValueTypes[inputIdx],
                                       /*ignoreFpPrecision=*/true))
      return emitOptionalError(
          loc, "The type of reduction-region's result type at index ", inputIdx,
          " differs from the op's corresponding init-value type: ",
          accumulatorSubShapes[inputIdx], " vs ", initValueTypes[inputIdx]);

    // Check C4.1.
    if (!tensorsHaveSameElType(
            inputArgTypes[inputIdx],
            block.getArgument(numInputs + inputIdx).getType(), true))
      return emitOptionalError(
          loc, "The element-type of reduction-region's argument at index ",
          numInputs + inputIdx, " is expected to be ",
          inputArgTypes[inputIdx].getElementType(), ", but got ",
          block.getArgument(numInputs + inputIdx).getType(), " as its type.");

    // Check C4.2.
    Type blockArgType = block.getArgument(numInputs + inputIdx).getType();
    auto blockArgTensorTy = blockArgType.cast<TensorType>();

    if (allInputsUnranked || !blockArgTensorTy.hasRank()) return success();

    auto argShape = blockArgTensorTy.getShape();
    if (argShape.size() > allowedDimensions.size())
      return emitOptionalError(
          loc, "The rank of reduction-region's argument at index ",
          numInputs + inputIdx,
          " is expected to be <= ", allowedDimensions.size(), ", got ",
          argShape.size());

    int64_t argShapeIdx = 0;
    for (int64_t outputShapeIdx = 0;
         outputShapeIdx < static_cast<int64_t>(allowedDimensions.size()) &&
         argShapeIdx < static_cast<int64_t>(argShape.size());
         outputShapeIdx++)
      if (allowedDimensions[outputShapeIdx] == ShapedType::kDynamic ||
          argShape[argShapeIdx] == ShapedType::kDynamic ||
          allowedDimensions[outputShapeIdx] == argShape[argShapeIdx])
        argShapeIdx++;

    if (argShapeIdx != static_cast<int64_t>(argShape.size()))
      return emitOptionalError(
          loc, "The shape of reduction-region's argument at index ",
          numInputs + inputIdx,
          " is not compatible with that of reduce-op's input-parameter "
          "at index ",
          inputIdx);
  }

  return success();
}

LogicalResult verifyReduceWindowOpInputsAndInferWindow(
    Optional<Location> location, SmallVector<TensorType> inputArgTypes,
    SmallVector<TensorType> initValueTypes,
    DenseIntElementsAttr windowDimensions,
    Optional<DenseIntElementsAttr> windowStrides,
    Optional<DenseIntElementsAttr> baseDilations,
    Optional<DenseIntElementsAttr> windowDilations,
    Optional<DenseIntElementsAttr> padding,
    Optional<DenseElementsAttr> windowReversal,
    SmallVector<int64_t>& windowDims,
    SmallVector<WindowDimension>& inferredWindow) {
  // Check for unranked tensors in input operands.
  uint64_t numInputs = inputArgTypes.size();
  int64_t rankedInputIdx = -1;
  for (uint64_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    if (inputArgTypes[inputIdx].hasRank()) {
      rankedInputIdx = inputIdx;
      break;
    }
  }
  bool allInputsUnranked = (rankedInputIdx == -1);

  // P1.
  if (!allInputsUnranked) {
    for (uint64_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
      if (failed(mlir::verifyCompatibleShape(inputArgTypes[rankedInputIdx],
                                             inputArgTypes[inputIdx]))) {
        return emitOptionalError(
            location, "expects all inputs to have compatible shapes. Shape at",
            " input-index ", inputIdx,
            " is not compatible with shape at input-index ", rankedInputIdx);
      }
    }
  }

  // P2.
  auto windowDimsOrErr =
      convert1DAttribute(windowDimensions, location, "window_dimensions");
  if (failed(windowDimsOrErr)) return failure();
  for (const auto inputType : inputArgTypes) {
    if (!inputType.hasRank()) continue;
    if (inputType.getRank() != static_cast<int64_t>((*windowDimsOrErr).size()))
      return emitOptionalError(
          location, "expects window-dimensions size == input rank, but got ",
          "window-dimensions size: ", (*windowDimsOrErr).size(),
          " and input: ", inputType, " with rank = ", inputType.getRank(), ".");
  }

  // P3.
  auto paddingOrErr = convertPaddingAttribute(padding, location);
  if (failed(paddingOrErr)) return failure();

  auto windowStridesOrErr =
      convert1DAttribute(windowStrides, location, "window_strides");
  if (failed(windowStridesOrErr)) return failure();
  auto baseDilationsOrErr =
      convert1DAttribute(baseDilations, location, "base_dilations");
  if (failed(baseDilationsOrErr)) return failure();
  auto windowDilationsOrErr =
      convert1DAttribute(windowDilations, location, "window_dilations");
  if (failed(windowDilationsOrErr)) return failure();
  auto windowReversalOrErr = convertWindowReversalAttribute(
      windowReversal, location, "window_reversal");
  if (failed(windowReversalOrErr)) return failure();

  auto windowOrErr = verifyWindowAttributesAndInferWindowDimensions(
      *windowDimsOrErr, *windowStridesOrErr, *paddingOrErr,
      /*lhsDilation=*/*baseDilationsOrErr,
      /*rhsDilation=*/*windowDilationsOrErr, *windowReversalOrErr, location);
  if (failed(windowOrErr)) return failure();

  windowDims.append(*windowDimsOrErr);
  inferredWindow.append(*windowOrErr);
  return success();
}

// Shape function can be called directly from autogenerated `build()` function,
// which may not guarantee the added region(s) in `odsState.regions` to be
// non-empty. Need check it here to avoid a crash for the ops that need regions
// in type inference, i.e. `IfOp/CaseOp/MapOp`.
LogicalResult verifyRegionNotEmpty(Optional<Location> location,
                                   Region& region) {
  if (region.empty())
    return emitOptionalError(location, "expect non-empty region");
  return success();
}

// Checks:
//  P1. Same sizes for input, kernel and output spatialDims.
//  P2. Spatial and non-spatial dimentions (for input,kernel, &output) should
//      be unique and in range [0, num_dims), where num_dims = rank of input
//      (lhs/rhs) tensors.
//
//  Note that the spatial + non-spatial dimensions may not cover all the
//  dimensions in the range [0,num) because of the presence of 'unknown'
//  dimensions (ref. `printConvolutionDimensions()`)
LogicalResult isSpatialDimensionsValid(
    Value lhs, int64_t inputBatchDimension, int64_t inputFeatureDimension,
    ArrayRef<int64_t> inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    ArrayRef<int64_t> kernelSpatialDimensions, int64_t outputBatchDimension,
    int64_t outputFeatureDimension, ArrayRef<int64_t> outputSpatialDimensions,
    Optional<Location> location) {
  uint64_t spatialDimNum = inputSpatialDimensions.size();
  // P1.
  if ((spatialDimNum != kernelSpatialDimensions.size()) ||
      (spatialDimNum != outputSpatialDimensions.size()))
    return emitOptionalError(location,
                             "expects the same size for input, kernel "
                             "and output spatial-dimensions, but got ",
                             spatialDimNum, ", ",
                             kernelSpatialDimensions.size(), ", and ",
                             outputSpatialDimensions.size(), " resp.");

  // P2.
  SmallVector<int64_t> inputDimNums(spatialDimNum + 2);
  inputDimNums[0] = inputBatchDimension;
  inputDimNums[1] = inputFeatureDimension;
  std::copy(inputSpatialDimensions.begin(), inputSpatialDimensions.end(),
            inputDimNums.begin() + 2);

  SmallVector<int64_t> windowDimNums(spatialDimNum + 2);
  windowDimNums[0] = kernelInputFeatureDimension;
  windowDimNums[1] = kernelOutputFeatureDimension;
  std::copy(kernelSpatialDimensions.begin(), kernelSpatialDimensions.end(),
            windowDimNums.begin() + 2);

  SmallVector<int64_t> OutputDimNums(spatialDimNum + 2);
  OutputDimNums[0] = outputBatchDimension;
  OutputDimNums[1] = outputFeatureDimension;
  std::copy(outputSpatialDimensions.begin(), outputSpatialDimensions.end(),
            OutputDimNums.begin() + 2);

  auto numDims = lhs.getType().cast<RankedTensorType>().getRank();
  const auto inRange = [numDims](int64_t i) { return 0 <= i && i < numDims; };

  if (!llvm::all_of(inputDimNums, inRange) ||
      !llvm::all_of(windowDimNums, inRange) ||
      !llvm::all_of(OutputDimNums, inRange))
    return emitOptionalError(location,
                             "expects input, kernel, and output "
                             "dimension-numbers to be in-range [0, ",
                             numDims, ").");

  if (hasDuplicates(inputDimNums))
    return emitOptionalError(
        location, "expects input dimension-numbers to be unique, got {",
        inputDimNums, "}.");

  if (hasDuplicates(windowDimNums))
    return emitOptionalError(
        location, "expects kernel dimension-numbers to be unique, got {",
        windowDimNums, "}.");

  if (hasDuplicates(OutputDimNums))
    return emitOptionalError(
        location, "expects output dimension-numbers to be unique, got {",
        OutputDimNums, "}.");

  return success();
}

// Verifies the following properties:
//  P1. The input, kernel, and output spatial-dimentions are valid.
//  P2. Given,
//          input-dimensions: b * input-spatial-dims * f
//          kernel-dimensions: kernel-spatial-dims * i * o
//          output-dimensions: b' * out-spatial-dims * f'
//            where b = input-batch-dim
//            where f = input-feature-dim
//            where i = kernel-input-feature-dim
//            where o = kernel-output-feature-dim
//            where b' = output-batch-dim
//            where f' = output-feature-dim
//      Check the following properties w.r.t feature_group_count (fgc) and
//      batch_group_count (bgc).
//        * fgc > 0, bgc > 0 and !(fgc > 1 && bgc > 1)
//        * dim(lhs, b) % bgc == 0
//        * dim(lhs, f) % fgc == 0 and
//          dim(lhs, f) / fgc = dim(rhs, i)
//        * dim(rhs, o) (or dim(output, f')) % bgc == 0 and
//          dim(rhs, o) (or dim(output, f')) % fgc == 0
LogicalResult verifyConvolutionAttributes(
    Value lhs, Value rhs, int64_t inputBatchDimension,
    int64_t inputFeatureDimension, ArrayRef<int64_t> inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    ArrayRef<int64_t> kernelSpatialDimensions, int64_t outputBatchDimension,
    int64_t outputFeatureDimension, ArrayRef<int64_t> outputSpatialDimensions,
    int64_t featureGroupCount, int64_t batchGroupCount,
    Optional<Location> location) {
  // P1.
  if (failed(isSpatialDimensionsValid(
          lhs, inputBatchDimension, inputFeatureDimension,
          inputSpatialDimensions, kernelInputFeatureDimension,
          kernelOutputFeatureDimension, kernelSpatialDimensions,
          outputBatchDimension, outputFeatureDimension, outputSpatialDimensions,
          location)))
    return failure();

  // P2.
  if (featureGroupCount <= 0)
    return emitOptionalError(
        location, "expects feature_group_count to be a positive number, got ",
        featureGroupCount, ".");

  if (batchGroupCount <= 0)
    return emitOptionalError(
        location, "expects batch_group_count to be a positive number, got ",
        batchGroupCount, ".");

  if (batchGroupCount > 1 && featureGroupCount > 1)
    return emitOptionalError(
        location,
        "expects batch_group_count and feature_group_count not to be both "
        "greater than 1. Got ",
        batchGroupCount, " and ", featureGroupCount, " resp.");

  auto lhsType = lhs.getType().cast<RankedTensorType>();
  const int64_t inputFeatures = lhsType.getShape()[inputFeatureDimension];
  const int64_t inputBatch = lhsType.getShape()[inputBatchDimension];

  auto rhsType = rhs.getType().cast<RankedTensorType>();
  const int64_t kernelInputFeatures =
      rhsType.getShape()[kernelInputFeatureDimension];
  const int64_t kernelOutputFeatures =
      rhsType.getShape()[kernelOutputFeatureDimension];

  if (!hlo::isDynamicDimSize(kernelOutputFeatures)) {
    if (kernelOutputFeatures % batchGroupCount != 0)
      return emitOptionalError(
          location, "expects output feature dimension size (",
          kernelOutputFeatures,
          ") to be a multiple of batch_group_count. Got batch_group_count = ",
          batchGroupCount, ".");

    if (kernelOutputFeatures % featureGroupCount != 0)
      return emitOptionalError(location,
                               "expects kernel output feature dimension (",
                               kernelOutputFeatures,
                               ") to be divisible by feature_group_count. For "
                               "feature_group_count = ",
                               featureGroupCount, ".");
  }

  if (!hlo::isDynamicDimSize(inputFeatures)) {
    if (inputFeatures % featureGroupCount != 0)
      return emitOptionalError(location, "expects input feature dimension (",
                               inputFeatures,
                               ") to be a multiple of feature_group_count. Got "
                               "feature_group_count = ",
                               featureGroupCount, ".");

    if (!hlo::isDynamicDimSize(kernelInputFeatures) &&
        inputFeatures / featureGroupCount != kernelInputFeatures)
      return emitOptionalError(
          location, "expects input feature dimension (", inputFeatures,
          ") / "
          "feature_group_count = kernel input feature dimension (",
          kernelInputFeatures,
          "). Got feature_group_count = ", featureGroupCount, ".");
  }

  if (!hlo::isDynamicDimSize(inputBatch) && inputBatch % batchGroupCount != 0)
    return emitOptionalError(location, "expects input batch dimension (",
                             inputBatch,
                             ") to be divisible by "
                             "batch_group_count. Got batch_group_count = ",
                             batchGroupCount, ".");

  return success();
}

// Checks if the precision config has a valid size, if provided.
LogicalResult verifyPrecisionConfig(Optional<Location> loc,
                                    Optional<ArrayAttr> maybeArrayAttr) {
  if (!maybeArrayAttr.has_value()) return success();
  auto arrayAttr = maybeArrayAttr.value();
  return !arrayAttr || arrayAttr.size() == 2 || arrayAttr.empty()
             ? success()
             : emitOptionalError(
                   loc, "expects precision config to be null or of size 2.");
}

LogicalResult inferDotShape(RankedTensorType lhs, RankedTensorType rhs,
                            SmallVector<int64_t>& result) {
  // vector dot vector
  if (1 == lhs.getRank() && 1 == rhs.getRank() &&
      verifyCompatibleDims(lhs.getDimSize(0), rhs.getDimSize(0))) {
    return success();
  }
  // matrix dot vector
  if (2 == lhs.getRank() && 1 == rhs.getRank() &&
      verifyCompatibleDims(lhs.getDimSize(1), rhs.getDimSize(0))) {
    result.push_back(lhs.getDimSize(0));
    return success();
  }
  // vector dot matrix
  if (1 == lhs.getRank() && 2 == rhs.getRank() &&
      verifyCompatibleDims(lhs.getDimSize(0), rhs.getDimSize(0))) {
    result.push_back(rhs.getDimSize(1));
    return success();
  }
  // matrix dot matrix
  if (2 == lhs.getRank() && 2 == rhs.getRank() &&
      verifyCompatibleDims(lhs.getDimSize(1), rhs.getDimSize(0))) {
    result.push_back(lhs.getDimSize(0));
    result.push_back(rhs.getDimSize(1));
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// Shape functions for ops.
//===----------------------------------------------------------------------===//

LogicalResult inferAbsOp(Optional<Location>, Value operand,
                         SmallVectorImpl<Type>& inferredReturnTypes) {
  auto operandTy = operand.getType().cast<ShapedType>();
  Type elementTy = operandTy.getElementType();
  if (auto complexTy = elementTy.dyn_cast<ComplexType>()) {
    elementTy = complexTy.getElementType();
  }

  Type resultTy;
  if (auto rankedOperandTy = operandTy.dyn_cast<RankedTensorType>()) {
    resultTy = RankedTensorType::get(operandTy.getShape(), elementTy,
                                     rankedOperandTy.getEncoding());
  } else if (operandTy.hasRank()) {
    resultTy = RankedTensorType::get(operandTy.getShape(), elementTy);
  } else {
    resultTy = UnrankedTensorType::get(elementTy);
  }
  inferredReturnTypes.push_back(resultTy);
  return success();
}

LogicalResult inferAfterAllOp(Dialect* dialect, Optional<Location> location,
                              SmallVectorImpl<Type>& inferredReturnTypes) {
  auto hloDialect = cast<HloDialectInterface>(dialect);
  inferredReturnTypes.push_back(hloDialect->createTokenType());
  return success();
}

LogicalResult inferAllToAllOp(
    Optional<Location> location, Value operand, int64_t splitDimension,
    int64_t concatDimension, int64_t splitCount,
    DenseIntElementsAttr replicaGroups,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  if (splitCount <= 0)
    return emitOptionalError(location, "AllToAll split_count must be > 0");

  if (failed(hlo::verifyReplicaGroups(location, replicaGroups,
                                      /*allGroupsMustHaveSameSize=*/true,
                                      /*useGlobalDeviceIds=*/false,
                                      splitCount)))
    return failure();

  if (splitDimension < 0)
    return emitOptionalError(location,
                             "AllToAll split_dimension cannot be negative");

  if (concatDimension < 0)
    return emitOptionalError(location,
                             "AllToAll concat_dimension cannot be negative");

  Type operandType = operand.getType();
  RankedTensorType operandRankedType = operandType.dyn_cast<RankedTensorType>();
  if (!operandRankedType) {
    inferredReturnShapes.emplace_back(
        operandType.cast<TensorType>().getElementType());
    return success();
  }

  int64_t inputRank = operandRankedType.getRank();
  if (splitDimension >= inputRank)
    return emitOptionalError(location, "AllToAll split_dimension ",
                             splitDimension,
                             " is out-of-bounds for input rank ", inputRank);
  if (concatDimension >= inputRank)
    return emitOptionalError(location, "AllToAll concat_dimension ",
                             concatDimension,
                             " is out-of-bounds for input rank ", inputRank);

  // If operand is ranked, size of split dimension should be a multiple of split
  // count.
  auto splitDimSize = operandRankedType.getDimSize(splitDimension);
  if (splitDimSize % splitCount != 0)
    return emitOptionalError(
        location, "split dimension has size ", splitDimSize,
        ", expected to be a multiple of split_count ", splitCount);
  SmallVector<int64_t> resultShape(operandRankedType.getShape().begin(),
                                   operandRankedType.getShape().end());
  resultShape[splitDimension] /= splitCount;
  resultShape[concatDimension] *= splitCount;
  inferredReturnShapes.emplace_back(resultShape,
                                    operandRankedType.getElementType());
  return success();
}

LogicalResult inferBatchNormGradOp(
    Optional<Location> location, Value operand, Value scale,
    int64_t featureIndex,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  if (failed(verifyBatchNorm(location, operand, scale, featureIndex)))
    return failure();
  auto operandType = operand.getType().cast<RankedTensorType>();
  inferredReturnShapes.emplace_back(operandType.cast<ShapedType>());

  const int64_t featureCount = operandType.getDimSize(featureIndex);
  SmallVector<int64_t> featureShape{featureCount};
  inferredReturnShapes.emplace_back(featureShape, operandType.getElementType());
  inferredReturnShapes.emplace_back(featureShape, operandType.getElementType());
  return success();
}

LogicalResult inferBatchNormInferenceOp(
    Optional<Location> location, Value operand, Value scale,
    int64_t featureIndex,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  if (failed(verifyBatchNorm(location, operand, scale, featureIndex)))
    return failure();
  auto operandType = operand.getType().cast<RankedTensorType>();
  inferredReturnShapes.emplace_back(operandType.cast<ShapedType>());
  return success();
}

LogicalResult inferBatchNormTrainingOp(
    Optional<Location> location, Value operand, Value scale,
    int64_t featureIndex,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  if (failed(verifyBatchNorm(location, operand, scale, featureIndex)))
    return failure();
  auto operandType = operand.getType().cast<RankedTensorType>();
  inferredReturnShapes.emplace_back(operandType.cast<ShapedType>());

  const int64_t featureCount = operandType.getDimSize(featureIndex);
  SmallVector<int64_t> featureShape{featureCount};
  inferredReturnShapes.emplace_back(featureShape, operandType.getElementType());
  inferredReturnShapes.emplace_back(featureShape, operandType.getElementType());
  return success();
}

LogicalResult inferBroadcastOp(
    Optional<Location> location, Value operand,
    DenseIntElementsAttr broadcastSizes,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  if (!operandType) return failure();

  // TODO: These should be expressed as type constraints.
  auto sizesRank = broadcastSizes.getType().getRank();
  if (sizesRank != 1)
    return emitOptionalError(location, "broadcast_sizes has rank ", sizesRank,
                             " instead of rank 1");

  for (int64_t size : broadcastSizes.getValues<int64_t>())
    if (size < 0)
      return emitOptionalError(location,
                               "Broadcast with negative dimension size ", size);
  SmallVector<int64_t> shapeValues(broadcastSizes.getValues<int64_t>());
  llvm::append_range(shapeValues, operandType.getShape());

  inferredReturnShapes.emplace_back(shapeValues, operandType.getElementType());
  return success();
}

// Used by IfOp and CaseOp
LogicalResult inferConditionalOp(Optional<Location> location,
                                 RegionRange branches,
                                 SmallVectorImpl<Type>& inferredReturnTypes) {
  if (branches.empty())
    return emitOptionalError(location, "expect at least one branch");
  for (auto region : branches)
    if (failed(verifyRegionNotEmpty(location, *region))) return failure();

  ValueTypeRange<OperandRange> branch0ResultTypes =
      branches[0]->front().getTerminator()->getOperandTypes();
  for (unsigned i = 0; i < branches.size(); ++i) {
    Twine branchName = "branch " + Twine(i);
    Region* region = branches[i];
    if (region->getNumArguments() != 0)
      return emitOptionalError(location, branchName,
                               " must have 0 arguments, but found ",
                               region->getNumArguments());

    auto branchResultTypes = region->front().getTerminator()->getOperandTypes();
    if (!hlo::isCompatibleForHloTypeInference(branch0ResultTypes,
                                              branchResultTypes))
      return emitOptionalError(location, "branch 0 and ", branchName,
                               " have mismatched return types: ",
                               branch0ResultTypes, " vs ", branchResultTypes);
  }
  for (auto resultType : branch0ResultTypes)
    inferredReturnTypes.push_back(resultType);
  return success();
}

LogicalResult inferCaseOp(Optional<Location> location, RegionRange branches,
                          SmallVectorImpl<Type>& inferredReturnTypes) {
  return inferConditionalOp(location, branches, inferredReturnTypes);
}

// The following properties are already enforced by the ODS:
//   P0. a.element_type is floating or complex
// We intend to verify the following properties
//   P1. The 'a' argument to Cholesky must have rank >= 2, got shape %s
//   P2. The two minor dimensions of 'a' must have equal size, got %s.
LogicalResult inferCholeskyOp(
    Optional<Location> location, Value a,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  Type aType = a.getType();
  RankedTensorType aRankedType = aType.dyn_cast<RankedTensorType>();
  if (!aRankedType) {
    inferredReturnShapes.emplace_back(
        aType.cast<TensorType>().getElementType());
    return success();
  }

  ArrayRef<int64_t> aShape = aRankedType.getShape();
  if (aShape.size() < 2) {
    return emitOptionalError(
        location, "argument 'a' must have rank >= 2, got shape ", aShape, ".");
  }

  int64_t lastDim = aShape[aShape.size() - 1];
  int64_t penultimateDim = aShape[aShape.size() - 2];
  if (!isDynamicDimSize(lastDim) && !isDynamicDimSize(penultimateDim) &&
      lastDim != penultimateDim) {
    return emitOptionalError(
        location, "minor dimensions of 'a' must have equal size, got shape ",
        aShape, ".");
  }
  inferredReturnShapes.emplace_back(aRankedType.getShape(),
                                    aRankedType.getElementType());
  return success();
}

LogicalResult inferClampOp(
    Optional<Location> location, Value min, Value operand, Value max,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto operandType = operand.getType().cast<RankedTensorType>();
  auto operandShape = operandType.getShape();
  auto minType = min.getType().cast<RankedTensorType>();

  auto minShape = minType.getShape();
  if (failed(verifyCompatibleShape(minType, operandType)) &&
      minType.getRank() != 0) {
    return emitOptionalError(
        location, "min shape [",
        llvm::make_range(minShape.begin(), minShape.end()),
        "] is not scalar and is not compatible to operand shape [",
        llvm::make_range(operandShape.begin(), operandShape.end()), "]");
  }

  auto maxType = max.getType().cast<RankedTensorType>();
  auto maxShape = maxType.getShape();
  if (failed(verifyCompatibleShape(maxType, operandType)) &&
      maxType.getRank() != 0) {
    return emitOptionalError(
        location, "max shape [",
        llvm::make_range(maxShape.begin(), maxShape.end()),
        "] is not scalar and is not compatible to operand shape [",
        llvm::make_range(operandShape.begin(), operandShape.end()), "]");
  }

  inferredReturnShapes.emplace_back(operandType.cast<ShapedType>());
  return success();
}

LogicalResult inferComplexOp(Optional<Location> location, Value lhs,
                             SmallVectorImpl<Type>& inferredReturnTypes) {
  TensorType operandType = lhs.getType().cast<TensorType>();
  ComplexType elementTy = ComplexType::get(operandType.getElementType());
  inferredReturnTypes.push_back(
      hlo::getSameShapeTensorType(operandType, elementTy));
  return success();
}

LogicalResult inferConcatenateOp(Optional<Location> location, ValueRange inputs,
                                 int64_t dimension,
                                 SmallVectorImpl<Type>& inferredReturnTypes) {
  if (dimension < 0)
    return emitOptionalError(location, "dimension ", dimension, " is negative");
  RankedTensorType firstRankedType;
  int firstRankedIndex = -1;
  for (uint64_t i = 0; i < inputs.size(); i++) {
    auto secondType = inputs[i].getType().dyn_cast<ShapedType>();
    if (!secondType.hasRank()) continue;

    if (!firstRankedType) {
      firstRankedType = secondType.cast<RankedTensorType>();
      firstRankedIndex = i;
      if (firstRankedType.getRank() == 0)
        return emitOptionalError(location,
                                 "rank-0 values cannot be concatenated");
      if (dimension >= firstRankedType.getRank())
        return emitOptionalError(location, "dimension ", dimension,
                                 " is out-of-bounds for input rank ",
                                 firstRankedType.getRank());
      continue;
    }
    if (firstRankedType.getRank() != secondType.getRank())
      return emitOptionalError(location, "operands (", firstRankedIndex,
                               ") and (", i, ") do not match rank");

    auto firstShape = firstRankedType.getShape();
    auto secondShape = secondType.getShape();
    for (int d = 0; d < firstRankedType.getRank(); ++d) {
      if (!isDynamicDimSize(firstShape[d]) &&
          !isDynamicDimSize(secondShape[d]) &&
          firstShape[d] != secondShape[d] && d != dimension) {
        return emitOptionalError(
            location, "shapes of operand (", firstRankedIndex, ") and (", i,
            ") do not match at non-concat "
            "index: (",
            llvm::make_range(firstShape.begin(), firstShape.end()), ") != (",
            llvm::make_range(secondShape.begin(), secondShape.end()),
            ") at non-concat index ", d);
      }
    }
  }

  auto elementType = inputs[0].getType().cast<ShapedType>().getElementType();
  if (!firstRankedType) {
    inferredReturnTypes.push_back(UnrankedTensorType::get(elementType));
    return success();
  }

  // Infer the most specific (size, bound) of all dimensions of the return type
  auto rank = firstRankedType.getRank();
  SmallVector<int64_t> inferredSizes(rank, ShapedType::kDynamic);
  SmallVector<int64_t> inferredBounds(rank, ShapedType::kDynamic);
  // Note: for the concatenate dimension, 0 should be the identity element:
  // Any dim size can keep unchanged when concatenated with 0
  inferredSizes[dimension] = 0;
  bool anyInputHaveBounds = false;

  // Note: unranked input types can't be ignored, consider these input types:
  // c0: (<5x?xf32>, <*xf32>) with concat dim 0 should infer <?x?xf32>
  // c1: (<5x?xf32>, <*xf32>) with concat dim 1 should infer <5x?xf32>
  // Instead, they should be replaced with dynamic tensors: tensor<?x...?x>
  for (const auto& it : llvm::enumerate(inputs.getTypes())) {
    RankedTensorType rankedType = it.value().dyn_cast<RankedTensorType>();
    SmallVector<int64_t> bounds;
    if (rankedType)
      bounds = to_vector(encodingToBounds(rankedType.getEncoding()));
    if (!bounds.empty()) anyInputHaveBounds = true;

    for (int dim = 0; dim < rank; ++dim) {
      std::pair<int64_t, int64_t> inferredDimAndBound;

      int64_t leftSize = inferredSizes[dim];
      int64_t rightSize =
          rankedType ? rankedType.getShape()[dim] : ShapedType::kDynamic;
      int64_t leftBound = inferredBounds[dim];
      int64_t rightBound = bounds.empty() ? ShapedType::kDynamic : bounds[dim];
      if (dim == dimension) {
        inferredDimAndBound = inferConcatenatedDimAndBound(
            leftSize, rightSize, leftBound, rightBound);
      } else {
        auto inferredDimAndBoundOrErr = inferMergedDimAndBound(
            location, dim, leftSize, rightSize, leftBound, rightBound);
        if (failed(inferredDimAndBoundOrErr)) return failure();
        inferredDimAndBound = *inferredDimAndBoundOrErr;
      }
      inferredSizes[dim] = inferredDimAndBound.first;
      inferredBounds[dim] = inferredDimAndBound.second;
    }
  }

  inferredReturnTypes.push_back(RankedTensorType::get(
      inferredSizes, elementType,
      boundsToEncoding(
          firstRankedType.getEncoding(),
          // Empty array as argument is an indicator to boundsToEncoding() that
          // there are no bounds at all in inputs, thus sparsity attributes will
          // be included in the return type
          anyInputHaveBounds ? inferredBounds : llvm::ArrayRef<int64_t>({}))));
  return success();
}

LogicalResult inferConstantOp(Optional<Location>, ElementsAttr value,
                              SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(value.getType());
  return success();
}

LogicalResult inferCreateTokenOp(Dialect* dialect, Optional<Location> location,
                                 SmallVectorImpl<Type>& inferredReturnTypes) {
  auto hloDialect = cast<HloDialectInterface>(dialect);
  inferredReturnTypes.push_back(hloDialect->createTokenType());
  return success();
}

LogicalResult inferDynamicSliceOp(
    Optional<Location> location, Value operand, ValueRange startIndices,
    DenseIntElementsAttr sliceSizes,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  int numSliceSizes = sliceSizes.getNumElements();
  int numStartIndices = startIndices.size();
  if (numStartIndices != numSliceSizes)
    return emitOptionalError(location, "has mismatched number of slice sizes (",
                             numSliceSizes, ") and number of start indices (",
                             numStartIndices, ")");
  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  if (!operandType) return failure();

  if (operandType.getRank() != numStartIndices)
    return emitOptionalError(
        location, "has mismatched number of start indices (", numStartIndices,
        ") and the rank of operand (", operandType.getRank(), ")");

  for (int i = 0; i < numSliceSizes; ++i) {
    int64_t sliceSize = sliceSizes.getValues<int64_t>()[i];
    if (sliceSize < 0)
      return emitOptionalError(
          location, "has negative size index to dynamic slice: ", sliceSize);
    if (!operandType.isDynamicDim(i)) {
      int64_t dimSize = operandType.getDimSize(i);
      if (sliceSize > dimSize)
        return emitOptionalError(location, "has slice size ", sliceSize,
                                 " greater than dimension size ", dimSize,
                                 " in dimension ", i, " of operand");
    }
  }

  inferredReturnShapes.emplace_back(sliceSizes.getValues<int64_t>(),
                                    operandType.getElementType());
  return success();
}

LogicalResult inferDynamicUpdateSliceOp(
    Optional<Location> location, Value operand, Value update,
    ValueRange startIndices,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto operandType = operand.getType().cast<ShapedType>();
  auto updateType = update.getType().cast<ShapedType>();

  // (C3)
  if (updateType.hasRank() && operandType.hasRank() &&
      updateType.getRank() != operandType.getRank())
    return emitOptionalError(
        location,
        "update rank does not match operand rank: ", updateType.getRank(),
        " vs ", operandType.getRank(), ".");

  // (C4)
  if (operandType.hasRank() &&
      (int64_t)startIndices.size() != operandType.getRank())
    return emitOptionalError(
        location, "expects number of start_indices to match operand rank: ",
        startIndices.size(), " vs ", operandType.getRank(), ".");

  // (C5)
  if (!startIndices.empty()) {
    auto firstIndexType = startIndices[0].getType().cast<ShapedType>();
    Type firstIndexElement = firstIndexType.getElementType();
    for (auto otherIndex : llvm::drop_begin(startIndices, 1)) {
      auto otherIndexType = otherIndex.getType().cast<ShapedType>();
      Type otherIndexElement = otherIndexType.getElementType();
      if (firstIndexElement != otherIndexElement)
        return emitOptionalError(
            location,
            "start indices must have same element type (encountered mismatch: ",
            firstIndexElement, " vs ", otherIndexElement, ")");
    }
  }

  // (C6)
  if (operandType.hasRank() && updateType.hasRank())
    for (auto [index, dims] : llvm::enumerate(
             llvm::zip(operandType.getShape(), updateType.getShape()))) {
      auto [operandDim, updateDim] = dims;
      if (hlo::isDynamicDimSize(updateDim)) continue;
      if (hlo::isStaticDimSize(operandDim)) {
        if (updateDim < 0 || updateDim > operandDim)
          return emitOptionalError(location, "expects size at dimension ",
                                   index, " of update to be in range [0, ",
                                   operandDim, "]. Got: ", updateDim, ".");
      } else {
        if (updateDim < 0)
          return emitOptionalError(
              location, "expects size at dimension ", index,
              " of update to be non-negative. Got: ", updateDim, ".");
      }
    }

  // (C1)
  if (operandType.hasRank()) {
    inferredReturnShapes.emplace_back(operandType.getShape(),
                                      operandType.getElementType());
  } else {
    inferredReturnShapes.emplace_back(operandType.getElementType());
  }
  return success();
}

LogicalResult inferGetTupleElementOp(
    Optional<Location> location, Value operand, int32_t index,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto operandType = operand.getType().dyn_cast<TupleType>();
  if (!operandType) return failure();
  if (index < 0 || index >= static_cast<int64_t>(operandType.size()))
    return emitOptionalError(location, "index ", index,
                             " is out of bounds of operand with size ",
                             operandType.size());

  inferredReturnTypes.push_back(operandType.getType(index));
  return success();
}

LogicalResult inferIsFiniteOp(MLIRContext* context, Optional<Location>, Value x,
                              SmallVectorImpl<Type>& inferredReturnTypes) {
  auto argTy = x.getType().cast<TensorType>();
  Builder b(context);
  inferredReturnTypes.push_back(
      hlo::getSameShapeTensorType(argTy, b.getI1Type()));
  return success();
}

LogicalResult inferGetDimensionSizeOp(
    MLIRContext* context, Optional<Location> location,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(
      RankedTensorType::get({}, IntegerType::get(context, 32)));
  return success();
}

LogicalResult inferIfOp(Optional<Location> location, RegionRange branches,
                        SmallVectorImpl<Type>& inferredReturnTypes) {
  return inferConditionalOp(location, branches, inferredReturnTypes);
}

LogicalResult inferMapOp(
    Optional<Location> location, ValueRange inputs,
    DenseIntElementsAttr dimensions, Region& computation,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  if (failed(verifyRegionNotEmpty(location, computation))) return failure();

  // Checks if the number of `operands` match the arity of the map `computation`
  // region.
  auto& computationBlock = computation.front();
  auto computationArgs = computationBlock.getArguments();
  if (inputs.size() != computationArgs.size())
    return emitOptionalError(location,
                             "expects number of operands to match the arity of "
                             "map computation, but got: ",
                             inputs.size(), " and ", computationArgs.size());

  // The parameters of computation should all be scalars and match the element
  // type of operands.
  for (const auto& indexedArg : llvm::enumerate(computationArgs)) {
    auto argType = indexedArg.value().getType().dyn_cast<RankedTensorType>();
    if (!argType || argType.getRank() != 0)
      return emitOptionalError(
          location,
          "computation arguments must be 0-rank tensor, but got: arg #",
          indexedArg.index(), " of type ", indexedArg.value().getType());
    auto operandElemTy = inputs[indexedArg.index()]
                             .getType()
                             .cast<TensorType>()
                             .getElementType();
    if (argType.getElementType() != operandElemTy) {
      return emitOptionalError(location,
                               "element type of operands and computation "
                               "arguments must match, but got: ",
                               operandElemTy, " and ",
                               argType.getElementType());
    }
  }

  // Mapped computation must return single output
  auto computationOutputs = computationBlock.getTerminator()->getOperands();
  if (computationOutputs.size() != 1)
    return emitOptionalError(location,
                             "computation must return single output, but got: ",
                             computationOutputs.size());

  // The output of computation must be scalar and have the same element type
  // as op result.
  auto computationOutputType =
      computationOutputs[0].getType().dyn_cast<RankedTensorType>();
  if (!computationOutputType || computationOutputType.getRank() != 0)
    return emitOptionalError(location,
                             "computation must return 0-rank tensor, but got: ",
                             computationOutputs[0].getType());

  // Checks that the requested map dimension numbers are monotonically
  // increasing.
  for (const auto& indexedValue :
       llvm::enumerate(dimensions.getValues<int64_t>())) {
    if (indexedValue.value() != static_cast<int64_t>(indexedValue.index()))
      return emitOptionalError(
          location,
          "requires monotonically increasing dimension numbers, but got: ",
          dimensions);
  }

  // Checks that number of dimensions of operands matches the size of
  // `dimensions` since we currently only support mapping across all
  // dimensions: i.e., scalar map functions.
  ArrayRef<int64_t> resultShape;
  bool allInputsUnranked = true;
  for (auto operand : inputs) {
    auto operandType = operand.getType().cast<TensorType>();
    if (operandType.hasRank()) {
      if (dimensions.size() !=
          static_cast<int64_t>(operandType.getShape().size()))
        return emitOptionalError(
            location,
            "applied to a subset of dimensions currently not supported: "
            "operand dimensions = ",
            operandType.getShape().size(),
            ", requested map dimensions size = ", dimensions.size());
      resultShape = operandType.getShape();
      allInputsUnranked = false;
    }
  }

  if (allInputsUnranked)
    inferredReturnShapes.emplace_back(computationOutputType.getElementType());
  else
    inferredReturnShapes.emplace_back(resultShape,
                                      computationOutputType.getElementType());
  return success();
}

LogicalResult inferPadOp(Optional<Location> location, Value operand,
                         Value paddingValue,
                         DenseIntElementsAttr edgePaddingLow,
                         DenseIntElementsAttr edgePaddingHigh,
                         DenseIntElementsAttr interiorPadding,
                         SmallVectorImpl<Type>& inferredReturnTypes) {
  auto inputType = operand.getType().cast<RankedTensorType>();
  auto padType = paddingValue.getType().cast<RankedTensorType>();

  if (padType.getRank() != 0)
    return emitOptionalError(location,
                             "padding value type should be a rank-0 "
                             "tensor, is rank ",
                             padType.getRank());

  int64_t rank = inputType.getRank();
  if (edgePaddingLow.getType().getNumElements() != rank)
    return emitOptionalError(location, "edge_padding_low length (",
                             edgePaddingLow.getType().getNumElements(),
                             ") must match operand rank (", rank, ")");

  if (edgePaddingHigh.getType().getNumElements() != rank)
    return emitOptionalError(location, "edge_padding_high length (",
                             edgePaddingHigh.getType().getNumElements(),
                             ") must match operand rank (", rank, ")");

  if (interiorPadding.getType().getNumElements() != rank)
    return emitOptionalError(location, "interior_padding length (",
                             interiorPadding.getType().getNumElements(),
                             ") must match operand rank (", rank, ")");

  auto inputShape = inputType.getShape();
  SmallVector<int64_t> resultShape(rank, ShapedType::kDynamic);
  ArrayRef<int64_t> inputBounds = encodingToBounds(inputType.getEncoding());
  SmallVector<int64_t> resultBounds(inputBounds.size(), ShapedType::kDynamic);

  for (int i = 0, e = inputShape.size(); i < e; i++) {
    int64_t paddingLowVal = edgePaddingLow.getValues<APInt>()[i].getSExtValue();
    int64_t paddingHighVal =
        edgePaddingHigh.getValues<APInt>()[i].getSExtValue();
    int64_t paddingInteriorVal =
        interiorPadding.getValues<APInt>()[i].getSExtValue();
    if (paddingInteriorVal < 0)
      return emitOptionalError(
          location,
          "Interior padding cannot be negative: ", paddingInteriorVal);

    bool isStaticDim = !hlo::isDynamicDimSize(inputShape[i]);
    bool isStaticBound =
        !inputBounds.empty() && !hlo::isDynamicDimSize(inputBounds[i]);
    if (isStaticDim || isStaticBound) {
      int64_t operandSizeOrBound = isStaticDim ? inputShape[i] : inputBounds[i];
      int64_t resultSizeOrBound =
          operandSizeOrBound + paddingLowVal + paddingHighVal +
          std::max<int64_t>(operandSizeOrBound - 1, 0LL) * paddingInteriorVal;

      if (resultSizeOrBound < 0) {
        auto sizeOrBound = isStaticDim ? "size" : "bound";
        return emitOptionalError(location, "Padding result in negative ",
                                 sizeOrBound, " for dimension ", i);
      }
      (isStaticDim ? resultShape : resultBounds)[i] = resultSizeOrBound;
    }
  }
  inferredReturnTypes.push_back(RankedTensorType::get(
      resultShape, inputType.getElementType(),
      boundsToEncoding(inputType.getEncoding(), resultBounds)));

  return success();
}

LogicalResult inferOptimizationBarrierOp(
    Optional<Location> location, ValueRange operand,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  for (auto inputArgType : operand.getTypes()) {
    inferredReturnTypes.emplace_back(inputArgType);
  }

  return success();
}

LogicalResult inferOutfeedOp(Dialect* dialect, Optional<Location> location,
                             SmallVectorImpl<Type>& inferredReturnTypes) {
  auto hloDialect = cast<HloDialectInterface>(dialect);
  inferredReturnTypes.push_back(hloDialect->createTokenType());
  return success();
}

LogicalResult inferPartitionIdOp(MLIRContext* context, Optional<Location>,
                                 SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(RankedTensorType::get(
      /*shape=*/{}, IntegerType::get(context, 32, IntegerType::Unsigned)));
  return success();
}

LogicalResult inferRealOp(Optional<Location>, Value operand,
                          SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(
      createRealType(operand.getType().cast<TensorType>()));
  return success();
}

LogicalResult inferReduceOp(
    Optional<Location> location, ValueRange inputs, ValueRange initValues,
    DenseIntElementsAttr dimensions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SmallVector<TensorType> inputArgTypes{llvm::map_range(
      inputs.getTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); })};
  SmallVector<TensorType> initValueTypes{llvm::map_range(
      initValues.getTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); })};

  SmallVector<int64_t> newDimensions;
  Attribute encoding;
  if (failed(verifyReduceOpInputsAndInferShape(location, inputArgTypes,
                                               initValueTypes, dimensions,
                                               newDimensions, encoding)))
    return failure();

  for (uint64_t inputIdx = 0; inputIdx < inputs.size(); ++inputIdx) {
    TensorType inputType = inputArgTypes[inputIdx];
    Type elementType = inputType.getElementType();
    if (inputType.hasRank())
      inferredReturnShapes.emplace_back(newDimensions, elementType, encoding);
    else
      inferredReturnShapes.emplace_back(elementType);
  }

  return success();
}

LogicalResult inferReduceWindowOp(
    Optional<Location> location, ValueRange inputs, ValueRange initValues,
    DenseIntElementsAttr windowDimensions,
    Optional<DenseIntElementsAttr> windowStrides,
    Optional<DenseIntElementsAttr> baseDilations,
    Optional<DenseIntElementsAttr> windowDilations,
    Optional<DenseIntElementsAttr> padding,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SmallVector<TensorType> inputArgTypes{llvm::map_range(
      inputs.getTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); })};
  SmallVector<TensorType> initValueTypes{llvm::map_range(
      initValues.getTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); })};

  SmallVector<int64_t> windowDims;
  SmallVector<WindowDimension> inferredWindow;
  if (failed(verifyReduceWindowOpInputsAndInferWindow(
          location, inputArgTypes, initValueTypes, windowDimensions,
          windowStrides, baseDilations, windowDilations, padding,
          /*windowReversal=*/std::nullopt, windowDims, inferredWindow)))
    return failure();

  for (size_t i = 0; i < inputArgTypes.size(); ++i) {
    if (!inputArgTypes[i].hasRank())
      inferredReturnShapes.emplace_back(inputArgTypes[i].getElementType());
    else
      inferredReturnShapes.emplace_back(
          inferWindowOutputShape(inputArgTypes[i].getShape(), inferredWindow),
          inputArgTypes[i].getElementType());
  }

  return success();
}

LogicalResult inferReturnOp(Optional<Location>, SmallVectorImpl<Type>&) {
  return success();
}

LogicalResult inferScatterOp(Optional<Location>, ValueRange inputs,
                             SmallVectorImpl<Type>& inferredReturnTypes) {
  llvm::append_range(inferredReturnTypes, inputs.getTypes());
  return success();
}

LogicalResult inferSelectOp(
    Optional<Location> location, Value pred, Value onTrue, Value onFalse,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto predType = pred.getType().cast<ShapedType>();
  auto trueType = onTrue.getType().cast<ShapedType>();
  auto falseType = onFalse.getType().cast<ShapedType>();

  // The operands `onTrue` and `onFalse` should have compatible types, i.e.,
  //   (a) have the same element type, and
  //   (b) have compatible shapes (i.e. the same shape and/or at least one
  //       dynamic shape)
  if (!hlo::compatibleShapeAndElementType(trueType, falseType))
    return emitOptionalError(
        location, "requires compatible types for non-predicate operands");

  // The predicate, if not-scalar, should have the same shape as the remaining
  // operands.
  bool predCannotBeScalar = predType.hasRank() && predType.getRank() != 0;
  if (predCannotBeScalar)
    if (failed(verifyCompatibleShape(predType, trueType)))
      return emitOptionalError(location,
                               "requires the same shape for all operands");

  // The output shape should be derived from the most specific parts of the
  // `onTrue` and `onFalse` (see documentation for details).
  SmallVector<Type> inferredReturnTypes;
  return hlo::inferMostSpecificTypeComponents(location, {trueType, falseType},
                                              inferredReturnShapes);
}

LogicalResult inferSelectAndScatterOp(
    Value operand, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(operand.getType());
  return success();
}

LogicalResult inferSendOp(Dialect* dialect, Optional<Location> location,
                          SmallVectorImpl<Type>& inferredReturnTypes) {
  auto hloDialect = cast<HloDialectInterface>(dialect);
  inferredReturnTypes.push_back(hloDialect->createTokenType());
  return success();
}

// The following properties are already enforced by the ODS:
//  type(start_indices) == type(limit_indices) == type(strides).
// Verify the following properties:
//  P1. Verify rank(start_indices) == 1.
//  P2. Verify size(start_indices) == rank(operand).
//  P3~5. Verify 0 <= start_indices[i] <= limit_indices[i] <= shape(operand)[i].
//  P6. Verify stride[i] > 0.
// Note: for P4, use the bound size than dim size for bounded dynamism case.
LogicalResult inferSliceOp(Optional<Location> location, Value operand,
                           DenseIntElementsAttr startIndices,
                           DenseIntElementsAttr limitIndices,
                           DenseIntElementsAttr strides,
                           SmallVectorImpl<Type>& inferredReturnTypes) {
  Type ty = operand.getType();
  RankedTensorType rankedTy = ty.dyn_cast<RankedTensorType>();
  if (!rankedTy) {
    // The operand type is unranked, so the best we can infer for the result
    // type is an unranked tensor with the same element type as the operand
    // type.
    inferredReturnTypes.assign({ty});
    return success();
  }

  ShapedType attrTy = startIndices.getType();
  // P1.
  // Note: ODS has type(start_indices) == type(limit_indices) == type(strides)
  // So this implies rank(limit_indices) == rank(strides) == 1 also.
  if (attrTy.getRank() != 1) {
    return emitOptionalError(location, "start_indices has rank ",
                             attrTy.getRank(), " instead of required rank 1");
  }

  // P2.
  int64_t rank = rankedTy.getRank();
  if (attrTy.getNumElements() != rank) {
    return emitOptionalError(
        location, "the number of elements in start_indices (",
        attrTy.getNumElements(), ") does not match the rank of the operand (",
        rank, ")");
  }

  SmallVector<int64_t, 4> start(startIndices.getValues<int64_t>());
  SmallVector<int64_t, 4> limit(limitIndices.getValues<int64_t>());
  SmallVector<int64_t, 4> strideVals(strides.getValues<int64_t>());

  ArrayRef<int64_t> inputBounds = encodingToBounds(rankedTy.getEncoding());
  SmallVector<int64_t> shape(rank, ShapedType::kDynamic);
  SmallVector<int64_t> resultBounds(inputBounds.size(), ShapedType::kDynamic);

  for (int64_t i = 0, e = rank; i != e; i++) {
    // P3.
    if (start[i] < 0)
      return emitOptionalError(location, "negative start index ", start[i],
                               " in dimension ", i);

    // P4.
    bool isStaticDim = !hlo::isDynamicDimSize(rankedTy.getDimSize(i));
    bool isStaticBound =
        !inputBounds.empty() && !hlo::isDynamicDimSize(inputBounds[i]);
    if (isStaticDim || isStaticBound) {
      int64_t operandSizeOrBound =
          isStaticDim ? rankedTy.getDimSize(i) : inputBounds[i];
      StringRef sizeOrBound = isStaticDim ? "size" : "bound";
      if (limit[i] > operandSizeOrBound)
        return emitOptionalError(location, "limit index ", limit[i],
                                 " is larger than dimension ", sizeOrBound, " ",
                                 operandSizeOrBound, " in dimension ", i);
    }

    // P5.
    if (start[i] > limit[i])
      return emitOptionalError(location, "start index ", start[i],
                               " is larger than limit index ", limit[i],
                               " in dimension ", i);
    // P6.
    if (strideVals[i] <= 0)
      return emitOptionalError(location, "stride must be positive but got ",
                               strideVals[i], " in dimension ", i);

    shape[i] = static_cast<int64_t>(
        llvm::divideCeil(limit[i] - start[i], strideVals[i]));
  }

  inferredReturnTypes.push_back(RankedTensorType::get(
      shape, rankedTy.getElementType(),
      boundsToEncoding(rankedTy.getEncoding(), resultBounds)));
  return success();
}

LogicalResult inferSortOp(
    Optional<Location>, ValueRange inputs,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  for (auto resultType : inputs.getTypes())
    inferredReturnShapes.emplace_back(resultType.cast<ShapedType>());
  return success();
}

LogicalResult inferTransposeOp(Optional<Location> loc, Value operand,
                               DenseIntElementsAttr permutation,
                               SmallVectorImpl<Type>& inferredReturnTypes) {
  auto type = operand.getType();
  auto rankedTy = type.dyn_cast<RankedTensorType>();
  if (!rankedTy) {
    inferredReturnTypes.emplace_back(type);
    return success();
  }
  int64_t rank = rankedTy.getRank();
  if (permutation.getType().getRank() != 1)
    return emitOptionalError(loc, "TransposeOp permutation has rank ",
                             permutation.getType().getRank(),
                             " instead of rank 1");

  if (permutation.size() != rank)
    return emitOptionalError(loc, "TransposeOp operand rank ", rank,
                             " does not match permutation size ",
                             permutation.size());

  std::vector<int64_t> range(rank);
  std::iota(range.begin(), range.end(), 0);
  if (!std::is_permutation(range.begin(), range.end(), permutation.begin()))
    return emitOptionalError(loc,
                             "attribute permutation must be a permutation"
                             " of [",
                             range, "] but got ", permutation);

  ArrayRef<int64_t> inputBounds = encodingToBounds(rankedTy.getEncoding());
  SmallVector<int64_t> resultShape;
  SmallVector<int64_t> resultBounds;
  ArrayRef<int64_t> inputShape = rankedTy.getShape();
  for (int64_t dim : permutation.getValues<int64_t>()) {
    resultShape.push_back(inputShape[dim]);
    if (!inputBounds.empty()) {
      resultBounds.push_back(inputBounds[dim]);
    }
  }

  inferredReturnTypes.push_back(RankedTensorType::get(
      resultShape, rankedTy.getElementType(),
      boundsToEncoding(rankedTy.getEncoding(), resultBounds)));
  return success();
}

LogicalResult inferTriangularSolveOp(
    Optional<Location> location, Value a, Value b, bool leftSide,
    bool isTransposeAInvalid,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  // ODS enforces that a and b are of same element type: float or complex.
  auto elementType = a.getType().cast<ShapedType>().getElementType();
  auto aType = a.getType().dyn_cast<RankedTensorType>();
  if (!aType) {
    inferredReturnShapes.emplace_back(elementType);
    return success();
  }

  auto aRank = aType.getRank();
  if (aRank < 2)
    return emitOptionalError(
        location, "operand 'a' must have rank >= 2, but got ", aType);

  if (aType.getDimSize(aRank - 2) != aType.getDimSize(aRank - 1))
    return emitOptionalError(location,
                             "two minor dimensions of operand 'a' must have "
                             "equal size, but got ",
                             aType);

  auto bType = b.getType().dyn_cast<RankedTensorType>();
  if (!bType) {
    inferredReturnShapes.emplace_back(elementType);
    return success();
  }

  auto bRank = bType.getRank();
  if (aRank != bRank)
    return emitOptionalError(location,
                             "operands must have equal rank, but got ", aType,
                             " and ", bType);

  // The shared dimension of a and b should match.
  if (aType.getDimSize(aRank - 1) !=
      bType.getDimSize(bRank - (leftSide ? 2 : 1)))
    return emitOptionalError(location,
                             "shared dimension of operands 'a' and 'b' does "
                             "not match, but got ",
                             aType, " and ", bType);

  // The leading batch dimensions of a and b must be equal.
  auto aBatchDims = aType.getShape().drop_back(2);
  auto bBatchDims = bType.getShape().drop_back(2);
  if (aBatchDims != bBatchDims)
    return emitOptionalError(
        location,
        "leading batch dimensions of the operands must be same, but got ",
        aType, " and ", bType);

  if (isTransposeAInvalid)
    return emitOptionalError(
        location, "Invalid transpose option value for triangular solve");

  inferredReturnShapes.emplace_back(bType.cast<ShapedType>());
  return success();
}

LogicalResult inferTupleOp(MLIRContext* context, Optional<Location>,
                           ValueRange val,
                           SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(TupleType::get(context, val.getTypes()));
  return success();
}

LogicalResult inferWhileOp(Optional<Location>, ValueRange operand,
                           SmallVectorImpl<Type>& inferredReturnTypes) {
  for (const auto& resultType : operand.getType())
    inferredReturnTypes.push_back(resultType);
  return success();
}

//===----------------------------------------------------------------------===//
// Verifiers for ops.
//===----------------------------------------------------------------------===//

LogicalResult verifyAllReduceOp(Optional<Location> location, Value operand,
                                DenseIntElementsAttr replicaGroups,
                                bool useGlobalDeviceIds, Region& computation) {
  if (failed(hlo::verifyReplicaGroups(location, replicaGroups,
                                      /*allGroupsMustHaveSameSize=*/false,
                                      useGlobalDeviceIds,
                                      /*expectedGroupSize=*/std::nullopt)))
    return failure();

  auto operandType = operand.getType().cast<TensorType>();
  bool operandTypeRanked = operandType.isa<RankedTensorType>();
  Block& block = computation.front();
  if (failed(hlo::verifyReducerShape(
          location, block, {operandType},
          {RankedTensorType::get({}, operandType.getElementType())},
          /*numInputs=*/1, /*allowedDimensions=*/{},
          /*allInputsUnranked=*/!operandTypeRanked)))
    return failure();

  return success();
}

/*
 * We intend to verify the following properties
 * P1. We cannot convert between complex and real types (cf xla)
 * P3. The dimensions of the operand and the target
 * shape must match, except that the shape with the smaller element bitwidth has
 * an appropriately-sized additional innermost dimension, e.g.
 * ... x f32 => [bitcast_convert] => ... x 4 x i8
 * ... x 4 x i8 => [bitcast_convert] => ... x f32
 */
LogicalResult verifyBitcastConvertOp(Optional<Location> location, Value operand,
                                     Value result) {
  auto operandTensorType = operand.getType().cast<TensorType>();
  auto targetTensorType = result.getType().cast<TensorType>();

  // P1.
  auto targetElt = targetTensorType.getElementType();
  auto operandElt = operandTensorType.getElementType();
  if (targetElt.isa<ComplexType>() != operandElt.isa<ComplexType>()) {
    return emitOptionalError(
        location, "cannot convert between real and complex types, but got: ",
        operandTensorType, " and ", targetTensorType);
  }

  auto targetEltBitwidth = hlo::potentiallyComplexBitwidth(targetElt);
  auto operandEltBitwidth = hlo::potentiallyComplexBitwidth(operandElt);

  // P2.
  auto operandType = operandTensorType.dyn_cast<RankedTensorType>();
  auto targetType = targetTensorType.dyn_cast<RankedTensorType>();
  if (!operandType || !targetType) return success();

  auto targetShape = targetType.getShape();
  auto operandShape = operandType.getShape();
  ArrayRef<int64_t> smallerEltShape, biggerEltShape;
  Type smallerElt, biggerElt;
  if (operandEltBitwidth < targetEltBitwidth) {
    smallerEltShape = operandShape;
    smallerElt = operandElt;
    biggerEltShape = targetShape;
    biggerElt = targetElt;
  } else {
    smallerEltShape = targetShape;
    smallerElt = targetElt;
    biggerEltShape = operandShape;
    biggerElt = operandElt;
  }

  ArrayRef<int64_t> smallerEltPrefix;
  auto smallerEltBitwidth = std::min(targetEltBitwidth, operandEltBitwidth);
  auto biggerEltBitwidth = std::max(targetEltBitwidth, operandEltBitwidth);
  if (operandEltBitwidth != targetEltBitwidth) {
    if (smallerEltShape.empty()) {
      return emitOptionalError(location,
                               "does not allow the smaller element type to be "
                               "part of a 0d tensor, but got: ",
                               operandType, " and ", targetType, ".");
    }
    smallerEltPrefix = smallerEltShape.drop_back();
    if (!hlo::isDynamicDimSize(smallerEltShape.back()) &&
        smallerEltShape.back() * smallerEltBitwidth != biggerEltBitwidth) {
      return emitOptionalError(
          location, "requires compatible bitwidths. ", "Got: ", operandType,
          " and ", targetType, ", but ", smallerEltBitwidth, " * ",
          smallerEltShape.back(), " != ", biggerEltBitwidth, ".");
    }
  } else {
    smallerEltPrefix = smallerEltShape;
  }

  for (auto it : llvm::zip(smallerEltPrefix, biggerEltShape)) {
    auto targetDim = std::get<0>(it);
    auto operandDim = std::get<1>(it);
    if (!hlo::isDynamicDimSize(targetDim) &&
        !hlo::isDynamicDimSize(operandDim)) {
      if (targetDim != operandDim) {
        return emitOptionalError(
            location,
            "operand and result shapes must match except "
            "for the innermost dimension of the shape with "
            "the smaller element type. Got: ",
            operandType, " and ", targetType, ".");
      }
    }
  }

  return success();
}

LogicalResult verifyBroadcastInDimOp(Optional<Location> location, Value operand,
                                     DenseIntElementsAttr broadcastDimensions,
                                     Value result) {
  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  if (!operandType) {
    // The following verification checks all depend on knowing the rank of
    // the operand. Bail out now if we don't know the rank of the operand.
    return success();
  }

  auto operandRank = operandType.getRank();
  if (!broadcastDimensions) {
    if (operandRank == 0) return success();
    return emitOptionalError(location,
                             "broadcast_dimensions is absent, but required "
                             "because operand has non-zero rank (",
                             operandRank, ")");
  }

  auto dimensionsType = broadcastDimensions.getType();
  auto dimensionsRank = dimensionsType.getRank();
  if (dimensionsRank != 1)
    return emitOptionalError(location, "broadcast_dimensions has rank ",
                             dimensionsRank, " instead of rank 1");

  auto dimensionsSize = dimensionsType.getNumElements();
  if (dimensionsSize != operandRank)
    return emitOptionalError(location, "broadcast_dimensions size (",
                             dimensionsSize, ") does not match operand rank (",
                             operandRank, ")");

  auto dimensions = llvm::to_vector(broadcastDimensions.getValues<int64_t>());
  if (hasDuplicates(dimensions))
    return emitOptionalError(location,
                             "broadcast_dimensions should not have duplicates");

  auto resultType = result.getType().cast<RankedTensorType>();
  auto resultRank = resultType.getRank();
  for (int i = 0; i != dimensionsSize; ++i) {
    auto dimIndex = dimensions[i];
    if (dimIndex >= resultRank)
      return emitOptionalError(location,
                               "broadcast_dimensions contains invalid value ",
                               dimIndex, " for result with rank ", resultRank);

    if (!operandType.isDynamicDim(i)) {
      auto dimSize = operandType.getDimSize(i);
      auto resultDimSize = resultType.getDimSize(dimIndex);
      if (dimSize != 1 && dimSize != resultDimSize)
        return emitOptionalError(
            location, "size of operand dimension ", i, " (", dimSize,
            ") is not equal to 1 or size of result dimension ", dimIndex, " (",
            resultDimSize, ")");
    }
  }

  return success();
}

LogicalResult verifyCollectivePermuteOp(
    Optional<Location> location, DenseIntElementsAttr sourceTargetPairs) {
  // Verifies the source target pairs attached to collective permute.
  auto type = sourceTargetPairs.getType().dyn_cast<RankedTensorType>();
  if (type.getRank() != 2)
    return emitOptionalError(location,
                             "expect source_target_pairs attribute to be of "
                             "rank 2, but got rank ",
                             type.getRank());
  if (type.getShape()[1] != 2)
    return emitOptionalError(
        location,
        "expect source_target_pairs attribute of shape (N, 2), but got (",
        type.getShape(), ")");
  // Check source target pairs for duplicate sources or targets.
  llvm::DenseSet<int64_t> sources;
  llvm::DenseSet<int64_t> targets;
  for (auto i = sourceTargetPairs.begin(), e = sourceTargetPairs.end(); i != e;
       ++i) {
    auto val = (*i).getSExtValue();
    if (val < 0)
      return emitOptionalError(
          location, "replica ids in source_target_pairs must be >= 0.");

    if (i.getIndex() % 2 == 0) {
      bool isUnique = sources.insert(val).second;
      if (!isUnique)
        return emitOptionalError(location, "duplicate sources not allowed.");
    } else {
      bool isUnique = targets.insert(val).second;
      if (!isUnique)
        return emitOptionalError(location, "duplicate targets not allowed.");
    }
  }
  return success();
}

/*
 * We intend to verify the following properties
 *  P1. Verify the input, kernel types.
 *  P2. Verify the convolution atributes.
 *  P3. Verify and collect the window atributes.
 *  P4. Verify precision_config attribute.
 *  P5. Verify the return shape.
 *      TODO(b/232574102): Verify the element-type of return-value.
 */
LogicalResult verifyConvolutionOp(
    Optional<Location> location, Value lhs, Value rhs,
    Optional<DenseIntElementsAttr> windowStrides,
    Optional<DenseIntElementsAttr> padding,
    Optional<DenseIntElementsAttr> lhsDilation,
    Optional<DenseIntElementsAttr> rhsDilation,
    Optional<DenseElementsAttr> windowReversal, int64_t inputBatchDimension,
    int64_t inputFeatureDimension, ArrayRef<int64_t> inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    ArrayRef<int64_t> kernelSpatialDimensions, int64_t outputBatchDimension,
    int64_t outputFeatureDimension, ArrayRef<int64_t> outputSpatialDimensions,
    int64_t featureGroupCount, int64_t batchGroupCount,
    Optional<ArrayAttr> precisionConfig, Value result) {
  auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
  if (!lhsType || !rhsType) return success();

  // P1.
  int numDims = lhsType.getRank();
  if (numDims != rhsType.getRank())
    return emitOptionalError(location,
                             "expects convolution arguments to have same "
                             "number of dimensions. Got: ",
                             lhsType, " and ", rhsType, ".");

  if (numDims < 2)
    return emitOptionalError(
        location,
        "expects convolution arguments to have >= 2 dimensions. Got: ", lhsType,
        " and ", rhsType, ".");

  // P2.
  if (failed(verifyConvolutionAttributes(
          lhs, rhs, inputBatchDimension, inputFeatureDimension,
          inputSpatialDimensions, kernelInputFeatureDimension,
          kernelOutputFeatureDimension, kernelSpatialDimensions,
          outputBatchDimension, outputFeatureDimension, outputSpatialDimensions,
          featureGroupCount, batchGroupCount, location)))
    return failure();

  if ((size_t)numDims != inputSpatialDimensions.size() + 2)
    return emitOptionalError(location, "expects convolution arguments to have ",
                             inputSpatialDimensions.size() + 2,
                             " dimensions. Got: ", numDims);

  // P3.
  SmallVector<int64_t> windowDimensions(kernelSpatialDimensions.size());
  for (size_t i = 0; i < windowDimensions.size(); i++)
    windowDimensions[i] = rhsType.getShape()[kernelSpatialDimensions[i]];

  auto paddingOrErr = hlo::convertPaddingAttribute(padding, location);
  if (failed(paddingOrErr)) return failure();

  // TODO: add missing tests for ConvolutionOp.
  auto windowStridesOrErr =
      hlo::convert1DAttribute(windowStrides, location, "window_strides");
  if (failed(windowStridesOrErr)) return failure();
  auto lhsDilationOrErr =
      hlo::convert1DAttribute(lhsDilation, location, "lhs_dilation");
  if (failed(lhsDilationOrErr)) return failure();
  auto rhsDilationOrErr =
      hlo::convert1DAttribute(rhsDilation, location, "rhs_dilation");
  if (failed(rhsDilationOrErr)) return failure();
  auto windowReversalOrErr = hlo::convertWindowReversalAttribute(
      windowReversal, location, "window_reversal");
  if (failed(windowReversalOrErr)) return failure();

  auto windowOrErr = hlo::verifyWindowAttributesAndInferWindowDimensions(
      windowDimensions, *windowStridesOrErr, *paddingOrErr, *lhsDilationOrErr,
      *rhsDilationOrErr, *windowReversalOrErr, location);
  if (failed(windowOrErr)) return failure();

  // P3.
  if (failed(verifyPrecisionConfig(location, precisionConfig)))
    return failure();

  // P5.
  SmallVector<int64_t> outputDimensions(lhsType.getShape().size(),
                                        ShapedType::kDynamic);

  // Infer the output spatial dimensions.
  auto numSpatialDims = inputSpatialDimensions.size();
  SmallVector<int64_t> inputSpatialDimVals(numSpatialDims);
  for (int64_t i = 0; i < static_cast<int64_t>(numSpatialDims); ++i)
    inputSpatialDimVals[i] = lhsType.getShape()[inputSpatialDimensions[i]];

  auto windowOutputShape =
      hlo::inferWindowOutputShape(inputSpatialDimVals, *windowOrErr);

  for (int64_t i = 0; i < static_cast<int64_t>(windowOrErr->size()); ++i)
    outputDimensions[outputSpatialDimensions[i]] = windowOutputShape[i];

  // Infer the output-batch-dimension and output-feature-dimension.
  const int64_t inputBatch = lhsType.getShape()[inputBatchDimension];
  const int64_t kernelOutputFeatures =
      rhsType.getShape()[kernelOutputFeatureDimension];

  outputDimensions[outputBatchDimension] = hlo::isDynamicDimSize(inputBatch)
                                               ? ShapedType::kDynamic
                                               : inputBatch / batchGroupCount;
  outputDimensions[outputFeatureDimension] = kernelOutputFeatures;

  auto resultType = result.getType().dyn_cast<RankedTensorType>();
  if (!resultType) return success();

  if (failed(verifyCompatibleShape(outputDimensions, resultType.getShape())))
    return emitOptionalError(
        location, "inferred shape '", dimSizesToString(outputDimensions), "' ",
        "is incompatible with return type of operation ", resultType, "");

  return success();
}

LogicalResult verifyDotOp(Optional<Location> location, Value lhs, Value rhs,
                          Optional<ArrayAttr> /*precisionConfig*/,
                          Value result) {
  auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
  auto resultType = result.getType().dyn_cast<RankedTensorType>();
  if (!lhsType || !rhsType) return success();

  SmallVector<int64_t> inferredShape;
  if (failed(inferDotShape(lhsType, rhsType, inferredShape)))
    return emitOptionalError(location, "failed to infer shape for lhs ",
                             lhsType, " and rhs ", rhsType);

  if (resultType &&
      failed(verifyCompatibleShape(inferredShape, resultType.getShape())))
    return emitOptionalError(
        location, "inferred shape '", dimSizesToString(inferredShape), "' ",
        "is incompatible with return type of operation ", resultType, "");

  return success();
}

LogicalResult verifyDotGeneralOp(Optional<Location> location, Value lhs,
                                 Value rhs,
                                 ArrayRef<int64_t> lhsBatchingDimensions,
                                 ArrayRef<int64_t> rhsBatchingDimensions,
                                 ArrayRef<int64_t> lhsContractingDimensions,
                                 ArrayRef<int64_t> rhsContractingDimensions,
                                 Optional<ArrayAttr> precisionConfig,
                                 Value result) {
  if (failed(verifyPrecisionConfig(location, precisionConfig)))
    return failure();

  if (lhsBatchingDimensions.size() != rhsBatchingDimensions.size())
    return emitOptionalError(location,
                             "lhs and rhs should have the same "
                             "number of batching dimensions");
  if (lhsContractingDimensions.size() != rhsContractingDimensions.size())
    return emitOptionalError(location,
                             "lhs and rhs should have the same "
                             "number of contracting dimensions");

  llvm::SmallDenseSet<int64_t> dimSet;
  auto checkDimsDistinct =
      [&](ArrayRef<int64_t> batchingDims, ArrayRef<int64_t> contractingDims,
          llvm::SmallDenseSet<int64_t>& dimSet, llvm::StringRef lhs,
          llvm::StringRef rhs) -> LogicalResult {
    auto dims = llvm::concat<const int64_t>(batchingDims, contractingDims);
    for (auto dim : dims) {
      auto [_, wasInserted] = dimSet.insert(dim);
      if (!wasInserted)
        return emitOptionalError(location, "has duplicated dimension from ",
                                 lhs, " and ", rhs, ": ", dim);
    }
    return success();
  };

  if (failed(checkDimsDistinct(lhsBatchingDimensions, lhsContractingDimensions,
                               dimSet, "lhs_batching_dimensions",
                               "lhs_contracting_dimensions")))
    return failure();

  dimSet.clear();

  if (failed(checkDimsDistinct(rhsBatchingDimensions, rhsContractingDimensions,
                               dimSet, "rhs_batching_dimensions",
                               "rhs_contracting_dimensions")))
    return failure();

  auto checkDimsInRange = [&](int64_t rank, ArrayRef<int64_t> dims,
                              llvm::StringRef dimName) -> LogicalResult {
    auto inRange = [&](int64_t i) -> bool { return 0 <= i && i < rank; };
    const auto* dimsNotInRange =
        std::find_if_not(dims.begin(), dims.end(), inRange);
    if (dimsNotInRange != dims.end())
      return emitOptionalError(location, dimName, " value: ", *dimsNotInRange,
                               " is out of range: ", "[0, ", rank, ")");
    return success();
  };
  auto lhsRankedType = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsRankedType = rhs.getType().dyn_cast<RankedTensorType>();

  if (lhsRankedType) {
    if (failed(checkDimsInRange(lhsRankedType.getRank(), lhsBatchingDimensions,
                                "lhs_batching_dimensions")) ||
        failed(checkDimsInRange(lhsRankedType.getRank(),
                                lhsContractingDimensions,
                                "lhs_contracting_dimensions")))
      return failure();
  }
  if (rhsRankedType) {
    if (failed(checkDimsInRange(rhsRankedType.getRank(), rhsBatchingDimensions,
                                "rhs_batching_dimensions")) ||
        failed(checkDimsInRange(rhsRankedType.getRank(),
                                rhsContractingDimensions,
                                "rhs_contracting_dimensions")))
      return failure();
  }
  if (lhsRankedType && rhsRankedType) {
    // Dimension sizes must be compatible for lhs/rhs.
    auto lhsShape = lhsRankedType.getShape();
    auto rhsShape = rhsRankedType.getShape();

    for (auto [lhs, rhs] :
         llvm::zip(lhsBatchingDimensions, rhsBatchingDimensions)) {
      if (hlo::isDynamicDimSize(lhsShape[lhs])) continue;
      if (hlo::isDynamicDimSize(rhsShape[rhs])) continue;
      if (lhsShape[lhs] != rhsShape[rhs])
        return emitOptionalError(location,
                                 "batching dimension sizes must "
                                 "match for lhs/rhs");
    }

    for (auto [lhs, rhs] :
         llvm::zip(lhsContractingDimensions, rhsContractingDimensions)) {
      if (hlo::isDynamicDimSize(lhsShape[lhs])) continue;
      if (hlo::isDynamicDimSize(rhsShape[rhs])) continue;
      if (lhsShape[lhs] != rhsShape[rhs])
        return emitOptionalError(location,
                                 "contracting dimension sizes must "
                                 "match for lhs/rhs");
    }
  }

  auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
  auto resultType = result.getType().dyn_cast<RankedTensorType>();
  if (!lhsType || !rhsType || !resultType) return success();

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  auto resultShape = resultType.getShape();

  // Infer the output dimensions of the operation.
  SmallVector<int64_t> dimensions;
  for (const int64_t lhsBatchingDim : lhsBatchingDimensions)
    dimensions.push_back(lhsShape[lhsBatchingDim]);
  for (int64_t i = 0; i < lhsType.getRank(); i++)
    if (!llvm::is_contained(lhsBatchingDimensions, i) &&
        !llvm::is_contained(lhsContractingDimensions, i))
      dimensions.push_back(lhsShape[i]);
  for (int64_t i = 0; i < rhsType.getRank(); i++)
    if (!llvm::is_contained(rhsBatchingDimensions, i) &&
        !llvm::is_contained(rhsContractingDimensions, i))
      dimensions.push_back(rhsShape[i]);

  if (failed(verifyCompatibleShape(dimensions, resultShape)))
    return emitOptionalError(
        location, "inferred shape '", dimSizesToString(dimensions), "' ",
        "is incompatible with return type of operation ", resultType, "");

  return success();
}

LogicalResult verifyDynamicBroadcastInDimOp(
    Optional<Location> location, Value operand, Value outputDimensions,
    DenseIntElementsAttr broadcastDimensions,
    Optional<DenseIntElementsAttr> knownExpandingDimensions,
    Optional<DenseIntElementsAttr> knownNonexpandingDimensions, Value result) {
  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  auto resultType = result.getType().dyn_cast<RankedTensorType>();

  // If either the operand or result are unranked, there is very little
  // to verify statically.
  if (!operandType || !resultType) return success();

  auto outputDimensionsType =
      outputDimensions.getType().cast<RankedTensorType>();
  auto outputDimensionsSize = outputDimensionsType.getDimSize(0);
  auto operandRank = operandType.getRank();
  auto resultRank = resultType.getRank();

  // Verify broadcast_dimensions.
  auto bcastDimensions = broadcastDimensions;
  auto bcastDimensionsType = broadcastDimensions.getType();
  auto bcastDimensionsRank = bcastDimensionsType.getRank();
  // TODO(laurenzo): Update the BroadcastDimAttr to constrain its rank to 1.
  if (bcastDimensionsRank != 1)
    return emitOptionalError(location, "broadcast_dimensions has rank ",
                             bcastDimensionsRank, " instead of rank 1");

  auto bcastDimensionsSize = bcastDimensionsType.getNumElements();
  if (bcastDimensionsSize != operandRank)
    return emitOptionalError(
        location, "broadcast_dimensions size (", bcastDimensionsSize,
        ") does not match operand rank (", operandRank, ")");

  if (resultRank < operandRank)
    return emitOptionalError(location, "result rank (", resultRank,
                             ") is less than operand rank (", operandRank, ")");

  for (int i = 0; i != bcastDimensionsSize; ++i) {
    auto dimIndex = bcastDimensions.getValues<int64_t>()[i];
    if (dimIndex >= resultRank)
      return emitOptionalError(location,
                               "broadcast_dimensions contains invalid value ",
                               dimIndex, " for result with rank ", resultRank);

    auto dimSize = operandType.getDimSize(i);
    auto resultDimSize = resultType.getDimSize(dimIndex);
    // Note: verifyCompatibleShapes doesn't consider size-1 broadcasting, so
    // we add a manual check for this.
    if (dimSize != 1 && failed(verifyCompatibleShape(dimSize, resultDimSize)))
      return emitOptionalError(location, "size of operand dimension ", i, " (",
                               dimSize,
                               ") is not compatible "
                               "with size of result dimension ",
                               dimIndex, " (", resultDimSize, ")");
  }

  if (outputDimensionsSize != resultRank)
    return emitOptionalError(location, "result rank (", resultRank,
                             ") is not equal to number of output dimensions (",
                             outputDimensionsSize, ")");

  // Verify that the known expanding and non-expanding dimensions are a subset
  // of the operand's dimensions.
  int64_t numKnownExpansionBehavior = 0;
  DenseSet<int64_t> knownExpansionBehavior;
  auto collectExpansionBehaviorDims =
      [&](const Optional<DenseIntElementsAttr>& attr) {
        if (!attr) return;
        for (const APInt& it : *attr) {
          numKnownExpansionBehavior++;
          knownExpansionBehavior.insert(it.getLimitedValue());
        }
      };
  collectExpansionBehaviorDims(knownExpandingDimensions);
  collectExpansionBehaviorDims(knownNonexpandingDimensions);
  if (knownExpansionBehavior.size() != numKnownExpansionBehavior)
    return emitOptionalError(
        location,
        "duplicate expansion hint for at least one operand dimension");
  for (int64_t i : knownExpansionBehavior)
    if (i < 0 || i >= operandRank)
      return emitOptionalError(location, "hint for expanding dimension ", i,
                               " does not refer to a "
                               "valid operand dimension");

  return success();
}

LogicalResult verifyDynamicReshapeOp(Optional<Location> location,
                                     Value outputShape, Value result) {
  auto resultType = result.getType().dyn_cast<RankedTensorType>();
  auto outputShapeType = outputShape.getType().dyn_cast<RankedTensorType>();
  if (resultType && outputShapeType && outputShapeType.hasStaticShape() &&
      outputShapeType.getDimSize(0) != resultType.getRank()) {
    return emitOptionalError(location,
                             "output should have a rank equal to the number of "
                             "elements in output_shape");
  }
  return success();
}

LogicalResult verifyIotaOp(Optional<Location> location, int64_t iotaDimension,
                           Value result) {
  auto shape = result.getType().cast<ShapedType>();
  if (!shape.hasRank()) return success();
  if (shape.getRank() == 0)
    return emitOptionalError(location, "does not support scalars.");

  if (iotaDimension >= shape.getRank() || iotaDimension < 0)
    return emitOptionalError(
        location,
        "iota dimension cannot go beyond the output rank or be negative.");
  return success();
}

// Verifies that operand rank matches start_indices/limit_indices/strides size
LogicalResult verifyRealDynamicSliceOp(Optional<Location> location,
                                       Value operand, Value startIndices,
                                       Value limitIndices, Value strides) {
  auto inputType = operand.getType().dyn_cast<RankedTensorType>();
  // If operand is unranked, there is very little to verify statically.
  if (!inputType) return success();
  int inputRank = inputType.getRank();

  auto startType = startIndices.getType().cast<RankedTensorType>();
  auto limitType = limitIndices.getType().cast<RankedTensorType>();
  auto stridesType = strides.getType().cast<RankedTensorType>();

  if (inputRank != startType.getNumElements())
    return emitOptionalError(
        location, "has mismatched number of operand rank (", inputRank,
        ") and start_indices size (", startType.getNumElements(), ")");

  if (inputRank != limitType.getNumElements())
    return emitOptionalError(
        location, "has mismatched number of operand rank (", inputRank,
        ") and limit_indices size (", limitType.getNumElements(), ")");

  if (inputRank != stridesType.getNumElements())
    return emitOptionalError(
        location, "has mismatched number of operand rank (", inputRank,
        ") and strides size (", stridesType.getNumElements(), ")");
  return success();
}

// We intend to verify the following properties
//  P1. Verify all `inputs` need to have compatible shapes.
//  P2. Verify that
//      1. the dimensions of reduce-op are in-bounds for the given shape.
//      2. the dimension-attribute have no duplicate entries.
//  P3. Verify the inner block defining the reducer function.
LogicalResult verifyReduceOp(Optional<Location> location, ValueRange inputs,
                             ValueRange initValues,
                             DenseIntElementsAttr dimensions, Region& body) {
  SmallVector<TensorType> inputArgTypes{llvm::map_range(
      inputs.getTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); })};
  SmallVector<TensorType> initValueTypes{llvm::map_range(
      initValues.getTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); })};

  // P1. & P2.
  SmallVector<int64_t> newDimensions;
  Attribute encoding;
  if (failed(verifyReduceOpInputsAndInferShape(location, inputArgTypes,
                                               initValueTypes, dimensions,
                                               newDimensions, encoding)))
    return failure();

  // P3.
  uint64_t numInputs = inputs.size();
  int64_t rankedInputIdx = -1;
  for (uint64_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    if (inputArgTypes[inputIdx].hasRank()) {
      rankedInputIdx = inputIdx;
      break;
    }
  }
  bool allInputsUnranked = (rankedInputIdx == -1);

  Block& block = body.front();
  if (failed(verifyReducerShape(location, block, inputArgTypes, initValueTypes,
                                numInputs, newDimensions, allInputsUnranked)))
    return failure();
  return success();
}

LogicalResult verifyReduceScatterOp(Optional<Location> location, Value operand,
                                    int64_t scatterDimension,
                                    DenseIntElementsAttr replicaGroups,
                                    bool useGlobalDeviceIds,
                                    Region& computation, Value result) {
  if (failed(hlo::verifyReplicaGroups(location, replicaGroups,
                                      /*allGroupsMustHaveSameSize=*/true,
                                      useGlobalDeviceIds,
                                      /*expectedGroupSize=*/std::nullopt)))
    return failure();
  auto operandType = operand.getType().cast<TensorType>();
  bool operandTypeRanked = operandType.isa<RankedTensorType>();
  Block& block = computation.front();
  if (failed(hlo::verifyReducerShape(
          location, block, {operandType},
          {RankedTensorType::get({}, operandType.getElementType())},
          /*numInputs=*/1, /*allowedDimensions=*/{},
          /*allInputsUnranked=*/!operandTypeRanked)))
    return failure();

  auto resultType = result.getType().cast<ShapedType>();
  if (!operandType.hasRank() || !resultType.hasRank()) return success();
  if (operandType.getRank() != resultType.getRank())
    return emitOptionalError(location,
                             "operand and result should have same rank");
  if (scatterDimension < 0) {
    return emitOptionalError(location, "expects scatter_dimension >= 0");
  }
  if (scatterDimension >= operandType.getRank())
    return emitOptionalError(
        location, "scatter dim should be less than operand/result rank");
  if (operandType.isDynamicDim(scatterDimension) ||
      resultType.isDynamicDim(scatterDimension))
    return success();

  if (operandType.getDimSize(scatterDimension) == 0)
    return emitOptionalError(location,
                             "operand scatter dimension cannot be zero");
  if (resultType.getDimSize(scatterDimension) == 0)
    return emitOptionalError(location,
                             "result scatter dimension cannot be zero");

  // If operand and result are both ranked, then the size of the scatter
  // dimension in the operand should be a multiple of the size of the scatter
  // dimension in the result.
  if ((operandType.getDimSize(scatterDimension) %
       resultType.getDimSize(scatterDimension)) != 0)
    return emitOptionalError(
        location, "operand scatter dimension has size ",
        operandType.getDimSize(scatterDimension),
        ", expected to be a multiple of result scatter dimension size ",
        resultType.getDimSize(scatterDimension));

  // Non scatter dimensions should be equal.
  for (uint64_t index : llvm::seq<uint64_t>(0, operandType.getRank())) {
    if (static_cast<int64_t>(index) == scatterDimension ||
        operandType.isDynamicDim(index) || resultType.isDynamicDim(index))
      continue;
    if (operandType.getDimSize(index) != resultType.getDimSize(index))
      return emitOptionalError(
          location, "non scatter dimensions should be same for operand (",
          operandType.getDimSize(index), ") and result (",
          resultType.getDimSize(index), ")");
  }
  return success();
}

// We intend to verify the following properties
//  P1. All `inputs` need to have compatible shapes.
//  P2. size-of(window_dimension) == rank-of(input),
//        where input is an element of 'inputs'.
//  P3. Verify and collect the window atributes.
//  P4. Verify the inner block defining the reducer function.
LogicalResult verifyReduceWindowOp(
    Optional<Location> location, ValueRange inputs, ValueRange initValues,
    DenseIntElementsAttr windowDimensions,
    Optional<DenseIntElementsAttr> windowStrides,
    Optional<DenseIntElementsAttr> baseDilations,
    Optional<DenseIntElementsAttr> windowDilations,
    Optional<DenseIntElementsAttr> padding, Region& body) {
  SmallVector<TensorType> inputArgTypes{llvm::map_range(
      inputs.getTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); })};
  SmallVector<TensorType> initValueTypes{llvm::map_range(
      initValues.getTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); })};
  uint64_t numInputs = inputs.size();

  // P1. ~ P3.
  SmallVector<int64_t> windowDims;
  SmallVector<WindowDimension> inferredWindow;
  if (failed(verifyReduceWindowOpInputsAndInferWindow(
          location, inputArgTypes, initValueTypes, windowDimensions,
          windowStrides, baseDilations, windowDilations, padding,
          /*windowReversal=*/std::nullopt, windowDims, inferredWindow)))
    return failure();

  // P4.
  // Check for unranked tensors in input operands.
  int64_t rankedInputIdx = -1;
  for (uint64_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    if (inputArgTypes[inputIdx].hasRank()) {
      rankedInputIdx = inputIdx;
      break;
    }
  }
  bool allInputsUnranked = (rankedInputIdx == -1);
  Block& block = body.front();
  if (failed(verifyReducerShape(location, block, inputArgTypes, initValueTypes,
                                numInputs, windowDims, allInputsUnranked)))
    return failure();

  return success();
}

LogicalResult verifySortOp(Optional<Location> location, ValueRange inputs,
                           int64_t dimension, Region& comparator) {
  auto operandTypes = inputs.getTypes();
  for (auto operandType : operandTypes) {
    auto operandShapedType = operandType.cast<ShapedType>();
    if (operandShapedType.hasRank()) {
      int64_t cmpDim = dimension;
      int64_t rank = operandShapedType.getRank();
      if (cmpDim < -rank || cmpDim >= rank)
        return emitOptionalError(
            location, "dimension attribute value must be in range [-", rank,
            ", ", rank, "), but found ", cmpDim);
      else
        break;  // ODS SameOperandsAndResultShape asserts inputs have same shape
    }
  }

  // Comparator must have 2 * N scalar arguments of same type as the N inputs.
  Block& block = comparator.front();
  size_t numOperands = operandTypes.size();
  if (block.getNumArguments() != 2 * numOperands)
    return emitOptionalError(location, "comparator block should have ",
                             2 * numOperands, " arguments");
  for (const auto& indexedOperandType : llvm::enumerate(operandTypes)) {
    int index = indexedOperandType.index();
    Type elementType =
        indexedOperandType.value().cast<ShapedType>().getElementType();
    Type tensorType = RankedTensorType::get({}, elementType);
    for (int i : {2 * index, 2 * index + 1}) {
      Type argType = block.getArgument(i).getType();
      if (argType != tensorType)
        return emitOptionalError(location, "comparator block argument #", i,
                                 " should be of type ", tensorType, " but got ",
                                 argType);
    }
  }

  // Comparator must return single 0-ranked tensor with element-type i1.
  auto comparatorResult = block.getTerminator()->getOperands();
  if (comparatorResult.size() != 1)
    return emitOptionalError(location,
                             "comparator must return single output but got ",
                             comparatorResult.size());
  auto comparatorResultType = comparatorResult[0].getType().cast<TensorType>();
  if ((comparatorResultType.hasRank() && comparatorResultType.getRank() != 0) ||
      !comparatorResultType.getElementType().isInteger(1))
    return emitOptionalError(location,
                             "comparator must return tensor<i1> but got ",
                             comparatorResult[0].getType());
  return success();
}

LogicalResult verifyWhileOp(Optional<Location> location, ValueRange operand,
                            Region& cond, Region& body) {
  auto operandTypes = operand.getTypes();
  auto condArgsTypes = cond.front().getArgumentTypes();
  auto bodyArgsTypes = body.front().getArgumentTypes();
  if (!hlo::isCompatibleForHloTypeInference(operandTypes, condArgsTypes))
    return emitOptionalError(location,
                             "expect operands are compatible with condition "
                             "block arguments but got ",
                             operandTypes, " vs ", condArgsTypes);
  if (!hlo::isCompatibleForHloTypeInference(operandTypes, bodyArgsTypes))
    return emitOptionalError(
        location,
        "expect operands are compatible with body block arguments but got ",
        operandTypes, " vs ", bodyArgsTypes);

  auto bodyReturnTypes = body.front().getTerminator()->getOperandTypes();
  if (!hlo::isCompatibleForHloTypeInference(operandTypes, bodyReturnTypes))
    return emitOptionalError(
        location,
        "expect operands are compatible with body block return types but got ",
        operandTypes, " vs ", bodyReturnTypes);

  auto condReturnTypes = cond.front().back().getOperandTypes();
  if (condReturnTypes.size() != 1)
    return emitOptionalError(
        location, "expect condition body returns a single value but got ",
        condReturnTypes.size());
  auto operandType = condReturnTypes[0].cast<TensorType>();
  if ((operandType.hasRank() && operandType.getRank() != 0) ||
      !operandType.getElementType().isInteger(1))
    return emitOptionalError(
        location,
        "expect condition block return a zero-ranked tensor of i1 but got ",
        condReturnTypes[0]);

  return success();
}

}  // end namespace hlo
}  // end namespace mlir
