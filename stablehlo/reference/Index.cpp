/* Copyright 2022 The StableHLO Authors.

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

#include "stablehlo/reference/Index.h"

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace stablehlo {

IndexSpaceIterator::IndexSpaceIterator(llvm::ArrayRef<int64_t> shape,
                                       int64_t counter)
    : shape_(shape), counter_(counter) {
  num_elements_ = 1;
  for (auto shapeElm : shape) num_elements_ *= shapeElm;
  indices_.resize(shape.size());
}

IndexSpaceIterator &IndexSpaceIterator::operator++() {
  if (counter_ == num_elements_)
    report_fatal_error(llvm::StringRef("Incrementing beyond the last index."));
  for (int64_t i = shape_.size() - 1; i >= 0; --i) {
    indices_[i] += 1;
    if (indices_[i] >= shape_[i]) {
      indices_[i] = 0;
    } else {
      break;
    }
  }

  counter_++;
  return *this;
}

IndexSpaceIterator IndexSpaceIterator::operator++(int) {
  IndexSpaceIterator tempIter = *this;
  ++*this;
  return tempIter;
}

}  // namespace stablehlo
}  // namespace mlir
