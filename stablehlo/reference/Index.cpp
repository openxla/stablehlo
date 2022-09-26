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

namespace mlir {
namespace stablehlo {

IndexSpaceIterator::IndexSpaceIterator(const std::vector<int64_t> &shape,
                                       bool lastIndex)
    : shape_(shape) {
  auto shapeSize = shape.size();
  if (lastIndex) {
    counter_ = 1;
    for (auto shapeElm : shape) counter_ *= shapeElm;
  } else {
    counter_ = 0;
    indices_.resize(shapeSize);
  }
}

IndexSpaceIterator &IndexSpaceIterator::operator++() {
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
