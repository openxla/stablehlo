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

#ifndef STABLEHLO_REFERENCE_TENSOR_H
#define STABLEHLO_REFERENCE_TENSOR_H

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/reference/Element.h"
#include "stablehlo/reference/Index.h"

namespace mlir {
namespace stablehlo {

namespace detail {

/// Underlying storage class for Tensor objects.
class Buffer : public llvm::RefCountedBase<Buffer> {
 public:
  /// \name Constructors
  /// @{
  explicit Buffer(ShapedType type);
  Buffer(ShapedType type, void *data);
  Buffer(ShapedType type, const void *data);
  Buffer(Buffer &&other) = default;
  /// @}

  /// Move assignment operator deleted in RefCountedBase
  Buffer &operator=(Buffer &&other) = delete;

  /// Returns type of the Buffer object.
  ShapedType getType() { return type_; }

  /// Returns the raw data as a byte array.
  char *getData() { return data_.data(); }

  /// Returns the size in bytes of the raw data.
  size_t getSize() { return data_.size(); }

 private:
  ShapedType type_;
  std::vector<char> data_;
};

}  // namespace detail

/// Helper class to access the tensor elements in a linearized layout.
class Tensor {
 public:
  /// \name Constructors
  /// @{
  Tensor();
  explicit Tensor(ShapedType type);
  explicit Tensor(DenseElementsAttr attr);
  Tensor(const Tensor &other) = default;
  /// @}

  /// Assignment operator.
  Tensor &operator=(const Tensor &other) = default;

  /// Returns type of the Tensor object.
  ShapedType getType() const;

  /// Returns the number of elements.
  int64_t getNumElements() const;

  /// Provides read access to the tensor element indexed at 'index'.
  Element get(ArrayRef<int64_t> index) const;

  /// Provides write access to the tensor element indexed at 'index'.
  ///
  /// \param index The multi-dimensional index to write to.
  /// \param element The Element object \a element is used to update the
  /// underlying storage pointed to by \a index.
  void set(ArrayRef<int64_t> index, const Element &element);

  /// Prints Tensor objects.
  void print(raw_ostream &os) const;
  void dump() const;

  /// Iterate over the index space of a Tensor object.
  IndexSpaceIterator index_begin() const;
  IndexSpaceIterator index_end() const;

 private:
  llvm::IntrusiveRefCntPtr<detail::Buffer> impl_;
};

/// Print utilities for Tensor objects.
inline raw_ostream &operator<<(raw_ostream &os, Tensor tensor) {
  tensor.print(os);
  return os;
}

/// Creates a Tensor using `type` as the static type and data provided as an
/// array of string literal which will be parsed using the corresponding element
/// type.
Tensor makeTensor(ShapedType type, llvm::ArrayRef<llvm::StringRef> strData);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_TENSOR_H
