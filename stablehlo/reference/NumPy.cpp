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

#include "stablehlo/reference/NumPy.h"

#include <cctype>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "llvm/ADT/bit.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {
namespace numpy {
namespace {

constexpr char kMagicString[] = "\x93NUMPY";
constexpr int kMagicStringSize = 6;
constexpr char kMajorVersion = 0x01;
constexpr char kMinorVersion = 0x00;

template <typename T>
struct IsComplexT : public std::false_type {};
template <typename T>
struct IsComplexT<std::complex<T>> : public std::true_type {};

}  // namespace

// Determine the NumPy type abbreviation for the given underlying data type T.
template <typename T>
static constexpr char getNumPyType() {
  if constexpr (std::is_same_v<T, bool>) return 'b';
  if constexpr (IsComplexT<T>::value) return 'c';
  if constexpr (std::is_floating_point_v<T>) return 'f';
  if constexpr (std::is_integral_v<T>) {
    if constexpr (std::is_signed_v<T>)
      return 'i';
    else
      return 'u';
  }

  llvm::report_fatal_error("Unknown type");
}

// Creates the NumPy dict header. This will serialize the `descr`,
// `fortran_order` and `shape` keys after the NumPy magic string. See
// `parseDescrHeader` for additional information on how the `descr` key is
// structured.
template <typename T>
static void buildNumpyHeaderDict(llvm::raw_fd_ostream& out,
                                 ArrayRef<int64_t> shape) {
  // For now, only little endian machines are supported.
  const auto descr = std::string(1, /*endianness=*/'<') +
                     std::string(1, getNumPyType<T>()) +
                     std::to_string(sizeof(T));
  const auto shapeString =
      std::accumulate(shape.begin(), shape.end(), std::string(""),
                      [](const std::string& dest, size_t dim) {
                        return dest + std::to_string(dim) + ",";
                      });
  std::string dict;
  std::stringstream outDict(dict);

  outDict << "{'descr': '" << descr << "', ";
  outDict << "'fortran_order': False, ";
  outDict << "'shape' : (" << shapeString << "), }";

  // The NumPy header is padded with spaces for proper alignment.
  // Account for newline and the magic string length.
  const auto headerSize = static_cast<int>(out.tell());
  const auto padding = 16 - (headerSize + 1) % 16;
  outDict << std::string(padding - 1, ' ') << '\n';

  const auto dictLength = static_cast<uint16_t>(outDict.str().size());
  out << (char)(dictLength & 0xff) << (char)((dictLength >> 8) & 0xff)
      << outDict.str();
}

template <typename T>
static void buildNumpyHeader(llvm::raw_fd_ostream& out,
                             ArrayRef<int64_t> shape) {
  out.write(kMagicString, kMagicStringSize);
  out.write(kMajorVersion);
  out.write(kMinorVersion);
  buildNumpyHeaderDict<T>(out, shape);
}

static llvm::Error parseFortranOrderKey(const std::string& header) {
  const std::size_t fortranOrderOffset = header.find("'fortran_order':");
  if (fortranOrderOffset == std::string::npos)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Failed to find fortran_order header.");

  if (header.find("False", fortranOrderOffset) == std::string::npos)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Only fortran_order: False is supported.");

  return llvm::Error::success();
}

// Parses the NumPy `descr` header, which is in the following format:
// 'descr': '<i4'
// Where the first character determines the endianness of the data (< for little
// endian, > for big endian), the next character determines the data type (i.e.
// unsigned int, float, bool, etc) and the last number(s) determine the data
// type size (8, 16, 32, etc). For now, we only support little endian values.
// Returns the size of the underlying type in bytes (i.e. for '<i4', this will
// return 4).
template <typename T>
static llvm::ErrorOr<int> parseDescrHeader(const std::string& header) {
  constexpr char kDescr[] = "'descr':";
  constexpr int kDescrSize = 8;

  const std::size_t descrOffset = header.find(kDescr);
  if (descrOffset == std::string::npos) return llvm::errc::invalid_argument;

  const std::size_t needleSize = header.find(',', descrOffset + kDescrSize + 1);
  std::string typeString = header.substr(
      descrOffset + kDescrSize, needleSize - (descrOffset + kDescrSize));

  if (typeString.front() != '\'' || typeString.back() != '\'')
    return llvm::errc::invalid_argument;

  // Strip quotes from type string (i.e. '<i8' to <i8).
  typeString = typeString.substr(1, typeString.size() - 2);

  if (typeString.size() < 3) return llvm::errc::invalid_argument;

  // Check that the serialized type string matches the expected type string.
  if (getNumPyType<T>() != typeString[1]) return llvm::errc::invalid_argument;

  return std::stoi(typeString.substr(2));
}

// Parses the `shape` key of the NumPy file format dictionary. Returns a vector
// representing the parsed shape or errors otherwise.
static llvm::ErrorOr<ArrayRef<int64_t>> parseShapeHeader(
    const std::string& header) {
  const std::size_t shapeOffset = header.find("'shape':");
  const std::size_t dimEnd = header.find(')', shapeOffset);
  if (shapeOffset == std::string::npos || dimEnd == std::string::npos)
    return llvm::errc::invalid_argument;

  // By convention, NumPy writers should include the shape key last (preserving
  // alphabetical ordering relative to other header keys), however we cannot
  // assume this to be unilaterally true. Therefore, apply a tight bound on the
  // regex matching for dimension integrals.
  std::regex dimRegex("[0-9]+");
  std::smatch dimMatch;
  std::string shapeString = header.substr(shapeOffset, shapeOffset - dimEnd);
  std::vector<int64_t> shape(4);

  while (std::regex_search(shapeString, dimMatch, dimRegex)) {
    shape.push_back(std::stoi(dimMatch[0]));
    shapeString = dimMatch.suffix();
  }

  return shape;
}

template <typename T>
static llvm::Error readNumpyHeader(std::ifstream& in, size_t& wordSize,
                                   ArrayRef<int64_t> shape) {
  char magicString[kMagicStringSize];
  if (!in.read(magicString, kMagicStringSize))
    return llvm::createStringError(llvm::errc::io_error,
                                   "Failed to read NumPy magic string.");

  if (strncmp(magicString, kMagicString, kMagicStringSize))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Invalid NumPy file format detected.");

  // Currently, only supports NumPy v1.0 format
  char majorVersion, minorVersion;
  in.read(&majorVersion, 1);
  in.read(&minorVersion, 1);

  if (majorVersion != kMajorVersion || minorVersion != kMinorVersion)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "Invalid NumPy version: %c.%c. Expected version to be %c.%c.",
        majorVersion, minorVersion, kMajorVersion, kMinorVersion);

  char headerSizeBuffer[2];
  if (!in.read(headerSizeBuffer, 2))
    return llvm::createStringError(llvm::errc::io_error,
                                   "Failed to read NumPy header size.");

  const int headerSize = (headerSizeBuffer[0]) | (headerSizeBuffer[1] << 8);
  std::string header(headerSize, '\0');
  if (!in.read(header.data(), headerSize) || header.back() != '\n')
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Invalid NumPy header.");

  header.erase(std::remove_if(header.begin(), header.end(),
                              [](char c) { return std::isspace(c); }),
               header.end());

  auto wordSizeOrError = parseDescrHeader<T>(header);
  if (!wordSizeOrError)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Failed to parse descr header.");

  auto fortranOrderKeyOrError = parseFortranOrderKey(header);
  if (fortranOrderKeyOrError)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Failed to parse fortran_order header.");

  auto shapeOrError = parseShapeHeader(header);
  if (!shapeOrError)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Failed to parse shape header.");

  shape = *shapeOrError;
  wordSize = *wordSizeOrError;

  return llvm::Error::success();
}

namespace {
template <typename T>
class ToNumpy {
 public:
  llvm::Error operator()(StringRef filename, ShapedType type,
                         const char* data) {
    int fd;
    if (llvm::sys::fs::openFileForWrite(filename, fd,
                                        llvm::sys::fs::CD_CreateAlways))
      return llvm::createStringError(llvm::errc::io_error,
                                     "Failed to open NumPy file.");

    llvm::raw_fd_ostream out(fd, /*shouldClose=*/true);

    buildNumpyHeader<T>(out, type.getShape());
    out.write(data, sizeof(T) * type.getNumElements());

    return llvm::Error::success();
  }
};

template <typename T>
class FromNumpy {
 public:
  llvm::ErrorOr<Tensor> operator()(StringRef filename, ShapedType type) {
    std::ifstream in(filename.str(), std::ifstream::binary);
    size_t wordSize = 0;

    if (readNumpyHeader<T>(in, wordSize, type.getShape()))
      return llvm::errc::invalid_argument;

    const int64_t numLoadedElements = std::accumulate(
        type.getShape().begin(), type.getShape().end(), 1, std::multiplies<>());

    if (numLoadedElements != type.getNumElements() || wordSize != sizeof(T))
      return llvm::errc::invalid_argument;

    const size_t dataBytesToRead = wordSize * numLoadedElements;
    std::vector<T> data(dataBytesToRead);
    in.read(reinterpret_cast<char*>(data.data()), dataBytesToRead);

    return Tensor(type,
                  HeapAsmResourceBlob::allocateAndCopyInferAlign<T>(data));
  }
};

template <template <typename Type> class Functor, typename... Args>
static decltype(auto) dispatchType(Type type, Args&&... args) {
  if (type.isSignlessInteger(1))
    return Functor<char>()(std::forward<Args>(args)...);
  if (type.isSignlessInteger(8))
    return Functor<int8_t>()(std::forward<Args>(args)...);
  if (type.isInteger(8)) return Functor<uint8_t>()(std::forward<Args>(args)...);
  if (type.isSignlessInteger(16))
    return Functor<int16_t>()(std::forward<Args>(args)...);
  if (type.isInteger(16))
    return Functor<uint16_t>()(std::forward<Args>(args)...);
  if (type.isSignlessInteger(32))
    return Functor<int32_t>()(std::forward<Args>(args)...);
  if (type.isInteger(32))
    return Functor<uint32_t>()(std::forward<Args>(args)...);
  if (type.isSignlessInteger(64))
    return Functor<uint64_t>()(std::forward<Args>(args)...);
  if (type.isInteger(64))
    return Functor<int64_t>()(std::forward<Args>(args)...);
  if (type.isF16()) return Functor<uint16_t>()(std::forward<Args>(args)...);
  if (type.isF32()) return Functor<float>()(std::forward<Args>(args)...);
  if (type.isF64()) return Functor<double>()(std::forward<Args>(args)...);
  if (auto complexTy = dyn_cast<ComplexType>(type)) {
    auto complexElemTy = complexTy.getElementType();

    // Use a double data type to serialize the real (32bit) and imaginary (32
    // bit) components of a complex number.
    if (complexElemTy.isF32())
      return Functor<double>()(std::forward<Args>(args)...);

#ifdef __SIZEOF_INT128__
    // Only some platforms support 128bit native data types required to
    // to serialize f64 complex element types.
    if (complexElemTy.isF64())
      return Functor<__int128>()(std::forward<Args>(args)...);
#endif
  }

  llvm::report_fatal_error("Unknown type");
}

}  // namespace

llvm::ErrorOr<Tensor> deserializeTensor(StringRef filename, ShapedType type) {
  if (llvm::endianness::native == llvm::endianness::big)
    llvm::report_fatal_error("Only little endian supported.");

  return dispatchType<FromNumpy>(type.getElementType(), filename, type);
}

llvm::Error serializeTensor(StringRef filename, ShapedType type,
                            const char* data) {
  if (llvm::endianness::native == llvm::endianness::big)
    llvm::report_fatal_error("Only little endian supported.");

  return dispatchType<ToNumpy>(type.getElementType(), filename, type, data);
}

}  // namespace numpy
}  // namespace stablehlo
}  // namespace mlir
