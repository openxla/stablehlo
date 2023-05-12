/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "stablehlo/integrations/python/PortableApi.h"

#include <string>

#include "stablehlo/api/PortableApi.h"

namespace py = pybind11;

namespace mlir {
namespace stablehlo {

void AddPortableApi(py::module& m) {
  //
  // Utility APIs.
  //

  m.def("get_api_version", []() { return getApiVersion(); });

  //
  // Serialization APIs.
  //

  m.def("get_current_version", []() { return getCurrentVersion(); });

  m.def("get_minimum_version", []() { return getMinimumVersion(); });

  m.def(
      "serialize_portable_artifact",
      [](std::string moduleStr, std::string targetVersion) -> py::bytes {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        if (failed(serializePortableArtifact(moduleStr, targetVersion, os))) {
          PyErr_SetString(PyExc_ValueError, "failed to serialize module");
          return "";
        }

        return py::bytes(buffer);
      },
      py::arg("module_str"), py::arg("target_version"));

  m.def(
      "deserialize_portable_artifact",
      [](std::string artifactStr) -> py::bytes {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        if (failed(deserializePortableArtifact(artifactStr, os))) {
          PyErr_SetString(PyExc_ValueError, "failed to deserialize module");
          return "";
        }

        return py::bytes(buffer);
      },
      py::arg("artifact_str"));
}

}  // namespace stablehlo
}  // namespace mlir
