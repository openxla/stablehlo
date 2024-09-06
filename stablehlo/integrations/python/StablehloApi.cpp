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

#include "stablehlo/integrations/python/StablehloApi.h"

#include <string>
#include <string_view>

#include "llvm/Support/raw_ostream.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "stablehlo/integrations/c/StablehloApi.h"

namespace py = pybind11;

namespace mlir {
namespace stablehlo {

// A helper class that implements `MlirStringCallback` by printing parts into a
// C++ string.
class StringWriterHelper {
 public:
  StringWriterHelper() : ss_(s_) {}

  static MlirStringCallback getMlirStringCallback() {
    return [](MlirStringRef string_ref, void *user_data) {
      auto *helper = static_cast<StringWriterHelper *>(user_data);
      helper->ss_ << llvm::StringRef(string_ref.data, string_ref.length);
    };
  }

  void *getUserData() { return static_cast<void *>(this); }

  const std::string &toString() {
    ss_.flush();
    return s_;
  }

 private:
  std::string s_;
  llvm::raw_string_ostream ss_;
};

static MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

static MlirStringRef toMlirStringRef(std::string_view s) {
  return mlirStringRefCreate(s.data(), s.size());
}

void AddStablehloApi(py::module &m) {
  // Portable API is a subset of StableHLO API
  AddPortableApi(m);

  //
  // Utility APIs.
  //
  py::enum_<MlirStablehloCompatibilityRequirement>(
      m, "StablehloCompatibilityRequirement")
      .value("NONE", MlirStablehloCompatibilityRequirement::NONE)
      .value("WEEK_4", MlirStablehloCompatibilityRequirement::WEEK_4)
      .value("WEEK_12", MlirStablehloCompatibilityRequirement::WEEK_12)
      .value("MAX", MlirStablehloCompatibilityRequirement::MAX);

  m.def(
      "get_version_from_compatibility_requirement",
      [](MlirStablehloCompatibilityRequirement requirement) -> py::str {
        StringWriterHelper accumulator;
        stablehloVersionFromCompatibilityRequirement(
            requirement, accumulator.getMlirStringCallback(),
            accumulator.getUserData());
        return accumulator.toString();
      },
      py::arg("requirement"));

  //
  // Serialization APIs.
  //
  m.def(
      "serialize_portable_artifact",
      [](MlirModule module, std::string_view target) -> py::bytes {
        StringWriterHelper accumulator;
        if (mlirLogicalResultIsFailure(
                stablehloSerializePortableArtifactFromModule(
                    module, toMlirStringRef(target),
                    accumulator.getMlirStringCallback(),
                    accumulator.getUserData()))) {
          PyErr_SetString(PyExc_ValueError, "failed to serialize module");
          return "";
        }

        return py::bytes(accumulator.toString());
      },
      py::arg("module"), py::arg("target"));

  m.def(
      "deserialize_portable_artifact",
      [](MlirContext context, std::string_view artifact) -> MlirModule {
        auto module = stablehloDeserializePortableArtifactNoError(
            toMlirStringRef(artifact), context);
        if (mlirModuleIsNull(module)) {
          PyErr_SetString(PyExc_ValueError, "failed to deserialize module");
          return {};
        }
        return module;
      },
      py::arg("context"), py::arg("artifact"));

  //
  // Reference APIs
  //
  m.def(
      "eval_module",
      [](MlirModule module,
         std::vector<MlirAttribute> &args) -> std::vector<MlirAttribute> {
        for (auto arg : args) {
          if (!mlirAttributeIsADenseElements(arg)) {
            PyErr_SetString(PyExc_ValueError,
                            "input args must be DenseElementsAttr");
            return {};
          }
        }

        int errorCode(0);
        MlirAttribute resultArrayAttr =
            stablehloEvalModule(module, args.size(), args.data(), &errorCode);

        if (errorCode != 0) {
          PyErr_SetString(PyExc_ValueError, "interpreter failed");
          return {};
        }

        std::vector<MlirAttribute> pyResults;
        for (int i = 0; i < mlirArrayAttrGetNumElements(resultArrayAttr); i++) {
          pyResults.push_back(mlirArrayAttrGetElement(resultArrayAttr, i));
        }
        return pyResults;
      },
      py::arg("module"), py::arg("args"));
}

void AddPortableApi(py::module &m) {
  //
  // Utility APIs.
  //
  m.def("get_api_version", []() { return stablehloGetApiVersion(); });

  m.def(
      "get_smaller_version",
      [](const std::string &version1, const std::string &version2) -> py::str {
        StringWriterHelper accumulator;
        if (mlirLogicalResultIsFailure(stablehloGetSmallerVersion(
                toMlirStringRef(version1), toMlirStringRef(version2),
                accumulator.getMlirStringCallback(),
                accumulator.getUserData()))) {
          PyErr_SetString(PyExc_ValueError,
                          "failed to convert version to stablehlo version");
          return "";
        }
        return accumulator.toString();
      },
      py::arg("version1"), py::arg("version2"));

  m.def("get_current_version", []() -> py::str {
    StringWriterHelper accumulator;
    stablehloGetCurrentVersion(accumulator.getMlirStringCallback(),
                               accumulator.getUserData());
    return accumulator.toString();
  });

  m.def("get_minimum_version", []() -> py::str {
    StringWriterHelper accumulator;
    stablehloGetMinimumVersion(accumulator.getMlirStringCallback(),
                               accumulator.getUserData());
    return accumulator.toString();
  });

  //
  // Serialization APIs.
  //
  m.def(
      "serialize_portable_artifact_str",
      [](std::string_view moduleStrOrBytecode,
         std::string_view targetVersion) -> py::bytes {
        StringWriterHelper accumulator;
        if (mlirLogicalResultIsFailure(
                stablehloSerializePortableArtifactFromStringRef(
                    toMlirStringRef(moduleStrOrBytecode),
                    toMlirStringRef(targetVersion),
                    accumulator.getMlirStringCallback(),
                    accumulator.getUserData()))) {
          PyErr_SetString(PyExc_ValueError, "failed to serialize module");
          return "";
        }
        return py::bytes(accumulator.toString());
      },
      py::arg("module_str"), py::arg("target_version"));

  m.def(
      "deserialize_portable_artifact_str",
      [](std::string_view artifact) -> py::bytes {
        StringWriterHelper accumulator;
        if (mlirLogicalResultIsFailure(stablehloDeserializePortableArtifact(
                toMlirStringRef(artifact), accumulator.getMlirStringCallback(),
                accumulator.getUserData()))) {
          PyErr_SetString(PyExc_ValueError, "failed to deserialize module");
          return "";
        }
        return py::bytes(accumulator.toString());
      },
      py::arg("artifact_str"));
}

}  // namespace stablehlo
}  // namespace mlir
