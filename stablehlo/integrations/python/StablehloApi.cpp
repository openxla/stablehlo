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

#include <stdexcept>
#include <string>
#include <string_view>

#include "llvm/Support/raw_ostream.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"
#include "nanobind/stl/vector.h"
#include "stablehlo/integrations/c/StablehloUnifiedApi.h"

namespace nb = nanobind;

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

static MlirStringRef toMlirStringRef(const nb::bytes &s) {
  return mlirStringRefCreate(static_cast<const char *>(s.data()), s.size());
}

void AddStablehloApi(nb::module_ &m) {
  // Portable API is a subset of StableHLO API
  AddPortableApi(m);

  //
  // Utility APIs.
  //
  nb::enum_<MlirStablehloCompatibilityRequirement>(
      m, "StablehloCompatibilityRequirement")
      .value("NONE", MlirStablehloCompatibilityRequirement::NONE)
      .value("WEEK_4", MlirStablehloCompatibilityRequirement::WEEK_4)
      .value("WEEK_12", MlirStablehloCompatibilityRequirement::WEEK_12)
      .value("MAX", MlirStablehloCompatibilityRequirement::MAX);

  m.def(
      "get_version_from_compatibility_requirement",
      [](MlirStablehloCompatibilityRequirement requirement) -> std::string {
        StringWriterHelper accumulator;
        stablehloVersionFromCompatibilityRequirement(
            requirement, accumulator.getMlirStringCallback(),
            accumulator.getUserData());
        return accumulator.toString();
      },
      nb::arg("requirement"));

  //
  // Serialization APIs.
  //
  m.def(
      "serialize_portable_artifact",
      [](MlirModule module, std::string_view target) -> nb::bytes {
        StringWriterHelper accumulator;
        if (mlirLogicalResultIsFailure(
                stablehloSerializePortableArtifactFromModule(
                    module, toMlirStringRef(target),
                    accumulator.getMlirStringCallback(),
                    accumulator.getUserData()))) {
          throw nb::value_error("failed to serialize module");
        }

        std::string serialized = accumulator.toString();
        return nb::bytes(serialized.data(), serialized.size());
      },
      nb::arg("module"), nb::arg("target"));

  m.def(
      "deserialize_portable_artifact",
      [](MlirContext context, std::string_view artifact) -> MlirModule {
        auto module = stablehloDeserializePortableArtifactNoError(
            toMlirStringRef(artifact), context);
        if (mlirModuleIsNull(module)) {
          throw nb::value_error("failed to deserialize module");
        }
        return module;
      },
      nb::arg("context"), nb::arg("artifact"));
  m.def(
      "deserialize_portable_artifact",
      [](MlirContext context, nb::bytes artifact) -> MlirModule {
        auto module = stablehloDeserializePortableArtifactNoError(
            toMlirStringRef(artifact), context);
        if (mlirModuleIsNull(module)) {
          throw nb::value_error("failed to deserialize module");
        }
        return module;
      },
      nb::arg("context"), nb::arg("artifact"));
  //
  // Reference APIs
  //
  m.def(
      "eval_module",
      [](MlirModule module,
         std::vector<MlirAttribute> &args) -> std::vector<MlirAttribute> {
        for (auto arg : args) {
          if (!mlirAttributeIsADenseElements(arg)) {
            throw nb::value_error("input args must be DenseElementsAttr");
          }
        }

        int errorCode(0);
        MlirAttribute resultArrayAttr =
            stablehloEvalModule(module, args.size(), args.data(), &errorCode);

        if (errorCode != 0) {
          throw nb::value_error("interpreter failed");
        }

        std::vector<MlirAttribute> pyResults;
        for (int i = 0; i < mlirArrayAttrGetNumElements(resultArrayAttr); i++) {
          pyResults.push_back(mlirArrayAttrGetElement(resultArrayAttr, i));
        }
        return pyResults;
      },
      nb::arg("module"), nb::arg("args"));
}

void AddPortableApi(nb::module_ &m) {
  //
  // Utility APIs.
  //
  m.def("get_api_version", []() { return stablehloGetApiVersion(); });

  m.def(
      "get_smaller_version",
      [](const std::string &version1,
         const std::string &version2) -> std::string {
        StringWriterHelper accumulator;
        if (mlirLogicalResultIsFailure(stablehloGetSmallerVersion(
                toMlirStringRef(version1), toMlirStringRef(version2),
                accumulator.getMlirStringCallback(),
                accumulator.getUserData()))) {
          throw nb::value_error(
              "failed to convert version to stablehlo version");
        }
        return accumulator.toString();
      },
      nb::arg("version1"), nb::arg("version2"));

  m.def("get_current_version", []() -> std::string {
    StringWriterHelper accumulator;
    stablehloGetCurrentVersion(accumulator.getMlirStringCallback(),
                               accumulator.getUserData());
    return accumulator.toString();
  });

  m.def("get_minimum_version", []() -> std::string {
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
         std::string_view targetVersion) -> nb::bytes {
        StringWriterHelper accumulator;
        if (mlirLogicalResultIsFailure(
                stablehloSerializePortableArtifactFromStringRef(
                    toMlirStringRef(moduleStrOrBytecode),
                    toMlirStringRef(targetVersion),
                    accumulator.getMlirStringCallback(),
                    accumulator.getUserData()))) {
          throw nb::value_error("failed to serialize module");
        }
        std::string serialized = accumulator.toString();
        return nb::bytes(serialized.data(), serialized.size());
      },
      nb::arg("module_str"), nb::arg("target_version"));
  m.def(
      "serialize_portable_artifact_str",
      [](nb::bytes moduleStrOrBytecode,
         std::string_view targetVersion) -> nb::bytes {
        StringWriterHelper accumulator;
        if (mlirLogicalResultIsFailure(
                stablehloSerializePortableArtifactFromStringRef(
                    toMlirStringRef(moduleStrOrBytecode),
                    toMlirStringRef(targetVersion),
                    accumulator.getMlirStringCallback(),
                    accumulator.getUserData()))) {
          throw nb::value_error("failed to serialize module");
        }
        std::string serialized = accumulator.toString();
        return nb::bytes(serialized.data(), serialized.size());
      },
      nb::arg("module_str"), nb::arg("target_version"));

  m.def(
      "deserialize_portable_artifact_str",
      [](std::string_view artifact) -> nb::bytes {
        StringWriterHelper accumulator;
        if (mlirLogicalResultIsFailure(stablehloDeserializePortableArtifact(
                toMlirStringRef(artifact), accumulator.getMlirStringCallback(),
                accumulator.getUserData()))) {
          throw nb::value_error("failed to deserialize module");
        }
        std::string serialized = accumulator.toString();
        return nb::bytes(serialized.data(), serialized.size());
      },
      nb::arg("artifact_str"));
  m.def(
      "deserialize_portable_artifact_str",
      [](const nb::bytes &artifact) -> nb::bytes {
        StringWriterHelper accumulator;
        if (mlirLogicalResultIsFailure(stablehloDeserializePortableArtifact(
                toMlirStringRef(artifact), accumulator.getMlirStringCallback(),
                accumulator.getUserData()))) {
          throw nb::value_error("failed to deserialize module");
        }
        std::string serialized = accumulator.toString();
        return nb::bytes(serialized.data(), serialized.size());
      },
      nb::arg("artifact_str"));
}

}  // namespace stablehlo
}  // namespace mlir
