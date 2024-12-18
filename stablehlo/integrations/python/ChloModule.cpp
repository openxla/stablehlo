/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string_view.h"
#include "stablehlo/integrations/c/ChloAttributes.h"
#include "stablehlo/integrations/c/ChloDialect.h"

namespace nb = nanobind;

namespace {

auto toPyString(MlirStringRef mlirStringRef) {
  return nb::str(mlirStringRef.data, mlirStringRef.length);
}

}  // namespace

NB_MODULE(_chlo, m) {
  m.doc() = "chlo main python extension";

  //
  // Dialects.
  //

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__chlo__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  //
  // Attributes.
  //

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "ComparisonDirectionAttr", chloAttributeIsAComparisonDirectionAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, std::string_view value, MlirContext ctx) {
            return cls(chloComparisonDirectionAttrGet(
                ctx, mlirStringRefCreate(value.data(), value.size())));
          },
          nb::arg("cls"), nb::arg("value"),
          nb::arg("context").none() = nb::none(),
          "Creates a ComparisonDirection attribute with the given value.")
      .def_property_readonly("value", [](MlirAttribute self) {
        return toPyString(chloComparisonDirectionAttrGetValue(self));
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "ComparisonTypeAttr", chloAttributeIsAComparisonTypeAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, std::string_view value, MlirContext ctx) {
            return cls(chloComparisonTypeAttrGet(
                ctx, mlirStringRefCreate(value.data(), value.size())));
          },
          nb::arg("cls"), nb::arg("value"),
          nb::arg("context").none() = nb::none(),
          "Creates a ComparisonType attribute with the given value.")
      .def_property_readonly("value", [](MlirAttribute self) {
        return toPyString(chloComparisonTypeAttrGetValue(self));
      });
}
