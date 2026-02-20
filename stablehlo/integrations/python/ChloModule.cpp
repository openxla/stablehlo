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

#include "llvm/ADT/STLExtras.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string_view.h"
#include "stablehlo/integrations/c/ChloAttributes.h"
#include "stablehlo/integrations/c/ChloDialect.h"

namespace nb = nanobind;

namespace {

// Returns a vector containing integers extracted from an attribute using the
// two provided callbacks.
std::vector<int64_t> attributePropertyVector(
    MlirAttribute attr, llvm::function_ref<intptr_t(MlirAttribute)> sizeFn,
    llvm::function_ref<int64_t(MlirAttribute, intptr_t)> getFn) {
  std::vector<int64_t> result;
  intptr_t size = sizeFn(attr);
  result.reserve(size);
  for (intptr_t i = 0; i < size; ++i) {
    result.push_back(getFn(attr, i));
  }
  return result;
}

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

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "RaggedDotDimensionNumbers", chloAttributeIsARaggedDotDimensionNumbers)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::vector<int64_t> &lhsBatchingDims,
             const std::vector<int64_t> &rhsBatchingDims,
             const std::vector<int64_t> &lhsContractingDims,
             const std::vector<int64_t> &rhsContractingDims,
             const std::vector<int64_t> &lhsRaggedDims,
             const std::vector<int64_t> &rhsGroupDims, MlirContext ctx) {
            return cls(chloRaggedDotDimensionNumbersGet(
                ctx, lhsBatchingDims.size(), lhsBatchingDims.data(),
                rhsBatchingDims.size(), rhsBatchingDims.data(),
                lhsContractingDims.size(), lhsContractingDims.data(),
                rhsContractingDims.size(), rhsContractingDims.data(),
                lhsRaggedDims.size(), lhsRaggedDims.data(), rhsGroupDims.size(),
                rhsGroupDims.data()));
          },
          nb::arg("cls"), nb::arg("lhs_batching_dimensions"),
          nb::arg("rhs_batching_dimensions"),
          nb::arg("lhs_contracting_dimensions"),
          nb::arg("rhs_contracting_dimensions"),
          nb::arg("lhs_ragged_dimensions"), nb::arg("rhs_group_dimensions"),
          nb::arg("context").none() = nb::none(),
          "Creates a RaggedDotDimensionNumbers attribute with the given "
          "dimension configuration.")
      .def_property_readonly(
          "lhs_batching_dimensions",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, chloRaggedDotDimensionNumbersGetLhsBatchingDimensionsSize,
                chloRaggedDotDimensionNumbersGetLhsBatchingDimensionsElem);
          })
      .def_property_readonly(
          "rhs_batching_dimensions",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, chloRaggedDotDimensionNumbersGetRhsBatchingDimensionsSize,
                chloRaggedDotDimensionNumbersGetRhsBatchingDimensionsElem);
          })
      .def_property_readonly(
          "lhs_contracting_dimensions",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self,
                chloRaggedDotDimensionNumbersGetLhsContractingDimensionsSize,
                chloRaggedDotDimensionNumbersGetLhsContractingDimensionsElem);
          })
      .def_property_readonly(
          "rhs_contracting_dimensions",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self,
                chloRaggedDotDimensionNumbersGetRhsContractingDimensionsSize,
                chloRaggedDotDimensionNumbersGetRhsContractingDimensionsElem);
          })
      .def_property_readonly(
          "lhs_ragged_dimensions",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, chloRaggedDotDimensionNumbersGetLhsRaggedDimensionsSize,
                chloRaggedDotDimensionNumbersGetLhsRaggedDimensionsElem);
          })
      .def_property_readonly("rhs_group_dimensions", [](MlirAttribute self) {
        return attributePropertyVector(
            self, chloRaggedDotDimensionNumbersGetRhsGroupDimensionsSize,
            chloRaggedDotDimensionNumbersGetRhsGroupDimensionsElem);
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "PrecisionAttr", chloAttributeIsAPrecisionAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::string &value, MlirContext ctx) {
            return cls(chloPrecisionAttrGet(
                ctx, mlirStringRefCreate(value.c_str(), value.size())));
          },
          nb::arg("cls"), nb::arg("value"),
          nb::arg("context").none() = nb::none(),
          "Creates a Precision attribute with the given value.")
      .def_property_readonly("value", [](MlirAttribute self) {
        return toPyString(chloPrecisionAttrGetValue(self));
      });
}
