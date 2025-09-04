/* Copyright 2025 The OpenXLA Authors.

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

#ifndef STABLEHLO_BUILDER_MLIRBUILDER_H_
#define STABLEHLO_BUILDER_MLIRBUILDER_H_

#include <functional>
#include <source_location>
#include <string>
#include <utility>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace mlir {

// Forward declare
class MlirBuilder;

// We need a value wrapper in order to hold onto the Builder that knows where
// all future ops that use this op should be inserted.
class MlirOp {
 public:
  MlirOp() : builder_(), value_() {}
  MlirOp(MlirBuilder& builder, Value value)
      : builder_(&builder), value_(value) {}
  MlirBuilder& getBuilder() { return *builder_; }
  Value getValue() const { return value_; }
  Type getType() const { return value_.getType(); }
  MLIRContext& getContext() { return *value_.getContext(); }

  // Op is only invalid if default constructor is used.
  // This is needed for conditionals, create a placeholder and set in if/else.
  bool isValid() const { return builder_ != nullptr; }

  std::string ToString() const {
    std::string valueAsString;
    llvm::raw_string_ostream stream(valueAsString);
    value_.print(stream);
    return valueAsString;
  }

 private:
  // Pointer not reference since this class must be copyable.
  MlirBuilder* builder_;
  Value value_;
};

// Base builder class that provides a reference to the MLIRContext and provides
// Utilities that all builder subclasses can use.
class MlirBuilder {
 public:
  MlirBuilder(MLIRContext& context, Location loc)
      : builder_(&context), loc_(loc) {}
  MlirBuilder(OpBuilder& builder, Location loc)
      : builder_(builder), loc_(loc) {}

  MlirBuilder(const MlirBuilder&) = delete;
  MLIRContext& getContext() { return *builder_.getContext(); }
  OpBuilder& getOpBuilder() { return builder_; }
  Location getLoc() { return loc_; }
  void setLoc(Location loc) { loc_ = loc; }

  // Forward to generated op builder using existing location / context.
  template <typename OpTy, typename... Args>
  MlirOp create(Args&&... args) {
    return MlirOp(*this,
                  OpTy::create(builder_, loc_, std::forward<Args>(args)...));
  }

  // Forward to generated op builder with no results using existing location /
  // context.
  template <typename OpTy, typename... Args>
  void create0(Args&&... args) {
    OpTy::create(builder_, loc_, std::forward<Args>(args)...);
  }

  template <typename OpTy, typename... Args>
  SmallVector<MlirOp> createVariadic(Args&&... args) {
    ValueRange values =
        OpTy::create(builder_, loc_, std::forward<Args>(args)...).getResults();
    SmallVector<MlirOp> ret;
    for (Value value : values) {
      ret.emplace_back(*this, value);
    }
    return ret;
  }

  // Forward to generated op builder using existing location / context.
  // Used for ops with multiple results, but with a known number of results.
  template <typename OpTy, int N, typename... Args>
  SmallVector<MlirOp> createN(Args&&... args) {
    SmallVector<MlirOp> ret = createVariadic<OpTy>(std::forward<Args>(args)...);
    if (ret.size() != N)
      llvm::report_fatal_error("Expected " + Twine(N) + " results from " +
                               OpTy::getOperationName() + ", got " +
                               Twine(ret.size()));
    return ret;
  }

  // Forward to generated op builder with no results using existing location /
  // context.
  template <typename OpTy, typename... Args>
  OpTy createUnwrapped(Args&&... args) {
    return OpTy::create(builder_, loc_, std::forward<Args>(args)...);
  }

  Type getTensorOfShape();

 protected:
  OpBuilder builder_;
  Location loc_;
};

class ModuleBuilder : public MlirBuilder {
 public:
  ModuleBuilder(MLIRContext& context, Location loc, StringRef name = "")
      : MlirBuilder(context, loc), module_(ModuleOp::create(loc)) {
    builder_.setInsertionPointToStart(module_->getBody());
    if (!name.empty()) module_->setName(name);
  }

  // Optional Location argument, populate with unknown location if not provided.
  explicit ModuleBuilder(MLIRContext& context, StringRef name = "")
      : ModuleBuilder(context, UnknownLoc::get(&context), name) {}

  // Note this method can only be called once.
  OwningOpRef<ModuleOp> build() { return std::move(module_); }

 private:
  OwningOpRef<ModuleOp> module_;
};

// Default Region Builder.
// This is a class that can be used in auto-generated bindings for ops that
// have a body region.
//
// A region builder is passed in a callback function, created with a reference
// to a region of the op being constructed. Ops with multiple regions will
// pass multiple RegionBuilders.
//
// stablehlo::Reduce({arg0, arg1}, [](RegionBuilder& rb){
//   auto type = makeTensorType(rb.getContext(), {}, ElementType::I64);
//   auto regArg0 = Argument(rb, type);
//   auto regArg1 = Argument(rb, type);
//   auto add = Add(regArg0, regArg1);
//   func::Return(rb, add);
// });
//
// For highly used ops with regions, it may make more sense for dialects to
// create more declarative builders for better UX (FuncBuilder, for example).
class RegionBuilder : public MlirBuilder {
 public:
  RegionBuilder(MlirBuilder& builder, Region& region)
      : MlirBuilder(builder.getOpBuilder(), builder.getLoc()),
        region_(&region) {
    // TODO: Only handles single-block regions.
    // Consider passing in an int region ID to handle multiple blocks.
    if (region_->getBlocks().empty()) builder_.createBlock(region_);
    builder_.setInsertionPointToStart(&region_->getBlocks().front());
  }

  // Create a region builder for a given block, do not emplace a new block.
  RegionBuilder(MlirBuilder& builder, Block& block)
      : MlirBuilder(builder.getOpBuilder(), builder.getLoc()),
        region_(block.getParent()) {
    // Build at end of block.
    builder_.setInsertionPointToEnd(&block);
  }

  Region& getRegion() { return *region_; }

  template <typename OpTy>
  OpTy getOp() {
    return region_->getParentOfType<OpTy>();
  }

 private:
  Region* region_;
};

// Add an argument to the region body.
MlirOp Argument(RegionBuilder& rb, Type type);

using RegionBuilderCallback = std::function<void(RegionBuilder&)>;

// A helper class for building ops that have a body region.
template <typename OpTy>
class RegionOpBuilder : public MlirBuilder {
 public:
  RegionOpBuilder(MlirBuilder& builder, OpTy op)
      : MlirBuilder(builder.getOpBuilder(), op.getLoc()), op_(op), region_() {
    region_ = &op_->getRegion(0);
    builder_.setInsertionPointToStart(&region_->emplaceBlock());
  }

  virtual ~RegionOpBuilder() = default;

  OpTy& build() { return op_; }

  RegionBuilder getRegionBuilder() {
    return RegionBuilder(*this, region_->getBlocks().front());
  }

 protected:
  // A callback that can be overridden by subclasses to be notified when the
  // signature of the region changes.
  //
  // This includes on calls to Return and Argument.
  virtual void notifySignatureChanged() {}

  // Protected op reference, users should only retrieve ops via `build`, but
  // builders can access the op in an incomplete state.
  OpTy getOp() { return op_; }

 private:
  OpTy op_;
  Region* region_;
};

///////////////
// Builtin Dialect Helpers
///////////////

Value unwrap(MlirOp const& value);
SmallVector<Value> unwrap(ArrayRef<MlirOp> values);
SmallVector<MlirOp> wrap(MlirBuilder& builder, ValueRange values);

// Change the builder associated with the MlirOp value.
MlirOp swap(MlirBuilder& builder, MlirOp& value);

// RAII class for temporarily changing the builder location.
class ScopedBuilderLocation {
 public:
  // Sets the builder location to `loc`.
  ScopedBuilderLocation(MlirBuilder& builder, Location loc)
      : builder_(builder), prev_(builder.getLoc()) {
    builder.setLoc(loc);
  }

  // Restores the builder location to the previous value.
  ~ScopedBuilderLocation() { builder_.setLoc(prev_); }

 protected:
  MlirBuilder& builder_;
  const Location prev_;
};

}  // namespace mlir

#endif  // STABLEHLO_BUILDER_MLIRBUILDER_H_
