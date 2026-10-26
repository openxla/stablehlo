// RUN: stablehlo-opt %s --stablehlo-refine-shapes --split-input-file --verify-diagnostics | FileCheck %s
// RUN: stablehlo-opt %s --stablehlo-refine-shapes --stablehlo-refine-shapes --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: module @operand_types
module @operand_types {
  // CHECK: func.func @main(%[[ARG:.*]]: tensor<4xf32>) -> tensor<4xf32>
  func.func @main(%arg0: tensor<4xf32>) -> tensor<?xf32> {
    %0 = stablehlo.bitcast_convert %arg0 : (tensor<4xf32>) -> tensor<?xf32>
    // CHECK: stablehlo.composite "test.identity" %[[ARG]] {composite_attributes = {tag = "keep"}, decomposition = @identity, version = 7 : i32} : (tensor<4xf32>) -> tensor<4xf32>
    %1 = stablehlo.composite "test.identity" %0 {composite_attributes = {tag = "keep"}, decomposition = @identity, version = 7 : i32} : (tensor<?xf32>) -> tensor<?xf32>
    return %1 : tensor<?xf32>
  }
  // CHECK: func.func private @identity(%[[ARG:.*]]: tensor<4xf32>) -> tensor<4xf32>
  // CHECK: return %[[ARG]] : tensor<4xf32>
  func.func private @identity(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    return %arg0 : tensor<?xf32>
  }
}

// -----

// CHECK-LABEL: module @shared_decomposition
module @shared_decomposition {
  // CHECK: func.func @main() -> (tensor<4xi32>, tensor<8xi32>, tensor<4xi32>)
  func.func @main() -> (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) {
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %c8 = stablehlo.constant dense<8> : tensor<i32>
    // CHECK: stablehlo.composite "test.iota" {{.*}} {decomposition = @[[FOUR:[a-zA-Z0-9_.]+]]} : (tensor<i32>) -> tensor<4xi32>
    %0 = stablehlo.composite "test.iota" %c4 {decomposition = @iota} : (tensor<i32>) -> tensor<?xi32>
    // CHECK: stablehlo.composite "test.iota" {{.*}} {decomposition = @[[EIGHT:[a-zA-Z0-9_.]+]]} : (tensor<i32>) -> tensor<8xi32>
    %1 = stablehlo.composite "test.iota" %c8 {decomposition = @iota} : (tensor<i32>) -> tensor<?xi32>
    // Reuse the specialization for the same shape constant.
    // CHECK: stablehlo.composite "test.iota" {{.*}} {decomposition = @[[FOUR]]} : (tensor<i32>) -> tensor<4xi32>
    %2 = stablehlo.composite "test.iota" %c4 {decomposition = @iota} : (tensor<i32>) -> tensor<?xi32>
    return %0, %1, %2 : tensor<?xi32>, tensor<?xi32>, tensor<?xi32>
  }
  func.func private @iota(%arg0: tensor<i32>) -> tensor<?xi32> {
    %shape = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %0 = stablehlo.dynamic_iota %shape, dim = 0 : (tensor<1xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
  // CHECK-DAG: func.func private @[[FOUR]](%{{.*}}: tensor<i32>) -> tensor<4xi32>
  // CHECK-DAG: func.func private @[[EIGHT]](%{{.*}}: tensor<i32>) -> tensor<8xi32>
}

// -----

// CHECK-LABEL: module @shared_operand_types
module @shared_operand_types {
  // CHECK: func.func @main({{.*}}) -> (tensor<4xf32>, tensor<8xf32>)
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<8xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = stablehlo.bitcast_convert %arg0 : (tensor<4xf32>) -> tensor<?xf32>
    %1 = stablehlo.bitcast_convert %arg1 : (tensor<8xf32>) -> tensor<?xf32>
    // CHECK: stablehlo.composite "test.identity" {{.*}} {decomposition = @[[FOUR:[a-zA-Z0-9_.]+]]} : (tensor<4xf32>) -> tensor<4xf32>
    %2 = stablehlo.composite "test.identity" %0 {decomposition = @identity} : (tensor<?xf32>) -> tensor<?xf32>
    // CHECK: stablehlo.composite "test.identity" {{.*}} {decomposition = @[[EIGHT:[a-zA-Z0-9_.]+]]} : (tensor<8xf32>) -> tensor<8xf32>
    %3 = stablehlo.composite "test.identity" %1 {decomposition = @identity} : (tensor<?xf32>) -> tensor<?xf32>
    return %2, %3 : tensor<?xf32>, tensor<?xf32>
  }
  // CHECK-DAG: func.func private @[[FOUR]](%{{.*}}: tensor<4xf32>) -> tensor<4xf32>
  // CHECK-DAG: func.func private @[[EIGHT]](%{{.*}}: tensor<8xf32>) -> tensor<8xf32>
  func.func private @identity(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    return %arg0 : tensor<?xf32>
  }
}

// -----

// CHECK-LABEL: module @shared_with_call
module @shared_with_call {
  // CHECK: func.func @main() -> (tensor<4xi32>, tensor<8xi32>)
  func.func @main() -> (tensor<?xi32>, tensor<?xi32>) {
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %c8 = stablehlo.constant dense<8> : tensor<i32>
    // CHECK: stablehlo.composite "test.iota" {{.*}} {decomposition = @[[IMPL:[a-zA-Z0-9_.]+]]} : (tensor<i32>) -> tensor<4xi32>
    %0 = stablehlo.composite "test.iota" %c4 {decomposition = @iota} : (tensor<i32>) -> tensor<?xi32>
    // CHECK: call @iota() : () -> tensor<8xi32>
    %1 = call @iota(%c8) : (tensor<i32>) -> tensor<?xi32>
    return %0, %1 : tensor<?xi32>, tensor<?xi32>
  }
  // CHECK: func.func private @iota() -> tensor<8xi32>
  func.func private @iota(%arg0: tensor<i32>) -> tensor<?xi32> {
    %shape = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %0 = stablehlo.dynamic_iota %shape, dim = 0 : (tensor<1xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
  // CHECK: func.func private @[[IMPL]](%{{.*}}: tensor<i32>) -> tensor<4xi32>
}

// -----

// CHECK-LABEL: module @call_before_composite
module @call_before_composite {
  // CHECK: func.func @main() -> (tensor<8xi32>, tensor<4xi32>)
  func.func @main() -> (tensor<?xi32>, tensor<?xi32>) {
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %c8 = stablehlo.constant dense<8> : tensor<i32>
    // CHECK: call @iota() : () -> tensor<8xi32>
    %0 = call @iota(%c8) : (tensor<i32>) -> tensor<?xi32>
    // CHECK: stablehlo.composite "test.iota" {{.*}} {decomposition = @[[IMPL:[a-zA-Z0-9_.]+]]} : (tensor<i32>) -> tensor<4xi32>
    %1 = stablehlo.composite "test.iota" %c4 {decomposition = @iota} : (tensor<i32>) -> tensor<?xi32>
    return %0, %1 : tensor<?xi32>, tensor<?xi32>
  }
  // CHECK: func.func private @iota() -> tensor<8xi32>
  func.func private @iota(%arg0: tensor<i32>) -> tensor<?xi32> {
    %shape = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %0 = stablehlo.dynamic_iota %shape, dim = 0 : (tensor<1xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
  // CHECK: func.func private @[[IMPL]](%{{.*}}: tensor<i32>) -> tensor<4xi32>
}

// -----

// CHECK-LABEL: module @nested_multiple_results
module @nested_multiple_results {
  // CHECK: func.func @main(%{{.*}}: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  func.func @main(%arg0: tensor<4xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = stablehlo.bitcast_convert %arg0 : (tensor<4xf32>) -> tensor<?xf32>
    // CHECK: stablehlo.composite "test.outer" {{.*}} : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
    %1:2 = stablehlo.composite "test.outer" %0 {decomposition = @outer} : (tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>)
    return %1#0, %1#1 : tensor<?xf32>, tensor<?xf32>
  }
  // CHECK: func.func private @outer(%{{.*}}: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  func.func private @outer(%arg0: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
    // CHECK: call @inner({{.*}}) : (tensor<4xf32>) -> tensor<4xf32>
    %0 = call @inner(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
    // CHECK: stablehlo.composite "test.inner" {{.*}} : (tensor<4xf32>) -> tensor<4xf32>
    %1 = stablehlo.composite "test.inner" %arg0 {decomposition = @identity} : (tensor<?xf32>) -> tensor<?xf32>
    return %0, %1 : tensor<?xf32>, tensor<?xf32>
  }
  func.func private @inner(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    return %arg0 : tensor<?xf32>
  }
  func.func private @identity(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    return %arg0 : tensor<?xf32>
  }
}

// -----

// CHECK-LABEL: module @token_and_dimension
module @token_and_dimension {
  // CHECK: func.func @main() -> (!stablehlo.token, tensor<4xf32>)
  func.func @main() -> (!stablehlo.token, tensor<?xf32>) {
    %token = stablehlo.create_token : !stablehlo.token
    %dim = stablehlo.constant dense<4> : tensor<i32>
    // CHECK: stablehlo.composite "test.token_iota" {{.*}} : (!stablehlo.token, tensor<i32>) -> (!stablehlo.token, tensor<4xf32>)
    %0:2 = stablehlo.composite "test.token_iota" %token, %dim {decomposition = @iota} : (!stablehlo.token, tensor<i32>) -> (!stablehlo.token, tensor<?xf32>)
    return %0#0, %0#1 : !stablehlo.token, tensor<?xf32>
  }
  // CHECK: func.func private @iota(%{{.*}}: !stablehlo.token, %{{.*}}: tensor<i32>) -> (!stablehlo.token, tensor<4xf32>)
  func.func private @iota(%arg0: !stablehlo.token, %arg1: tensor<i32>) -> (!stablehlo.token, tensor<?xf32>) {
    %shape = stablehlo.reshape %arg1 : (tensor<i32>) -> tensor<1xi32>
    %0 = stablehlo.dynamic_iota %shape, dim = 0 : (tensor<1xi32>) -> tensor<?xf32>
    return %arg0, %0 : !stablehlo.token, tensor<?xf32>
  }
}

// -----

// CHECK-LABEL: module @runtime_dimension
module @runtime_dimension {
  // CHECK: func.func @main(%[[DIM:.*]]: tensor<i32>) -> tensor<?xf32>
  func.func @main(%arg0: tensor<i32>) -> tensor<?xf32> {
    // CHECK: stablehlo.composite "test.iota" %[[DIM]] {decomposition = @iota} : (tensor<i32>) -> tensor<?xf32>
    %0 = stablehlo.composite "test.iota" %arg0 {decomposition = @iota} : (tensor<i32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  // CHECK: func.func private @iota(%{{.*}}: tensor<i32>) -> tensor<?xf32>
  func.func private @iota(%arg0: tensor<i32>) -> tensor<?xf32> {
    %shape = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %0 = stablehlo.dynamic_iota %shape, dim = 0 : (tensor<1xi32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// -----

module @recursive_decomposition {
  func.func @main() -> tensor<i32> {
    %0 = stablehlo.composite "test.recursive" {decomposition = @recursive} : () -> tensor<i32>
    return %0 : tensor<i32>
  }
  // expected-error@+1 {{Function recursive is being refined recursively}}
  func.func private @recursive() -> tensor<i32> {
    %0 = stablehlo.composite "test.recursive" {decomposition = @recursive} : () -> tensor<i32>
    return %0 : tensor<i32>
  }
}
