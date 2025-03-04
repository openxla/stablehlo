// RUN: stablehlo-opt %s -stablehlo-compatibility-expander=target=1.8.0 --mlir-print-debuginfo | FileCheck %s
// RUN: stablehlo-opt %s -stablehlo-compatibility-expander=target=1.8.0 --mlir-print-debuginfo | stablehlo-translate --serialize --target=1.8.0 | stablehlo-translate -deserialize -mlir-print-debuginfo | FileCheck %s

// Test that FileLineColRange locations are converted to FileLineColLoc
// locations, including in nested location contexts, block args, module op, etc.
// Ex: loc("file.mlir":2:21 to :30) ==> loc("file.mlir":2:21)

#loc3 = loc("file.mlir":2:21 to :30)
module {
  func.func @main(%arg0: tensor<i32> loc("file.mlir":2:21 to :30)) -> tensor<i32> {
    %c = stablehlo.constant dense<1> : tensor<i32> loc(#loc4)
    %0 = stablehlo.add %arg0, %c : tensor<i32> loc(#loc5)
    return %0 : tensor<i32> loc(#loc6)
  } loc(#loc9)
} loc(#loc)
#loc = loc("file.mlir":0:0 to :3)
#loc1 = loc("file.mlir":1:1 to :2)
#loc2 = loc("file.mlir":2:19 to :20)
#loc4 = loc("file.mlir":2:8 to :10)
#loc5 = loc("file.mlir":4:10 to :12)
#loc6 = loc("file.mlir":3:3 to :5)
#loc7 = loc("WrappedLocation.call"(#loc1))
#loc8 = loc("WrappedLocation.callsite"(#loc2))
#loc9 = loc(callsite(#loc7 at #loc8))

// CHECK:      #[[LOC3:.*]] = loc("file.mlir":2:21)
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func @main{{.*}}tensor<i32> loc("file.mlir":2:21)
// CHECK-NEXT:     stablehlo.constant {{.*}} loc(#[[LOC4:.*]])
// CHECK-NEXT:     stablehlo.add {{.*}} : tensor<i32> loc(#[[LOC5:.*]])
// CHECK-NEXT:     return {{.*}} loc(#[[LOC6:.*]])
// CHECK-NEXT:   } loc(#[[LOC9:.*]])
// CHECK-NEXT: } loc(#[[LOC:.*]])
// CHECK-NEXT: #[[LOC]]     = loc("file.mlir":0:0)
// CHECK-NEXT: #[[LOC1:.*]] = loc("file.mlir":1:1)
// CHECK-NEXT: #[[LOC2:.*]] = loc("file.mlir":2:19)
// CHECK-NEXT: #[[LOC4]]    = loc("file.mlir":2:8)
// CHECK-NEXT: #[[LOC5]]    = loc("file.mlir":4:10)
// CHECK-NEXT: #[[LOC6]]    = loc("file.mlir":3:3)
// CHECK-NEXT: #[[LOC7:.*]] = loc("WrappedLocation.call"(#[[LOC1]]))
// CHECK-NEXT: #[[LOC8:.*]] = loc("WrappedLocation.callsite"(#[[LOC2]]))
// CHECK-NEXT: #[[LOC9]] = loc(callsite(#[[LOC7]] at #[[LOC8]]))

