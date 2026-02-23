# [RFC] StableHLO Support for Mesh-Axes Replica Groups

Status: In Review<br/>
Initial version: 02/23/2026<br/>
Last updated: 02/23/2026<br/>
Discussion thread: N/A

## Overview

This RFC proposes extending StableHLO to support a new representation for
`replica_groups`. The traditional representation of replica groups uses a dense
tensor (or integer arrays). The new representation utilizes a mesh and
axes-based description, which is more concise and interpretable.

Currently, collective operations like `all_gather`, `all_reduce`,
`reduce_scatter`, `all_to_all`, and `collective_broadcast` accept the dense
integer array for `replica_groups`. We propose extending this definition to
accept the new mesh-axes format, alongside the existing representation.

## Proposal

To support mesh-axes Replica Groups natively in StableHLO, we add new attributes
to capture mesh and axis configurations, and we update the existing definition
of replica groups to allow for either the legacy `I64ElementsAttr` or the new
`ReplicaGroupMeshAxes` format.

### New Attributes

#### 1. SubAxisInfo

Encodes information about how a sub-axis is derived from a full axis. It
encapsulates two integers, a `pre_size` and a `size`.

```ebnf
SubAxisInfo ::= '(' pre_size ')' size
```

#### 2. AxisRef

Represents a reference to either a full mesh axis or a split sub-axis. It is
primarily described by a string `name` and optional `sub_axis_info`.

```mlir
#stablehlo.axis_ref<name = "foo", sub_axis_info = (1)2>
```

#### 3. ReplicaGroupMeshAxes

Represents an overall mesh-axes based replica group. It contains a
`FlatSymbolRefAttr` for the `mesh_name` and an `ArrayAttr` containing `AxisRef`
attributes for the `axes`.

```mlir
#stablehlo.replica_group_mesh_axes<
  mesh_name = @mesh,
  axes = [
    #stablehlo.axis_ref<name = "foo">,
    #stablehlo.axis_ref<name = "bar", sub_axis_info = (1)2>
  ]
>
```

### Updates to Operations

The type definition for `replica_groups` currently used by collective operations
will be updated to `AnyAttrOf<[I64ElementsAttr,
StableHLO_ReplicaGroupMeshAxes]>`.

Affected operations that will now accept the `replica_group_mesh_axes` layout: -
`stablehlo.all_gather` - `stablehlo.all_reduce` - `stablehlo.reduce_scatter` -
`stablehlo.all_to_all` - `stablehlo.collective_broadcast`

#### Example

```mlir
// Using the new format inside a standard collective op
%result = "stablehlo.all_reduce"(%operand) ({
^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "stablehlo.return"(%0) : (tensor<f32>) -> ()
}) {
  replica_groups = #stablehlo.replica_group_mesh_axes<
    mesh_name = @mesh,
    axes = [
      #stablehlo.axis_ref<name = "foo">,
      #stablehlo.axis_ref<name = "bar", sub_axis_info = (1)2>
    ]
  >
} : (tensor<1024xf32>) -> tensor<1024xf32>
```

## Rollout Plan

The rollout logic can be broken into corresponding CL phases:

1.  **Schema & Types**: Add the StableHLO and MHLO IR definitions for
    `#stablehlo.replica_group_mesh_axes` and its dependent types
    (`#stablehlo.axis_ref`, `#stablehlo.sub_axis_info`). Update operations to
    widen the types to include either representation format. Checks are
    explicitly added to ensure new representation is not fully un-lowered
    (temporarily) until the following steps.
2.  **Verification & Constraints**: Add logic + full verification for the new
    format.
3.  **Roundtripping & Integration**: Add round-tripping support into translation
    paths. Native operations and XLA pipelines will be updated to map this new
    IR format natively.
