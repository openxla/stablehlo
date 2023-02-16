# StableHLO roadmap

At the time of writing, StableHLO is ready to supersede MHLO/HLO as compiler
interface. It can be produced by TensorFlow, JAX and PyTorch, it can be consumed
by XLA and IREE, and it has all the public features provided by MHLO/HLO
as well as additional functionality.

This document describes the next steps for the StableHLO project, categorizing
the ongoing work reflected in the issue tracker and arranging this work into
planned deliverables.

## Milestones

In 2023, we are planning two big milestones: 1) StableHLO v0.9 which will
provide an initial version of the opset and initial compatibility
guarantees, 2) StableHLO v1.0 which will implement high-priority improvements
and start providing full compatibility guarantees.

**StableHLO v0.9** will mirror MHLO/HLO, augmented with a specification for
statically-shaped ops and initial compatibility guarantees. Per
[compatibility RFC](https://github.com/openxla/stablehlo/blob/main/rfcs/20220912-compatibility.md),
this release will provide 1 month of forward and backward compatibility. These
modest guarantees will enable gaining experience with dialect evolution and
allow some time for cleanups before full guarantees will be in effect. We are
planning to release StableHLO v0.9 in Q1 2023.

**StableHLO v1.0** will implement high-priority improvements, including
cleaning up the frontend contract (with the goal for StableHLO programs to only
include ops from the StableHLO dialect, rather than today's mixture of dialects
and unregistered attributes) and providing a reference implementation. Having
obtained these improvements and resolved key specification compliance issues,
StableHLO will be ready for full compatibility guarantees - 5 years of forward
and backward compatibility. We are planning to release StableHLO v1.0 in
H2 2023.

## Workstreams

In order to organize the development towards the aforementioned milestones,
we have categorized the tickets in the issue tracker into multiple workstreams
and tied these workstreams to the milestones. A limited number of tickets
(less than 10%) are not assigned to any specific workstream and aren't part of
any specific milestone.

(P0) [Compatibility implementation](https://github.com/orgs/openxla/projects/4)
workstream is dedicated to implementing
[the compatibility RFC](https://github.com/openxla/stablehlo/blob/main/rfcs/20220912-compatibility.md)
along with compatibility test suite. Most of this work is expected to be
completed in StableHLO v0.9, and the rest will be done in StableHLO v1.0.

(P0) [Frontend contract](https://github.com/orgs/openxla/projects/6) workstream
consists in implementing 100% of the features which are used by StableHLO
frontends but are not yet in the StableHLO specification. The goal of this
workstream is to ensure that StableHLO programs only include ops from the
StableHLO dialect, rather than today's mixture of dialects and unregistered
attributes. We are planning to complete all or almost all work in this
workstream in StableHLO v1.0.

(P0) [Reference implementation](https://github.com/orgs/openxla/projects/7)
workstream organizes the work on implementing
[an interpreter](https://github.com/openxla/stablehlo/blob/main/docs/reference.md)
for 100% of StableHLO ops as defined in the StableHLO specification. We are
planning to complete all or almost all work in this workstream in
StableHLO v1.0.

(P0) [Documentation](https://github.com/orgs/openxla/projects/12) workstream is
dedicated to providing all the information that StableHLO producers or consumers
might need. The StableHLO specification is a major deliverable, as well as
reference for StableHLO API and StableHLO serialization format. Critical pieces
of the workstream will be delivered in StableHLO v1.0, with lower-priority items
addressed on a rolling basis.

(P1) [Conformance suite](https://github.com/orgs/openxla/projects/8) workstream
consists in delivering a test suite that compares reference implementation with
implementations provided by StableHLO backends. Tests for the reference
implementation will provide a conformance suite of sorts, so this workstream
does not have P0 priority. However, further augmenting this suite with
additional interesting test cases will likely be a useful area for future work.

(P1) [Specification compliance](https://github.com/orgs/openxla/projects/9)
workstream ensures that 100% of StableHLO ops are implemented in the StableHLO
dialect as defined in the StableHLO specification. The StableHLO dialect is
already reasonably compliant, so this workstream does not have P0 priority,
but a lot of minor items still remain (especially in the corner cases of
verifier implementation) and will be addressed on a rolling basis.

(P1) [New features](https://github.com/orgs/openxla/projects/10) workstream
concludes the StableHLO roadmap and consists of a ragtag collection of new
functionality for the StableHLO opset (not the StableHLO dialect or the
StableHLO bindings - that would be other workstreams). A few of these new
features are something that we are planning to deliver in StableHLO v1.0, but
the majority are currently lower-priority items which are not part of any
specific milestone.

(P1) [Public API](https://github.com/orgs/openxla/projects/5) workstream is
dedicated to delivering C/C++/Python bindings for the StableHLO dialect.
Existing C++/Python bindings are already fairly reasonable, so this workstream
does not have P0 priority. However, quite a bit of work still remains to be
done, especially around providing stability for these bindings - which is
something that is not currently covered by the compatibility RFC but will
likely be a useful area for future work.
