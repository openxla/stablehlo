# StableHLO roadmap

At the time of writing, StableHLO is ready to supersede MHLO/HLO as compiler
interface. It can be produced by TensorFlow, JAX and PyTorch, it can be consumed
by XLA and several 3p PJRT plugins, and it has all the public features provided
by MHLO/HLO as well as additional functionality.

This document describes the next steps for the StableHLO project, categorizing
the ongoing work reflected in the issue tracker and arranging this work into
planned deliverables.

## Current Milestones

Our current milestones follow two main trends:

1. Maximize the benefit of StableHLO for the entire OpenXLA community.
2. Unify the developer experience for all OpenXLA members.

- **MHLO Deprecation**: In Q4'24 we have began exploring internal deprecation of
  MHLO, migrating useful passes including canonicalization and folder patterns
  to StableHLO. Once the migration process is proven trivial internally, we plan
  to share an RFC with timelines for external migrations to StableHLO. This will
  likely happen in Q1'25, and we plan to give ample time and support to teams to
  migrate to StableHLO in H1'25.
- **Migrate hardware independent optimizations to StableHLO**: Following the
  trend above, we want StableHLO to be the best place to consolidate hardware
  independent graph simplifications, so that all PJRT plugins including those
  which convert from StableHLO to a non-XLA compiler IR can see max benefits.
  Part of this goal involves consolidating patterns used in
  [Google AI Edge](https://io.google/2024/explore/18c47ed9-a8f7-4cd5-aec2-80457d839942/),
  the [JAX-Enzyme](https://github.com/EnzymeAD/Enzyme-JAX) project, and other
  projects all in the StableHLO repo. Some of this consolidation has already
  began, but the workstream will largely pick up and complete in Q1'25.
- **OpenXLA Componentization**: We have began creating dedicated components in
  openxla/xla for HLO which resembles the StableHLO repo setup
  ([ref](https://github.com/openxla/xla/tree/main/xla/hlo)), as well as started
  moving all OpenXLA backends behind PJRT plugins. We are additionally investing
  in fixing prominent UX issues we discover in these PJRT plugins, including
  things like having precise StableHLO version communication in StableHLO
  plugins, so new features can be used immediately by new plugins
  ([ref](https://github.com/openxla/xla/commit/84ad5fab88e3979b9c43ce93089e0ef537d14b88)).
- **Make composites work e2e**: In Q3'24 we added composites to HLO, enabling
  full compiler stack support for abstractions. In Q4'24 we taught the XLA
  inliner about composites and added passes in HLO/StableHLO for inlining
  unknown composites with their decompositions. We are now investigating adding
  dedicated JAX APIs for generating composites from the framework (PyTorch APIs
  already exist), as well as adding Colab documentation on how to properly use
  composites, to be completed in Q4'24.

## Past Milestones

In H1 2024, we released **StableHLO v1.0** which implemented high-priority
improvements, including finishing the public opset spec, reference interpreter,
extending forward and backward compatibility to support on-device deployment
with compatibility unit testing, extensibility via composite operations, spec'ed
dynamism support, released a full testdata suite with evaluated golden results,
and more.
