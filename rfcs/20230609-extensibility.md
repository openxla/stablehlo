# [RFC] StableHLO Extensibility

Status: Approved<br/>
Initial version: 6/9/2023<br/>
Last updated: 3/8/2024<br/>
Discussion thread: [openxla-discuss](https://groups.google.com/a/openxla.org/g/openxla-discuss/c/Ao5K8fvXoEk/m/OaddRrgyAgAJ).

## Summary

For full details and RFC discussion, see:
[[RFC] StableHLO Extensibility](https://docs.google.com/document/d/1bSyyLA-p1F7KjZgjo563F1WFsPwcZc4eaH5WyQfbsi0/edit#heading=h.kfv34azf3j5k).

In its role as a portability layer between ML frameworks and ML compilers,
StableHLO provides a common vocabulary of well-understood ops along with
compatibility guarantees for them. However, this all works only for a closed set
of ops within the StableHLO dialect. In this document, we propose to offer a
mechanism to create portable abstractions of StableHLO ops.
