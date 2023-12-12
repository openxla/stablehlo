# [RFC] Increase StableHLO Compatibility Guarantees

Status: Under review<br/>
Initial version: 6/23/2023<br/>
Last updated: 6/23/2023<br/>
Discussion thread: [openxla-discuss](https://groups.google.com/a/openxla.org/g/openxla-discuss/c/rfd30zKR9uU/m/khMs-1ZEAAAJ).

## Summary

The original StableHLO Compatibility [RFC](https://github.com/openxla/stablehlo/blob/main/rfcs/20220912-compatibility.md)
contained backward and forwards compatibility of 5 years within a major release,
and backward compatibility for serialized artifacts across 1 major release. This
guarantee was then reduced in a follow up [RFC](https://groups.google.com/a/openxla.org/g/openxla-discuss/c/yYjTDAsoygQ/m/WOk9JHNaAQAJ).

I propose we bring back Stability guarantees to the same level, since this is
critical for mobile ML deployments where the execution environment is not
tightly controlled by the model author. Sharing details below-

* ML Models deployed on-device (eg., Android) need strict backward and forward
  compatibility guarantees.
  * A deployed ML Model should never break due to a software update. This could
    be an update to the ML runtime, Mobile OS, or the App itself. OEMs regularly
    update phones, which can break functionality if the Opset changes.
  * ML models are often long-lived. Even when the application is updated, the
    model it uses may be older or the application team may not have access to
    the source model it uses. Said differently, a mobile ML runtime needs to
    support older versions of StableHLO Ops.
  * There are a significant number of users who use old Mobile/Android phones,
    often 5+ years. App developers should be able to target older phones, should
    they choose to for deploying their ML features.

Due to the above it's essential that Opset definitions are maintained long-term.
It's reasonable for us as a community to iterate on the Opset and utilize the
VHLO mechanism to version them. But once we have ML models deployed in the
market, especially on Mobile phones it's not feasible to update the execution
environment.
