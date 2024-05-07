# [RFC] Increase StableHLO Compatibility Guarantees

Status: Approved<br/>
Initial version: 6/23/2023<br/>
Last updated: 4/30/2024<br/>
Discussion thread: [openxla-discuss][openxla-discuss-post].

## Summary

**Proposal:** For StableHLO v1.0 Release, provide 5yrs backward compatibility,
2yrs forward compatibility.

ML Models deployed on-device (eg., Android) need strict backward and forward
compatibility guarantees.

* A deployed ML Model should never break due to a software update. This could
  be an update to the ML runtime, Mobile OS, or the App itself. OEMs regularly
  update phones, which can break functionality if the Opset changes.
* ML models are often long-lived. Even when the application is updated, the
  model it uses may be older or the application team may not have access to
  the source model it uses. Said differently, a mobile ML runtime needs to
  support older versions of StableHLO Ops.
* There are a significant number of users who use old Mobile/Android phones,
  often **5+ years**. App developers should be able to target older phones,
  should they choose to for deploying their ML features.
* Release cycles for new on-device toolkits are roughly annual, so a
  **2yr forward compatiblity** window is needed to target the latest release. We
  see this requirement for other users in the StableHLO community on annual
  release cycles.

Due to the above it's essential that Opset definitions are maintained long-term.
It's reasonable for us as a community to iterate on the Opset and utilize the
VHLO mechanism to version them. But once we have ML models deployed in the
market, especially on Mobile phones it's not feasible to update the execution
environment.

## OpenXLA Discuss Summary

_For full discussion see the [OpenXLA Discuss post][openxla-discuss-post]._

### Sep 18, 2023

As mentioned in TensorFlow Lite update on StableHLO use, TFLite would like to
consume StableHLO as an input source and leverage VHLO versioning facilities. In
order to achieve this goal, we propose the following changes to the StableHLO
compatibility - please let us know of any feedback you have!

#### Proposed VHLO Compatibility Guarantees

Up until this point the StableHLO project has focused on compatibility
guarantees from the opset perspective, without providing specific guarantees for
implementation details like the VHLO dialect. However, on the TFLite side, we
found VHLO to be really useful, and we would like to propose to formalize some
of its properties - for the most part these properties are already maintained in
practice, and this RFC proposes to formally document and maintain them:

1. VHLO op version number must only change by increment, if and only if there is
a change to Operator behavior. (i.e. `add_v1 → add_v2`).
1. VHLO ops must not be deleted within the compatibility window.
1. VHLO ops must always be convertible to StableHLO ops within the compatibility
window using machinery maintained in the openxla/stablehlo repository (i.e. not
an external tool).

#### Proposed Documentation Enhancement

For developers that could be interacting directly with this serialized VHLO, we
propose a documentation enhancement. Namely, there must be an easy way to access
documentation detailing the changes between different versions of the same op..
The exact mechanism can be determined in a follow-up RFC.

### Feb 21, 2024

_An edit to compatibility duration given more analysis of models in production._

Today TFLite offers open ended compatibility, and we are aware of some android
apps that are shipping model assets created more than 4 years ago. (e.g.
mobilenetV3 remains to be very popular) We propose to extend the backward
compatibility window of 5 years. Additionally, the TFlite team will follow up by
soliciting feedback from their developer community, as well as exploring other
mechanisms to meet their developers platform stability requirements and assess
if there is any scope to further refine the compatibility window in a future
RFC.

[openxla-discuss-post]: https://groups.google.com/a/openxla.org/g/openxla-discuss/c/rfd30zKR9uU/m/khMs-1ZEAAAJ
