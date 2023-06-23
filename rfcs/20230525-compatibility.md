# Reduce StableHLO v1.0 Compatibility Guarantees RFC

Status: Approved<br/>
Initial version: 5/25/2023<br/>
Last updated: 6/8/2023<br/>
Discussion thread: [openxla-discuss](https://groups.google.com/a/openxla.org/g/openxla-discuss/c/yYjTDAsoygQ)

## Summary

In the compatibility RFC from December 2022 we proposed 5 years of forward and
backward compatibility for StableHLO v1.0. This was an aspirational goal, based
on conversations with potential users, which seemed like a reasonable
guesstimate at the time. The RFC was approved, and it has become the plan of
record.

Since then, we have gained practical experience with StableHLO compatibility,
and upon further conversations, we haven't yet come across actual use cases for
5 years of compatibility guarantees. Without such use cases, we started
thinking whether such robust guarantees would be worth it, given that they come
at a high maintenance cost.

With that in mind, we propose to not provide 5 years of guarantees for
StableHLO v1.0, and instead gradually increase the current guarantees with RFCs
based on the community’s needs, keeping StableHLO compatibility guarantees tied
to existing use cases. To that end - If you are interested in something more
than the current 1mo forward / 6mo backward compatibility, let's discuss!
