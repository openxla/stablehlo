# Increase backward compatibility guarantees to 6 months

Status: Approved<br/>
Initial version: 3/9/2023<br/>
Last updated: 3/24/2023<br/>
Discussion thread: [GitHub](https://github.com/openxla/stablehlo/pull/1306)

## Summary

This RFC is cloned from an informal pull request:
[https://github.com/openxla/stablehlo/pull/1306](https://github.com/openxla/stablehlo/pull/1306).

We have recently had the first official release of StableHLO, which fully
implemented the compatibility guarantees established by the compatibility RFC
accepted in December.

These guarantees ended up being more compelling than we anticipated in the RFC
discussions, and there are stakeholders who are interested in using them in
production right away. To enable that, we have received a request to bump up the
backward compatibility guarantees to 6 months (from 1 month for the 0.x.x
series, as established by the original RFC).

I would like to proposes to fulfil this request to further strengthen the
practical usefulness of the StableHLO opset, given that the additional
maintenance cost looks acceptable (5 extra months of maintaining
backward-compatible VHLO ops).

This RFC does not affect the long-term compatibility guarantees established for
the 1.x.x series and onwards, which remain at 5 years of forward and
backward compatibility.
