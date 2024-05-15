# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## StableHLO Opset Changes

All changes to the StableHLO opset including new ops, types, or attributes must
be reviewed via an RFC. We aim for StableHLO opset changes to take ~2 weeks
if feedback is actively addressed. This allows adequate time for the community
to review all proposals.

### 1. Write an RFC

An RFC should outline the proposed spec changes, as well as the rationale, and
alternatives considered if relevant. This can be shared as a markdown file in
the [`rfcs/`](https://github.com/openxla/stablehlo/tree/main/rfcs) directory and
shared as a PR.

For example, see the [`collective_broadcast` RFC](https://github.com/openxla/stablehlo/pull/1809).

### 2. Notify OpenXLA Discuss

To signal boost your RFC, post on [OpenXLA Discuss](https://groups.google.com/a/openxla.org/g/openxla-discuss).
This will ensure the proper reviewers see the RFC. While there is no formal
process for posts, we tend to recommend keeping RFC discussion on the PR to keep
feedback centralized in the repository.

For example, see the [`collective_broadcast` post](https://groups.google.com/a/openxla.org/g/openxla-discuss/c/Q7JFyoiVFPU/m/dUH_LmJlCgAJ).

### 3. Work with project maintainer for final approval

As denoted in [`governance.md`](https://github.com/openxla/stablehlo/blob/main/docs/governance.md),
while we work towards instating StableHLO module maintainers, the interim review
process requires approval from Google project maintainers. A member of the
StableHLO team will help drive final approval.

### 4. Send PRs for the opset change

Once an RFC is approved, PRs which implement the approved proposal may be sent,
reviewed, and merged.

A few things to consider when adding new features:

- Spec, Type Inference, Verifiers: Steps for adding to [`spec.md`](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
as well as related op implementation can be found in
[`spec_checklist.md`](https://github.com/openxla/stablehlo/blob/main/docs/spec_checklist.md).
- Verifiers: Steps for modifying or implementing op verifiers can also be found
in the spec checklist.
- Type Inference: Type inference design principles and testing details can be
found in [`type_inference.md`](https://github.com/openxla/stablehlo/blob/main/docs/type_inference.md).
- Compatibility: Tips on managing forward/backward compatibility are in
[`vhlo_checklist.md`](https://github.com/openxla/stablehlo/blob/main/docs/vhlo.md#contributing-incompatible-changes).
- Reference: Steps for adding interpreter support can be found in
[`reference_checklist.md`](https://github.com/openxla/stablehlo/blob/main/docs/reference_checklist.md).
- Tests: For each of the above modifications, consider positive and negative
test cases.

Some examples to help guide changes:

- Adding a new op: [`collective_broadcast`](https://github.com/openxla/stablehlo/pull/1856).
- Adding new types: [`f8E4M3FNUZ` and `f8E5M2FNUZ`](https://github.com/openxla/stablehlo/pull/1379).
- Expanding type support: [Quantized `ReduceOp`](https://github.com/openxla/stablehlo/pull/1796).
