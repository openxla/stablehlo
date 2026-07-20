# StableHLO Repo Source-of-Truth Migration

Status: In Review<br/>
Initial version: 7/20/2026<br/>
Last updated: 7/20/2026<br/>
Discussion thread: [GitHub](https://github.com/openxla/stablehlo/pull/2980)

## Summary

We propose aligning StableHLO's contribution process more closely with XLA and
other repositories under the OpenXLA organization. Our motivations are to
increase version-history transparency and to reduce maintenance overhead.

## Background

Unlike other OpenXLA projects, StableHLO is currently hosted "GitHub-first,"
i.e. it uses GitHub as the source of truth for its weekly integrations with
Google3 (Google's monorepo) and related projects such as XLA. While this
approach does have some upsides, it comes with some serious downsides as well.
We have come to the conclusion that the downsides now far outweigh the upsides,
so we propose changing the source of truth from GitHub to Google3. This will
match the way all other OpenXLA projects are hosted, e.g.
[XLA](https://github.com/openxla/xla) and
[Shardy](https://github.com/openxla/shardy).

## Motivation

Aligning StableHLO's contribution process with that of XLA would provide three
major benefits:
- **Transparency:** Currently, any changes made to StableHLO in Google3 get
  squashed together and integrated into GitHub as part of the weekly LLVM
  integration PR, which results in an obfuscated version history. These
  integration PRs have names of the form
  `Integrate LLVM at llvm/llvm-project@<commit_hash>`, and in addition to
  integrating new LLVM changes, they also integrate any new StableHLO changes
  made in Google3. The StableHLO team does try to make our more notable code
  changes through GitHub in order to mitigate this issue, but the current system
  still misses a lot of context in the version history.
- **Maintenance:** The current hosting structure makes StableHLO's weekly
  integrations much more labor-intensive than necessary, which wastes a lot of
  the StableHLO team's time that could otherwise be spent on improving
  StableHLO. Switching the source of truth to Google3 would allow StableHLO to
  benefit from a great deal of tooling
  (e.g.  [Copybara](https://github.com/google/copybara)) designed to facilitate
  such integrations. Streamlining integrations would allow the StableHLO team to
  spend our time more productively, improving StableHLO's development velocity.
- **Velocity:** We currently rely on ad hoc patching and validation to derisk
  potentially high-impact PRs before approval. Reusing XLA's approach will help
  us to validate changes across the stack more easily, putting less burden on
  reviewers and expediting overall review time.

## What Would Change

Changing the source of truth from GitHub to Google3 would make StableHLO more
consistent with related projects such as [XLA](https://github.com/openxla/xla)
and [Shardy](https://github.com/openxla/shardy), so contributors familiar with
those projects should feel right at home. Overall, from a contributor
perspective, very little would change; OSS contributions would of course still
be welcome, and the contribution process would look mostly the same. The main
differences you might notice are:
- There would be one additional CI check running on your StableHLO PRs. This
  check would confirm that each PR passed Google's internal integration tests,
  which would help expedite both our review process and our weekly integrations.
  (Currently, these tests run as part of the weekly integration process; in the
  event that they fail, the team member in charge of the integration is left to
  debug the failure without knowing which PR was responsible.)
- As discussed under [Motivation](#motivation), you'd be able to see a much
  more complete picture of StableHLO's version history through Git and GitHub.
- As discussed under [Motivation](#motivation), the StableHLO team would have
  more time for active development work due to the streamlined integrations.

## What Wouldn't Change

- Under the proposed hosting system, most contributors should see very little
  change in their work with StableHLO.
- Obviously, GitHub pull requests will remain both welcome and encouraged
  regardless of the source of truth.
- StableHLO remains a priority, as do its OSS community and ecosystem.
