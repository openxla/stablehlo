# Integrate Scripts

A collection of scripts used to integrate the StableHLO repository into the
rest of the OpenXLA repos. These scripts are run ~2x/wk by a StableHLO oncall
rotation at Google to ensure new changes can propagate to the rest of the
ecosystem in a reasonable amount of time.

## Integrate Process

### Bump LLVM Revision

First we bump LLVM to match that of XLA, and apply any patches the the XLA
LLVM integrate had to apply to StableHLO to build LLVM:

```
$ ./build_tools/integrate/llvm_bump_revision.sh
```

### Integarte into OpenXLA Repositories

The StableHLO oncall then integrates the change into the google monorepo, which
propagates the new StableHLO features to XLA, Shardy, JAX, TF, etc, including
any changes or patches that were needed to build these projects with the new
feature.

_Note: this is the only step that must be carried out by a Google team member._

### Tag the integrated StableHLO commit and bump StableHLO version numbers

This step takes care of a few things:
1. Add a tag for the integrated StableHLO version
2. Bump the patch version in [Version.h](https://github.com/openxla/stablehlo/tree/main/stablehlo/dialect/Version.h#L41)
3. Bump the 4w and 12w forward compatibility requirement versions in
   [Version.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/Version.cpp#L75)

```
Usage: ./build_tools/integrate/stablehlo_tag_and_bump_version.sh [-t <COMMIT_TO_TAG>]
   -t  Specify a commit to tag, must be an integrated StableHLO commit
       available on https://github.com/search?q=repo%3Aopenxla%2Fxla+integrate+stablehlo&type=commits
       If not specifed, will only bump the 4w and 12w versions.

$ ./build_tools/integrate/stablehlo_tag_and_bump_version.sh -t 37487a8e
```
