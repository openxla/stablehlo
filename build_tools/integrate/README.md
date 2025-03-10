# Integrate Scripts

A collection of scripts used to integrate the StableHLO repository into the
rest of the OpenXLA repos. These scripts are run ~2x/wk by a StableHLO oncall
rotation at Google to ensure new changes can propagate to the rest of the
ecosystem in a reasonable amount of time.

## Integrate Process

### Bump LLVM Revision

The XLA repo has constant integrates of LLVM and carries a patch file for
StableHLO in [temporary.patch](https://github.com/openxla/xla/blob/main/third_party/stablehlo/temporary.patch),
which contains any changes needed to build StableHLO at a new LLVM revision.
To sync to XLA's LLVM revision and apply necessary patches, use:

```sh
$ ./build_tools/integrate/llvm_bump_revision.sh
Using LLVM commit: b3134fa2338388adf8cfb2d77339d0b042eab9f6
Updating LLVM Commit & SHA256
Bumping commit to: b3134fa2338388adf8cfb2d77339d0b042eab9f6
Bumping sha256 to: b6024606092290b0838735c26ad1c5c239b3e931136b420af8680e3a1156e759
Patch file openxla/xla/third_party/stablehlo/temporary.patch is empty
Skipping patch apply
Commit changes with message:
git add .
git commit -m "Integrate LLVM at llvm/llvm-project@b3134fa23383"
```

### Integrate into OpenXLA Repositories

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

```sh
Usage: ./build_tools/integrate/stablehlo_tag_and_bump_version.sh [-t <COMMIT_TO_TAG>]
   -t  Specify a commit to tag, must be an integrated StableHLO commit
       available on https://github.com/search?q=repo%3Aopenxla%2Fxla+integrate+stablehlo&type=commits
       If not specified, will only bump the 4w and 12w versions.

$ ./build_tools/integrate/stablehlo_tag_and_bump_version.sh -t 37487a8e
Bumping 4w and 12w compatibility window values
New WEEK_4 Version: 1, 7, 8
New WEEK_12 Version: 1, 6, 0
      return Version(1, 7, 8);  // WEEK_4 ANCHOR: DO NOT MODIFY
      return Version(1, 6, 0);  // WEEK_12 ANCHOR: DO NOT MODIFY
Bumping version Version(1, 8, 4) -> Version(1, 8, 5)
Using commit:
Integrate LLVM at llvm/llvm-project@246b57cb2086 (#2620)

Is this the correct commit? [y] y

Creating tagged release v1.8.4 at 37487a8e
$ git tag -a v1.8.4 37487a8e -m "StableHLO v1.8.4"

Most recent tags:
v1.8.4
v1.8.3
v1.8.2

If this is incorrect, can undo using:
$ git tag -d v1.8.4

Bumping revision to: Version(1, 8, 5)
$ sed -i "s/Version(1, 8, 4)/Version(1, 8, 5)/" ./stablehlo/dialect/Version.h

NEXT STEPS
  Push tag to upstream using:
  $ git push upstream v1.8.4

  Commit and patch bump changes:
  $ git add .
  $ git commit -m "Bump patch version after integrate 1.8.4 -> 1.8.5"
```
