# StableHLO Evolution Process

## Motivation

HLO/MHLO are supported by a wide variety of ML frameworks and compilers,
including IREE, JAX, ONNX, PyTorch, TensorFlow and XLA. With StableHLO, we are
aiming to build on this success and create an amazing portability layer between
ML frameworks and ML compilers.

To that end, we are establishing the StableHLO evolution process to provide
StableHLO users with well-defined means of following and influencing the
evolution of the StableHLO opset and accompanying infrastructure (the CHLO
opset, bytecode serialization/deserialization, etc).

## Scope

This process governs the evolution of StableHLO and CHLO opsets and bytecode,
but doesn't apply to other changes in the StableHLO repository.

For example, adding a new op to the StableHLO dialect or a new operand to an
existing op is governed by the process. However, the following functionality and
infrastructure is out of scope of the process:
  * Interpreter (for the ops whose spec has been written and approved,
    interpreter must conform to the spec; for the ops whose spec is not yet
    available, it's best effort - either way, the implementation of the
    interpreter is out of scope).
  * Verification (same as interpreter).
  * Type inference (same as interpreter).
  * Prettyprinting (not part of the opset).
  * C/C++/Python APIs (might be governed by the process in the future, but
    for now these APIs are treated as an implementation detail).

## RFCs

All changes within the scope of the process must happen through RFCs - pull
requests to this repository that add markdown files to the `rfcs/` directory.
The name of the markdown file must start with the date when the RFC was proposed
in the `yyyymmdd` format, e.g. `20220912-compatibility.md`. If the markdown file
needs supporting materials, e.g. images, they must go into an eponymous
directory, e.g. `20220912-compatibility`.

At the moment, there is no particular template that RFCs must adhere to, because
the process is fairly young. In the future, we may decide to introduce one.

RFCs are encouraged to be accompanied by prototypes (either in the same pull
request or through other means). This is a plus, but not a requirement. E.g. in
addition to a markdown file, a compatibility RFC may come with a pass or passes
which implement proposed compatibility protocols. For another example, an RFC
that proposes to add an op to the StableHLO opset, may come with changes to the
dialect definition, the interpreter, etc.

## Participants

* *RFC authors*: Community members who propose an RFC by sending an RFC pull
  request. Authors are responsible for driving the RFC through the process by
  actively participating in community discussion and addressing community
  feedback.
* *Community members*: Everyone is welcome to provide feedback about RFCs
  through comments on RFC pull requests. All feedback provided this way is
  responded to during review.
* *Process lead*: is responsible for making decisions about RFCs based on
  community discussion. Per [governance.md](docs/governance.md): "During the
  bootstrapping phase of the project in 2022, Google engineers will assume
  responsibility for the technical leadership of the project. It is a high
  priority to create a path for community members to take on technical
  leadership roles". At the moment, @burmako serves in the role of process lead.

## Process

1. Authors propose an RFC by sending an RFC pull request. See above for the
requirements and the guidelines for these pull requests. At any time, authors
can withdraw the RFC by closing the pull request.

1. After an RFC is proposed, it enters a 14-day period of community discussion
where community members are encouraged to provide feedback via comments on the
RFC pull request. The goal of the community discussion is to comprehensively
study the proposal and hopefully achieve consensus about it.

1. After the community discussion period, the process lead leaves a review on
the RFC pull request, summarizing the discussion and providing a recommendation
which can be one of the following:
  * *Approve*: the markdown document is merged. If the RFC pull request also
    includes an implementation, it additionally needs to go through the usual
    code review process (same for subsequent pull requests which implement
    the RFC).
  * *Reject*: the pull request is closed.
  * *Request changes*: the pull request stays open, and authors are requested to
    address specific feedback. Once authors have addressed the feedback, they
    indicate that in a comment on the pull request, and the RFC enters another
    community discussion period.

## Future work

This is the first version of the StableHLO evolution process. It addresses much
of the community feedback about the HLO/MHLO evolution process (ensures
visibility into ongoing work, establishes timelines for reviews, provides
transparency of decision making).

In the future, we will further work on improving the process. The highest
priority is creating a path for community members to take on technical
leadership roles. This will be an important discussion topic at OpenXLA
community meetings in Q4 2022.
