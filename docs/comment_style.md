# Comment style

Comments record information that the code cannot express clearly. They explain
contracts, constraints, and intent; they do not narrate the implementation.

This guide applies to first-party C++ code under `include/`, `src/`, `plugins/`,
`tests/`, `benchmarks/`, and `examples/`. Do not reformat vendored code under
`third_party/` to match it.

## Language and tone

- Write comments in English.
- Be direct and specific. Prefer facts over introductions such as "Note that"
  or "This function is responsible for".
- Use complete sentences for API documentation and multi-line explanations.
  Short labels may be sentence fragments.
- Refer to identifiers with backticks. Use the same domain terms as the public
  API and documentation.
- Keep comments close to the code whose behavior they constrain.
- When touching existing code, improve comments relevant to the change. Do not
  rewrite unrelated legacy comments solely to make them conform to this guide.

## What to document

Write a comment when it answers at least one of these questions:

- Why is the implementation shaped this way?
- Which precondition or invariant is not encoded by the type system?
- What are the units, tensor axis order, image coordinate system, pixel format,
  stride, or memory layout?
- Who owns the memory, how long is it valid, and when is work synchronized?
- Which backend, plugin ABI, compatibility, or numerical behavior must remain
  stable?
- What contract is a test protecting, and why is its tolerance appropriate?

Do not comment code that already states the same fact:

```cpp
// Increment the index.
++index;
```

Name the consequence when preserving an unusual choice:

```cpp
// `std_vals` is a divisor: 255 scales 8-bit pixels to [0, 1].
arg.std_vals = {255.0F, 255.0F, 255.0F};
```

## Public API

Use Doxygen comments for public declarations under `include/ai_core/`. Follow
the existing style: `/** ... */` for types and functions, and `///<` for a short
field or enumerator note. Document the observable contract rather than the
implementation:

- accepted inputs, tensor names, dtype, shape, layout, and units;
- output ownership and lifetime;
- host or device memory requirements and buffer aliasing;
- synchronous or asynchronous completion and, where applicable, CUDA stream
  behavior;
- thread-safety and execution-context ownership;
- errors, invalid states, and non-obvious side effects.

Do not add boilerplate `@param` or `@return` entries when the signature and the
summary already make them clear. Simple accessors need no comment unless they
have unusual lifetime or state semantics. Use Doxygen commands such as
`@param`, `@return`, `@throws`, and `@par Thread safety` when they add contract
information that is not obvious from the declaration.

Do not add metadata-only file banners. A file-level comment is useful when it
explains the role or constraints of the whole file, not merely its filename,
author, date, or version.

## Internal implementation

Use `//` in implementation files. Comments are most useful around:

- thread/block mapping and boundary handling;
- buffer aliasing, alignment, and allocation lifetime;
- tensor shape/layout conversion and model-specific output decoding;
- plugin discovery, registration, and backend-specific behavior;
- precision choices and numerical stability;
- compatibility behavior that looks accidental;
- non-obvious performance tradeoffs.

Avoid a header comment for every private function. Prefer a short explanation
at the decision or expression it applies to.

## Tests

Test names should describe the behavior. Add comments only for information the
name and assertions cannot convey, such as:

- why a fixture uses a boundary-sized or deliberately misaligned input;
- why comparison is approximate rather than exact;
- which historical regression, model contract, or backend difference is
  covered;
- why an apparently redundant assertion protects a separate contract.

Large test files may use short Markdown-style section headings. Do not use rows
of punctuation as visual separators.

## TODO comments

A TODO must state a concrete remaining action and include an issue reference
when one exists:

```cpp
// TODO(#123): Reject tensors whose shape does not match the model metadata.
```

Do not use TODOs for vague design wishes. Put larger design work in an issue or
under `docs/` and link to it from the relevant code if necessary.

## Longer explanations

Keep model-output decoding details, pipeline architecture, plugin integration,
migration history, and benchmark or tuning rationale under `docs/`. Source
comments should state the local constraint and link to the document when the
background is needed to change it safely.

## Review checklist

- Could a clearer name or type remove the need for this comment?
- Does the comment explain a contract or reason instead of restating code?
- Are tensor names, dtypes, shapes/layouts, ownership, and synchronization
  explicit where needed?
- Does a compatibility comment identify the observable consequence?
- Is the comment still true on every control-flow path it describes?
- Is unrelated code absent from the comment-only change?

## Project terminology

Use these terms consistently:

| Prefer | Avoid or qualify |
| --- | --- |
| host memory, device memory | CPU memory, GPU-side data |
| CUDA stream | CUDA flow |
| inference engine, backend | inference module, inference framework |
| preprocessor, inference engine, postprocessor | handler, worker, processor without a stage name |
| plugin | component when dynamic registration/loading matters |
| execution context | inference stream unless referring to `TrtInferStream` |
| tensor name, dtype, shape, layout | tensor metadata without saying which part |
| NCHW, NHWC, HWC, CHW | channel-first, channel-last when the exact order matters |
| pixel format, channel order | image type |
| row stride | row step, pitch unless using a backend API's term |
| buffer owns storage | buffer holds a pointer |
| view does not own storage | borrowed buffer |
| synchronous, asynchronous | sync, async in explanatory prose |
