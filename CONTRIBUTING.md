# Contributing to DIGM

Thanks for your interest. DIGM is a small research project, so the process is simple.

## Questions and bug reports

Open an issue and include:

- your vLLM version (`pip show vllm`), GPU and model;
- the exact command and `DIGM_*` settings you ran;
- the last 20 lines of output.

## Pull requests

- **Small fixes** (typos, docs, clear bugs): open a pull request directly.
- **Anything larger** (new features, restructuring, another vLLM version): open an issue first so we can agree on the approach.

Before you open a pull request:

1. Run the quick check from the README and paste its summary lines in the pull request.
   The line `identity eviction reproduces Full KV exactly : N/N` must still be N out of N.
2. Sign off your commits with `git commit -s`.
3. If you used an AI assistant, say so. You should be able to explain every line you changed.

## Licence of contributions

By opening a pull request you agree that:

- your contribution is licensed under this repository's licence (see `LICENSE.md`), and
- the maintainers may also release it under a different licence later, for example Apache-2.0.

The method is covered by a pending patent (see the README).

## Commercial use and collaboration

Open an issue titled "Commercial use" or "Collaboration" and the authors will get back to you.
