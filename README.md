# research

![Runnable notes on inference systems: a clay notebook with pop-ups for silicon, kernels, economics, schedulers, and RL and training](banner.png)

Notes on how AI models run on real hardware, written so you can rerun them. Each post picks one
question about inference (the work of serving a trained model to its users), explains the mechanism
in plain words, draws it, and ships the code behind its numbers. Change a shape, swap a model, and
see what moves.

## What's inside

- **Heterogeneous inference**: splitting one model's work across different kinds of chips. Which part belongs where, and what does the handoff cost?
- **Kernels**: the small GPU programs that do the math. Is this one fast, is it correct, and how would you know?
- **Profilers**: the tools that show where a GPU spends its time. What are they telling you to fix?
- **Schedulers**: the part of a serving engine that decides which requests run together. Who waits, and what does batching buy?
- **RL and training infrastructure**: the loops that train models, and what keeps their GPUs busy.
- **Economics**: what a token costs on NVIDIA, AMD, Google TPU, AWS Trainium, Cerebras, and d-Matrix silicon, with the conditions written down.

## What every post has

- **An explainer** you can read in a few minutes, with each term defined where it first shows up.
- **A diagram** of the one idea that matters.
- **Code** in `code/`, with a README that says what it needs, how to run it, and what it printed.
- **Sources** for every number, so you can check the conditions yourself.


## Latest

<a href="posts/2026-10-25-pm-bad-guesser-same-answer/"><img src="posts/2026-10-25-pm-bad-guesser-same-answer/images/fun.png" alt="Cover: a grey robot on a heap of crumpled guesses throws another one away while a blue robot at a desk stamps cards calmly" width="720"></a>

**[Why a bad guesser can't change what your model says](posts/2026-10-25-pm-bad-guesser-same-answer/)** · October 25, 2026

HOW IT WORKS

## Earlier posts

| | |
|---|---|
| <a href="posts/2026-09-28-pm-decode-wants-sram/"><img src="posts/2026-09-28-pm-decode-wants-sram/images/fun.png" alt="" width="200"></a> | **[More batch won't make decode compute-bound](posts/2026-09-28-pm-decode-wants-sram/)**<br>September 28, 2026<br>At 4,096 tokens of context, one decode step of a 72B dense model cannot reach a B200's ridge point at any batch size, because the KV read grows with the batch and caps arithmetic intensity at 116 FLOP/byte against a… |

Spotted a mistake, or got a different result? [Open an issue](https://github.com/antje/research/issues/new).
