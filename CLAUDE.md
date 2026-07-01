# scHopfield - project notes for Claude Code

Worked with the global `research-paper` skill family (research-paper, lit-novelty,
paper-red-team, auto-review-loop, paper-writing, experiment-bridge, run-experiment),
installed in `~/.claude/skills/` - nothing to install here. The machine run policy
below is read by `run-experiment` / `experiment-bridge`.

## GPU

Local dev workstation (this machine, shared across projects): one **RTX 3090 (24 GB)**,
**shared with an ollama service** that holds ~18 GB when a model is loaded.

- Check first: `nvidia-smi --query-gpu=memory.free,memory.used,utilization.gpu --format=csv`.
- ollama **not** holding a model (GPU mostly free): you MAY run several jobs in parallel
  (watch the free-memory budget against each job's footprint).
- ollama **actively** holding its ~18 GB: run jobs **one at a time**.
- ollama **idle but still** pinning ~18 GB: free it with `ollama stop <model>`, then use
  the memory. Never starve ollama while it is in use.
- Train on CUDA, not CPU (confirm no silent fallback).
- Use this project's own Python environment (per `pyproject.toml` / `requirements.txt`;
  never bare pip into the system Python).

Remote / Vast.ai / Modal are not configured; runs are local.

## Before automated runs

Check `git status` and commit or stash in-progress work before running skills that modify
code or generate files (`experiment-bridge`, `paper-writing`), so their changes do not
co-mingle with existing WIP. The skills commit only when you ask.
