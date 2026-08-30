# I-008 — The preflight reported checkpointing healthy while no training run wrote one

**Organic finding, not injected.** Found by auditing the codebase before provisioning
cloud hardware. This is the most instructive defect in the repository so far, because
it was located *inside the safety mechanism built to prevent exactly this class of
error*.

- **Incident ID:** I-008
- **Phase:** C preparation (local, CPU)
- **Oracle:** invariant assertion (end-to-end capability probe)
- **Result:** fixed; the check now proves the capability instead of inferring it

## 1. Scope and topology

Single process, CPU. Affects the `cli.py train` path and the `torchrun` entrypoint —
that is, every path that would have run on GCP.

## 2. What was wrong

`save_checkpoint` was called from exactly one place in the repository:
`training/resume_experiment.py`, the harness for the B3/B4 oracles.

Neither `cli.py cmd_train` nor `distributed/entrypoint.py` called it. So
`checkpoint.every_steps` and `checkpoint.keep_last` were:

- declared in every config,
- type-checked and validated by the config schema,
- asserted by `scripts/preflight_cloud.py`,
- and **honoured by nothing**.

Demonstration, before the fix — `local_smoke.toml` declares `every_steps = 10`:

```text
$ uv run pretrainmodel train --config configs/local_smoke.toml
run local_smoke-…: 20 steps, final_loss=0.486737, stopped=max_steps

$ ls checkpoints/local_smoke
NO CHECKPOINT DIRECTORY EXISTS

$ preflight: checkpoint_cadence::configs/two_node_l4.toml  ok=True
```

## 3. Why Phase B did not catch it

Phase B proved the checkpoint *contract* thoroughly: atomic commit markers,
incomplete-checkpoint rejection, provenance guards, bit-exact resume, resharding
across world sizes. All of it correct, all of it exercised — by a test harness that
called `save_checkpoint` directly.

The gap is the difference between **"the mechanism is correct"** and **"the mechanism
is installed"**. Only the second survives a preemption, and the Phase B evidence
spoke only to the first. A reader of `artifacts/oracles/exact-resume.json` would
reasonably conclude the training path checkpoints; it did not.

## 4. Why the preflight made it worse

The preflight was written specifically so that unknowns would not be silently
converted into assurances before money was spent. Its manual items are deliberately
recorded as UNANSWERED rather than defaulted to ok.

Its `checkpoint_cadence` check then did the exact thing it was built to prevent: it
read `cfg.checkpoint.every_steps > 0` and reported healthy. **Reading a config value
proves a config value.** The check was well intentioned, cheap, and worse than
nothing — it converted "nobody has verified this" into a green tick immediately
before provisioning.

## 5. Consequence had it shipped

Phase C is planned on **spot instances**, which are preempted without warning. The
plan explicitly treats preemption as an asset: a free organic interruption to
recover from, stronger evidence than the injected failure in I-001a.

That entire argument depended on checkpointing working in the training path. Without
it, a preemption at any point destroys the whole run, the recovery experiment is
impossible, and the preflight would have reported the configuration sound
beforehand.

## 6. Intervention

1. **`training/checkpointing.py`** — `PeriodicCheckpointer`, installed as the
   training loop's `on_step` hook, honouring `every_steps` and `keep_last`, plus
   `resume_if_available()` for restart. Policy lives outside `train()` so the loop
   stays identical between a control run and a resumed one.
2. **Wired into both real paths** — `cli.py train --resume` and the torchrun
   entrypoint `--resume`.
3. **The preflight now proves the capability** —
   `checkpoint_capability_writes_one` runs a real two-step training run into a
   temporary directory and asserts a committed checkpoint lands on disk. The
   config-reading check remains, relabelled to say what it does and does not cover.

Verified after the fix:

```text
run local_smoke-…: 20 steps, checkpoints: [10, 20]
  step-00000010: COMMITTED step=10 cursor=40
  step-00000020: COMMITTED step=20 cursor=80
$ … --resume
resumed from step 20 (RNG state restored per rank; bit-exact continuation is possible.)
```

## 7. Verification oracle and result

**Oracle:** invariant assertion — a real run must produce a committed checkpoint.
**Result:** pass. Regression tests in
`tests/integration/test_checkpointing_path.py` assert the cadence, the recorded
loader cursor, `keep_last` pruning, the disabled case, and resume from a stopped run.

## 8. Cost and time lost

About 40 minutes of local work. $0.

## 9. Counterfactual

Undetected, this surfaces as: a spot node is reclaimed four hours into the Phase C
run, `--resume` finds nothing, and the entire reservation is lost — after a green
preflight said checkpointing was configured. The likely diagnosis in the moment
("preemption happened before the first checkpoint interval") is wrong and would have
sent the next attempt down the path of shortening the interval, which fixes nothing.

## 10. Limitations and claim boundary

- Single process, CPU. The capability probe runs at world size 1; that a *sharded*
  run checkpoints is covered separately by the B2 contract tests, but the two have
  not been exercised together in one run.
- The probe asserts a checkpoint is **committed**, not that resuming from it
  reproduces the run. Bit-exactness is I-001a's job.
- Generalisation worth stating plainly: every other config field should be assumed
  guilty until a test exercises the behaviour it names. `determinism.async_checkpoint`
  is currently in exactly the same position — declared, validated, and implemented
  nowhere.
