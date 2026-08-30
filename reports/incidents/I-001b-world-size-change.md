# I-001b — Checkpoint resharding across a world-size change

**Injected experiment.** The world-size change is deliberate. Not an organic
incident.

- **Incident ID:** I-001b
- **Phase:** B4 (local, CPU/Gloo, FSDP2 + `torch.distributed.checkpoint`)
- **Oracle:** invariant assertion (coverage) plus a **refusal** by the exact oracle
- **Result:** all four reshards loaded; the epoch stayed exactly-once; the exact
  oracle correctly refused every case

## 1. Scope and topology

FSDP2 over an explicit CPU `DeviceMesh`, Gloo, one host. Save under one world size,
load under another: **1→2, 2→1, 2→4, 4→2**. 20 steps, checkpoint at step 10.

Local multi-process. **Not multi-node evidence.**

## 2. Hypothesis and predeclared detection signal

**Hypothesis.** Node loss is the normal operational scenario: a job dies and comes
back at whatever capacity is available. Resuming at a different world size must
continue over the *remaining* data without replaying or skipping any of it — and it
cannot be bit-exact, because both the collective reduction order and the per-rank
RNG change.

**Signals, declared before the experiment.**

1. DCP loads the sharded state at the new world size, and the shard layout actually
   changes (a replicated load would not be resharding).
2. The union of samples consumed before and after obeys the exactly-once invariant
   over the prefix of the epoch that was consumed.
3. The exact-equality oracle **refuses to run**, rather than passing or failing.

Signal 3 is the unusual one and it was chosen deliberately. See §5.

## 3. Injection

`resume_world_size != control_world_size`. Nothing else is changed.

## 4. Detection evidence

`artifacts/checkpoints/reshard-matrix.json`:

```text
1 -> 2   refused=True  coverage=True  local_frac 1.000 -> 0.500   consumed 120
2 -> 1   refused=True  coverage=True  local_frac 0.500 -> 1.000   consumed 120
2 -> 4   refused=True  coverage=True  local_frac 0.500 -> 0.250   consumed 240
4 -> 2   refused=True  coverage=True  local_frac 0.250 -> 0.500   consumed 240
```

`local_frac` is the fraction of global parameter elements resident on each rank. It
tracks `1 / world_size` exactly, which is the evidence that the state is genuinely
partitioned rather than replicated.

## 5. Diagnosis

Two design consequences fell out of this experiment, and both are load-bearing.

**RNG continuity cannot survive a reshard.** Per-rank RNG state is saved per rank.
Resuming at a larger world size has no state for the new ranks — rank 3 of 4 did not
exist when 2 ranks saved. The loader therefore reseeds deterministically from
`(seed, step, rank)` and reports `rng_continuity=False`.

That flag is not advisory. `compare_exact` **raises** `OracleNotApplicableError`
when it is false, so the code cannot be talked into certifying bit-exactness for a
resharded run. This makes spec 8.1 and 8.2 mechanically distinct rather than a
matter of discipline: refusing says *the wrong oracle was chosen*, whereas reporting
"fail" would invite someone to loosen a tolerance until it passed.

**The epoch permutation must not depend on world size.** It is derived from
`(seed, epoch)` alone. Had it depended on world size, the reshard would have
silently reshuffled the epoch underneath the resume, and "consumed exactly once"
would not even be well defined across the boundary — the two runs would disagree
about what the samples *are*.

One consequence worth recording: `drop_last` makes the epoch's exact membership
world-size dependent (246 samples yields 246 usable at world size 2 but 244 at world
size 4). The coverage check therefore compares against the consumed *prefix* of the
shared epoch order, which is identical across world sizes.

## 6. Intervention

None required. The reshard path works and the guard rails hold.

## 7. Verification oracle and result

**Oracle:** invariant assertion (exactly-once coverage after resume) plus a required
refusal from the exact oracle. **Result:** pass on all four reshards.

## 8. Cost and time lost

Roughly 90 s of laptop CPU for the four-case matrix. $0.

## 9. Counterfactual

Without the coverage invariant, a reshard that resumed every rank at the *same*
per-rank cursor would replay or skip a `world_size`-dependent slice of the epoch and
report nothing. Without the refusal, a resharded run compared under an exact oracle
would report "fail", and the natural next move — widening a tolerance until it
passes — would produce a meaningless equivalence claim backed by a number.

## 10. Limitations and claim boundary

- One host, CPU, Gloo. **Not multi-node evidence.** Resharding across physical
  hosts is Phase C.
- **Loss trajectories are not compared across the reshard.** Changing world size
  with fixed `grad_accum_steps` changes the global batch size (4→8, 8→16, and so
  on), which changes the optimisation trajectory for real reasons. Judging such a
  run against a seed-variance band built at a different global batch would conflate
  the reshard with the batch-size change. A like-for-like comparison requires
  holding the global batch constant by compensating `grad_accum_steps`, which is
  itself a config change and therefore a separate experiment. What is proven here
  is the mechanical claim: state loads and reshards, and the epoch is still
  consumed exactly once.
- Synthetic fixtures. No model-quality claim follows.
