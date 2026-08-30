# I-001a — Partial resume state, detected by bit-exact equality

**Injected experiment.** The defects below were induced deliberately to validate a
detector. This is not an organic incident and must never be described as one.

- **Incident ID:** I-001a
- **Phase:** B3 (local, CPU/Gloo, 2 ranks, FSDP2)
- **Oracle:** exact equality
- **Result:** both injected defects detected; clean resume bit-exact

## 1. Scope and topology

Two Gloo ranks as separate OS processes on one host, FSDP2 sharding over an
explicit CPU `DeviceMesh` (each rank holds 50% of parameters, verified as DTensor).
20 steps, checkpoint at step 10, fp32, deterministic algorithms, synchronous
checkpointing, fixed data order, dropout 0.

Local multi-process. **Not multi-node evidence.**

The three phases run in genuinely separate process groups: control, interrupt
(which stops at the checkpoint), and resume. An in-process "resume" would inherit
warm allocator state, live RNG objects and an already-built optimizer — none of
which survive a real crash — so it can pass while the actual recovery path is
broken.

## 2. Hypothesis and predeclared detection signal

**Hypothesis.** Resume state is a checklist, and a partially restored checkpoint
does not crash. It produces a *plausible* loss curve, so any tolerance-based
comparison accepts it.

**Detection signal, declared before injection.** After the resume boundary, the
resumed run must be **exactly equal** to the uninterrupted control in all four of:

1. every model parameter
2. every optimizer state tensor
3. the loss sequence
4. the consumed sample IDs

Declared in advance: the comparison is exact equality, not a tolerance. A tolerance
was ruled out on the grounds that every defect below stays inside any tolerance
loose enough to absorb ordinary float noise.

## 3. Injection

Two defects, each a bug people actually ship:

| Defect | What it simulates |
|---|---|
| `reset_optimizer` | optimizer state not restored — AdamW moments and the per-parameter step counter restart, so bias correction is wrong |
| `ignore_loader_state` | dataloader cursor not restored — the epoch replays from the top |

## 4. Detection evidence

`artifacts/oracles/exact-resume.json`:

```text
defect=none                 -> pass    0 differences, losses and IDs identical
defect=reset_optimizer      -> fail  176 tensor differences, loss diverges
defect=ignore_loader_state  -> fail  132 tensor differences, loss and IDs diverge
```

The clean case compared **186 tensors** and found zero differences.

## 5. Diagnosis

The two defects are distinguishable by *which* sub-check fails, which is what makes
the gate diagnostic rather than merely alarming:

| `sample_id_match` | `loss_match` | Localises to |
|---|---|---|
| true | false | optimizer / scheduler state — the data stream is intact |
| false | false | dataloader cursor — the run is consuming the wrong samples |
| true | true | no fault |

`reset_optimizer` leaves sample IDs matching because the data pipeline is fine; only
the trajectory is wrong. `ignore_loader_state` breaks both, because the wrong data
produces both wrong samples and a wrong trajectory.

## 6. Intervention

None required: the defects are injected and the detector is the deliverable. In a
real run the gate is the pre-flight check on the recovery path — it is run once,
deliberately, before relying on resume in an expensive setting.

## 7. Verification oracle and result

**Oracle:** exact equality under the declared constraints. **Result:** pass — the
clean resume is bit-exact, and both injected defects are detected.

## 8. Cost and time lost

Roughly 60 s of laptop CPU for the full three-case matrix. $0.

## 9. Counterfactual

Without exact equality, both defects pass. `reset_optimizer` in particular yields a
loss curve that resumes slightly higher and re-converges — indistinguishable by eye
from normal post-resume behaviour, and readily explained away as "the optimizer
warming back up." That explanation is wrong, and on a long run it silently degrades
the trajectory for as many steps as it takes AdamW's moments to re-accumulate.

## 10. Limitations and claim boundary

- One host, CPU, Gloo, 2 ranks. **Not multi-node evidence.**
- Bit-exactness is claimed **only** under the declared constraints: fp32,
  deterministic algorithms, synchronous checkpointing, fixed data order, unchanged
  world size. It is **not** claimed for bf16, asynchronous checkpointing, or any
  world-size change — see I-001b.
- Dropout is 0 in this config, so RNG restoration is not load-bearing for *this*
  gate. The RNG state is saved and restored regardless, but this experiment does
  not prove that path; a dropout-enabled variant would be needed to exercise it.
- Synthetic fixtures. No model-quality claim follows.
