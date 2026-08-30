# I-006 — FSDP2 pre-backward hook dropped by a view-returning forward

**Organic finding, not injected.** This was a latent defect in this repository's own
model, surfaced by a framework diagnostic during Phase B bring-up and fixed before
it produced a wrong result. It is recorded because it is the only non-induced defect
in this project, and because its failure mode is exactly the kind this project
exists to catch: silently wrong, not loud.

- **Incident ID:** I-006
- **Phase:** B1 (FSDP2 bring-up, local CPU/Gloo)
- **Oracle:** framework diagnostic, then exact-equality regression
- **Result:** fixed before any artifact was produced with the defective code

## 1. Scope and topology

Two Gloo ranks on one host, FSDP2 over a CPU `DeviceMesh`. Not multi-node evidence.

## 2. Hypothesis and detection signal

No hypothesis was declared in advance — this was not an experiment. The signal was
an unsolicited `UserWarning` emitted by `torch.distributed.fsdp` on the first
sharded forward pass:

> FSDP2-wrapped module returned a view tensor. An in-place op on this view will
> silently drop the pre-backward hook and skip the all-gather, which can cause
> backward to fail or produce wrong gradients.

## 3. What was wrong

`SpatiotemporalTransformer.forward` ended with:

```python
out.reshape(b, s, self.horizon_steps, self.num_features).permute(0, 2, 1, 3)
```

`permute` returns a **view**. FSDP2 registers its pre-backward hook on the returned
tensor; if a caller later performs an in-place operation on that view, the hook is
dropped, the parameter all-gather before backward never runs, and gradients are
computed against sharded rather than gathered parameters.

## 4. Detection evidence

The warning fired on every rank on the first `fully_shard` forward. It is visible in
the Phase B bring-up logs and does not appear after the fix.

## 5. Diagnosis

The dangerous property is that this is **conditional and silent**. Nothing fails at
the point of the view; the corruption requires a later in-place op, and its symptom
is wrong gradient values — not a crash, not a shape error, not a NaN. On a single
GPU there is no all-gather to skip and the bug does not exist at all, so it would
have appeared only after moving to sharded execution, and would have presented as
"the multi-node run converges worse than the single-GPU reference" — a symptom with
a dozen plausible and wrong explanations (learning rate, batch size, network,
sharding strategy).

## 6. Intervention

Append `.contiguous()` to the forward's return value, materialising a dense tensor
so no downstream in-place op can detach the hook. The call is documented in-line as
load-bearing rather than cosmetic, since `.contiguous()` reads like a performance
tweak and would otherwise be a tempting deletion.

## 7. Verification oracle and result

The bit-exact resume oracle (I-001a) runs against the fixed model under FSDP2 and
passes with zero differences across 186 tensors. That does not by itself prove this
bug is gone — the oracle compares two runs of the same code — but it does establish
that the sharded forward and backward produce a self-consistent, reproducible
trajectory. **Result:** fixed; no artifact in this repository was produced by the
defective version.

## 8. Cost and time lost

Under 10 minutes, entirely because the framework warned. $0.

## 9. Counterfactual

Had the warning been suppressed or ignored, the defect would have survived until a
model change introduced an in-place op on the output — at which point gradients
would have quietly been wrong on sharded runs only. The natural debugging path
(compare against the single-GPU reference, suspect the learning rate, suspect the
network) leads away from the actual cause, and the single-GPU reference would have
looked perfect throughout.

## 10. Limitations and claim boundary

- Found by reading a framework warning, not by a detector this project built. No
  credit is claimed for the detection mechanism.
- The fix is verified by absence of the warning and by the bit-exactness oracle
  passing. A direct test — asserting gradient equality between sharded and
  unsharded execution — is **not** implemented and would be the stronger evidence.
- One host, CPU, Gloo. Not multi-node evidence.
