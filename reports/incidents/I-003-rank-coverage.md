# I-003 — Silent sample duplication across ranks

**Injected experiment.** This failure was induced deliberately to validate a
detector. It is not an organic incident and must never be described as one.

- **Incident ID:** I-003
- **Phase:** A2 (local, CPU/Gloo)
- **Oracle:** invariant assertion
- **Result:** detected

## 1. Scope and topology

Four Gloo ranks as separate OS processes on a single host
(`distinct_hosts = 1`). Synthetic 96-sample universe, `seed = 2027`,
`shuffle = true`, `drop_last = true`, one epoch.

This is a local multi-process test environment. It is **not** multi-node
evidence; the equivalent experiment across physical hosts is Phase C.

## 2. Hypothesis and predeclared detection signal

**Hypothesis.** A sharding defect that causes ranks to consume overlapping or
duplicated samples produces no crash, no loss anomaly and no throughput anomaly.
It is invisible to every metric normally watched during training.

**Detection signal, declared before injection.** The exactly-once invariant over
one epoch:

```text
multiset_union(consumed_ids_by_rank) == expected_epoch_ids
and consumed_ids[i] disjoint from consumed_ids[j] for all i != j
```

Failure must name the offending sample IDs. A count-only check was declared
insufficient in advance, because two of the four defects below preserve a
plausible total count.

## 3. Injection

Four defects were injected independently via
`pretrainmodel.incidents.inject.inject_sampler_defect`:

| Defect | What it simulates |
|---|---|
| `duplicate_within_rank` | one rank replays its first sample |
| `duplicate_across_ranks` | two ranks are handed the same sample |
| `drop_sample` | off-by-one in a shard-length calculation |
| `full_dataset_per_rank` | every rank iterates the entire dataset |

The last is the realistic one: it is what happens when a `DistributedSampler` is
forgotten, and it is the reason this project exists.

## 4. Detection evidence

Clean runs at world size 1, 2 and 4:

- `artifacts/coverage/world-size-1.json`
- `artifacts/coverage/world-size-2.json`
- `artifacts/coverage/world-size-4.json`

Each records `passed: true`, `consumed_count == expected_count == 96`, empty
duplicate/missing/unexpected lists and no rank overlaps.

Negative control, `full_dataset_per_rank` at world size 4:

- `artifacts/coverage/negative-control-full-dataset-per-rank.json`

```text
expected_count   96
consumed_count   384          <- 4x inflation
duplicated_total 96
rank_overlaps    6 pairs      <- every pair of 4 ranks
passed           false
```

Every defect is additionally covered by an automated test:
`tests/unit/test_coverage.py` (verifier logic) and
`tests/distributed/test_rank_coverage.py` (real process group).

## 5. Diagnosis

`full_dataset_per_rank` is the instructive case. Consumed count is exactly
`world_size` times the expected count, and all six rank pairs overlap completely.
The signature distinguishes it from the other defects:

- inflated total **and** universal pairwise overlap → no sharding at all
- correct total **and** one overlapping pair → sharding present, boundary wrong
- correct-looking total **and** a within-rank repeat → sampler replay
- deficit of one **and** no overlap → off-by-one truncation

Note that the loss curve for `full_dataset_per_rank` would look entirely healthy.
Nothing in the training signal reveals it. Only the assertion does.

## 6. Intervention

None required: the defects are injected, and the detector is the deliverable. In
a real run the invariant is checked at epoch end and a violation aborts before
the checkpoint is committed, so a corrupted data trajectory cannot be resumed
and inherited.

## 7. Verification oracle and result

**Oracle:** invariant assertion (exact multiset equality plus pairwise
disjointness). **Result:** pass — all four injected defects were detected, and
all three clean world sizes satisfied the invariant.

## 8. Cost and time lost

Approximately 21 s of laptop CPU across the full distributed suite. $0.

## 9. Counterfactual

Without this invariant, `full_dataset_per_rank` would have trained on a 4×
duplicated epoch while reporting a normal loss curve, normal throughput and a
normal step count. The error would surface, if ever, as unexplained
overfitting — after the compute was already spent, and with no way to attribute
it. The equivalent defect at world size 8 on rented GPUs is the same bug and the
same silence, at cost.

## 10. Limitations and claim boundary

- One host, CPU, Gloo. Not multi-node evidence.
- Synthetic 96-sample universe; no real dataset was involved and no model-quality
  claim follows from this report.
- The invariant covers sample *identity*, not sample *content*: it proves each
  window was consumed once, not that the values inside it were correct.
- Defects were injected at the sampler boundary. A defect below that boundary
  (for example inside a shard reader) would need a different detector.
