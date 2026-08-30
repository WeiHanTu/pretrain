# I-002 — Length-preserving shard corruption

**Injected experiment.** Induced deliberately to validate a detector. Not an
organic incident.

- **Incident ID:** I-002
- **Phase:** A2 (local, CPU)
- **Oracle:** invariant assertion
- **Result:** detected before optimizer step 0

## 1. Scope and topology

Single process, CPU. Synthetic dataset of 3 shards × 40 timesteps × 8 sensors,
each shard's SHA-256 and byte length recorded in `shards.json` at creation.

## 2. Hypothesis and predeclared detection signal

**Hypothesis.** A shard modified after preprocessing — a partial write, a bad
disk, an interrupted sync, an object-store retry that wrote stale bytes — will
not announce itself. Training will proceed and produce a loss curve.

**Detection signal, declared before injection.** `ShardManifest.verify()` runs
before the first optimizer step and must reject the dataset, naming the offending
file and reporting `sha256 mismatch`. Declared in advance: a **size check alone
is insufficient**, so the injected corruption is length-preserving on purpose.

## 3. Injection

`corrupt_file_bytes()` XORs 16 bytes at offset 100 of shard 1 in place. File
length is unchanged and the test asserts that it is unchanged, so the digest is
the only thing that can catch it.

A no-op corruption would make the detector appear to pass while nothing was ever
broken, so corrupting an empty file raises rather than silently doing nothing
(`test_corrupting_an_empty_file_is_an_error`).

## 4. Detection evidence

`tests/unit/test_shard.py`:

- `test_corrupted_shard_is_rejected_and_named` — asserts length is preserved,
  then asserts `ManifestError` naming the shard path and `sha256 mismatch`;
  restores the bytes and re-verifies clean.
- `test_truncated_shard_is_rejected` — size mismatch path.
- `test_missing_shard_is_rejected` — absent file path.
- `test_verify_reports_every_bad_shard_not_just_the_first` — a two-fault dataset
  reports both faults.
- `test_index_needs_no_shard_bytes` — the sample index is a pure function of the
  manifest, so corruption cannot silently change what the epoch *is*.

## 5. Diagnosis

Three distinct failure modes are distinguished by the manifest check rather than
collapsed into one "bad data" error:

| Symptom | Cause |
|---|---|
| size mismatch | truncated or partially written file |
| sha256 mismatch, size correct | in-place modification or bit rot |
| missing shard | incomplete sync or wrong dataset root |

The distinction matters operationally: truncation points at an interrupted write,
digest mismatch at storage or a stale copy.

## 6. Intervention

Verification runs before the first optimizer step, so the run aborts having spent
no compute. The remedy is to re-fetch the shard and re-verify against the same
manifest; the manifest itself is never regenerated to make the error go away,
which would simply record the corruption as canonical.

## 7. Verification oracle and result

**Oracle:** invariant assertion (SHA-256 and byte length against the recorded
manifest). **Result:** pass — corruption, truncation and absence are each
detected and named.

## 8. Cost and time lost

Under 1 s of CPU. $0. In a cloud run the value is the compute *not* spent: the
gate fires before step 0 rather than after a full run on bad data.

## 9. Counterfactual

Without the digest gate, the corrupted shard would have been read as valid
floats. The corrupted region decodes to arbitrary values, including NaN, which
would surface as a loss spike or a silent NaN propagation attributed to the
learning rate or to instability — a plausible, wrong diagnosis that costs hours
and possibly a restart.

## 10. Limitations and claim boundary

- Detects corruption **at rest, before a run**. It does not detect corruption
  that occurs mid-run after verification, which would need periodic re-checking
  or a checksummed read path.
- Synthetic fixtures only; no statement about any real dataset.
- SHA-256 over whole files: it identifies the offending file, not the offending
  byte range.
- Single process. Verifying that every rank sees identical bytes across hosts is
  a Phase C concern.
