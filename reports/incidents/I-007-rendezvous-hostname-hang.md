# I-007 — c10d rendezvous hangs forever when a node cannot resolve its own hostname

**Organic finding, not injected.** Encountered while rehearsing the real multi-node
launch path locally, before any paid hardware was provisioned. Recorded because the
whole point of the rehearsal was to find this class of failure for free, and it did.

- **Incident ID:** I-007
- **Phase:** C preparation (local, CPU/Gloo)
- **Oracle:** manual inspection, then an automated preflight
- **Result:** diagnosed, worked around, and converted into a fail-fast check

## 1. Scope and topology

Two ranks on one host via `torchrun`, Gloo. Not multi-node evidence — this is the
launch *path* being exercised, not a network boundary being crossed.

## 2. Context

Every distributed run in Phases A and B used `file://` rendezvous, which works only
because all ranks share a filesystem. Real multi-node uses TCP: ranks find each
other through `MASTER_ADDR`/`MASTER_PORT`. Those are different code paths, and the
TCP one had never executed in this repository.

Rehearsing it was cheap. Discovering its failure modes on rented GPUs would not be.

## 3. What happened

```bash
torchrun --nnodes=1 --nproc-per-node=2 \
         --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29511 \
         -m pretrainmodel.distributed.entrypoint --config configs/local_smoke.toml
```

The job produced no training output and did not exit. It emitted, on a widening
retry interval:

```text
[c10d] The hostname of the client socket cannot be retrieved. err=-3
[c10d] The IPv6 network addresses of (claude, 35137) cannot be retrieved
       (gai error: -3 - Temporary failure in name resolution)
```

Retries continued at ~1s, 4s, 5s, 12s, 28s, 40s intervals. It never failed.

## 4. Diagnosis

The host's own hostname is not resolvable:

```text
$ hostname
claude
$ getent hosts claude
(no output)
$ grep claude /etc/hosts
(no match)
```

`c10d` resolves the local hostname while constructing its rendezvous store. The
lookup fails, and the store retries with backoff **indefinitely** rather than
raising. Nothing in the process ever errors, so nothing downstream can catch it.

Three properties make this expensive:

1. **It hangs rather than fails.** On a billed instance the meter runs for the
   entire hang, with no upper bound and no error to alert on.
2. **The symptom points away from the cause.** "The job just sits at startup"
   suggests a firewall, a wrong `MASTER_ADDR`, a peer that never joined, a security
   group — anywhere but `/etc/hosts` on the machine you are already logged into.
3. **The obvious defence does not work.** `init_process_group(timeout=...)` bounds
   the *collective*, but torchrun performs this lookup while building its store,
   before any timeout this process controls is in effect.

## 5. Intervention

Two changes, plus a workaround.

**Workaround (immediate).** Static rendezvous with an explicit IP avoids the c10d
store's hostname lookup entirely:

```bash
torchrun --nnodes=1 --nproc-per-node=2 \
         --master-addr=127.0.0.1 --master-port=29520 \
         -m pretrainmodel.distributed.entrypoint --config configs/local_smoke.toml --rehearsal
```

This completed: 20 steps, `world_size=2`, `distinct_hosts=1`, `multi_node=False`.

**Fix 1 — fail fast.** `distributed.launch.network_preflight()` checks hostname
resolution before any rendezvous is attempted and refuses to launch with the actual
remedy in the message. A multi-minute silent hang becomes an immediate, actionable
error. It is fatal only when the run claims distinct hosts; a loopback single-host
run does not need resolution and is allowed to proceed.

**Fix 2 — prevent it on the cloud nodes.** `infra/gcp/startup.sh` appends the
hostname to `/etc/hosts` at boot when it does not already resolve.

## 6. Verification oracle and result

**Oracle:** manual inspection for the diagnosis; automated assertion for the fix.
**Result:** `network_preflight()` reports `resolves=False, fatal=True` on this host
with the remedy quoted, and the preflight gate
(`artifacts/cloud/preflight.json`) fails its `hostname_resolves` check accordingly.

## 7. Cost and time lost

About 20 minutes of local debugging. $0.

Avoided cost is the point. Two spot L4 instances hanging for one unattended hour is
a small charge; the same hang across an afternoon of "why won't it start" is the
failure mode this project exists to demonstrate competence against.

## 8. Counterfactual

Without the rehearsal, this would have been discovered on the first paid two-node
launch — the moment when the priority is measurement, the meter is running, and the
natural suspects (firewall rules, VPC routing, the peer node) are all wrong and all
expensive to investigate.

## 9. Limitations and claim boundary

- Encountered in a sandboxed VM whose `/etc/hosts` is read-only. A stock GCP image
  normally resolves its own hostname, so this specific trigger may not recur there;
  the *class* of failure — rendezvous hanging instead of failing — does recur, which
  is why the check ships rather than just the workaround.
- The preflight verifies hostname resolution only. It does not verify that
  `MASTER_ADDR` is reachable from a peer, that the rendezvous port is open, or that
  a firewall permits collectives. Those need two real hosts and belong to Phase C.
- One host, CPU, Gloo. **Not multi-node evidence.**
