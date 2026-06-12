"""Benchmarks: block_lms frequency-domain equalizer (DD-02 gate).

Canonical DD-02 workload: ``block_lms/bps+cs/16qam/N1e5/C2`` — BPS CPR with
per-symbol cycle-slip correction enabled, which exercises the per-block
D2H/H2D synchronization points that DD-02 removes.

Two block sizes are tracked:

* ``bench_block_lms`` (block_size=256) — the deliberate stress case: on GPU
  the per-block work is too small to amortize the ~30-50 kernel launches per
  iteration, so wall time is dominated by fixed launch overhead.  This is the
  DD-02 gate workload (kept ID-stable across baselines) and the case CUDA
  graph capture (DD-02 step 3) is meant to rescue.
* ``bench_block_lms_large`` (block_size=2048) — the recommended GPU operating
  point with launch overhead amortized.  Guards against regressions that
  scale with block size (per-element work, intermediate-tensor growth) which
  the overhead-bound 256 case would mask, and provides the throughput ceiling
  that graph capture at small block sizes is judged against.
"""

import pytest

from commstools.equalization import block_lms
from workloads import mimo_equalizer_workload

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)
N_SYM = 100_000

CPR_CONFIGS = [
    ("no-cpr", dict()),
    ("bps", dict(cpr_type="bps")),
    ("bps+cs", dict(cpr_type="bps", cpr_cycle_slip_correction=True)),
]


def _bench_block_lms(benchmark, xp, sync, cpr_kwargs, block_size):
    linewidth = 1e4 if cpr_kwargs else 0.0
    samples, syms = mimo_equalizer_workload(
        n_sym=N_SYM, order=16, sps=2, linewidth_hz=linewidth
    )
    x = xp.asarray(samples)
    t = xp.asarray(syms)

    def run():
        r = block_lms(
            x,
            t,
            num_taps=21,
            sps=2,
            modulation="qam",
            order=16,
            block_size=block_size,
            **cpr_kwargs,
        )
        sync()
        return r

    benchmark.pedantic(run, **ROUNDS)


@pytest.mark.parametrize("label,cpr_kwargs", CPR_CONFIGS)
def bench_block_lms(benchmark, backend_device, xp, sync, label, cpr_kwargs):
    _bench_block_lms(benchmark, xp, sync, cpr_kwargs, block_size=256)


@pytest.mark.parametrize("label,cpr_kwargs", CPR_CONFIGS)
def bench_block_lms_large(benchmark, backend_device, xp, sync, label, cpr_kwargs):
    _bench_block_lms(benchmark, xp, sync, cpr_kwargs, block_size=2048)
