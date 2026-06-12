"""Benchmarks: block_lms frequency-domain equalizer (DD-02 gate).

Canonical DD-02 workload: ``block_lms/bps+cs/16qam/N1e5/C2`` — BPS CPR with
per-symbol cycle-slip correction enabled, which exercises the per-block
D2H/H2D synchronization points that DD-02 removes.
"""

import pytest

from commstools.equalization import block_lms
from workloads import mimo_equalizer_workload

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)
N_SYM = 100_000


@pytest.mark.parametrize(
    "label,cpr_kwargs",
    [
        ("no-cpr", dict()),
        ("bps", dict(cpr_type="bps")),
        ("bps+cs", dict(cpr_type="bps", cpr_cycle_slip_correction=True)),
    ],
)
def bench_block_lms(benchmark, backend_device, xp, sync, label, cpr_kwargs):
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
            block_size=256,
            **cpr_kwargs,
        )
        sync()
        return r

    benchmark.pedantic(run, **ROUNDS)
