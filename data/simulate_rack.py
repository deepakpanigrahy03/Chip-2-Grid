import csv
import random

def load_trace(filename):
    """Load a saved trace file: timestamp_ns | energy_uj | power_watts.
    Power is clamped at zero: the raw watts column is derived upstream as
    delta_energy / delta_time between consecutive hardware counter samples,
    which is numerically unstable when two samples land close in time --
    counter jitter near-idle can produce small negative deltas. Physical
    power cannot be negative, so this clamps the artifact at its source
    rather than let it propagate into the summed aggregate."""
    trace = []
    with open(filename) as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3 and parts[2] not in ('', None):
                try:
                    trace.append(max(0.0, float(parts[2])))
                except ValueError:
                    continue
    return trace

# Steady-state sources: dense traces, dcgm_field156 attribution method only.
# NOTE: run_52/56/88 were used in an earlier version of this script and were
# removed after discovering they use a different, incompatible attribution
# method (spbm_package_v1) with incomplete GPU energy columns. See
# figure2_methodology.tex Section 2-3 for the full finding.
dense_files = ['run_821_gpu_trace.txt', 'run_904_gpu_trace.txt',
               'run_902_gpu_trace.txt', 'run_823_gpu_trace.txt', 'run_1114_gpu_trace.txt']
dense_traces = [load_trace(f) for f in dense_files]

# Retry-pattern sources: contain the real idle-to-spike jump
retry_files = ['run_769_gpu_trace.txt', 'run_223_gpu_trace.txt', 'run_230_gpu_trace.txt']
retry_traces = [load_trace(f) for f in retry_files]

N = 32
SIM_LENGTH = 300  # 300 samples * ~100ms native sampling interval = 30 seconds
MAX_OFFSET = 100  # random start offset window, in samples
SYNC_TIME = 100   # the instant everyone's retry lands

def build_task_series(rng, base_trace, offset, length):
    """Place a real trace at a random start offset within the simulation
    window. Draws a random WINDOW from within the source trace (not always
    its start) to capture steady-state behavior rather than every task
    showing the same startup transient."""
    series = [0.0] * length
    if len(base_trace) > length:
        max_start = len(base_trace) - length
        window_start = rng.randint(0, max_start)
        windowed = base_trace[window_start:window_start + length]
    else:
        windowed = base_trace
    for i, val in enumerate(windowed):
        idx = offset + i
        if idx < length:
            series[idx] = val
    return series

def run_simulation(seed):
    """Run one full independent-vs-correlated simulation for a given seed.
    Returns (independent_aggregate, correlated_aggregate) as lists of
    per-timestep summed power across all 32 simulated task slots."""
    rng = random.Random(seed)

    # --- INDEPENDENT scenario: 32 tasks, random independent start times ---
    independent_series = []
    for _ in range(N):
        trace = rng.choice(dense_traces)
        offset = rng.randint(0, MAX_OFFSET)
        independent_series.append(build_task_series(rng, trace, offset, SIM_LENGTH))
    independent_aggregate = [sum(s[t] for s in independent_series) for t in range(SIM_LENGTH)]

    # --- CORRELATED scenario: same 32 tasks, but 12 of them forced to
    #     retry-spike together (models a synchronized retry storm) ---
    correlated_series = []
    for i in range(N):
        if i < 12:
            trace = rng.choice(retry_traces)
            correlated_series.append(build_task_series(rng, trace, SYNC_TIME, SIM_LENGTH))
        else:
            trace = rng.choice(dense_traces)
            offset = rng.randint(0, MAX_OFFSET)
            correlated_series.append(build_task_series(rng, trace, offset, SIM_LENGTH))
    correlated_aggregate = [sum(s[t] for s in correlated_series) for t in range(SIM_LENGTH)]

    return independent_aggregate, correlated_aggregate

def summarize(agg):
    """Return (baseline, peak, max_ramp_w_per_100ms) for one aggregate series."""
    baseline = sum(agg[:50]) / 50  # first 5 seconds, before any spike
    peak = max(agg)
    ramp = max(agg[t] - agg[t - 1] for t in range(1, len(agg)))
    return baseline, peak, ramp

# --- Run all 5 seeds, matching the seeds reported in the methodology doc ---
SEEDS = [42, 1, 2, 3, 4]
results = []

print(f"{'Seed':>6} {'Indep. peak (W)':>16} {'Corr. peak (W)':>16} {'Ratio':>8} {'Indep. ramp (kW/s)':>20} {'Corr. ramp (kW/s)':>20}")
for seed in SEEDS:
    indep_agg, corr_agg = run_simulation(seed)
    indep_baseline, indep_peak, indep_ramp = summarize(indep_agg)
    corr_baseline, corr_peak, corr_ramp = summarize(corr_agg)
    ratio = corr_peak / indep_peak if indep_peak else float('nan')
    indep_ramp_kw_s = indep_ramp * 10 / 1000
    corr_ramp_kw_s = corr_ramp * 10 / 1000  # 100ms steps -> per-second, W -> kW
    results.append({
        'seed': seed,
        'indep_baseline': indep_baseline, 'indep_peak': indep_peak,
        'corr_baseline': corr_baseline, 'corr_peak': corr_peak,
        'ratio': ratio, 'indep_ramp_kw_s': indep_ramp_kw_s, 'corr_ramp_kw_s': corr_ramp_kw_s,
    })
    print(f"{seed:>6} {indep_peak:>16.1f} {corr_peak:>16.1f} {ratio:>7.2f}x {indep_ramp_kw_s:>19.3f} {corr_ramp_kw_s:>19.3f}")
    if seed == SEEDS[0]:
        # Save the primary reported run (seed 42) as the per-timestep CSV,
        # matching the original script's behavior.
        with open("station3_simulation_output.csv", "w") as f:
            f.write("time_step,independent_watts,correlated_watts\n")
            for t in range(SIM_LENGTH):
                f.write(f"{t},{indep_agg[t]:.1f},{corr_agg[t]:.1f}\n")

avg_indep_peak = sum(r['indep_peak'] for r in results) / len(results)
avg_corr_peak = sum(r['corr_peak'] for r in results) / len(results)
avg_ratio = avg_corr_peak / avg_indep_peak
avg_indep_ramp = sum(r['indep_ramp_kw_s'] for r in results) / len(results)
avg_corr_ramp = sum(r['corr_ramp_kw_s'] for r in results) / len(results)
min_indep_ramp = min(r['indep_ramp_kw_s'] for r in results)
max_indep_ramp = max(r['indep_ramp_kw_s'] for r in results)
min_corr_ramp = min(r['corr_ramp_kw_s'] for r in results)
max_corr_ramp = max(r['corr_ramp_kw_s'] for r in results)

print(f"{'Average':>6} {avg_indep_peak:>16.1f} {avg_corr_peak:>16.1f} {avg_ratio:>7.2f}x {avg_indep_ramp:>19.3f} {avg_corr_ramp:>19.3f}")
print(f"Indep. ramp range across seeds: {min_indep_ramp:.3f}-{max_indep_ramp:.3f} kW/s")
print(f"Corr.  ramp range across seeds: {min_corr_ramp:.3f}-{max_corr_ramp:.3f} kW/s")
print("Saved station3_simulation_output.csv (seed 42 run)")