import csv
import random

random.seed(4)  # reproducible

def load_trace(filename):
    """Load a saved trace file: timestamp_ns | energy_uj | power_watts"""
    trace = []
    with open(filename) as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3 and parts[2] not in ('', None):
                try:
                    trace.append(float(parts[2]))
                except ValueError:
                    continue
    return trace

# Your 5 real dense traces
dense_files = ['run_821_gpu_trace.txt', 'run_904_gpu_trace.txt',
               'run_902_gpu_trace.txt', 'run_823_gpu_trace.txt', 'run_1114_gpu_trace.txt']
dense_traces = [load_trace(f) for f in dense_files]

# Your 3 real retry traces (contain the idle-to-spike jump)
retry_files = ['run_769_gpu_trace.txt', 'run_223_gpu_trace.txt', 'run_230_gpu_trace.txt']
retry_traces = [load_trace(f) for f in retry_files]

N = 32
SIM_LENGTH = 300  # 300 samples * 100ms = 30 seconds of simulated rack time
MAX_OFFSET = 100  # random start offset window, in samples

def build_task_series(base_trace, offset, length):
    """Place a real trace at a random start offset, using a random window
    from WITHIN the source trace (not always its beginning) to capture
    steady-state behavior instead of startup transients."""
    series = [0.0] * length
    if len(base_trace) > length:
        max_start = len(base_trace) - length
        window_start = random.randint(0, max_start)
        windowed = base_trace[window_start:window_start + length]
    else:
        windowed = base_trace
    for i, val in enumerate(windowed):
        idx = offset + i
        if idx < length:
            series[idx] = val
    return series

# --- INDEPENDENT scenario: 32 tasks, random independent start times ---
independent_series = []
for _ in range(N):
    trace = random.choice(dense_traces)
    offset = random.randint(0, MAX_OFFSET)
    independent_series.append(build_task_series(trace, offset, SIM_LENGTH))

independent_aggregate = [sum(s[t] for s in independent_series) for t in range(SIM_LENGTH)]

# --- CORRELATED scenario: same 32 tasks, but 12 of them forced to retry-spike together ---
correlated_series = []
SYNC_TIME = 100  # the instant everyone's retry lands
for i in range(N):
    if i < 12:  # these 12 all get a retry pattern, synchronized
        trace = random.choice(retry_traces)
        correlated_series.append(build_task_series(trace, SYNC_TIME, SIM_LENGTH))
    else:  # the rest behave independently
        trace = random.choice(dense_traces)
        offset = random.randint(0, MAX_OFFSET)
        correlated_series.append(build_task_series(trace, offset, SIM_LENGTH))

correlated_aggregate = [sum(s[t] for s in correlated_series) for t in range(SIM_LENGTH)]

# --- Summary numbers for Station 3 ---
def summarize(name, agg):
    baseline = sum(agg[:50]) / 50  # first 5 seconds, before any spike
    peak = max(agg)
    ramp_w_per_100ms = max(agg[t] - agg[t-1] for t in range(1, len(agg)))
    print(f"{name}: baseline={baseline:.1f}W, peak={peak:.1f}W, "
          f"max single-step ramp={ramp_w_per_100ms:.1f}W per 100ms "
          f"({ramp_w_per_100ms*10/1000:.3f} kW/sec)")

summarize("Independent (32 tasks)", independent_aggregate)
summarize("Correlated (12 of 32 synchronized)", correlated_aggregate)

# Save both curves for plotting in the actual figure
with open("station3_simulation_output.csv", "w") as f:
    f.write("time_step,independent_watts,correlated_watts\n")
    for t in range(SIM_LENGTH):
        f.write(f"{t},{independent_aggregate[t]:.1f},{correlated_aggregate[t]:.1f}\n")
print("Saved station3_simulation_output.csv")
