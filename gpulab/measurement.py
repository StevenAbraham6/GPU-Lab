import statistics
import time
from .catalog import metric

def stats(values):
    values = sorted(float(v) for v in values)
    med = statistics.median(values)
    return dict(median=med, minimum=min(values), maximum=max(values),
                p95=values[min(len(values)-1, int(.95*(len(values)-1)))],
                cv_pct=100*statistics.pstdev(values)/med if med else 0, samples=values)

def timed_loop(fn, cfg, work, unit, name, settings, minimum=None):
    """fn must complete one dispatch and return device seconds; energy uses wall work."""
    until = time.perf_counter() + cfg["warmup"]
    while time.perf_counter() < until:
        fn()
    rates, intervals = [], []
    for _ in range(cfg["repeats"]):
        start = time.time(); start_m = time.perf_counter()
        count, device_s = 0, 0.0
        while count == 0 or time.perf_counter()-start_m < (minimum or cfg["duration"]):
            seconds = fn()
            if seconds <= 0:
                raise RuntimeError("Non-positive device timer")
            device_s += seconds; count += 1
        end = time.time()
        rates.append(work*count/device_s)
        intervals.append(dict(start=start, end=end, work=work*count, work_unit=unit.split("/")[0], dispatches=count,
                              device_seconds=device_s, wall_rate=work*count/(end-start)))
    summary = stats(rates)
    return dict(name=name, status="passed", settings=settings,
                metrics=[metric("throughput", summary["median"], unit, statistics=summary)],
                intervals=intervals, timing="device events for throughput; continuous-loop wall time for efficiency")

def event_seconds(event):
    event.wait()
    seconds=(event.profile.end-event.profile.start)*1e-9
    if seconds<=0:raise RuntimeError('Device event has no positive profiling duration')
    return seconds
