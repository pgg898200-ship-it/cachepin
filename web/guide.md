# Choose a backup-request delay from a trace

Racecurve estimates the outcomes of racing one primary and one backup for an idempotent call. It preserves the association between attempts for the same logical request. The static browser explorer reads the file on your device. A C++17 streaming kernel and Python utilities are in [the source download](source.zip).

## Record one row per request

The browser accepts simple numeric CSV or the four observation status keywords and intervals below. Preserve arrival order. Provide at least two rows and one of these exact headers.

```csv
primary_us,backup_us
70,35
110,20
```

Use complete paired times when both branches finished. Times are durations from each branch's own dispatch.

```csv
winner,elapsed_us,old_delay_us
primary,70,50
backup,90,50
```

For compact logs, record the original launch delay, with launches occurring exactly when the primary exceeds that delay. The primary wins ties. For dispatch jitter or cleanup spans, import instrumented OpenTelemetry traces with the Python adapter.

```csv
a_status,a_us,b_status,b_us
exact,70,at_least,20
unstarted,,interval,89..90
```

Columns a and b represent primary and backup. Status exact records a completion, at_least a lower bound, censored an unfinished duration strictly above the indicated integer, unstarted an unknown nonnegative service time, and interval an inclusive lower..upper range.

## Interpret the curve

For fixed primary duration P, backup duration B and delay D, completion is min(P, D+B). A launch occurs when P>D. A miss occurs when completion exceeds the deadline. The monotone response formula allows evaluations at both endpoints of an interval and preserves uncertainty from censored losers.

The explorer calibrates on the first 70% in file order and shows the remaining requests. It chooses among up to 16,385 equally spaced integer delays and a primary-only option. For a given extra-request budget, it minimizes the upper bound on misses, then upper-bound launches, then delay. The budget applies to calibration. Held-out traffic may exceed it. Completed traces yield exact observed rates under replay. The uncertainty bands from censored logs represent missing observations. They do not express sampling uncertainty.

## Reproduce the native kernel

Extract the source ZIP. In its top directory run:

```sh
make test
./racecurve 6000 1 6001 examples/loopback.csv > curve.csv
python3 examples/live_ab.py --output fresh-live-ab.json
```

The native kernel also ingests sorted group,primary_us,backup_us for all-of fanout, reports aggregate attempt occupancy under instant cancellation and exposes a mergeable C++ API. The browser handles one leaf per row and reports request counts. The source includes the paired collector, benchmark, direct replay oracle, stronger indexed baselines, and an OTLP JSON adapter for tagged spans.

These calculations assume fixed service times under changed launch timing. The localhost experiment used controlled stalls, serialized logical requests, and fully completed losers. Deployment decisions still depend on real congestion, throttling, transport delays, errors, side effects and billing.
