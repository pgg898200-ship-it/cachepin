# Inspect a hedged request in Perfetto

[Open the measured trace in Perfetto](https://ui.perfetto.dev/#!/?url=https://pgg898200-ship-it.github.io/cachepin/racecurve/shared-slots.trace.json) · [Download the trace](https://pgg898200-ship-it.github.io/cachepin/racecurve/shared-slots.trace.json) · [All 36,800 requests and aggregate results](https://pgg898200-ship-it.github.io/cachepin/racecurve/)

Expand a run and compare **Replica queue**, **Wait for shared slot** and **In service** on the primary and backup lanes. The response ends at the first completed copy. Each request has a marker at the 6 ms deadline.

The specimen takes request ID equal to the offered rate from each two-second run, near its midpoint. It includes both recorded seeds, both policies and the 400/s and 1,600/s loads. Each request has its own zero at its planned arrival. Lanes from different processes are independent runs aligned for comparison. Each interval preserves its recorded duration. Missing backup lanes mean the run launched no backup for that selected request.

At 1,600/s on seed 127995, request 1,600 finished in 0.792 ms in the primary-only run and 347.378 ms in the hedged run. On seed 127996 it took 11.489 ms and 513.420 ms. These are individual requests on one host, chosen by ordinal. Use the full dataset to estimate rates or percentiles.

## Regenerate

From a full clone of CachePin, extract the recorded CSVs at the pinned experiment revision and run the standard-library exporter.

```sh
git archive 8f963f09edf68942f9ac6e304e7635d4cf41a27e experiments/shared-slots/measured | tar -x
python examples/racecurve-trace/export.py experiments/shared-slots/measured /tmp/shared-slots.trace.json
```

Each slice carries its arm, seed, offered rate and request ID. To inspect the imported durations with the Perfetto trace processor, run this query.

```sql
SELECT extract_arg(arg_set_id, 'args.arm') AS policy,
       extract_arg(arg_set_id, 'args.seed') AS seed,
       extract_arg(arg_set_id, 'args.offered_rps') AS requests_per_second,
       round(dur / 1e6, 3) AS latency_ms
FROM slice
WHERE name = 'Time to first result'
ORDER BY requests_per_second, seed, policy;
```
