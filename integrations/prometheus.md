# What can separate Prometheus histograms tell me about a hedge timer?

If you measured both replicas for every request but retained only cumulative bucket counts, Racecurve can bound deadline misses and backup launches for a proposed delay. It requires complete service durations measured from each replica's own dispatch, in seconds, on the same request cohort. A shadow experiment where both copies finish can provide these observations. Ordinary winner-only logging, selectively launched backups, and canceled attempts need the [censored-trace workflow](../BOUNDS.md). Matching totals alone cannot establish matching cohorts.

## Run the saved example

From the Racecurve source directory, use Python 3.10 or later.

```sh
python3 integrations/prometheus_bounds.py integrations/moss-prometheus.json --deadline-s 60 --delay-s 0 10 20 60 120
```

The [saved instant vector](moss-prometheus.json) came from a live Prometheus 3.5.0 scrape of bucket counts derived from GPUHedge's 36 held-out RunPod/Modal pairs. The original experiment recorded both providers. PGG generated the histogram and performed the comparison. [Upstream source, license and selection of rounds](../external/README.md)

| Timer | Launches from histogram | Misses from histogram | Misses from paired replay |
| --- | ---: | ---: | ---: |
| Immediate | 36 | 0–5 | 0 |
| 10 s | 11 | 0–6 | 1 |
| 20 s | 11 | 0–11 | 2 |
| 60 s | 11 | 11 | 11 |
| 120 s | 2 | 11 | 11 |
| Primary only | 0 | 11 | 11 |

At 10 s the histogram rules out a 25% duplicate budget (nine launches). Eleven primaries exceed the deadline and six backups exceed the remaining 50 s. Assuming independent replica times would predict 11 × 6 / 36 = 1.83 misses in expectation. The histogram permits any integer from zero through six; the actual pairing gives one. At 20 s, the backup threshold of 40 s also falls between bucket edges, widening the range. At 60 s and later, primary-only achieves the same miss count with zero duplicates.

## Export your own counts

For a dedicated completed shadow run, take a single instant query after both replicas finish. Supply the metric name and a selector covering that run. Both histograms must count exactly one observation per arm per request from a common start, with no counter resets or unrelated requests. The example below assumes a local Prometheus server and application-defined labels. Aggregating per-interval rates, mixing cohorts, or counting observed cancellation times as service times changes the question.

```sh
curl -fsSG http://localhost:9090/api/v1/query \
  --data-urlencode 'query=sum by (arm, le) (shadow_duration_seconds_bucket{experiment="my-run"})' > buckets.json
python3 integrations/prometheus_bounds.py buckets.json --deadline-s 0.1 --delay-s 0.01 0.05
```

Use the arm label values primary and backup. Both arms need a +Inf total; their finite bucket layouts can differ. Keep integer counts. The importer rejects unequal totals, repeated boundaries, decreasing counts, extra labels and fractional counts. [Prometheus query response format](https://prometheus.io/docs/prometheus/latest/querying/api/)

## Check the bounds

For a timer d and deadline s, a miss requires both P > s and B > s − d. A launch requires P > d. At an exact bucket edge, subtract the cumulative count from the total. Between edges, use the two adjacent counts. Service durations are nonnegative. For primary and backup late-count intervals [a, A] and [b, B] on N requests, their intersection can contain from max(0, a + b − N) through min(A, B) requests. These are attainable endpoints at each individual delay consistent with the buckets and arbitrary pairing. Across delays the same rows must stay paired. The bounds apply to the observed cohort under fixed service durations; queue feedback and sampling uncertainty require separate measurements.

To repeat the live scrape with a Prometheus binary on your PATH, run the command below. The driver verifies the pinned upstream trace checksum, starts a local exporter and server, queries the real API, and checks the output against the complete pairs. Use `--source` for an offline copy of the upstream JSONL and `--vector-out` to save the response.

```sh
python3 external/check_prometheus.py --prometheus prometheus
```
