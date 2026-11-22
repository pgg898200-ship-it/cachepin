# RaceCurve command-line tools

Choose a backup-request delay from paired HTTP timings, or determine how tightly a canceled-race log bounds its possible performance. Requires Python 3.10 or later. The wheel uses only the Python standard library at runtime.

```sh
python3 -m pip install https://huggingface.co/spaces/pgg-engineering/racecurve/resolve/main/racecurve_tools-0.1.0-py3-none-any.whl
racecurve-demo
```

The demo ships with 1,000 complete primary/backup HTTP timings from a constructed loopback workload. It calibrates on the first 700 rows with a 6,000 µs deadline and 25% extra-request budget. The selected 1,177 µs delay meets the deadline on all 300 held-out requests in replay, with 42 backup launches. Primary-only misses 27 deadlines. This models fixed branch durations when launch timing changes. The package does not dispatch requests or observe queueing feedback.

| Command | Input | Output |
| --- | --- | --- |
| `racecurve-demo pairs.csv --slo-us 6000 --budget-pct 25` | Ordered CSV with header `primary_us,backup_us` | JSON with selected delay, calibration, holdout and primary-only comparison |
| `racecurve-bounds races.csv --slo-us 6000 --delay-us 0 1000 2000` | CSV with header `winner,elapsed_us,old_delay_us`, or explicit interval observations | Sharp empirical response-sum and deadline-miss bounds for each delay |
| `racecurve-otlp trace.jsonl` | Caller-side OTLP JSON trace batches with documented `racecurve.*` attributes | Observation CSV suitable for piping into `racecurve-bounds -` |

The selector considers integer delays from zero through the deadline, plus a primary-only option. It rejects policies above the calibration launch budget, minimizes deadline misses, then chooses fewer launches and the smallest delay. A `null` delay represents primary-only. Holdout rows can exceed the calibration launch budget. Full native fanout analysis, observation formats and instrumentation examples are in the [project](https://huggingface.co/spaces/pgg-engineering/racecurve).

To build the wheel from the source distribution, run `python3 packaging/build_wheel.py` from the project root with setuptools, wheel and pip installed. The build stages the original source modules and trace in a temporary directory. The wheel is written to `dist/`.
