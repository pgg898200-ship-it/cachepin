# Does a 25% duplicate cap protect a 6 ms deadline?

The original shared-slot workload selected a 1,177 µs timer under a calibration budget. During a busier run it duplicated nearly every request. This experiment enforces the budget when the dispatcher decides whether to send the backup. At timer decision n, it permits at most floor(n/4) cumulative backups including the proposed launch. The primary must still be outstanding. Unused allowance carries forward, so shorter intervals can exceed 25%. Each new arm starts with zero credit.

All three policies run the same service demands and regularly scheduled arrivals through the [two-slot executor](../README.md). Each arm lasts two seconds plus drain. Three seeds rotate the execution order so each policy runs first, second and third once per rate. The deadline runs from planned arrival to first completion. Both copies occupy their slot until completion. Timestamp and winner checks run on every row. The runner also verifies the cumulative budget at every request ID.

## Reproduce

From the repository root with Python 3.10+ and a C++17 compiler, run

```sh
python3 experiments/shared-slots/compare_cap.py --output fresh-cap-run
```

The output contains 27 compressed CSVs and a JSON summary with per-run p50, p95, p99, misses, launches, queue waits and drain. Matching the seeds reproduces demand and offered load. Timer jitter and host scheduling affect observed response times. The harness simulates service with timed sleeps on real worker threads.

## Local results

Counts sum three runs per rate. The host used Linux x86-64 and GCC 14.2.0. [Per-run counts and latency percentiles](local-runs.csv)

| Requests/s | Requests per policy | Primary-only misses | Fixed timer misses | Capped timer misses | Fixed launches | Capped launches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | 2,400 | 204 | 1 | 1 | 339 | 332 |
| 1,600 | 9,600 | 2,147 | 9,491 | 4,625 | 9,520 | 2,400 |
| 2,600 | 15,600 | 15,346 | 15,510 | 12,127 | 15,533 | 3,900 |

At 1,600/s, enforcing the duplicate budget reduced misses relative to the fixed timer, but turning backups off prevented another 2,478 misses. At 2,600/s the capped arm had fewer misses than primary-only, while every capped run had worse p99 latency (976–1,074 ms versus 512–538 ms). Optimizing a single miss threshold can hide a longer tail among the requests that miss.

The cap controls cumulative launches. It does not reserve execution slots, cancel losing work, or adapt the timer to queue depth. These short runs cover one shared resource and one constructed workload.

## Second PGG run on Hugging Face

On a second execution environment the three arms at 1,600 requests/s missed 2,042 (primary-only), 9,432 (fixed) and 3,915 (capped) out of 9,600 deadlines apiece. The capped policy launched 2,400 backups, versus 9,519 for the fixed timer. At 400/s, misses were 204, 1 and 1 out of 2,400 respectively. At 2,600/s, misses were 15,332, 15,508 and 11,994 out of 15,600. Both runs were operated by PGG.

The [budgeted_shared_slots dataset](https://huggingface.co/datasets/pgg-engineering/racecurve-hedge-traces) contains the 82,800 request rows from the second run. Group rows by rate_rps, seed and policy. A request misses when first_ns minus planned_ns exceeds the deadline in nanoseconds. A nonnegative backup_submit_ns records a launch. Computing the same statistics at a different deadline requires no timer simulation.
