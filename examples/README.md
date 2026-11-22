# Paired loopback trace

`loopback.csv` contains 1,000 paired HTTP response times measured on one Linux 6.18.57 x86_64 host with an AMD EPYC 9V74 processor and CPython 3.12.14. Run this command from the repository root to collect the same seeded workload.

```sh
python3 collect.py --pairs 1000 --seed 42 --output fresh.csv
```

Sample SHA-256 is `6a85c3dacfeb110224b2db1c22950e6788906ed5b79da463d670e0ad53a434d1`.

Each row preserves arrival order. The two columns are complete HTTP durations in microseconds from separate client threads, each timed on a monotonic clock. After one warm-up, the client dispatches the same request on persistent connections to two separate local Python server processes. A barrier releases both threads. Actual starts can differ due to scheduling. The collector waits for both responses before the next pair.

The servers deliberately sleep for 100–500 µs of per-replica jitter. Roughly one in 12 primary calls adds 6 ms and one in 20 backup calls adds 4 ms. Approximately one in 50 request IDs adds a further 4 ms on both servers. The seed determines the injected delays. Measured durations include scheduling and HTTP overhead and will vary from run to run.

## Held-out replay

Using the first 700 rows only, enumerate every integer delay from 0 to 6,000 µs. Discard delays that launch a backup for more than 25% of requests. Minimize deadline misses, break ties by fewer backup launches, then by the smallest delay. The selected delay was **1,177 µs**. The table shows the result of applying this fixed delay to the final 300 rows.

| Replay condition | Requests | Missed 6 ms deadline | Backup launches | p99 latency (µs) |
| --- | ---: | ---: | ---: | ---: |
| Calibration, primary only | 700 | 68 | 0 | 6,849 |
| Calibration, selected delay | 700 | 0 | 95 | 4,819 |
| Holdout, primary only | 300 | 27 | 0 | 6,953 |
| Holdout, selected delay | 300 | 0 | 42 | 5,338 |

The calculation launches a backup when the measured primary duration exceeds the delay, predicts completion at `min(primary_us, delay_us + backup_us)`, counts a miss when completion exceeds 6,000 µs, and uses nearest-rank p99. A delayed launch is assumed to have the service duration measured during simultaneous dispatch. Load-dependent queueing, cancellation effects and transport across hosts require a separate live validation. These samples contain one logical request per row without fanout grouping.

## Prospective live A/B

Before generating a second workload we fixed the 1,177 µs delay, 6 ms deadline, 1,000 request IDs, workload seed 127994 and arm-order seed 30513. For each request ID, the client ran a primary-only arm and a delayed-hedge arm in seeded random order. Each arm submitted a real HTTP request to the primary. In the hedged arm, if the primary was still outstanding when the timer woke, the client submitted the same request to the second local server. Timing starts before submission and stops when the caller observes the first successful HTTP response. Each branch runs to completion, and the client drains any remaining response before starting the next arm.

```sh
python3 examples/live_ab.py --output fresh-live-ab.json
```

| Live measurement | Primary only | Fixed hedge |
| --- | ---: | ---: |
| Requests missing the 6 ms deadline | 80 / 1,000 | 9 / 1,000 |
| p50 observed HTTP latency | 789 µs | 810 µs |
| p95 | 6,766 µs | 2,435 µs |
| p99 | 7,839 µs | 5,692 µs |
| Backup requests actually submitted | 0 | 126 |

The hedge turned 71 deadline misses into successes. Zero matched requests changed from a success to a miss. The median rose by 21 µs and 512 of 1,000 hedged trials took longer than their matched primary-only trials. Backup submission occurred 1,221–3,182 µs after trial start, with a median of 1,294 µs.

The test completed 2,000 sequential trial arms in 2.86 seconds including loser drain. The two services fully executed all 126 backup requests. The experiment measured latency with controlled stalls and one logical request in flight. Sustained service capacity, remote-network behavior and savings from cancellation remain unmeasured. [Raw results and paired observations](../results/prospective.json)
