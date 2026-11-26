# When a second replica competes for the same slots

This experiment tests Racecurve v0.1.0's published 1,177 µs hedge delay at three offered loads, using real threads and two shared execution slots. It measures what happens when a fast backup still has to wait for capacity and completed losers still consume it.

[Follow-up: enforcing the 25% backup budget under load](budget/README.md)

## Protocol

The primary and backup each have a FIFO request queue and two worker threads. All four workers acquire permits from the same two-slot FIFO semaphore. Each admitted branch sleeps for its service duration while holding its permit. This models a constrained shared executor or downstream connection pool. The study runs entirely on one host. There is no HTTP, GPU, remote provider or busy-loop compute in the measurements.

Before observing results, we fixed the delay (1,177 µs), deadline (6,000 µs), planned arrival rates (400, 1,600 and 2,600 requests/s), and a two-second arrival window per arm. Both arms receive identical regularly spaced requests with identical per-request service demands for each seed. Seed 127995 runs control first, then hedged. Seed 127996 reverses the order. Each arm starts with empty queues in a new process and drains before the next arm. Roughly one in 12 primaries stalls for 6 ms, one in 20 backups stalls for 4 ms, and one in 50 request IDs adds 4 ms to both arms. Every branch has 100–500 µs of additional jitter. The service-demand generator matches the published collector.

A dispatcher schedules requests independently of completions. When it wakes late it catches up to the original schedule. The hedge timer is relative to actual primary submission. It launches a backup only if the primary is still outstanding at dispatch. Once launched, both branches consume their full service time. The logical deadline starts at the planned arrival, including dispatch lag and all queueing. Each worker records result availability under a shared lock. The caller may consume the earliest completed result at that point.

## Reproduce

The runner requires Python 3.10+ and a C++17 compiler with threads.

```sh
python3 run.py --output fresh-run
```

Identical seeds reproduce the offered schedule and service demands. OS timing, CPU competition and timer wakeups affect the measurements. The runner recalculates aggregate metrics and checks monotonic per-branch timestamps, launch eligibility, input counts and first-winner times. No provider credentials are required.

Each run writes gzipped per-request CSV with planned, submit, dequeue, service start and service end timestamps relative to its run epoch, in nanoseconds. A negative timestamp marks an unlaunched backup. The first completion timestamp is the logical response. The JSON summary records hashes for each CSV. All percentiles use nearest rank.

Throughput in the two-second arrival window counts logical completions by the end of that window. Throughput including drain divides logical requests by time from the first scheduled arrival to the last finished branch, including losers. These measures differ under overload. Slot utilization divides occupied slot time by two times that same wall duration. Total queue wait includes both the per-replica queue and the shared-slot wait after dequeue.

## Measured results

AMD EPYC 9V74 host, Linux 6.18.57 x86_64, GCC 14.2.0 and CPython 3.12.14. The 12 two-second cases completed 36,800 logical requests across both arms. Each row below sums misses and launches across the two seeds. Ranges show the two individual runs.

| Offered requests/s | Arm | 6 ms misses | Backup launches | p50, ms | p99, ms | Primary queue p50, ms | Completions/s in window | Completions/s incl. drain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | control | 161/1,600 (10.1%) | 0/1,600 | 0.55–0.62 | 6.68–10.62 | 0.03 | 400 | 400 |
| 400 | hedge | 21/1,600 (1.3%) | 261/1,600 | 0.55–0.57 | 5.00–8.85 | 0.03–0.04 | 400 | 400 |
| 1,600 | control | 1,722/6,400 (26.9%) | 0/6,400 | 1.77–2.36 | 20.60–22.81 | 0.45–1.02 | 1,596 | 1,594 |
| 1,600 | hedge | 6,324/6,400 (98.8%) | 6,350/6,400 | 347.38–513.67 | 771.42–1,071.85 | 674.53–751.75 | 1,002–1,100 | 979–1,085 |
| 2,600 | control | 10,212/10,400 (98.2%) | 0/10,400 | 283.09–325.78 | 588.54–661.87 | 282.50–325.04 | 1,944–2,023 | 1,950–2,004 |
| 2,600 | hedge | 10,338/10,400 (99.4%) | 10,371/10,400 | 1,212.88–1,358.39 | 2,477.04–2,524.48 | 1,537.93–1,582.84 | 1,091–1,197 | 1,066–1,094 |

At 400/s, the hedge reduced deadline misses in both runs. Low-load p99 improved in the first run and worsened in the second (6.68 to 8.85 ms). At 1,600/s, the primary-only pool nearly kept pace. The fixed hedge launched backups for 99% of requests and primary median queue wait rose from 0.45–1.02 ms to 675–752 ms. At 2,600/s, both arms fell behind. Full loser execution and shared capacity let copies of queued work compete with later arrivals.

Dispatch timing varied. Across runs, p99 launch lag for primary arrivals ranged from 0.55 to 11.01 ms, and the worst single lag was 22.25 ms. At 2,600/s one hedged run submitted its last request after the nominal two-second arrival window. These short constant-rate bursts include startup, drain and host scheduling disturbances. They establish one observed load-dependent reversal and give no deployment-specific safe rate or universal hedging threshold. [Per-run measurements and raw CSV filenames](measured/results.json)
