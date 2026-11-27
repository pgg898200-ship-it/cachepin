# When a backup request backfires

Racecurve · PGG Engineering · 2026-11-22

In a two-slot test, a backup after 1.177 ms cut 6 ms deadline misses from 10.1% to 1.3% at 400 requests/s. At 1,600 requests/s, misses rose from 26.9% to 98.8%.

Two replicas shared two worker slots on one host. Both copies ran to completion. The experiment measured 36,800 requests over 12 runs at three offered loads, with two workload seeds and two policies. The clock started at the planned arrival and stopped when the first copy finished.

| Offered load | Requests per arm | Primary misses | Hedged misses | Backups launched |
| --- | --- | --- | --- | --- |
| 400/s | 1,600 | 161 (10.1%) | 21 (1.3%) | 261 |
| 1,600/s | 6,400 | 1,722 (26.9%) | 6,324 (98.8%) | 6,350 |
| 2,600/s | 10,400 | 10,212 (98.2%) | 10,338 (99.4%) | 10,371 |

At 1,600 requests/s, the timer launched 6,350 backups for 6,400 requests. Those copies competed for the same slots as the primaries. Services with independent capacity or effective cancellation can behave differently.

[Run the analysis in Colab](https://colab.research.google.com/github/pgg898200-ship-it/cachepin/blob/f93cc10e6770a7a839f5bb05d9de2c6a3a08fa99/experiments/shared-slots/hedging-under-load.ipynb) · [Trace files](https://huggingface.co/datasets/pgg-engineering/racecurve-hedge-traces/tree/efd1d491158d8b5ac4ff787f05dd05196700d60c) · [Experiment protocol](https://github.com/pgg898200-ship-it/cachepin/tree/8f963f09edf68942f9ac6e304e7635d4cf41a27e/experiments/shared-slots) · [Racecurve explorer](https://huggingface.co/spaces/pgg-engineering/racecurve) · [Chart](https://pgg898200-ship-it.github.io/cachepin/racecurve/deadline-misses.png)

## What changes with a 25% backup budget?

On November 26, PGG tested a dispatcher that allows at most one backup per four requests cumulatively. It checks the budget when each timer fires. Both copies still run to completion on the same two slots.

At 1,600 requests/s, across three seeds on the Hugging Face host.

| Policy | Requests | 6 ms misses | Backups |
| --- | ---: | ---: | ---: |
| Primary only | 9,600 | 2,042 | 0 |
| Fixed timer with cap | 9,600 | 3,915 | 2,400 |
| Fixed timer | 9,600 | 9,432 | 9,519 |

The cap prevented 5,517 misses compared with the fixed timer. Disabling backups prevented another 1,873 at this load. The result depends on offered load. At 400 requests/s, both hedge policies missed one of 2,400 deadlines and primary-only missed 204.

[Download the 82,800 request rows (CSV.gz)](https://huggingface.co/datasets/pgg-engineering/racecurve-hedge-traces/resolve/6f3fd69316fe59fba97b6d866d38736f0b87d0bb/data/budgeted-shared-slots.csv.gz). Open DuckDB in the download directory and run the query below. Change the rate or the deadline in nanoseconds to compare policies for another target. [Three-policy experiment protocol](https://github.com/pgg898200-ship-it/cachepin/tree/5b598e9d687fa2669db11d9c8b5d9b2df7758df1/experiments/shared-slots/budget)

```sql
SELECT policy, count(*) AS requests,
       count_if(first_ns - planned_ns > 6000000) AS misses,
       count_if(backup_submit_ns >= 0) AS backups
FROM read_csv('budgeted-shared-slots.csv.gz')
WHERE rate_rps = 1600
GROUP BY policy
ORDER BY misses;
```
