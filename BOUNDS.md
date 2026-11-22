# What can a canceled race tell you?

Suppose a backup won in 8 µs after a 5 µs delay. It took 3 µs of service. The primary would have taken at least 9 µs under an integer clock and a primary-wins-ties policy. With a new delay of 15 µs, the same request could finish anywhere from 9 to 18 µs. At the original delay its response is exactly 8 µs. A canceled branch carries information even when its completion time is unavailable.

```sh
printf 'winner,elapsed_us,old_delay_us\nbackup,8,5\n' | python3 bounds.py - --slo-us 10 --delay-us 5 15
```

The output for delay 15 contains `"sum_response_us":[9,18]` and `"slo_misses":[0,1]`. Divide each endpoint by `requests` to obtain the exact empirical mean response in µs or miss fraction. An upper sum of `null` means no finite upper bound. Miss bounds always have finite endpoints. The index retains integers through calculation and JSON serialization.

## Input contracts

Use `winner,elapsed_us,old_delay_us` when each row records a completed race, the row-specific configured hedge delay and which replica answered. Use `primary` or `backup` as the winner. Assume the primary starts at zero, the backup starts exactly at the logged delay only if the primary has not completed, and the primary wins when both would finish at the same time. All times are nonnegative whole microseconds. For a primary win at time e after a delay d, the unobserved backup could take any duration from max(0,e−d) onward. A backup win implies e≥d, backup service e−d, and primary service at least e+1. Race logs with asynchronous dispatch, rounded timestamps that obscure ties, or errors that can terminate a branch early need explicit interval observations.

Use `a_status,a_us,b_status,b_us` for individual observations of primary and backup service durations. Use a shared time unit and deterministic comparison convention for all rows.

| Status | Value | Permitted service times |
| --- | --- | --- |
| exact | t | t |
| interval | lo..hi | every integer from lo through hi, inclusive |
| censored | t | t+1, t+2, … (known unfinished through t) |
| at_least | t | t, t+1, … (completion at the censoring instant is possible) |
| unstarted | empty field | 0, 1, … |

## What the bounds cover

For each row, assume its two potential service durations stay fixed when you change the delay. Model immediate launches at their prescribed time, instantaneous observation, no load feedback and no cross-request constraints. Separate histories can carry separate bounds. These are identification bounds on the recorded sample. The CSV does not estimate sampling uncertainty, queue feedback, dispatch overhead, or workload shifts. The original log must include all in-scope requests for its mean to describe that traffic.

For a delay d, response is min(P,d+B). It increases monotonically with both durations, as does the indicator that response exceeds the SLO. Across rows, each lower endpoint can be attained at once, as can any finite upper endpoint under these assumptions. Consequently both sums and miss counts are sharp. When both upper durations on one row are unbounded, the mean has an unbounded upper endpoint. For finite queries, arbitrarily large finite values suffice to realize all upper miss counts. Cross-delay differences and ratios need a paired analysis; subtracting separately bounded averages can overstate their uncertainty.

The index preprocesses N rows in O(N log N) time and O(N) space for a fixed SLO. Each delay takes O(log N) time. For finite bounds, min(P,d+B) rises with unit slope until max(0,P−B). A prefix sum of sorted breakpoints accumulates the entire response curve. The SLO comparison uses a separate sorted list of integer delays at which each row first exceeds the deadline. The CLI requires only Python 3.10+ and its standard library.
