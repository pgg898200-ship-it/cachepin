# GPUHedge's MOSS-TTS trace

Racecurve replayed all 54 RunPod/Modal pairs from GPUHedge's July 2026 experiment. Its authors submitted the same fixed text-to-speech prompt concurrently to three serverless clouds, validated each generated WAV, and let each arm finish or reach a 300-second observation cap. RunPod used an RTX 4090 with CPU audio tokenization. Modal used an L40S with GPU audio tokenization. Every selected arm completed successfully. We used the published cold-round client wall times, which include provider queues and are rounded to 0.01 seconds. The separate warm companion calls and the Cerebrium arm were excluded. This fixed choice avoids missing or warm-contaminated Cerebrium measurements.

The upstream repository carries an Apache 2.0 license. The importer fetches the upstream data on request, verifies its SHA-256 and converts seconds to integer microseconds. This release includes the derived aggregate results. [Source, methods and license at the pinned revision](https://github.com/mireklzicar/gpuhedge/tree/669c8c9327c95164cfd317a4cc4c7c123f4afcc4)

## A 25% backup budget changed which policy was feasible

We used GPUHedge's 60-second deadline and chronological split, taking rounds 1–18 for calibration and 19–54 for evaluation. The original authors examined all 54 rounds during their study. The later partition provides a retrospective check of the fixed selection rule. For each integer 0.1-second delay through 300 seconds, Racecurve counted deadline misses and backup launches. On calibration rows, we chose the fewest misses subject to the 25% launch cap, then broke ties by fewer launches and smaller delay.

| Policy on the 36 later rounds | Missed 60 s deadline | Backup launches | p95, nearest rank |
| --- | ---: | ---: | ---: |
| RunPod primary only | 11 / 36 | 0 / 36 | 120.07 s |
| Primary p75, 116.1 s | 11 / 36 | 4 / 36 | 120.07 s |
| Primary p95, 121.7 s | 11 / 36 | 1 / 36 | 120.07 s |
| Selected 121.7 s delay | 11 / 36 | 1 / 36 | 120.07 s |
| GPUHedge's fixed 10 s delay, same provider pair | 1 / 36 | 11 / 36 | 52.18 s |

The percentile rules use only primary durations from the first 18 rounds. Take the nearest-rank percentile and round up to the next 0.1-second grid point when needed. With ties suppressing launches, p75 caps calibration launches at 25%. Both percentile delays exceeded the 60-second deadline, so even an instantaneous backup at those settings could never rescue a deadline miss. The fitted delay matched p95 exactly. Primary-only achieved the same held-out misses and p95 with zero duplicates.

The calibrated delay achieved zero rescues. At 10 seconds, the replay rescued ten misses, but its 11 launches on 36 later rounds (30.6%) exceeded the launch budget. On the 18 calibration rounds, the same comparator needed seven launches. Only four were feasible under the cap. Loosening the budget leads to a different engineering decision. [Full numerical results](results.json)

This exercise compares fixed-delay replays of measured simultaneous starts. It assumes each provider's recorded response time would carry over to a delayed start. GPUHedge separately supports state-aware queue cutover, billed-cost estimates and live cancellation. Racecurve's overlap time cannot establish GPU execution time, saved spend or the behavior of a new cloud deployment. The experiment covers one prompt and one evening. [GPUHedge methodology](https://github.com/mireklzicar/gpuhedge/blob/669c8c9327c95164cfd317a4cc4c7c123f4afcc4/benchmarks/2026-07-moss/methodology.md)

## Reproduce

From the Racecurve root, run the native analysis with a standard-library Python driver.

```sh
make racecurve
python3 external/test_moss.py
python3 external/moss.py > fresh-moss.json
```

Use `--source /path/to/moss_rounds.jsonl` to run offline. Optionally use `--csv /tmp/moss.csv` to write the 54 converted pairs for other tools. The driver checks the native output against direct enumeration at all 3,001 delays on both partitions.

We also ran the pinned GPUHedge implementation. Its 60-second miss and backup launch counts agreed at all **6,002 partition/delay combinations**. To repeat this check, install PyYAML in your environment and run the following commands.

```sh
git clone https://github.com/mireklzicar/gpuhedge.git /tmp/gpuhedge-moss
git -C /tmp/gpuhedge-moss checkout --detach 669c8c9327c95164cfd317a4cc4c7c123f4afcc4
python3 external/check_gpuhedge.py /tmp/gpuhedge-moss
```
