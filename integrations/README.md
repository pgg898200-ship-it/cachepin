# Import caller-side OpenTelemetry traces

Run these commands from the RaceCurve repository root.

```sh
python3 integrations/otlp.py integrations/race.otlp.jsonl | python3 bounds.py - --slo-us 10 --delay-us 5 15
python3 -m unittest discover -s integrations -p 'test_*.py'
```

In the included synthetic fixture, the backup launches at 5 µs and answers at 8 µs. The primary's cancellation cleanup finishes at 105 µs. The converter records the primary as taking **at least 8 µs** and the backup as taking **exactly 3 µs**. A 15 µs delay gives a response interval of 8–18 µs. This import allows ties at the measurement boundary. The compact winner-log format has a separate, explicit primary-wins-ties contract.

## Emit four application attributes

Record all three spans on the coordinator's clock. Make one parent span for each request race and make each launched attempt its direct child. Emit each attempt's actual dispatch timestamp as its span start. End the winning child at the instant the coordinator observes its usable result. A late shadow completion can record its actual response time if it was allowed to finish.

| Span | Attribute | Value |
| --- | --- | --- |
| Race parent | `racecurve.winner` | `primary` or `backup` |
| Race parent | `racecurve.backup_started` | Boolean recording actual dispatch |
| Attempt child | `racecurve.role` | `primary` or `backup` |
| Attempt child | `racecurve.outcome` | `complete` if usable completion was measured, or `pending` if still outstanding at the winning decision |

Example with an OpenTelemetry Python SDK tracer. The application supplies the timestamps and outcomes at dispatch, decision and cleanup. Save the parent context so independently scheduled tasks attach to the same race.

```python
from opentelemetry import trace

tracer = trace.get_tracer("myapp.hedging")
race = tracer.start_span("hedge.call", start_time=primary_dispatch_ns)
context = trace.set_span_in_context(race)
primary = tracer.start_span("hedge.primary", context=context, start_time=primary_dispatch_ns,
                            attributes={"racecurve.role": "primary"})
backup = tracer.start_span("hedge.backup", context=context, start_time=backup_dispatch_ns,
                           attributes={"racecurve.role": "backup"})
# In this example the backup wins. Record the result before async cleanup.
backup.set_attribute("racecurve.outcome", "complete")
backup.end(end_time=decision_ns)
primary.set_attribute("racecurve.outcome", "pending")
race.set_attribute("racecurve.winner", "backup")
race.set_attribute("racecurve.backup_started", True)
race.end(end_time=decision_ns)
primary.end(end_time=cleanup_ns)
```

Export a standard OTLP/HTTP JSON trace request or newline-delimited OTLP JSON trace batches. The converter joins on both trace ID and direct parent span ID across resources, scopes and batches in any order. Repeated identical tagged observations are deduplicated. A missing launched attempt, an orphan, conflicting duplicate, or inconsistent observation timing fails the entire import before output. Unrelated untagged spans are ignored.

The importer holds the tagged records in memory. Split large runs on whole trace boundaries. The covered population consists of exported completed two-way races with these tags. Dropping whole traces or sampling on winner or latency can bias reported traffic averages. Ordinary server spans, failed attempts, untagged retries and client cancellation messages alone do not establish these service-time observations. For comparisons on multiple machines, use caller-side timestamps throughout. Durations use the actual child dispatch times and full integer epoch nanoseconds. Whole-microsecond completed measurements stay exact. Fractional measurements export an inclusive `interval` from floor to ceiling, for example `2..3` µs. A `pending` loser gives a lower bound from its own launch to the winning decision, rounded down. These outward bounds can widen the integer-grid answer by less than one microsecond per measured endpoint. A scheduled delay can differ from the actual launch offset when the coordinator is busy. Compare the resulting observations under the candidate launch delay you intend to model.

The OpenTelemetry specification defines the underlying span fields and file format. The four `racecurve.*` attributes above are application-defined. [Trace protocol](https://github.com/open-telemetry/opentelemetry-proto/blob/main/opentelemetry/proto/trace/v1/trace.proto) · [OTLP JSON encoding](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding) · [File exporter](https://opentelemetry.io/docs/specs/otel/protocol/file-exporter/)
