# You're on call

Six exercises for an AI infrastructure on-call rotation. Calculate whether a hedge earns its extra load, decide when a streaming gateway may emit bytes, and distinguish modeled prefix reuse from billed cache reads. Each answer links to the experiment, source or fixture.

[Play the course in LiaScript](https://liascript.github.io/course/?https://pgg898200-ship-it.github.io/cachepin/on-call/course.md) · [Read, print or link a single case](https://pgg898200-ship-it.github.io/cachepin/on-call/)

## Evaluate a model

From a checkout of CachePin, use Inspect AI's multiple-choice solver and scorer. The mock provider exercises the full input and grading path locally.

```sh
uvx --from inspect-ai inspect eval examples/on-call/task.py --model mockllm/model
```

For a model evaluation, replace `mockllm/model` with a configured Inspect provider and model. The default mock emits a fixed placeholder. Its score has no interpretation as model performance. The six educational questions and their answers are public, so use them for a smoke test or teaching exercise. Each sample includes three choices and one capital-letter target. The solver presents the input and choices to the model. Explanations and sources live in metadata.

## Teach the cases in Moodle

Download the [GIFT question bank](https://pgg898200-ship-it.github.io/cachepin/on-call/on-call.gift.txt). In the course question bank choose **Import**, select **GIFT format** and upload the file. Each of the six questions has three choices, one correct answer and source-linked feedback. Results stay in the educator’s learning system.

## Reuse the cases

[Download the JSONL](../../site/on-call/cases.jsonl) or [fork the course Markdown](../../site/on-call/course.md). Modify the JSONL and run `python examples/on-call/build.py` to regenerate the printable field guide, the LiaScript course and the Moodle question bank together. Keep measurements attached to their workload and preserve the source links. The repository license applies.
