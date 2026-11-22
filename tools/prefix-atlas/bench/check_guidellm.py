"""Exercise the indexed replay export with the installed GuideLLM 0.8.1 reader."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from guidellm.data.deserializers.trace_common import TraceDatasetDeserializer
from guidellm.schemas.data.deserializers.file import FileDataArgs
from guidellm.schemas.data.deserializers.trace_mooncake import MooncakeTraceFormatArgs
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parents[1]


def run(work):
    trace = work / "trace.jsonl"
    archive = work / "trace.ppk"
    replay = work / "replay.jsonl"
    rows = [
        dict(timestamp=1778977459236 + delay, input_length=32, output_length=16,
             hash_ids=[2**63, tip]) for delay, tip in [(0, 11), (1020, 23), (2020, 11)]
    ]
    trace.write_text("".join(json.dumps(r) + "\n" for r in rows))
    subprocess.run([ROOT / "prefixpack", "pack", trace, archive], check=True)
    selected = subprocess.check_output([ROOT / "prefixpack", "get", archive, "0", "1", "2"])
    subprocess.run([sys.executable, ROOT / "replay.py", "--block-size", "16", "-", replay],
                   input=selected, check=True)
    vocab = {v: i for i, v in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    processor = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    config = MooncakeTraceFormatArgs(source=FileDataArgs(kind="json_file", path=replay),
                                     hash_id_block_size=16)
    dataset = TraceDatasetDeserializer()(config, lambda: processor)
    turns = [json.loads(row["conversation_turns"])["turns"][0]["columns"] for row in dataset]
    assert [t["relative_timestamp_column"][0] for t in turns] == [0, 1.02, 2.02]
    assert all(t["prompt_tokens_count_column"] == [32] for t in turns)
    assert all(t["output_tokens_count_column"] == [16] for t in turns)
    prompts = [processor.encode(t["text_column"][0], add_special_tokens=False) for t in turns]
    assert all(len(p) == 32 for p in prompts)
    assert prompts[0][:16] == prompts[1][:16] and prompts[0][16:] != prompts[1][16:]
    assert prompts[0] == prompts[2]
    print("GuideLLM reader verified 3 timed requests, token counts and prefix identities")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        run(Path(directory))
