"""Black-box contract tests. Build ./prefixpack first, then run python3 -m unittest."""

import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path


BINARY = Path(__file__).resolve().with_name("prefixpack")


def encoded_row(ids, row_number):
    record = {"timestamp": row_number, "hash_ids": ids, "input_length": len(ids)}
    return json.dumps(record, separators=(",", ":")).encode() + b"\n"


class PrefixPackCLI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name)
        self.source = directory / "input.jsonl"
        self.archive = directory / "trace.ppk"
        self.output = directory / "restored.jsonl"

    def invoke(self, *args, success=True):
        result = subprocess.run(
            [BINARY, *map(str, args)], capture_output=True, timeout=20
        )
        error = result.stderr.decode(errors="replace")
        if success:
            self.assertEqual(result.returncode, 0, error)
        else:
            self.assertNotEqual(result.returncode, 0, error)
        return result.stdout

    def pack(self, rows):
        self.source.write_bytes(b"".join(rows))
        self.invoke("pack", self.source, self.archive)
        self.source.unlink()

    def check_recovery(self, rows, query):
        self.pack(rows)
        self.invoke("verify", self.archive)
        self.invoke("unpack", self.archive, self.output)
        self.assertEqual(self.output.read_bytes(), b"".join(rows))
        if query:
            actual = self.invoke("get", self.archive, *query)
            self.assertEqual(actual, b"".join(rows[index] for index in query))

    def test_mixed_json_preserves_original_spelling_and_terminators(self):
        rows = [
            b'  {"hash_ids" : [0, 12, 18446744073709551615], "id":"a"}\t\r\n',
            b'{"before":{"hash_ids":[5,6]},"hash_ids":[0,12,8],"after":true}\n',
            b'{"hash_ids":[-4,0.00,2e7,"3",null,[]]}\r\n',
            b'{"hash_ids":[18446744073709551616],"x":1}\n',
            b'{"hash_ids":[],"hash_ids":[9]}\n',
            b'{"hash_ids":null,"note":"\\\"hash_ids\\\":[123]"}\n',
            '["\u2603","\U0001f30c"]\r\n'.encode(),
            b'null\n',
            b'  -0.00e+3 \t\n',
            b'{"hash_ids":[0,12,13],"escaped":"\\u2603"}',
        ]
        self.check_recovery(rows, [8, 9, 1, 0, 9, 2, 3, 4, 5, 6, 7, 0])

    def test_interleaved_prefixes_and_random_gather(self):
        randomizer = random.Random(947137)
        roots = [[randomizer.getrandbits(63) for _ in range(56)] for _ in range(8)]
        rows = []
        for i in range(320):
            stem = roots[i % len(roots)]
            ids = stem[: 28 + (i // 8) % 29] + [i % 3, randomizer.getrandbits(60)]
            rows.append(encoded_row(ids, i))
        rows.extend([rows[5], rows[12], rows[5]])
        query = [randomizer.randrange(len(rows)) for _ in range(80)]
        self.check_recovery(rows, query)

    def test_deep_hash_array_with_short_and_empty_siblings(self):
        prefix = list(range(32_768))
        branches = [prefix, [], prefix + [888], [42], prefix[:-1], prefix + [777]]
        rows = [encoded_row(ids, i) for i, ids in enumerate(branches)]
        self.check_recovery(rows, [5, 4, 3, 2, 1, 0, 5])

    def test_single_unterminated_raw_row(self):
        self.check_recovery([b'"abc"'], [0, 0])

    def test_same_integer_path_multiple_json_spellings(self):
        rows = [
            b'{"hash_ids":[1,2,3]}\n',
            b'{"hash_ids": [ 1 ,2,\t 3 ] }\r\n',
            b'{ "\\u0068ash_ids":[1, 2, 3], "timestamp":0 }\n',
            b'{"hash_ids"\t:\t[1,2,3],"nested":{"hash_ids":[9,8,7]}}',
        ]
        self.check_recovery(rows, [3, 1, 2, 0])

    def test_empty_input(self):
        self.check_recovery([], [])
        self.invoke("get", self.archive, 0, success=False)

    def test_invalid_row_indices(self):
        self.pack([b'null\n', b'0\n'])
        for index in ["-1", "2", "18446744073709551616", "0x1", "1junk"]:
            with self.subTest(index=index):
                self.invoke("get", self.archive, index, success=False)

    def test_verify_rejects_truncation_and_damaged_header(self):
        self.pack([encoded_row(list(range(n)), n) for n in range(50)])
        archive = self.archive.read_bytes()
        damaged = bytearray(archive)
        damaged[0] ^= 0xFF
        body_damage = bytearray(archive)
        body_damage[len(archive) // 2] ^= 0x01
        for label, payload in [
            ("empty", b""),
            ("one_byte", archive[:1]),
            ("half", archive[: len(archive) // 2]),
            ("last_byte_missing", archive[:-1]),
            ("bad_header", damaged),
            ("payload_bit_flip", body_damage),
        ]:
            with self.subTest(malformed=label):
                self.archive.write_bytes(payload)
                self.invoke("verify", self.archive, success=False)
                self.output.write_bytes(b"existing file")
                self.invoke("unpack", self.archive, self.output, success=False)
                self.assertEqual(self.output.read_bytes(), b"existing file")


if __name__ == "__main__":
    unittest.main()
