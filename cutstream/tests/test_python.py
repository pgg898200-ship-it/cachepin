import asyncio
import os
import select
import subprocess
import sys
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

from cutstream import Plan


class PythonApi(unittest.TestCase):
    def test_binary_and_index(self):
        s = Plan([b"\x00!", b"a\x00!", b"a\x00!"]).stream()
        self.assertEqual(s.feed(b"old:a\x00"), b"old:")
        self.assertEqual(s.pending, b"a\x00")
        self.assertEqual(s.feed(b"!unread"), b"")
        self.assertEqual((s.match, s.consumed, s.done), (1, 1, True))
        self.assertEqual(s.feed(b"ignored"), b"")
        self.assertEqual((s.match, s.consumed, s.finish()), (1, 0, b""))

    def test_branches(self):
        plan = Plan([b"END"])
        a, b = plan.stream(), plan.stream()
        self.assertEqual(a.feed(b"E"), b"")
        self.assertEqual(b.feed(b"hello"), b"hello")
        self.assertEqual(a.feed(b"X"), b"EX")
        self.assertEqual(b.feed(b"EN"), b"")
        self.assertEqual((b.finish(), b.finish(), b.done), (b"EN", b"", True))

    def test_read_limit(self):
        def pieces():
            yield b"ab"
            yield b"c!discard"
            self.fail("upstream read past match")
        self.assertEqual(list(Plan([b"!"]).filter(pieces())), [b"ab", b"c"])

    def test_native_types(self):
        for bad in (b"end", ["end"], "end", [0]):
            with self.assertRaises(TypeError):
                Plan(bad)
        with self.assertRaises(ValueError):
            Plan([b""])
        s = Plan([]).stream()
        with self.assertRaises(TypeError):
            s.feed("text")
        self.assertEqual(s.feed(b"pass"), b"pass")

    def load_gateway(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "gateway.py"
        spec = importlib.util.spec_from_file_location("gateway", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_gateway(self):
        mod = self.load_gateway()

        async def events():
            for s in ["😀hi", "世", "界!", "unread"]:
                if s == "unread":
                    self.fail("gateway read past matching byte")
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=s))])

        async def run():
            return "".join([x async for x in mod.visible_text(events(), Plan(["界".encode()]))])
        self.assertEqual(asyncio.run(run()), "😀hi世")

    def test_selected_gateway_choice(self):
        mod = self.load_gateway()

        def choice(index, content, reason=None):
            return SimpleNamespace(index=index, delta=SimpleNamespace(content=content),
                                   finish_reason=reason)

        async def events(last, reason):
            yield SimpleNamespace(choices=[])
            yield SimpleNamespace(choices=[choice(1, None), choice(0, "wrong")])
            yield SimpleNamespace(choices=[choice(1, "😀hi</")])
            yield SimpleNamespace(choices=[choice(0, "tool>", "stop")])
            yield SimpleNamespace(choices=[choice(0, None), choice(1, last, reason)])
            self.fail("read past the selected choice's terminal event")

        async def collect(last, reason):
            stream = mod.visible_text(events(last, reason), Plan([b"</tool>"]), choice_index=1)
            return [text async for text in stream]

        self.assertEqual(asyncio.run(collect("tool>hidden", None)), ["😀hi"])
        self.assertEqual(asyncio.run(collect("to", "length")), ["😀hi", "</to"])


class CliInput(unittest.TestCase):
    def invoke(self, data, *args):
        return subprocess.run([sys.executable, "-m", "cutstream", "--stop", "it is", *args],
                              input=data, capture_output=True)

    def test_ollama_split_stop(self):
        # Deltas from ollama-python's test_client_generate_stream.
        data = (b'{"response":"Because "}\n{"response":"it "}\n'
                b'{"response":"is."}\ninvalid unread tail\n')
        result = self.invoke(data, "--input-format", "ollama-generate")
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"Because ", b""))

    def test_ollama_done_flushes_text_only(self):
        data = ('{"thinking":"it is"}\n{"model":"it is","response":"雪"}\n'
                '{"response":"it ","done":true}\ninvalid unread tail\n').encode()
        result = self.invoke(data, "--input-format", "ollama-generate")
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "雪it ".encode(), b""))

    def test_ollama_failure_preserves_only_committed_output(self):
        prefix = b'{"response":"visible it "}\n'
        for tail in (b"", b'{"error":"generation failed"}\n', b"bad json\n",
                     b'{"message":{"content":"chat"}}\n', b'{"response":null}\n'):
            with self.subTest(tail=tail):
                result = self.invoke(prefix + tail, "--input-format", "ollama-generate")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b"visible ")
                self.assertIn(b"cutstream:", result.stderr)

    def test_raw_default_preserves_binary(self):
        self.assertEqual(self.invoke(b"\xff\x00it is hidden").stdout, b"\xff\x00")

    @unittest.skipUnless(os.name == "posix", "select on a pipe requires POSIX")
    def test_ollama_emits_before_eof(self):
        command = [sys.executable, "-m", "cutstream", "--input-format", "ollama-generate",
                   "--stop", "it is"]
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as child:
            try:
                child.stdin.write('{"response":"雪 it "}\n'.encode())
                child.stdin.flush()
                self.assertTrue(select.select([child.stdout], [], [], 5)[0])
                self.assertEqual(os.read(child.stdout.fileno(), 64), "雪 ".encode())
                child.stdin.write(b'{"response":"is hidden"}\n')
                child.stdin.flush()
                self.assertEqual(child.wait(timeout=5), 0)
                self.assertEqual(child.stdout.read(), b"")
            finally:
                if child.poll() is None:
                    child.kill()
