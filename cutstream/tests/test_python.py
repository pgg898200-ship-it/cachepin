import asyncio
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
