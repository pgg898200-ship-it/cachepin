"""Build the printable field drill and the LiaScript course from Inspect samples."""
import json
from html import escape
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "on-call"
BASE = "https://pgg898200-ship-it.github.io/cachepin/on-call/"
COURSE = "https://liascript.github.io/course/?" + BASE + "course.md"
READER = "https://github.com/pgg898200-ship-it/cachepin/tree/main/examples/on-call"


def read_cases():
    cases = [json.loads(line) for line in (OUT / "cases.jsonl").read_text().splitlines()]
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["id"].replace("-", "").isalnum()
        assert len(set(case["choices"])) == len(case["choices"])
        assert case["target"] in "ABC" and len(case["choices"]) == 3
    return cases


def render_card(case):
    meta = case["metadata"]
    choices = []
    for letter, choice in zip("ABC", case["choices"]):
        choices.append(f'<li><button type="button" data-choice="{letter}">'
                       f'<span>{letter}</span> {escape(choice)}</button></li>')
    answer = case["choices"]["ABC".index(case["target"])]
    return f"""<article id="{case['id']}" data-target="{case['target']}">
<header><small>{escape(meta['product'])}</small><h2>{escape(meta['title'])}</h2></header>
<p>{escape(case['input'])}</p>
<ol class="choices">{''.join(choices)}</ol>
<details><summary>Reveal answer</summary><div class="result">
<strong class="verdict" aria-live="polite"></strong><b>{escape(answer)}</b>
<p>{escape(meta['explanation'])}</p><a href="{escape(meta['source'])}">{escape(meta['source_label'])}</a>
</div></details>
<a class="permalink" href="#{case['id']}">Link to this case</a></article>"""


def lesson(cases):
    lines = ["""<!--
author: PGG Engineering
version: 1.0.0
language: en
comment: Six operating decisions about hedging, streaming and prompt caching.
-->

# You're on call

Make a call before opening the evidence. The six cases use measured request timings, literal stream bytes and an offline cache replay. Use the next arrow to begin.

[Print the casebook or run the Inspect AI version](""" + BASE + ")"]
    for case in cases:
        m = case["metadata"]
        lines += ["", "## " + m["title"], "", escape(case["input"], quote=False), ""]
        for letter, choice in zip("ABC", case["choices"]):
            mark = "X" if letter == case["target"] else " "
            lines.append(f"- [({mark})] " + escape(choice, quote=False))
        lines += ["**********", "", escape(m["explanation"], quote=False), "",
                  f"[{m['source_label']}]({m['source']})", "", "**********"]
    return "\n".join(lines) + "\n"


def gift_text(text):
    """Quote GIFT metacharacters while keeping stream bytes literal."""
    return "".join("\\" + c if c in "\\~=#{}:" else c for c in text)


def question_bank(cases):
    questions = []
    for case in cases:
        m = case["metadata"]
        lines = [f"::{case['id']}::[plain]{gift_text(case['input'])} {{"]
        for letter, choice in zip("ABC", case["choices"]):
            marker = "=" if letter == case["target"] else "~"
            lines.append(marker + gift_text(choice))
        lines += ["####" + gift_text(m["explanation"] + " " + m["source"]), "}"]
        questions.append("\n".join(lines))
    return "\n\n".join(questions) + "\n"


def main():
    cases = read_cases()
    template = Template(Path(__file__).with_name("page.html").read_text())
    nav = "".join(f'<a href="#{c["id"]}">{escape(c["metadata"]["title"])}</a>' for c in cases)
    html = template.substitute(course=COURSE, reader=READER, nav=nav,
                               cards="\n".join(map(render_card, cases)))
    (OUT / "index.html").write_text(html)
    (OUT / "course.md").write_text(lesson(cases))
    (OUT / "on-call.gift.txt").write_text(question_bank(cases))


if __name__ == "__main__":
    main()
