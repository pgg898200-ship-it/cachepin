---
title: Cutstream
emoji: ✂️
colorFrom: yellow
colorTo: green
sdk: static
app_file: index.html
tags:
  - streaming
  - inference
  - stop-sequences
  - aho-corasick
  - cpp
short_description: Replay stop-sequence filtering one chunk at a time
---

# Cutstream

Compare stop-sequence handling at each output chunk. Edit the stop strings and payloads, or replay a split stop, same-byte tie, binary delimiter, or incomplete stop at EOF. The byte counters show how much output a gateway can forward to its caller at each step.

This Space runs an independent JavaScript reference implementation in your browser. Its counters measure bytes, with no wall-clock speed estimate. Cutstream's native C++ library compiles a dictionary shared across streams. The native project includes tests and a reproducible throughput benchmark.

One line represents one chunk or stop sequence. Literal text is converted to UTF-8. Use `\xNN` for arbitrary bytes, `\n` for newline, `\r` for carriage return, `\t` for tab, and `\\` for backslash. Earliest completion wins. At the same ending byte the longest match wins, then the earliest dictionary entry.

Open `index.html` in a browser to run the complete demo offline.

[C++ library, tests and benchmarks](https://github.com/pgg898200-ship-it/cachepin/tree/cutstream-v0.1/cutstream)
