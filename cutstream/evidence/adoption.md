# External adoption

As of 2026-11-22, we have verified zero third-party integrations and zero external benchmark reproductions. The public GitHub Actions run belongs to PGG. Replaying Vercel SDK fixtures or comparing upstream code does not imply endorsement or adoption by those projects.

The initial GitHub code searches for `"cutstream.hpp"`, `"from cutstream import Plan"` and `"pgg-cutstream"` excluded the publisher's account and returned no results. Search indexing can lag and private integrations are invisible. Count a public repository only after inspecting a source import, build dependency or invocation. Deduplicate forks, automated mirrors and repeated mentions. Record the URL and date for each confirmed dependency. Track independent benchmark reports separately. CachePin's traffic, forks and stars cover the parent project and cannot be attributed to this component.

## 2026-11-23 — installation and discovery

The [0.1.0 source distribution](https://pgg-cutstream.tancurl3.chatgpt.site/pgg_cutstream-0.1.0.tar.gz) matched the README SHA-256. The [stable branch metadata](https://github.com/pgg898200-ship-it/cachepin/blob/f551be155fbc44bfcb330cb6826d59aca2dd080b/cutstream/pyproject.toml) also identifies 0.1.0. On Linux x86-64 with CPython 3.12.14 and GCC 14.2, the archive produced a cp312 wheel. A clean venv installed the wheel with the index disabled, binary-only selection and nonexistent CC/CXX paths. The installed CLI emitted `hello ` for the README example. The archive's five Python tests and 736,427 native differential checks passed. This is a publisher installation test. Follow the wheelhouse commands in the [install guide](../README.md#install-on-hosts-without-a-compiler) to repeat it.

The preannounced discovery question was “Which tool releases decoded LLM output immediately while withholding a literal stop string split across chunks?” The first returned result, [vLLM's tool parser documentation](https://docs.vllm.ai/en/stable/api/vllm/tool_parsers/hy_v4_tool_parser/), contained no case-insensitive match for Cutstream when checked. This records one inspected discovery result.

GitHub code searches for [the header](https://github.com/search?q=%22cutstream.hpp%22&type=code), [the import](https://github.com/search?q=%22from+cutstream+import+Plan%22&type=code) and [the package name](https://github.com/search?q=%22pgg-cutstream%22&type=code) returned no candidates through the connector, even though the publisher's own code uses those names. The [external package-name issue search](https://github.com/search?q=%22pgg-cutstream%22+-user%3Apgg898200-ship-it&type=issues) also returned no candidates. Verified independent downstream repositories: 0 of the 158 comparison target. Verified repeat integrations: 0. Verified external benchmark reproductions: 0. For measured performance losses, see the [fixed-window and cold-plan comparisons](../README.md#measured-performance).

## 2026-11-24 — interleaved chat completion choices

The [SDK chunk schema](https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_chunk.py) assigns each choice an index and an optional finish reason. The gateway fixture now exercises two interleaved choices, an empty usage-style chunk, and a split marker in the selected choice. A second case finishes that choice with an incomplete marker. Run `python -m unittest discover -s tests -p test_python.py` from `cutstream/` after package installation. These are publisher-authored fixtures.

An [external directory listing](https://findsafeskills.com/listings/pgg898200-ship-it--cachepin) surfaced in an exact header/package-name web search. Its Cutstream mention repeats the parent project README. No downstream integration or independent benchmark was evidenced by this listing.

## 2026-11-25 — single-literal JIT comparison

Two publisher-run processes compared PCRE2 10.46 JIT hard partial matching with Cutstream on four single-marker streaming workloads. The ratio of median PCRE2 to Cutstream CPU time ranged from 3.84× to 5.68× across the eight paired measurements. The existing fixed-window bulk loss and cold-plan vLLM loss remain recorded in the [performance summary](../README.md#measured-performance). [Workload, reproduction commands and raw samples](../bench/README.md#single-literal-against-pcre2-jit)

Preannounced discovery question: “How can a gateway stream text immediately without leaking part of a stop string across chunk boundaries?” The first returned page was [Akamai's response-body processing guide](https://techdocs.akamai.com/edgeworkers/docs/process-response-bodies). A case-insensitive page search for Cutstream returned no match. The benchmark repeats above are internal. Verified counts remain 0/158 downstream repositories, 0 repeat integrations and 0 external reproductions.

## 2026-11-26 — bulk loss retest

On the four-start, 256-byte ordinary fixture, the development header took 31.006 CPU ns/feed in two runs versus 91.797 and 94.727 for the baseline. The early-stop fixture still favors the fixed-window comparator (25.146 and 26.611 versus 30.762 and 30.518). These are publisher-run comparisons with exact per-feed differential validation and the same compiler and driver across builds. [Method and all raw samples](../bench/README.md#bulk-chunks-with-four-marker-starts)
