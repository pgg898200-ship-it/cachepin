#include "cutstream.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Markers = std::vector<std::string>;
using cutstream::Cursor;
using cutstream::Plan;

std::size_t checks = 0;

std::string hex(std::string_view bytes) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (unsigned char c : bytes) {
        result += digits[c >> 4];
        result += digits[c & 15];
    }
    return result;
}

void check(bool ok, const char* what, const Markers& markers = {},
           std::string_view observed = {}) {
    ++checks;
    if (ok) return;
    std::string detail = std::string(what) + " observed(hex)=" + hex(observed) + " markers(hex)=";
    for (const auto& marker : markers) detail += "[" + hex(marker) + "]";
    throw std::runtime_error(detail);
}

struct Stop {
    int id = -1;
    std::size_t begin;
    std::size_t end;
};

// Evaluate every marker independently on a complete string. The tuple is
// earliest end, longest winning marker, then the first registration.
Stop offline(std::string_view bytes, const Markers& markers) {
    Stop best{-1, bytes.size(), bytes.size()};
    for (std::size_t i = 0; i < markers.size(); ++i) {
        const auto pos = bytes.find(markers[i]);
        if (pos == std::string_view::npos) continue;
        const auto end = pos + markers[i].size();
        if (best.id == -1 || end < best.end || (end == best.end && pos < best.begin))
            best = {static_cast<int>(i), pos, end};
    }
    return best;
}

// EOF permits all observed bytes. Any future that suppresses an observed
// byte has a winning marker crossing this boundary. Truncating that future
// immediately after the marker yields one of these completing continuations.
std::size_t safe_length(const std::string& observed, const Markers& markers) {
    std::size_t safe = observed.size();
    for (const auto& marker : markers) {
        const auto max_overlap = std::min(observed.size(), marker.size() - 1);
        for (std::size_t k = 1; k <= max_overlap; ++k) {
            if (observed.compare(observed.size() - k, k, marker, 0, k) != 0) continue;
            const auto possible = offline(observed + marker.substr(k), markers);
            safe = std::min(safe, possible.begin);
        }
    }
    return safe;
}

// Independent exhaustive future oracle for the small binary suite.
void enumerate_futures(std::string& bytes, const Markers& markers,
                       std::size_t remaining, std::size_t& safe) {
    const auto result = offline(bytes, markers);
    if (result.id >= 0) {
        safe = std::min(safe, result.begin);
        return;
    }
    if (remaining == 0) return;
    for (char c : {'a', 'b'}) {
        bytes.push_back(c);
        enumerate_futures(bytes, markers, remaining - 1, safe);
        bytes.pop_back();
    }
}

std::string emit_views(const cutstream::Result& result) {
    return std::string(result.prefix) + std::string(result.input);
}

struct Trial {
    const Plan& plan;
    const Markers& markers;
    Cursor cursor{};
    std::string observed;
    std::string output;

    Trial(const Plan& plan_, const Markers& markers_) : plan(plan_), markers(markers_) {}

    void feed(const std::string& chunk) {
        if (cursor.done()) return;
        const auto old_size = observed.size();
        const std::string full = observed + chunk;
        const auto stop = offline(full, markers);
        const auto result = plan.feed(cursor, chunk);
        check(result.match == stop.id, "wrong marker ID", markers, full);
        check(result.consumed == stop.end - old_size, "wrong consumed count", markers, full);
        check(cursor.done() == (stop.id >= 0), "wrong terminal state", markers, full);
        observed = full.substr(0, stop.end);
        output += emit_views(result);
        const auto safe = stop.id >= 0 ? stop.begin : safe_length(observed, markers);
        check(output == observed.substr(0, safe), "incorrect/maximally delayed release", markers, observed);
        const auto expected_pending = stop.id >= 0 ? std::string() : observed.substr(safe);
        check(plan.pending(cursor) == expected_pending, "wrong pending bytes", markers, observed);
    }

    void finish() {
        const std::string pending(plan.pending(cursor));
        check(plan.finish(cursor) == pending, "EOF did not return precisely pending bytes", markers, observed);
        output += pending;
        const auto stop = offline(observed, markers);
        check(output == observed.substr(0, stop.begin), "incorrect complete stream output", markers, observed);
        check(cursor.done(), "EOF did not terminate", markers, observed);
        check(plan.pending(cursor).empty(), "pending bytes after EOF", markers, observed);
        check(plan.finish(cursor).empty(), "second EOF returned bytes", markers, observed);
    }
};

void fragmented(const Markers& markers, const std::vector<std::string>& chunks) {
    const Plan plan(markers);
    Trial trial(plan, markers);
    trial.feed("");
    for (const auto& chunk : chunks) trial.feed(chunk);
    trial.finish();
}

std::vector<std::string> binary_words(unsigned max_length) {
    std::vector<std::string> words{""};
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (words[i].size() == max_length) break;
        const std::string prefix = words[i];
        words.push_back(prefix + "a");
        words.push_back(prefix + "b");
    }
    return words;
}

void exhaustive() {
    const auto vocabulary = binary_words(2);
    const auto streams = binary_words(5);
    std::vector<Markers> sets{{}};
    for (std::size_t i = 1; i < vocabulary.size(); ++i) {
        sets.push_back({vocabulary[i]});
        for (std::size_t j = 1; j < vocabulary.size(); ++j)
            sets.push_back({vocabulary[i], vocabulary[j]});
    }
    // Longer words admit a marker strictly inside an unreachable longer one.
    sets.push_back({"abba", "bb", "bba"});
    sets.push_back({"abab", "baba", "bab", "aab", "aba"});
    for (const auto& markers : sets) {
        const Plan plan(markers);
        std::size_t max_length = 0;
        for (const auto& m : markers) max_length = std::max(max_length, m.size());
        for (const auto& stream : streams) {
            if (offline(stream, markers).id < 0) {
                auto extended = stream;
                auto safe = stream.size();
                enumerate_futures(extended, markers, max_length, safe);
                check(safe_length(stream, markers) == safe, "future oracle disagreement", markers, stream);
            }
            const unsigned partitions = 1u << (stream.empty() ? 0 : stream.size() - 1);
            for (unsigned split = 0; split < partitions; ++split) {
                Trial trial(plan, markers);
                trial.feed("");
                std::size_t start = 0;
                for (std::size_t j = 1; j < stream.size(); ++j) {
                    if ((split & (1u << (j - 1))) == 0) continue;
                    trial.feed(stream.substr(start, j - start));
                    trial.feed("");
                    start = j;
                }
                trial.feed(stream.substr(start));
                trial.finish();
            }
        }
    }
}

void edge_cases() {
    for (const auto& markers : std::vector<Markers>{{""}, {"ok", ""}}) {
        bool rejected = false;
        try { const Plan plan(markers); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "empty marker accepted", markers);
    }
    fragmented({}, {"", "hello", "", "world"});
    fragmented({"ba", "aba", "aba", "xxaba"}, {"xx", "ab", "a", "after"});
    fragmented({"abcde", "cd"}, {"a", "b", "c", "", "d,ignored"});
    fragmented({"abcdef", "cd"}, {"ab", "c", "q", "abc"});
    fragmented({"aaaaab", "aaaac"}, {"aaaaaaa", "aa", "", "ax", "aaa", "ac!"});
    fragmented({"abcde", "cdef"}, {"no abc", "d", ""});
    fragmented({"abx", "cd", "bq"}, {"ab", "cd"}); // Match begins in the deciding chunk.
    fragmented({std::string("\0\xff\0", 3), std::string("\xff\0", 2)},
               {std::string("!\0", 2), std::string("\xff", 1), std::string("\0!", 2)});
    const std::string utf8 = u8"é漢💡";
    std::vector<std::string> pieces{"x"};
    for (char byte : utf8) pieces.emplace_back(1, byte);
    pieces.push_back("after");
    fragmented({u8"漢💡", u8"é漢💡"}, pieces);
    const std::string long_marker = std::string(600, 'x') + "y";
    fragmented({long_marker}, {std::string(650, 'x'), "x", std::string(93, 'x'), "y!"});

    const Markers markers{"</tool>", "<", "stop", "stork", "tork"};
    const Plan plan(markers);
    Trial left(plan, markers), right(plan, markers);
    left.feed("left st");
    right.feed("right sto");
    left.feed("x");
    right.feed("rk extra");
    right.finish();
    left.feed("stop");
    left.finish();

    // A marker set using every byte exercises the full DFA alphabet.
    Markers all_bytes;
    for (int i = 0; i < 256; ++i) {
        std::string marker(1, static_cast<char>(i));
        marker += '!';
        all_bytes.push_back(marker);
    }
    const Plan all_plan(all_bytes);
    for (int i = 0; i < 256; ++i) {
        Trial trial(all_plan, all_bytes);
        trial.feed(std::string(1, static_cast<char>(i)));
        trial.feed("!extra");
        trial.finish();
    }

    const Plan reusable({"halt"});
    Cursor cursor;
    std::string chunk = "literalha";
    auto part = reusable.feed(cursor, chunk);
    check(part.input == "literal" && part.input.data() == chunk.data(),
          "literal output does not borrow input");
    std::fill(chunk.begin(), chunk.end(), 'X');
    check(reusable.pending(cursor) == "ha", "pending bytes depend on input lifetime");
    part = reusable.feed(cursor, "lt!");
    check(part.match == 0 && part.consumed == 2, "cross-chunk match");
    part = reusable.feed(cursor, "more");
    check(part.consumed == 0 && part.match < 0 && emit_views(part).empty(),
          "feed after stop was not inert");
    cursor.reset();
    check(!cursor.done() && reusable.pending(cursor).empty(), "reset retained prior state");
    part = reusable.feed(cursor, "ha");
    check(emit_views(part).empty() && reusable.finish(cursor) == "ha", "reset failed to restart");
    part = reusable.feed(cursor, "more");
    check(part.consumed == 0 && part.match < 0 && emit_views(part).empty(),
          "feed after EOF was not inert");
}

void randomized() {
    std::mt19937 rng(0xf10dd15u);
    const auto rand_bytes = [&](std::size_t length) {
        std::string bytes;
        for (std::size_t i = 0; i < length; ++i)
            bytes.push_back(static_cast<char>(rng() % 5));
        return bytes;
    };
    for (int run = 0; run < 1200; ++run) {
        Markers markers;
        const unsigned count = rng() % 10;
        for (unsigned i = 0; i < count; ++i) {
            if (i && rng() % 5 == 0) markers.push_back(markers[rng() % i]);
            else markers.push_back(rand_bytes(1 + rng() % 12));
        }
        auto stream = rand_bytes(rng() % 100);
        if (!markers.empty() && rng() % 2 == 0)
            stream.insert(rng() % (stream.size() + 1), markers[rng() % markers.size()]);
        const Plan plan(markers);
        Trial trial(plan, markers);
        for (std::size_t offset = 0; offset < stream.size() && !trial.cursor.done();) {
            std::string chunk = stream.substr(offset, 1 + rng() % 14);
            offset += chunk.size();
            trial.feed(chunk);
            // A cursor must survive transport buffer reuse on the next read.
            std::fill(chunk.begin(), chunk.end(), static_cast<char>(0xfe));
            if (rng() % 3 == 0) trial.feed("");
        }
        trial.finish();
    }
}

} // namespace

int main() {
    try {
        edge_cases();
        exhaustive();
        randomized();
        std::cout << checks << " differential checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
