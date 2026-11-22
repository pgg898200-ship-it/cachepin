// Build from the repository root: c++ -std=c++17 -O3 -DNDEBUG -Iinclude bench/bench.cpp -o /tmp/cutbench
#include "cutstream.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using cutstream::Result;
using View = std::string_view;
using Clock = std::chrono::steady_clock;

struct PatternSet {
    std::vector<std::string> words;
    std::array<std::vector<int>, 256> starts;
    std::size_t max_len = 0;
    int single_start = -1;

    explicit PatternSet(std::vector<std::string> patterns) : words(std::move(patterns)) {
        int n_starts = 0;
        for (std::size_t i = 0; i < words.size(); ++i) {
            auto first = static_cast<unsigned char>(words[i][0]);
            if (starts[first].empty()) {
                ++n_starts;
                single_start = first;
            }
            starts[first].push_back(static_cast<int>(i));
            max_len = std::max(max_len, words[i].size());
        }
        if (n_starts != 1) single_start = -1;
    }
};

struct BufferCursor {
    std::string held;
    bool done = false;
};

struct Cut {
    using Cursor = cutstream::Cursor;
    cutstream::Plan plan;
    explicit Cut(const PatternSet& p) : plan(p.words) {}
    Result feed(Cursor& c, View v) { return plan.feed(c, v); }
    View finish(Cursor& c) { return plan.finish(c); }
};

// Native baselines share immutable pattern data. The bounded join workspace is
// reused by one calling thread. Each cursor owns only its unresolved bytes.
template<bool FixedWindow> struct Rescan {
    using Cursor = BufferCursor;
    const PatternSet& p;
    std::string joined;
    explicit Rescan(const PatternSet& p_) : p(p_) {}

    Result feed(Cursor& c, View chunk) {
        if (c.done) return {};
        const std::size_t old = c.held.size();
        View text = chunk;
        if (old) {
            joined.assign(c.held);
            joined.append(chunk);
            text = joined;
        }
        std::size_t stop_end = View::npos;
        int match = -1;
        std::size_t safe = text.size();
        auto record = [&](int id, std::size_t end) {
            if (end < stop_end ||
                (end == stop_end && (p.words[id].size() > p.words[match].size() ||
                 (p.words[id].size() == p.words[match].size() && id < match)))) {
                stop_end = end;
                match = id;
            }
        };

        if constexpr (FixedWindow) {
            // Only a match ending inside the new chunk can be the first stop.
            for (std::size_t i = 0; i < p.words.size(); ++i) {
                const auto& word = p.words[i];
                const auto start = old >= word.size() ? old - word.size() + 1 : 0;
                auto pos = text.find(word, start);
                if (pos != View::npos) record(static_cast<int>(i), pos + word.size());
            }
            safe = text.size() - std::min(text.size(), p.max_len - 1);
        } else {
            // Candidate-first exact-prefix scanner. For one starting byte use
            // libc memchr via string_view::find(char); otherwise index by byte.
            for (std::size_t pos = 0; pos < text.size() && pos < stop_end; ++pos) {
                if (p.single_start >= 0) {
                    pos = text.find(static_cast<char>(p.single_start), pos);
                    if (pos == View::npos || pos >= stop_end) break;
                }
                const auto first = static_cast<unsigned char>(text[pos]);
                const auto remaining = text.size() - pos;
                for (int id : p.starts[first]) {
                    const auto& word = p.words[id];
                    const auto count = std::min(remaining, word.size());
                    if (text.compare(pos, count, word.data(), count) != 0) continue;
                    if (remaining >= word.size()) record(id, pos + word.size());
                    else safe = std::min(safe, pos);
                }
            }
        }
        if (match >= 0) {
            c.done = true;
            return {{}, text.substr(0, stop_end - p.words[match].size()), stop_end - old, match};
        }
        c.held.assign(text.substr(safe));
        return {{}, text.substr(0, safe), chunk.size(), -1};
    }
    View finish(Cursor& c) {
        if (c.done) return {};
        c.done = true;
        return c.held;
    }
};

struct Case {
    std::string name;
    std::size_t streams, bytes, chunk;
    PatternSet patterns;
    std::vector<std::string> payloads;
};

static std::uint32_t next(std::uint32_t& x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

Case make_case(std::string name, std::size_t n_stops, std::size_t n_streams,
               std::size_t chunk, bool prefixes, bool mixed_starts = false) {
    std::vector<std::string> words{"<|im_end|>"};
    for (std::size_t i = 1; i < n_stops; ++i) {
        char start = mixed_starts ? "<[/#"[i % 4] : '<';
        words.push_back(start + std::string("|agent_event_") + std::to_string(i) + "|>");
    }
    Case c{std::move(name), n_streams, 256, chunk, PatternSet(std::move(words)), {}};
    c.payloads.reserve(c.streams);
    std::uint32_t rng = 314159265;
    constexpr View alphabet = "The node recorded its current score and returned a short reply. ";
    constexpr View partial = "k<|agent_event_!|> <|im_enX ";
    for (std::size_t i = 0; i < c.streams; ++i) {
        std::string data(c.bytes, ' ');
        for (std::size_t j = 0; j < data.size(); ++j)
            data[j] = prefixes ? partial[(j+i) % partial.size()] : alphabet[next(rng) % alphabet.size()];
        if (i % 4 == 0) {
            auto& stop = c.patterns.words[next(rng) % c.patterns.words.size()];
            data.replace(data.size() - stop.size(), stop.size(), stop);
        }
        c.payloads.push_back(std::move(data));
    }
    return c;
}

struct Stats {
    std::uint64_t calls = 0, input_bytes = 0, output_bytes = 0, match_sum = 0;
    std::uint64_t total_held = 0, active_calls = 0, max_held = 0, checksum = 14695981039346656037ull;
    double avg_held() const { return active_calls ? double(total_held)/active_calls : 0; }
};

void hash(View v, std::uint64_t& h) {
    for (unsigned char ch : v) h = (h ^ ch) * 1099511628211ull;
}

template<class Engine> Stats verify(const Case& spec, bool exact_release) {
    Engine engine(spec.patterns);
    Cut oracle(spec.patterns);
    Stats stats;
    for (const auto& bytes : spec.payloads) {
        typename Engine::Cursor cur;
        Cut::Cursor oracle_cur;
        std::string emitted, reference;
        for (std::size_t off = 0; off < bytes.size(); off += spec.chunk) {
            View v = View(bytes).substr(off, spec.chunk);
            auto out = engine.feed(cur, v);
            auto truth = oracle.feed(oracle_cur, v);
            emitted.append(out.prefix).append(out.input);
            reference.append(truth.prefix).append(truth.input);
            if (out.match != truth.match || out.consumed != truth.consumed ||
                (exact_release && emitted != reference) ||
                reference.compare(0, emitted.size(), emitted) != 0)
                throw std::runtime_error("differential mismatch in " + spec.name);
            stats.calls++;
            stats.input_bytes += out.consumed;
            if (out.match >= 0) {
                stats.match_sum += std::uint64_t(out.match + 1);
                break;
            }
            auto held = off + out.consumed - emitted.size();
            stats.total_held += held;
            stats.max_held = std::max<std::uint64_t>(stats.max_held, held);
            stats.active_calls++;
        }
        emitted.append(engine.finish(cur));
        reference.append(oracle.finish(oracle_cur));
        if (emitted != reference) throw std::runtime_error("EOF mismatch in " + spec.name);
        stats.output_bytes += emitted.size();
        hash(emitted, stats.checksum);
    }
    return stats;
}

template<class Engine> double time_run(const Case& spec, const Stats& expected) {
    Engine engine(spec.patterns);
    std::vector<typename Engine::Cursor> cursors(spec.streams);
    std::uint64_t matches = 0, emitted = 0;
    auto begin = Clock::now();
    for (std::size_t off = 0; off < spec.bytes; off += spec.chunk) {
        for (std::size_t i = 0; i < spec.streams; ++i) {
            auto out = engine.feed(cursors[i], View(spec.payloads[i]).substr(off, spec.chunk));
            emitted += out.prefix.size() + out.input.size();
            matches += std::uint64_t(out.match + 1);
        }
    }
    for (auto& cur : cursors) emitted += engine.finish(cur).size();
    auto end = Clock::now();
    if (emitted != expected.output_bytes || matches != expected.match_sum)
        throw std::runtime_error("timing mismatch in " + spec.name);
    return std::chrono::duration<double,std::nano>(end-begin).count() / expected.calls;
}

int main(int argc, char** argv) {
    if (argc > 2) return 2;
    const int samples = argc == 2 ? std::stoi(argv[1]) : 7;
    if (samples < 1 || samples > 100) throw std::invalid_argument("samples must be 1-100");
    const std::vector<Case> cases{
        make_case("single-tiny",1,2048,1,false),
        make_case("single-text",1,4096,4,false),
        make_case("four-text",4,4096,4,false,true),
        make_case("many-text",32,4096,4,false,true),
        make_case("many-shared-start",32,4096,4,false),
        make_case("single-large",1,4096,64,false),
        make_case("four-large",4,4096,64,false,true),
        make_case("many-prefix",32,4096,4,true),
        make_case("single-prefix",1,4096,4,true)
    };
    std::cout << "case,engine,stops,streams,chunk_bytes,feed_calls,input_bytes,median_ns_per_feed,"
              << "mean_held_bytes,max_held_bytes,output_bytes,output_fnv64,samples,sample_ns_per_feed\n";
    for (const auto& spec : cases) {
        std::array<Stats,3> stats{verify<Cut>(spec,true),verify<Rescan<false>>(spec,true),
                                  verify<Rescan<true>>(spec,false)};
        std::array<std::vector<double>,3> times;
        for (int round = -1; round < samples; ++round) {
            for (int shift = 0; shift < 3; ++shift) {
                int id = (round + 1 + shift) % 3;
                double ns = id == 0 ? time_run<Cut>(spec,stats[id]) :
                            id == 1 ? time_run<Rescan<false>>(spec,stats[id]) :
                                      time_run<Rescan<true>>(spec,stats[id]);
                if (round >= 0) times[id].push_back(ns);
            }
        }
        constexpr std::array<const char*,3> names{"cutstream","grouped-rescan","fixed-find"};
        for (std::size_t id = 0; id < names.size(); ++id) {
            auto t = times[id];
            std::sort(t.begin(), t.end());
            double median = (t[(t.size()-1)/2] + t[t.size()/2]) * 0.5;
            auto& s = stats[id];
            std::cout << spec.name << ',' << names[id] << ',' << spec.patterns.words.size() << ','
                      << spec.streams << ',' << spec.chunk << ',' << s.calls << ',' << s.input_bytes
                      << ',' << std::fixed << std::setprecision(3) << median << ',' << s.avg_held()
                      << ',' << s.max_held << ',' << s.output_bytes << ',' << std::hex
                      << s.checksum << std::dec << ',' << samples << ',';
            for (std::size_t i = 0; i < times[id].size(); ++i) {
                if (i) std::cout << ' ';
                std::cout << times[id][i];
            }
            std::cout << '\n';
        }
    }
}
