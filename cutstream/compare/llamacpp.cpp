// Driven by llamacpp.py. The .inc files are extracted from pinned upstream
// source. Each implementation materializes the output as an owned string.
#include "cutstream.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "common.inc"

struct Params { std::vector<std::string> antiprompt; };
struct Task { Params params; };

struct Slot {
    Task const* task;
    int stop = 0;
    std::string stopping_word;
    bool has_next_token = true;
    std::string pending;
    enum { STOP_TYPE_WORD = 1 };
    #define GGML_ASSERT(x) assert(x)
    #include "slot.inc"
    #undef GGML_ASSERT

    std::string feed(std::string_view delta) {
        if (!has_next_token) return {};
        pending.append(delta);
        auto pos = find_stopping_strings(pending, delta.size(), true);
        if (pos != std::string::npos) return pending.substr(0, pos);
        if (find_stopping_strings(pending, delta.size(), false) != std::string::npos) return {};
        // Retain the buffer capacity. Only unsent text is kept for this experiment.
        auto out = pending;
        pending.clear();
        return out;
    }
    std::string finish() { return has_next_token ? pending : std::string(); }
};

struct Cut {
    const cutstream::Plan* plan;
    cutstream::Cursor cursor;
    std::string feed(std::string_view delta) {
        auto r = plan->feed(cursor, delta);
        std::string owned(r.prefix);
        owned.append(r.input);
        return owned;
    }
    std::string finish() { return std::string(plan->finish(cursor)); }
};

constexpr unsigned streams = 512;
template<class State>
double run(State fresh, const std::vector<std::string>& chunks, const std::string& expected) {
    std::vector<State> slots(streams, fresh);
    uint64_t checksum = 0;
    const auto begin = std::clock();
    for (const auto& delta : chunks) {
        for (auto& s : slots) {
            auto out = s.feed(delta);
            for (unsigned char ch : out) checksum += ch;
        }
    }
    for (auto& s : slots) for (unsigned char ch : s.finish()) checksum += ch;
    const auto end = std::clock();
    uint64_t correct = 0;
    for (unsigned char ch : expected) correct += ch;
    if (checksum != correct * streams) throw std::runtime_error("byte checksum mismatch");
    return (end - begin) * (1e9 / CLOCKS_PER_SEC) / (streams * chunks.size());
}

template<class State>
std::string replay(State state, const std::vector<std::string>& chunks) {
    std::string out;
    for (const auto& c : chunks) out += state.feed(c);
    return out + state.finish();
}

int main() {
    const std::vector<std::string> sdk = {
        #include "chunks.inc"
    };
    std::cout << "case,streams,deltas,engine,cpu_ns_per_feed_median,samples_cpu_ns_per_feed\n";
    for (bool near : {false, true}) {
        std::vector<std::string> chunks = sdk;
        if (near) { // Keep recorded SDK deltas; insert artificial near misses and one split stop.
            for (std::size_t i = 20; i < chunks.size(); i += 21) chunks[i] += "<|im_en";
            chunks.push_back("ok</to");
            chunks.push_back("ol_response>after");
        }
        const Task task{{{"<|end_of_message|>", "</tool_response>", "<|im_end|>", "</answer>"}}};
        cutstream::Plan plan(task.params.antiprompt);
        std::string expected;
        for (const auto& s : chunks) expected += s;
        if (near) expected.resize(expected.find("</tool_response>"));
        Slot llama{&task};
        Cut cut{&plan, {}};
        if (replay(llama, chunks) != expected || replay(cut, chunks) != expected)
            throw std::runtime_error("output mismatch");
        std::vector<double> timings[2];
        for (int trial = 0; trial < 12; ++trial) {
            for (int j = 0; j < 2; ++j) {
                int id = (trial + j) % 2;
                double ns = id ? run(llama, chunks, expected) : run(cut, chunks, expected);
                if (trial) timings[id].push_back(ns);
            }
        }
        for (int id = 0; id < 2; ++id) {
            auto sorted = timings[id];
            std::sort(sorted.begin(), sorted.end());
            std::cout << (near ? "sdk-plus-near-misses" : "vercel-openai-text") << ','
                      << streams << ',' << chunks.size() << ','
                      << (id ? "llama-stop-search" : "cutstream-owned") << ','
                      << std::fixed << std::setprecision(2) << sorted[sorted.size()/2] << ',';
            for (std::size_t i = 0; i < timings[id].size(); ++i) {
                if (i) std::cout << ' ';
                std::cout << timings[id][i];
            }
            std::cout << '\n';
        }
    }
}
