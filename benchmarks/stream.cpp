// Linux stress benchmark. Source samples are generated on demand and discarded.
#include "../src/racecurve.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <sys/resource.h>

namespace {
using racecurve::Curve;
using racecurve::Outcome;
using racecurve::time_us;
using Clock = std::chrono::steady_clock;
constexpr time_us slo = 7500, step = 1;
constexpr size_t points = 16384;
constexpr std::uint64_t fanout = 8;
constexpr std::array<size_t, 7> probes{0, 1, 127, 1024, 4000, 7000, 16383};

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

std::pair<time_us, time_us> sample(std::uint64_t i) {
    auto h = mix(i);
    time_us base = (h >> 32) % 3000;
    auto duration = [base](std::uint32_t seed) -> time_us {
        return base + (seed % 29 == 0 ? seed % 100000 : seed % 3000);
    };
    return {duration(static_cast<std::uint32_t>(h)), duration(static_cast<std::uint32_t>(mix(h)))};
}

std::array<Outcome, probes.size()> oracle(std::uint64_t n) {
    std::array<Outcome, probes.size()> out{};
    std::array<bool, probes.size()> group_misses{};
    for (std::uint64_t i = 0; i < n; ++i) {
        auto [p, b] = sample(i);
        for (size_t j = 0; j < probes.size(); ++j) {
            auto delay = static_cast<time_us>(probes[j] * step);
            auto end = std::min(p, delay + b);
            auto& x = out[j];
            x.elapsed += end;
            x.launched += p > delay;
            x.backup += end > delay ? end - delay : 0;
            x.misses += end > slo;
            group_misses[j] |= end > slo;
            if ((i + 1) % fanout == 0 || i + 1 == n) {
                x.group_misses += group_misses[j];
                group_misses[j] = false;
            }
        }
    }
    return out;
}

double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
}

int main(int argc, char** argv) {
    std::uint64_t n = argc > 1 ? std::stoull(argv[1]) : 10000000;
    unsigned workers = argc > 2 ? std::stoul(argv[2]) : 1;
    unsigned repeats = argc > 3 ? std::stoul(argv[3]) : 5;
    if (!n || n > racecurve::max_time || !workers || workers > 64 || !repeats) return 2;
    auto expected = oracle(n);
    auto groups = (n + fanout - 1) / fanout;
    std::cout << "leaves,points,workers,trial,ingest_ms,merge_ms,snapshot_ms,maxrss_kib,all_field_hash\n";
    for (unsigned trial = 1; trial <= repeats; ++trial) {
        auto start = Clock::now();
        std::vector<Curve> shards;
        shards.reserve(workers);
        for (unsigned w = 0; w < workers; ++w) shards.emplace_back(slo, step, points);
        auto ingest = [&](unsigned w) {
            for (auto g = groups * w / workers; g < groups * (w + 1) / workers; ++g) {
                for (auto i = g * fanout; i < std::min(n, (g + 1) * fanout); ++i) {
                    auto [p, b] = sample(i);
                    shards[w].add(p, b);
                }
                shards[w].end_group();
            }
        };
        std::vector<std::thread> threads;
        for (unsigned w = 1; w < workers; ++w) threads.emplace_back(ingest, w);
        ingest(0);
        for (auto& thread : threads) thread.join();
        auto ingested = Clock::now();
        for (unsigned w = 1; w < workers; ++w) shards[0].merge(shards[w]);
        auto merged = Clock::now();
        auto out = shards[0].results();
        auto finished = Clock::now();
        for (size_t j = 0; j < probes.size(); ++j)
            if (!(out[probes[j]] == expected[j])) throw std::runtime_error("direct replay disagreed");
        std::uint64_t hash = 0;
        for (auto& x : out)
            for (auto v : {x.launched, x.elapsed, x.backup, x.misses, x.group_misses})
                hash = (hash ^ static_cast<std::uint64_t>(v)) * 0x100000001b3ULL;
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage)) return 1;
        std::cout << n << ',' << points << ',' << workers << ',' << trial << ','
            << ms(start, ingested) << ',' << ms(ingested, merged) << ',' << ms(merged, finished) << ','
            << usage.ru_maxrss << ',' << hash << '\n';
    }
}
