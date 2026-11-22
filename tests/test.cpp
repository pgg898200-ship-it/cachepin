#include "reference.hpp"
#include <iostream>
#include <stdexcept>

using namespace reference;
void check(const std::vector<Pair>& trace, time_us slo, time_us step, size_t points) {
    auto truth = replay(trace, slo, step, points);
    if (truth != compile(trace, slo, step, points) || truth != Sorted(trace, slo).scan(step, points))
        throw std::runtime_error("curve differs from direct replay");
}
void check_merge() {
    racecurve::Curve merged(12, 2, 101), worker(12, 2, 101);
    const auto trace = seeded_trace(203);
    for (size_t i = 0; i < trace.size(); ++i) {
        auto row = trace[i];
        auto& shard = i % 16 < 8 ? merged : worker;
        shard.add(row.p, row.b);
        if (row.last) shard.end_group();
    }
    merged.merge(worker);
    if (merged.samples() != 203 || merged.groups() != 26 ||
        merged.results() != replay(trace, 12, 2, 101))
        throw std::runtime_error("incorrect distributed merge");
    try { merged.merge(racecurve::Curve(15, 2, 101)); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("merge accepted incompatible grids");
}
int main() {
    check_merge();
    for (time_us a = 0; a < 6; ++a)
        for (time_us b = 0; b < 6; ++b)
            for (time_us c = 0; c < 6; ++c)
                for (time_us d = 0; d < 6; ++d)
                    for (time_us slo = 0; slo < 8; ++slo)
                        for (time_us step : {1u, 2u, 5u})
                            check({{a,b,false},{c,d,true},{b,c,true}}, slo, step, 11);
    for (size_t n : {1, 42, 2177}) {
        auto data = seeded_trace(n);
        for (auto slo : {0u, 2500u, 8000u, racecurve::max_time})
            for (auto step : {1u, 11u, 808u}) check(data, slo, step, 135);
    }
    auto max = racecurve::max_time;
    std::mt19937 rng(129871);
    for (int trial = 0; trial < 30000; ++trial) {
        time_us width = 1 + rng() % max;
        size_t k = 1 + std::min<time_us>(max / width, 8);
        auto boundary = static_cast<time_us>((k - 1) * width);
        time_us lower = boundary ? boundary - 1 : 0;
        time_us upper = boundary < max ? boundary + 1 : max;
        check({{boundary, 0, false}, {upper, 1, false}, {lower, 0, true},
               {static_cast<time_us>(rng() % max), static_cast<time_us>(rng() % max), true}},
               static_cast<time_us>(rng() % max), width, k);
    }
    check({{max, max, false}, {0, max, true}, {max, 0, true}}, max - 1, max, 2);
    std::cout << "All differential checks passed.\n";
}
