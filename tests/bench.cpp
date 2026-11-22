#include "reference.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>

namespace {
using clock_type = std::chrono::steady_clock;
template<class F>
double measure(F run, const std::vector<racecurve::Outcome>& expected) {
    auto start = clock_type::now();
    auto output = run();
    double ms = std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
    if (output != expected) throw std::runtime_error("different benchmark outputs");
    return ms;
}
double median(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}
}
int main(int argc, char** argv) {
    size_t n = argc > 1 ? std::stoul(argv[1]) : 200000;
    size_t repeats = argc > 2 ? std::stoul(argv[2]) : 5;
    if (!n || !repeats || n > racecurve::max_time) return 2;
    auto data = reference::seeded_trace(n);
    std::cout << "pairs,points,step_us,slo_us,repetitions,compiled_ms,direct_ms,sorted_build_scan_ms,sorted_warm_scan_ms\n";
    for (size_t k : {1, 16, 256, 1024}) {
        constexpr auto slo = 7500u, step = 32u;
        reference::Sorted sorted(data, slo);
        auto expected = sorted.scan(step, k);
        auto compiled = [&] { return reference::compile(data, slo, step, k); };
        auto replay = [&] { return reference::replay(data, slo, step, k); };
        auto presort = [&] { return reference::Sorted(data, slo).scan(step, k); };
        auto scan = [&] { return sorted.scan(step, k); };
        // Warm up and rotate the measurement order to reduce ordering effects.
        measure(compiled, expected); measure(replay, expected); measure(presort, expected); measure(scan, expected);
        std::vector<double> times[4];
        for (size_t i = 0; i < repeats; ++i)
            for (size_t j = 0; j < 4; ++j) {
                auto m = (i + j) % 4;
                times[m].push_back(m == 0 ? measure(compiled, expected) : m == 1 ? measure(replay, expected) :
                    m == 2 ? measure(presort, expected) : measure(scan, expected));
            }
        std::cout << std::fixed << std::setprecision(4) << n << ',' << k << ',' << step << ',' << slo << ',' << repeats;
        for (auto& t : times) std::cout << ',' << median(t);
        std::cout << std::endl;
    }
}
