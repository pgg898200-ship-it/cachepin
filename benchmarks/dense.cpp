// Competing exact index for a trace with a known bounded microsecond domain.
#include "../tests/reference.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
using racecurve::Outcome;
using racecurve::time_us;
using racecurve::total;

class Dense {
    struct Counts {
        std::uint32_t primary = 0, gain = 0, rescue = 0, group_rescue = 0;
    };
    std::vector<Counts> ticks_;
    total n_ = 0, p_sum_ = 0, gain_sum_ = 0, gains_ = 0;
    total misses_ = 0, group_misses_ = 0, rescues_ = 0, group_rescues_ = 0;
public:
    Dense(const std::vector<reference::Pair>& pairs, time_us slo, time_us bound)
        : ticks_(static_cast<size_t>(std::max(bound, slo)) + 2) {
        bool group_missed = false;
        total group_limit = 0;
        for (auto [p, b, last] : pairs) {
            if (p > bound) throw std::invalid_argument("observed value exceeds bound");
            ++n_;
            p_sum_ += p;
            ++ticks_[p].primary;
            auto gain = p - std::min(p, b);
            gains_ += gain > 0;
            gain_sum_ += gain;
            ticks_[gain].gain += gain > 0;
            if (p > slo) {
                ++misses_;
                auto limit = static_cast<total>(slo) - b;
                if (limit >= 0) {
                    ++ticks_[static_cast<size_t>(limit) + 1].rescue;
                    ++rescues_;
                }
                group_limit = group_missed ? std::min(limit, group_limit) : limit;
                group_missed = true;
            }
            if (last && group_missed) {
                ++group_misses_;
                if (group_limit >= 0) {
                    ++ticks_[static_cast<size_t>(group_limit) + 1].group_rescue;
                    ++group_rescues_;
                }
                group_missed = false;
            }
        }
    }
    std::vector<Outcome> scan(time_us step, size_t points) const {
        total launched = n_, gains = gains_, ps = p_sum_, gs = gain_sum_;
        total misses = misses_ - rescues_, group_misses = group_misses_ - group_rescues_;
        size_t t = 0;
        std::vector<Outcome> out(points);
        for (size_t i = 0; i < points; ++i) {
            auto d = static_cast<total>(i * step);
            while (t < ticks_.size() && static_cast<total>(t) <= d) {
                const auto& c = ticks_[t];
                auto tick = static_cast<total>(t);
                launched -= c.primary;
                ps -= tick * c.primary;
                gains -= c.gain;
                gs -= tick * c.gain;
                misses += c.rescue;
                group_misses += c.group_rescue;
                ++t;
            }
            auto saved = gs - gains * d;
            out[i] = {launched, p_sum_ - saved, ps - launched * d - saved, misses, group_misses};
        }
        return out;
    }
};

template<class F>
double time_ms(F f, const std::vector<Outcome>& expected) {
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    auto out = f();
    auto elapsed = Clock::now() - start;
    if (out != expected) throw std::runtime_error("incorrect result");
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

int main() {
    constexpr size_t n = 200000, trials = 9;
    constexpr time_us slo = 7500, step = 32, bound = 103000;
    auto input = reference::seeded_trace(n);
    std::cout << "pairs,points,step_us,slo_us,repetitions,compiled_ms,dense_build_scan_ms,dense_warm_scan_ms\n";
    std::cout << std::fixed << std::setprecision(4);
    for (size_t points : {1, 16, 256, 1024}) {
        auto expected = reference::Sorted(input, slo).scan(step, points);
        Dense index(input, slo, bound);
        std::vector<double> times[3];
        for (size_t r = 0; r <= trials; ++r) {
            for (size_t j = 0; j < 3; ++j) {
                auto method = (r + j) % 3;
                double elapsed;
                if (method == 0) elapsed = time_ms([&]{ return reference::compile(input, slo, step, points); }, expected);
                else if (method == 1) elapsed = time_ms([&]{ return Dense(input, slo, bound).scan(step, points); }, expected);
                else elapsed = time_ms([&]{ return index.scan(step, points); }, expected);
                if (r) times[method].push_back(elapsed);
            }
        }
        std::cout << n << ',' << points << ',' << step << ',' << slo << ',' << trials;
        for (auto& t : times) {
            std::sort(t.begin(), t.end());
            std::cout << ',' << t[trials / 2];
        }
        std::cout << '\n';
    }
}
