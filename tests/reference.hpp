#pragma once
#include "../src/racecurve.hpp"
#include <random>

namespace reference {
struct Pair { racecurve::time_us p, b; bool last; };
using racecurve::Outcome;
using racecurve::time_us;

inline std::vector<Outcome> replay(const std::vector<Pair>& trace, time_us slo, time_us step, size_t points) {
    std::vector<Outcome> out(points);
    for (size_t i = 0; i < points; ++i) {
        const auto d = static_cast<std::int64_t>(i * step);
        bool group_miss = false;
        auto& x = out[i];
        for (auto [p, b, last] : trace) {
            auto end = std::min<std::int64_t>(p, d + b);
            x.launched += p > d;
            x.elapsed += end;
            x.backup += std::max<std::int64_t>(0, end - d);
            x.misses += end > slo;
            group_miss |= end > slo;
            if (last) { x.group_misses += group_miss; group_miss = false; }
        }
    }
    return out;
}

// An independent event sweep with sorted raw endpoints. The index can be
// rescanned for another delay grid at the same SLO without re-reading pairs.
struct Sorted {
    std::vector<time_us> primary, gain, rescue_end, group_end;
    std::int64_t primary_sum = 0, gain_sum = 0, misses = 0, group_misses = 0;

    Sorted(const std::vector<Pair>& trace, time_us slo) {
        std::int64_t limit = -1;
        bool missed = false;
        for (auto [p, b, last] : trace) {
            primary.push_back(p);
            primary_sum += p;
            if (p > b) { gain.push_back(p - b); gain_sum += p - b; }
            if (p > slo) {
                ++misses;
                auto cutoff = static_cast<std::int64_t>(slo) - b;
                if (cutoff >= 0) rescue_end.push_back(static_cast<time_us>(cutoff + 1));
                limit = missed ? std::min(limit, cutoff) : cutoff;
                missed = true;
            }
            if (last) {
                group_misses += missed;
                if (missed && limit >= 0) group_end.push_back(static_cast<time_us>(limit + 1));
                missed = false;
            }
        }
        for (auto* v : {&primary, &gain, &rescue_end, &group_end}) std::sort(v->begin(), v->end());
    }
    std::vector<Outcome> scan(time_us step, size_t points) const {
        std::vector<Outcome> out(points);
        size_t pi = 0, gi = 0, ri = 0, fi = 0;
        std::int64_t ps = primary_sum, gs = gain_sum;
        for (size_t i = 0; i < points; ++i) {
            std::int64_t d = i * step;
            while (pi < primary.size() && primary[pi] <= d) ps -= primary[pi++];
            while (gi < gain.size() && gain[gi] <= d) gs -= gain[gi++];
            while (ri < rescue_end.size() && rescue_end[ri] <= d) ++ri;
            while (fi < group_end.size() && group_end[fi] <= d) ++fi;
            auto launched = static_cast<std::int64_t>(primary.size() - pi);
            auto saved = gs - static_cast<std::int64_t>(gain.size() - gi) * d;
            out[i] = {launched, primary_sum - saved, ps - launched * d - saved,
                misses - static_cast<std::int64_t>(rescue_end.size() - ri),
                group_misses - static_cast<std::int64_t>(group_end.size() - fi)};
        }
        return out;
    }
};
inline std::vector<Outcome> compile(const std::vector<Pair>& trace, time_us slo, time_us step, size_t points) {
    racecurve::Curve curve(slo, step, points);
    for (auto x : trace) {
        curve.add(x.p, x.b);
        if (x.last) curve.end_group();
    }
    return curve.results();
}
inline std::vector<Pair> seeded_trace(size_t n) {
    std::mt19937 rng(73842);
    std::vector<Pair> pairs;
    pairs.reserve(n);
    // Shared base duration introduces correlation. Rare attempt-local tails
    // represent skew. All work is simulated integer time for this benchmark.
    for (size_t i = 0; i < n; ++i) {
        time_us base = rng() % 3000;
        auto duration = [&]() -> time_us {
            auto extra = rng() % 29 == 0 ? rng() % 100000 : rng() % 3000;
            return base + extra;
        };
        pairs.push_back({duration(), duration(), (i + 1) % 8 == 0 || i + 1 == n});
    }
    return pairs;
}
}
