#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace racecurve {
using time_us = std::uint32_t;
using total = std::int64_t;
constexpr time_us max_time = std::numeric_limits<std::int32_t>::max();

struct Outcome {
    total launched = 0, elapsed = 0, backup = 0, misses = 0, group_misses = 0;
    bool operator==(const Outcome& rhs) const {
        return launched == rhs.launched && elapsed == rhs.elapsed &&
            backup == rhs.backup && misses == rhs.misses && group_misses == rhs.group_misses;
    }
};

class Curve {
    struct Bin {
        total primary_count = 0, primary_sum = 0;
        total gain_count = 0, gain_sum = 0;
        total rescues = 0, group_rescues = 0;

        void accumulate(const Bin& x) {
            primary_count += x.primary_count;
            primary_sum += x.primary_sum;
            gain_count += x.gain_count;
            gain_sum += x.gain_sum;
            rescues += x.rescues;
            group_rescues += x.group_rescues;
        }
    };
    time_us slo_, step_;
    std::vector<Bin> bins_;
    std::uint64_t divisor_multiplier_;
    unsigned divisor_shift_;
    total samples_ = 0, elapsed_ = 0, misses_ = 0, groups_ = 0, group_misses_ = 0;
    bool open_ = false, group_has_miss_ = false;
    std::int64_t group_limit_ = 0;

    std::size_t clipped_bin(time_us t) const {
        auto quotient = (t * divisor_multiplier_) >> divisor_shift_;
        return std::min<std::size_t>(quotient, bins_.size() - 1);
    }

public:
    Curve(time_us slo, time_us step, std::size_t points) : slo_(slo), step_(step) {
        if (slo > max_time || !step || step > max_time || !points ||
            points - 1 > max_time / step)
            throw std::invalid_argument("grid or SLO outside supported microsecond range");
        // Divide 31-bit unsigned ticks by the invariant step using one multiply
        // and shift. With f=floor(log2(step)), m=ceil(2^(32+f)/step),
        // t*(m*step-2^(32+f)) < 2^31*step <= 2^(32+f). Thus the
        // rounding error cannot advance the integer quotient by one.
        unsigned f = 0;
        for (auto s = step; s > 1; s >>= 1) ++f;
        divisor_shift_ = 32 + f;
        auto scale = std::uint64_t{1} << divisor_shift_;
        divisor_multiplier_ = (scale + step - 1) / step;
        bins_.resize(points);
    }
    total samples() const { return samples_; }
    total groups() const { return groups_; }
    time_us delay(std::size_t i) const { return static_cast<time_us>(i * step_); }

    void add(time_us p, time_us b) {
        if (p > max_time || b > max_time || samples_ == max_time)
            throw std::overflow_error("trace exceeds supported integer range");
        ++samples_;
        open_ = true;
        elapsed_ += p;
        // (x - d)+ is active up to and including delay x - 1.
        if (p) {
            auto& bin = bins_[clipped_bin(p - 1)];
            ++bin.primary_count;
            bin.primary_sum += p;
        }
        // Zero gains add zero to both counters. Keeping this path branchless
        // avoids a poorly predictable winner test on similarly fast replicas.
        time_us gain = p - std::min(p, b);
        auto& gain_bin = bins_[clipped_bin(gain - (gain > 0))];
        gain_bin.gain_count += gain > 0;
        gain_bin.gain_sum += gain;
        if (p > slo_) {
            ++misses_;
            auto limit = static_cast<std::int64_t>(slo_) - b;
            if (limit >= 0) ++bins_[clipped_bin(static_cast<time_us>(limit))].rescues;
            group_limit_ = group_has_miss_ ? std::min(group_limit_, limit) : limit;
            group_has_miss_ = true;
        }
    }

    // Call once after the last leaf of each parallel fanout request.
    void end_group() {
        if (!open_) return;
        ++groups_;
        if (group_has_miss_) {
            ++group_misses_;
            if (group_limit_ >= 0)
                ++bins_[clipped_bin(static_cast<time_us>(group_limit_))].group_rescues;
        }
        open_ = group_has_miss_ = false;
    }

    // Each worker owns complete groups. Repeated merges count the input again.
    void merge(const Curve& other) {
        if (open_ || other.open_) throw std::logic_error("merge requires closed groups");
        if (slo_ != other.slo_ || step_ != other.step_ || bins_.size() != other.bins_.size())
            throw std::invalid_argument("merge requires identical SLO and delay grid");
        if (samples_ + other.samples_ > max_time)
            throw std::overflow_error("merged trace exceeds supported integer range");
        samples_ += other.samples_;
        elapsed_ += other.elapsed_;
        misses_ += other.misses_;
        groups_ += other.groups_;
        group_misses_ += other.group_misses_;
        for (std::size_t i = 0; i < bins_.size(); ++i) bins_[i].accumulate(other.bins_[i]);
    }

    std::vector<Outcome> results() const {
        if (open_) throw std::logic_error("end_group required before snapshot");
        std::vector<Outcome> output(bins_.size());
        Bin tail;
        for (std::size_t i = bins_.size(); i-- > 0;) {
            tail.accumulate(bins_[i]);
            auto d = static_cast<total>(delay(i));
            auto saved = tail.gain_sum - d * tail.gain_count;
            output[i] = {tail.primary_count, elapsed_ - saved,
                tail.primary_sum - d * tail.primary_count - saved,
                misses_ - tail.rescues, group_misses_ - tail.group_rescues};
        }
        return output;
    }
};
}
