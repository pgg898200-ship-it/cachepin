#ifndef CUTSTREAM_HPP
#define CUTSTREAM_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cutstream {

class Plan;

// Each live stream needs one cursor. Keep it with the plan that created its
// state. Separate cursors can concurrently read the same immutable plan.
class Cursor {
public:
    bool done() const noexcept { return state_ == finished; }
    void reset() noexcept { state_ = 0; }

private:
    friend class Plan;
    static constexpr std::uint32_t finished = UINT32_MAX;
    std::uint32_t state_ = 0;
};

static_assert(sizeof(Cursor) == 4, "A cursor is one 32-bit state index");

// Emit prefix then input. prefix borrows the plan; input borrows the caller's
// chunk. consumed includes a matched stop, and excludes trailing chunk bytes.
struct Result {
    std::string_view prefix;
    std::string_view input;
    std::size_t consumed = 0;
    int match = -1;
};

class Plan {
public:
    explicit Plan(std::vector<std::string> stops) : stops_(std::move(stops)) {
        if (stops_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::length_error("too many stop markers");
        std::vector<int> viable;
        viable.reserve(stops_.size());
        for (std::size_t i = 0; i < stops_.size(); ++i) {
            if (stops_[i].empty())
                throw std::invalid_argument("stop markers must be nonempty");
            viable.push_back(static_cast<int>(i));
        }
        compile(viable);

        // A marker that contains a stop ending strictly before its own end
        // can never win. Even its prefixes can cause unnecessary holdback.
        std::size_t kept = 0;
        for (int id : viable) {
            std::uint32_t state = 0;
            const auto& stop = stops_[id];
            for (std::size_t i = 0; i + 1 < stop.size(); ++i) {
                state = step(state, static_cast<unsigned char>(stop[i]));
                if (nodes_[state].match >= 0) break;
            }
            if (nodes_[state].match < 0) viable[kept++] = id;
        }
        if (kept != viable.size()) {
            viable.resize(kept);
            compile(viable);
        }
    }

    // The returned bytes are held until a future feed disproves the prefix or
    // finish commits it. Moving, assigning or destroying the plan invalidates
    // outstanding plan-backed views. Use the same plan for a live cursor.
    std::string_view pending(const Cursor& cursor) const noexcept {
        return cursor.done() ? std::string_view{} : spelling(cursor.state_);
    }

    Result feed(Cursor& cursor, std::string_view chunk) const noexcept {
        if (cursor.done()) return {};
        if (nodes_.size() == 1) return {{}, chunk, chunk.size(), -1};

        const std::string_view old = spelling(cursor.state_);
        std::uint32_t state = cursor.state_;
        std::size_t pos = 0;
        int matched = -1;
        while (pos < chunk.size()) {
            if (single_start_ && state == 0 &&
                static_cast<unsigned char>(chunk[pos]) != start_) {
                const void* hit = std::memchr(chunk.data() + pos, start_, chunk.size() - pos);
                if (!hit) {
                    pos = chunk.size();
                    break;
                }
                pos = static_cast<std::size_t>(static_cast<const char*>(hit) - chunk.data());
            }
            state = step(state, static_cast<unsigned char>(chunk[pos++]));
            matched = nodes_[state].match;
            if (matched >= 0) break;
        }
        cursor.state_ = matched < 0 ? state : Cursor::finished;

        // The literal region is a prefix of old || chunk[0:pos]. Remove the
        // held or matched suffix, subtracting from chunk first to avoid overflow.
        const std::size_t withheld = matched < 0 ? nodes_[state].depth : stops_[matched].size();
        if (withheld <= pos)
            return {old, chunk.substr(0, pos - withheld), pos, matched};
        return {old.substr(0, old.size() - (withheld - pos)), {}, pos, matched};
    }

    // Flush any unfinished prefix as literal bytes and close this stream.
    std::string_view finish(Cursor& cursor) const noexcept {
        const auto literal = pending(cursor);
        cursor.state_ = Cursor::finished;
        return literal;
    }

private:
    struct Node {
        std::uint32_t depth;
        int spelling_id;
        int match;
    };

    std::vector<std::string> stops_;
    std::vector<Node> nodes_;
    std::vector<std::uint32_t> table_;
    std::array<std::uint16_t, 256> classes_{};
    std::size_t stride_ = 1;
    bool single_start_ = false;
    unsigned char start_ = 0;

    std::string_view spelling(std::uint32_t state) const noexcept {
        if (!state) return {};
        const auto& node = nodes_[state];
        return {stops_[node.spelling_id].data(), node.depth};
    }

    std::uint32_t step(std::uint32_t state, unsigned char byte) const noexcept {
        return table_[static_cast<std::size_t>(state) * stride_ + classes_[byte]];
    }

    std::uint32_t add_node(std::uint32_t depth, int id) {
        if (nodes_.size() >= Cursor::finished || stride_ > table_.max_size() - table_.size())
            throw std::length_error("stop automaton too large");
        auto index = static_cast<std::uint32_t>(nodes_.size());
        nodes_.push_back({depth, id, -1});
        table_.resize(table_.size() + stride_);
        return index;
    }

    // Build a byte-class DFA. Class zero groups all bytes absent from markers.
    // The temporary failure links disappear after transitions are compiled.
    void compile(const std::vector<int>& ids) {
        classes_.fill(0);
        stride_ = 1;
        std::array<bool, 256> starts{};
        unsigned start_count = 0;
        for (int id : ids) {
            const auto& stop = stops_[id];
            const auto first = static_cast<unsigned char>(stop.front());
            if (!starts[first]) {
                starts[first] = true;
                start_ = first;
                ++start_count;
            }
            for (unsigned char c : stop)
                if (!classes_[c]) classes_[c] = static_cast<std::uint16_t>(stride_++);
        }
        single_start_ = start_count == 1;
        nodes_.clear();
        table_.clear();
        add_node(0, -1);
        for (int id : ids) {
            std::uint32_t state = 0;
            for (unsigned char c : stops_[id]) {
                const auto slot = static_cast<std::size_t>(state) * stride_ + classes_[c];
                if (!table_[slot]) {
                    const auto next = add_node(nodes_[state].depth + 1, id);
                    table_[slot] = next;
                }
                state = table_[slot];
            }
            // IDs are inserted in original order, so the first duplicate wins.
            if (nodes_[state].match < 0) nodes_[state].match = id;
        }

        struct Link { std::uint32_t state, failure; };
        std::vector<Link> queue;
        queue.reserve(nodes_.size() - 1);
        for (std::size_t c = 1; c < stride_; ++c)
            if (table_[c]) queue.push_back({table_[c], 0});
        for (std::size_t i = 0; i < queue.size(); ++i) {
            const auto [state, failure] = queue[i];
            // An exact terminal at this node is longer than every suffix.
            if (nodes_[state].match < 0) nodes_[state].match = nodes_[failure].match;
            const auto row = static_cast<std::size_t>(state) * stride_;
            const auto back = static_cast<std::size_t>(failure) * stride_;
            for (std::size_t c = 1; c < stride_; ++c) {
                auto& next = table_[row + c];
                if (next) queue.push_back({next, table_[back + c]});
                else next = table_[back + c];
            }
        }
    }
};

} // namespace cutstream

#endif
