// Exact all-capacity stack-distance histogram for forward root-to-tip page scans.
// Build with c++ -std=c++17 -O3 curve.cpp -o curve. See README-curve.md.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Count = std::uint64_t;
using Node = int;

Count add(Count a, Count b) {
    if (b > std::numeric_limits<Count>::max() - a)
        throw std::overflow_error("page count exceeds 64 bits");
    return a + b;
}

class Tree {
    std::vector<std::vector<Node>> up;
    std::vector<int> depth;
    std::vector<Node> child, sibling;
public:
    std::vector<Count> pages;
    std::vector<int> begin, end;

    explicit Tree(int n) : depth(n+1), child(n+1), sibling(n+1),
                           pages(n+1), begin(n+1), end(n+1) {
        int levels = 1;
        while ((Count{1} << levels) <= static_cast<Count>(n)) ++levels;
        up.assign(levels, std::vector<Node>(n+1));
    }
    void attach(Node id, Node parent, Count span) {
        pages[id] = add(pages[parent], span);
        depth[id] = depth[parent] + 1;
        sibling[id] = child[parent];
        child[parent] = id;
        up[0][id] = parent;
        for (std::size_t k=1; k<up.size(); ++k)
            up[k][id] = up[k-1][up[k-1][id]];
    }
    void index() {
        int clock = 0;
        // Open/close events avoid using the native call stack for deep traces.
        std::vector<int> stack{0};
        while (!stack.empty()) {
            int entry = stack.back();
            stack.pop_back();
            if (entry < 0) {
                end[~entry] = clock;
                continue;
            }
            begin[entry] = clock++;
            stack.push_back(~entry);
            for (Node i=child[entry]; i; i=sibling[i]) stack.push_back(i);
        }
    }
    Node lift(Node id, int delta) const {
        for (int k=0; delta; ++k, delta >>= 1)
            if (delta & 1) id = up[k][id];
        return id;
    }
    Node child_toward(Node ancestor, Node descendant) const {
        return lift(descendant, depth[descendant] - depth[ancestor] - 1);
    }
    bool contains(Node ancestor, Node descendant) const {
        return begin[ancestor] <= begin[descendant] && begin[descendant] < end[ancestor];
    }
    Node lca(Node a, Node b) const {
        if (contains(a,b)) return a;
        if (contains(b,a)) return b;
        for (int k=static_cast<int>(up.size())-1; k>=0; --k)
            if (!contains(up[k][a], b)) a = up[k][a];
        return up[0][a];
    }
};

// A point stores its most recent request epoch. Subtree maximum yields the
// most recent request whose page path crosses a given child edge.
class RangeMax {
    std::size_t n;
    std::vector<int> tree;
public:
    explicit RangeMax(std::size_t size) : n(size), tree(2*size) {}
    void set(int position, int epoch) {
        std::size_t i = position + n;
        tree[i] = epoch;
        while ((i >>= 1)) tree[i] = std::max(tree[2*i], tree[2*i+1]);
    }
    int get(int begin, int end) const {
        int result = 0;
        for (std::size_t l=begin+n, r=end+n; l<r; l/=2, r/=2) {
            if (l & 1) result = std::max(result, tree[l++]);
            if (r & 1) result = std::max(result, tree[--r]);
        }
        return result;
    }
};

// Number of pages still owned by each epoch of the unbounded recency stack.
class Weights {
    std::vector<Count> slots;
public:
    explicit Weights(int requests) : slots(requests+1) {}
    void credit(int epoch, Count weight) {
        for (std::size_t i=epoch; i<slots.size(); i+=i&-i) slots[i] = add(slots[i], weight);
    }
    void debit(int epoch, Count weight) {
        for (std::size_t i=epoch; i<slots.size(); i+=i&-i) slots[i] -= weight;
    }
    Count before(int epoch) const {
        Count sum = 0;
        for (int i=epoch-1; i; i-=i&-i) sum += slots[i];
        return sum;
    }
};

std::vector<Count> capacities(int argc, char** argv) {
    if (argc == 1) return {};
    if (argc != 3 || std::string(argv[1]) != "--prefix-capacities")
        throw std::runtime_error("usage: curve [--prefix-capacities PAGES,PAGES,... | --prefix-curve] < trace.tree");
    std::string csv(argv[2]);
    std::vector<Count> values;
    std::size_t pos = 0;
    for (;;) {
        auto comma = csv.find(',', pos);
        auto item = csv.substr(pos, comma == std::string::npos ? comma : comma-pos);
        if (item.empty() || item.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("capacities must be comma-separated nonnegative page counts");
        values.push_back(std::stoull(item));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

int main(int argc, char** argv) {
    try {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);
        const bool full_curve = argc == 2 && std::string(argv[1]) == "--prefix-curve";
        const auto caps = full_curve ? std::vector<Count>{} : capacities(argc, argv);
        const bool atomic = full_curve || !caps.empty();
        long long nn, rr;
        if (!(std::cin>>nn>>rr) || nn<0 || rr<0 || nn>=std::numeric_limits<int>::max()
            || rr>=std::numeric_limits<int>::max())
            throw std::runtime_error("expected nonnegative node and request counts below INT_MAX");
        int n=static_cast<int>(nn), requests=static_cast<int>(rr);
        Tree tree(n);
        for (int id=1; id<=n; ++id) {
            int parent;
            std::string span;
            if (!(std::cin>>parent>>span) || parent<0 || parent>=id ||
                span.empty() || span.find_first_not_of("0123456789")!=std::string::npos)
                throw std::runtime_error("invalid parent or edge length at node "+std::to_string(id));
            Count weight=std::stoull(span);
            if (!weight) throw std::runtime_error("edge lengths must be positive");
            tree.attach(id, parent, weight);
        }
        tree.index();
        RangeMax recent(static_cast<std::size_t>(n)+1);
        Weights live(requests);
        std::vector<Node> tips(requests+1);
        std::vector<std::pair<int,Count>> debits;
        std::map<Count,Count> histogram;
        // Under atomic prefix lookup, each matching run contributes
        // clamp(capacity - newer_pages, 0, length) hits per request.
        std::map<Count,std::int64_t> slope_changes;
        Count distinct=0, total=0;
        for (int t=1; t<=requests; ++t) {
            Node tip;
            if (!(std::cin>>tip) || tip<0 || tip>n) throw std::runtime_error("invalid request tip at request "+std::to_string(t));
            tips[t] = tip;
            total = add(total, tree.pages[tip]);
            Node processed = 0;
            Count cold = 0;
            debits.clear();
            while (processed != tip) {
                Node next = tree.child_toward(processed, tip);
                int epoch = recent.get(tree.begin[next], tree.end[next]);
                if (!epoch) {
                    cold = tree.pages[tip] - tree.pages[processed];
                    break;
                }
                Node shared = tree.lca(tip, tips[epoch]);
                Count length = tree.pages[shared] - tree.pages[processed];
                if (!atomic) {
                    Count threshold = distinct - live.before(epoch);
                    histogram[threshold] = add(histogram[threshold], length);
                } else {
                    Count newer = distinct - live.before(epoch+1);
                    ++slope_changes[newer];
                    --slope_changes[newer + length];
                }
                debits.emplace_back(epoch, length);
                processed = shared;
            }
            // All thresholds use the stack at start of request.
            for (auto [epoch, weight] : debits) live.debit(epoch, weight);
            live.credit(t, tree.pages[tip]);
            distinct = add(distinct, cold);
            recent.set(tree.begin[tip], t);
        }
        std::string trailing;
        if (std::cin>>trailing) throw std::runtime_error("unexpected trailing input");
        if (!atomic) {
            std::cout << "cold " << distinct << '\n';
            for (auto [threshold, hits] : histogram) std::cout << threshold << ' ' << hits << '\n';
        } else if (full_curve) {
            std::cout << "capacity_pages,hits,hits_per_additional_page,page_references\n";
            slope_changes.try_emplace(0, 0);
            Count previous=0, hits=0;
            std::int64_t slope=0;
            for (auto [capacity, change] : slope_changes) {
                if (change == 0 && capacity != 0) continue;
                hits += (capacity - previous) * static_cast<Count>(slope);
                previous = capacity;
                slope += change;
                std::cout << capacity << ',' << hits << ',' << slope << ',' << total << '\n';
            }
        } else {
            std::cout << "capacity_pages,hits,page_references\n";
            auto event = slope_changes.begin();
            Count previous=0, hits=0;
            std::int64_t slope=0;
            for (Count cap : caps) {
                while (event != slope_changes.end() && event->first <= cap) {
                    hits += (event->first - previous) * static_cast<Count>(slope);
                    previous = event->first;
                    slope += event->second;
                    ++event;
                }
                hits += (cap - previous) * static_cast<Count>(slope);
                previous = cap;
                std::cout << cap << ',' << hits << ',' << total << '\n';
            }
        }
    } catch (const std::exception& err) {
        std::cerr << "curve: " << err.what() << '\n';
        return 1;
    }
}
