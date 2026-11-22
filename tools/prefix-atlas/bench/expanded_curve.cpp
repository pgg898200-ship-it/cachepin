// Independent exact baseline: each page has a last-access position. One set
// bit per resident page marks that position in the unbounded LRU stack.
#include <algorithm>
#include <cstdint>
#include <string>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using Size = std::size_t;
Size number() {
    Size x;
    std::cin >> std::ws;
    if (std::cin.peek() == '-' || !(std::cin >> x)) throw std::runtime_error("invalid unsigned number");
    return x;
}
Size plus(Size a, Size b) {
    if (a > std::numeric_limits<Size>::max() - b) throw std::runtime_error("too many pages");
    return a+b;
}

// Packed bitmap with word-level Fenwick ranks. Words allow last-position
// updates inside one word to share all Fenwick ancestors.
class Rank {
    std::vector<std::uint64_t> bits;
    std::vector<Size> sums;
    void update(Size word, int delta) {
        for (Size i=word+1; i<sums.size(); i+=i&-i) sums[i]+=delta;
    }
public:
    explicit Rank(Size touches) : bits(touches / 64 + (touches % 64 != 0)), sums(bits.size()+1) {}
    void place(Size current) {
        bits[current/64] |= std::uint64_t(1) << (current%64);
        update(current / 64, 1);
    }
    Size before(Size old) const {
        Size word = old / 64;
        auto bit = std::uint64_t(1) << (old%64);
        Size older = __builtin_popcountll(bits[word] & (bit-1));
        for (Size i=word; i; i-=i&-i) older += sums[i];
        return older;
    }
    void move(Size old, Size current) {
        Size from = old / 64, to = current / 64;
        bits[from] ^= std::uint64_t(1) << (old%64);
        bits[to] |= std::uint64_t(1) << (current%64);
        if (from != to) {
            update(from, -1);
            update(to, 1);
        }
    }
};

struct Edge { Size parent, begin, end, depth; };

std::vector<Size> read_capacities(int argc, char** argv) {
    if (argc == 1) return {};
    if (argc != 3 || std::string(argv[1]) != "--prefix-capacities")
        throw std::runtime_error("expected --prefix-capacities <CSV>");
    std::string csv(argv[2]);
    std::vector<Size> caps;
    Size pos=0;
    for (;;) {
        Size comma=csv.find(',',pos);
        std::string item=csv.substr(pos, comma-pos);
        if (item.empty() || item.find_first_not_of("0123456789")!=std::string::npos)
            throw std::runtime_error("invalid capacity");
        caps.push_back(std::stoull(item));
        if (comma==std::string::npos) break;
        pos=comma+1;
    }
    std::sort(caps.begin(),caps.end());
    caps.erase(std::unique(caps.begin(),caps.end()),caps.end());
    return caps;
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    try {
        const auto caps=read_capacities(argc,argv);
        Size n=number(), requests=number(), unique=0, touches=0;
        std::vector<Edge> edges(plus(n,1));
        for (Size i=1; i<=n; ++i) {
            Size parent=number(), span=number();
            if (parent>=i || span==0) throw std::runtime_error("invalid edge");
            edges[i] = {parent, unique, plus(unique,span), plus(edges[parent].depth,span)};
            unique += span;
        }
        std::vector<Size> tips(requests);
        for (auto &t:tips) {
            t=number();
            if (t>n) throw std::runtime_error("invalid tip");
            touches=plus(touches,edges[t].depth);
        }
        std::cin >> std::ws;
        if (std::cin.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing input");
        Rank order(touches);
        std::vector<Size> path, last(unique, std::numeric_limits<Size>::max());
        std::vector<Size> histogram(plus(unique,1));
        Size cold=0, pos=0;
        for (Size tip:tips) {
            path.clear();
            for (Size n=tip; n; n=edges[n].parent) path.push_back(n);
            if (caps.empty()) {
                for (auto i=path.rbegin(); i!=path.rend(); ++i) {
                    for (Size page=edges[*i].begin; page<edges[*i].end; ++page,++pos) {
                        Size old=last[page];
                        if (old == std::numeric_limits<Size>::max()) {
                            order.place(pos);
                            ++cold;
                        } else {
                            ++histogram[cold-order.before(old)];
                            order.move(old,pos);
                        }
                        last[page]=pos;
                    }
                }
            } else {
                // Lookup all prefix blocks before modifying recency. On a
                // never-seen page all descendants must also be cold.
                bool first_cold=false;
                for (auto i=path.rbegin(); i!=path.rend() && !first_cold; ++i) {
                    for (Size page=edges[*i].begin; page<edges[*i].end; ++page) {
                        Size old=last[page];
                        if (old == std::numeric_limits<Size>::max()) {
                            first_cold=true;
                            break;
                        }
                        ++histogram[cold-order.before(old)];
                    }
                }
                // Release in reverse page order so the shallowest page is MRU.
                for (Size edge:path) {
                    for (Size end=edges[edge].end; end>edges[edge].begin; --end,++pos) {
                        Size page=end-1, old=last[page];
                        if (old == std::numeric_limits<Size>::max()) {
                            order.place(pos);
                            ++cold;
                        } else order.move(old,pos);
                        last[page]=pos;
                    }
                }
            }
        }
        if (caps.empty()) {
            std::cout << "cold " << cold << '\n';
            for (Size i=1; i<histogram.size(); ++i)
                if (histogram[i]) std::cout << i << ' ' << histogram[i] << '\n';
        } else {
            std::cout << "capacity_pages,hits,page_references\n";
            Size i=0, hits=0;
            for (Size cap:caps) {
                while (i<histogram.size() && i<=cap) hits+=histogram[i++];
                std::cout << cap << ',' << hits << ',' << touches << '\n';
            }
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
