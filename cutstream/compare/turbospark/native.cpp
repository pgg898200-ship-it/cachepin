#include "cutstream.hpp"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Event { std::size_t stream; std::string chunk; };
struct Digest {
    std::uint64_t value = 14695981039346656037ULL;
    std::size_t bytes = 0;
    void add(const std::string& output, bool stopped) {
        for (unsigned char c : output) value = (value ^ c) * 1099511628211ULL;
        value = (value ^ (256u + stopped)) * 1099511628211ULL;
        bytes += output.size();
    }
};

int main(int argc, char** argv) {
    if (argc != 3) return 1;
    std::ifstream stops_file(std::string(argv[1]) + "/stops.txt");
    std::ifstream events_file(std::string(argv[1]) + "/events.tsv");
    if (!stops_file || !events_file) return 2;
    std::vector<std::string> stops;
    std::string line;
    while (std::getline(stops_file, line)) stops.push_back(line);
    std::vector<Event> events;
    std::size_t streams = 0;
    while (std::getline(events_file, line)) {
        const auto tab = line.find('\t');
        const auto stream = static_cast<std::size_t>(std::stoul(line.substr(0, tab)));
        if (streams <= stream) streams = stream + 1;
        events.push_back({stream, line.substr(tab + 1)});
    }
    const cutstream::Plan plan(std::move(stops));
    const auto budget_ns = std::stoull(argv[2]);
    std::uint64_t elapsed = 0, runs = 0;
    Digest expected;
    do {
        std::vector<cutstream::Cursor> cursors(streams);
        Digest digest;
        const auto start = std::chrono::steady_clock::now();
        for (const auto& event : events) {
            auto& cursor = cursors[event.stream];
            const auto result = plan.feed(cursor, event.chunk);
            std::string owned;
            owned.reserve(result.prefix.size() + result.input.size());
            owned.append(result.prefix);
            owned.append(result.input);
            digest.add(owned, cursor.done());
        }
        for (auto& cursor : cursors) {
            const bool stopped_on_marker = cursor.done();
            const std::string owned(plan.finish(cursor));
            digest.add(owned, stopped_on_marker);
        }
        elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (runs && (expected.value != digest.value || expected.bytes != digest.bytes)) return 3;
        expected = digest;
        ++runs;
    } while (elapsed < budget_ns);
    std::cout << expected.value << ' ' << expected.bytes << ' ' << elapsed << ' ' << runs << '\n';
}
