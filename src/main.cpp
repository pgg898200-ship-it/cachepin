#include "racecurve.hpp"
#include <charconv>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace {
unsigned long long integer(std::string_view s) {
    unsigned long long n = 0;
    auto [end, error] = std::from_chars(s.data(), s.data() + s.size(), n);
    if (error != std::errc{} || end != s.data() + s.size()) throw std::invalid_argument("invalid unsigned integer");
    return n;
}
racecurve::time_us microseconds(std::string_view s) {
    auto n = integer(s);
    if (n > racecurve::max_time) throw std::out_of_range("time or grid count out of range");
    return static_cast<racecurve::time_us>(n);
}
std::string_view field(std::string_view& row) {
    auto pos = row.find(',');
    if (pos == row.npos) throw std::invalid_argument("missing field");
    auto value = row.substr(0, pos);
    row.remove_prefix(pos + 1);
    return value;
}
void strip_cr(std::string& s) {
    if (!s.empty() && s.back() == '\r') s.pop_back();
}
}
int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    const char* usage = "racecurve SLO_US STEP_US POINTS [trace.csv|-]\n"
        "Input header: primary_us,backup_us OR group,primary_us,backup_us\n"
        "Delays: 0, STEP_US, ..., (POINTS-1)*STEP_US. Group IDs must be sorted.\n";
    if (argc == 2 && std::string_view(argv[1]) == "--help") { std::cout << usage; return 0; }
    if (argc < 4 || argc > 5) { std::cerr << usage; return 2; }
    size_t line = 0;
    try {
        auto points = microseconds(argv[3]);
        racecurve::Curve curve(microseconds(argv[1]), microseconds(argv[2]), points);
        std::ifstream file;
        std::istream* in = &std::cin;
        if (argc == 5 && std::string_view(argv[4]) != "-") {
            file.open(argv[4]);
            if (!file) throw std::runtime_error("unable to open trace");
            in = &file;
        }
        std::string row;
        ++line;
        std::getline(*in, row);
        strip_cr(row);
        bool grouped = row == "group,primary_us,backup_us";
        if (!grouped && row != "primary_us,backup_us") throw std::runtime_error("unrecognized CSV header");
        unsigned long long previous = 0;
        while (std::getline(*in, row)) {
            ++line;
            strip_cr(row);
            std::string_view fields(row);
            if (grouped) {
                auto group = integer(field(fields));
                if (group < previous) throw std::runtime_error("group IDs must be nondecreasing");
                if (group != previous) curve.end_group();
                previous = group;
            }
            auto primary = microseconds(field(fields));
            curve.add(primary, microseconds(fields));
            if (!grouped) curve.end_group();
        }
        if (in->bad()) throw std::runtime_error("trace read failed");
        if (!curve.samples()) throw std::runtime_error("empty trace");
        curve.end_group();
        auto values = curve.results();
        std::cout << "delay_us,leaves,groups,launches,leaf_misses,group_misses,elapsed_us,backup_us,total_work_us\n";
        for (size_t i = 0; i < values.size(); ++i) {
            auto& x = values[i];
            std::cout << curve.delay(i) << ',' << curve.samples() << ',' << curve.groups() << ','
                << x.launched << ',' << x.misses << ',' << x.group_misses << ',' << x.elapsed
                << ',' << x.backup << ',' << x.elapsed + x.backup << '\n';
        }
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "racecurve";
        if (line) std::cerr << " at input line " << line;
        std::cerr << ": " << error.what() << '\n';
        return 2;
    }
}
