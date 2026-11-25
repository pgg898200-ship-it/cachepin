// Optional comparator for the benchmark's single literal fixtures.
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <memory>

struct PcreLiteral {
    using Cursor = BufferCursor;
    std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)> code{nullptr, pcre2_code_free};
    std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> data{nullptr, pcre2_match_data_free};
    std::string joined;

    explicit PcreLiteral(const PatternSet& patterns) {
        if (patterns.words.size() != 1) throw std::invalid_argument("one literal required");
        const auto& word = patterns.words.front();
        int error;
        PCRE2_SIZE offset;
        code.reset(pcre2_compile(reinterpret_cast<PCRE2_SPTR>(word.data()), word.size(),
                                 PCRE2_LITERAL, &error, &offset, nullptr));
        if (!code || pcre2_jit_compile(code.get(), PCRE2_JIT_PARTIAL_HARD) != 0)
            throw std::runtime_error("PCRE2 literal JIT compilation failed");
        data.reset(pcre2_match_data_create_from_pattern(code.get(), nullptr));
        if (!data) throw std::bad_alloc();
    }

    Result feed(Cursor& cursor, View chunk) {
        if (cursor.done) return {};
        const auto old = cursor.held.size();
        View text = chunk;
        if (old) {
            joined.assign(cursor.held).append(chunk);
            text = joined;
        }
        int rc = pcre2_jit_match(code.get(), reinterpret_cast<PCRE2_SPTR>(text.data()),
                                 text.size(), 0, PCRE2_PARTIAL_HARD, data.get(), nullptr);
        if (rc == PCRE2_ERROR_NOMATCH) {
            cursor.held.clear();
            return {{}, text, chunk.size(), -1};
        }
        if (rc < 0 && rc != PCRE2_ERROR_PARTIAL) throw std::runtime_error("PCRE2 match failed");
        auto range = pcre2_get_ovector_pointer(data.get());
        auto visible = text.substr(0, range[0]);
        if (rc == PCRE2_ERROR_PARTIAL) {
            cursor.held.assign(text.substr(range[0]));
            return {{}, visible, chunk.size(), -1};
        }
        cursor.done = true;
        return {{}, visible, range[1] - old, 0};
    }

    View finish(Cursor& cursor) {
        if (cursor.done) return {};
        cursor.done = true;
        return cursor.held;
    }
};
