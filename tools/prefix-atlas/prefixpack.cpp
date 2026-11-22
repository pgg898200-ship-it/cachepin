// Prefix Atlas trace archive. Build with make. Format version 1 is little endian.
#include <zstd.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {
using U = uint64_t;
using Bytes = std::string;
using View = std::string_view;
constexpr U max_row = 64ULL << 20;
constexpr U max_frame = 80ULL << 20;
constexpr U max_dir = 128ULL << 20;
constexpr size_t target_frame = 128 << 10;
constexpr size_t rows_per_frame = 64;
constexpr View magic = "PATLAS01";
constexpr View trailer = "PATLEND1";

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void put64(Bytes &s, U n) {
    for (unsigned i = 0; i < 8; ++i) { s.push_back(char(n & 255)); n >>= 8; }
}
U take64(View &s) {
    require(s.size() >= 8, "incomplete index");
    U n = 0;
    for (unsigned i = 0; i < 8; ++i) n |= U(uint8_t(s[i])) << (8*i);
    s.remove_prefix(8);
    return n;
}
void put(Bytes &s, U n) {
    while (n >= 128) { s.push_back(char((n & 127) | 128)); n >>= 7; }
    s.push_back(char(n));
}
U take(View &s) {
    U n = 0;
    for (unsigned shift = 0; shift <= 63; shift += 7) {
        require(!s.empty(), "incomplete integer");
        const auto b = uint8_t(s.front()); s.remove_prefix(1);
        require(shift < 63 || b <= 1, "integer overflow");
        n |= U(b & 127) << shift;
        if (b < 128) return n;
    }
    throw std::runtime_error("invalid integer");
}
View slice(View &s, U n) {
    require(n <= s.size(), "invalid field length");
    auto v = s.substr(0, n); s.remove_prefix(n); return v;
}
void put_field(Bytes &s, View part) {
    put(s, part.size()); s.append(part);
}
void write(std::ostream &out, View s) {
    out.write(s.data(), s.size());
    require(bool(out), "output write failed");
}
Bytes read_at(std::ifstream &in, U pos, U n) {
    Bytes result(n, '\0');
    in.seekg(pos); in.read(result.data(), n);
    require(bool(in), "archive read failed"); return result;
}
size_t checked(size_t n) {
    if (ZSTD_isError(n)) throw std::runtime_error(ZSTD_getErrorName(n));
    return n;
}
struct Compressor {
    std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> ctx{ZSTD_createCCtx(), ZSTD_freeCCtx};
    Compressor() {
        require(bool(ctx), "zstd allocation failed");
        checked(ZSTD_CCtx_setParameter(ctx.get(), ZSTD_c_checksumFlag, 1));
        checked(ZSTD_CCtx_setParameter(ctx.get(), ZSTD_c_compressionLevel, 3));
    }
    Bytes frame(View raw) {
        Bytes dest(ZSTD_compressBound(raw.size()), '\0');
        dest.resize(checked(ZSTD_compress2(ctx.get(), dest.data(), dest.size(), raw.data(), raw.size())));
        return dest;
    }
};
Bytes inflate(View src, U limit) {
    auto len = ZSTD_getFrameContentSize(src.data(), src.size());
    require(len <= limit, "invalid decompressed size");
    require(checked(ZSTD_findFrameCompressedSize(src.data(), src.size())) == src.size(), "invalid frame boundary");
    Bytes dest(len, '\0');
    require(checked(ZSTD_decompress(dest.data(), len, src.data(), src.size())) == len, "frame size mismatch");
    return dest;
}
// Detect a regular decimal ID list. Everything outside the list remains literal.
Bytes encode(View row) {
    auto raw = [&] { Bytes out(1, char(0)); out.append(row); return out; };
    auto key = row.find("\"hash_ids\"");
    if (key == View::npos) return raw();
    size_t p = key + 10;
    auto skip_space = [&] { while (p < row.size() && (row[p]==' ' || row[p]=='\t')) ++p; };
    skip_space();
    if (p == row.size() || row[p++] != ':') return raw();
    skip_space();
    if (p == row.size() || row[p++] != '[') return raw();
    const size_t head = p;
    Bytes runs;
    U start = 0, last = 0, length = 0;
    View delimiter;
    bool have_delimiter = false;
    auto flush = [&] { put(runs, start); put(runs, length); };
    for (;;) {
        if (p >= row.size() || row[p] < '0' || row[p] > '9') return raw();
        U value;
        const char *first = row.data() + p;
        auto parsed = std::from_chars(first, row.data()+row.size(), value);
        if (parsed.ec != std::errc() || (*first=='0' && parsed.ptr-first != 1)) return raw();
        p = size_t(parsed.ptr-row.data());
        if (length && last != std::numeric_limits<U>::max() && value == last + 1) ++length;
        else { if (length) flush(); start = value; length = 1; }
        last = value;
        if (p == row.size()) return raw();
        if (row[p] == ']') break;
        const size_t sep = p;
        if (row[p++] != ',') return raw();
        skip_space();
        auto current = row.substr(sep, p-sep);
        if (have_delimiter && delimiter != current) return raw();
        delimiter = current; have_delimiter = true;
    }
    flush();
    Bytes out(1, char(1));
    put(out, row.size()); put_field(out, row.substr(0,head)); put_field(out, delimiter);
    put_field(out, row.substr(p)); out += runs;
    return out.size() < row.size()+1 ? out : raw();
}
Bytes decode(View record) {
    auto kind = slice(record, 1)[0];
    if (kind == 0) { require(record.size() <= max_row, "oversize literal"); return Bytes(record); }
    require(kind == 1, "unknown row encoding");
    const U expected = take(record);
    require(expected <= max_row, "oversize row");
    auto head = slice(record, take(record));
    auto delimiter = slice(record, take(record));
    auto tail = slice(record, take(record));
    require(head.size()+tail.size() <= expected, "invalid row size");
    Bytes out(head); out.reserve(expected);
    bool first = true;
    while (!record.empty()) {
        U start = take(record), count = take(record);
        require(count && count-1 <= std::numeric_limits<U>::max()-start && count <= expected-out.size(), "invalid run");
        for (U j=0; j<count; ++j) {
            char digits[24];
            auto end = std::to_chars(digits, digits+sizeof digits, start+j).ptr;
            U needed = U(end-digits) + (first ? 0 : delimiter.size());
            require(needed <= expected-out.size(), "run exceeds row length");
            if (!first) out.append(delimiter);
            out.append(digits,end); first = false;
        }
    }
    require(out.size()+tail.size() == expected, "row length mismatch");
    out.append(tail); return out;
}

struct AtomicFile {
    std::string name, target;
    std::ofstream stream;
    explicit AtomicFile(const std::string &path) : name(path+".tmp.XXXXXX"), target(path) {
        int fd = mkstemp(name.data()); require(fd >= 0, "cannot create output file"); close(fd);
        stream.open(name, std::ios::binary);
        if (!stream) { std::remove(name.c_str()); throw std::runtime_error("cannot open output"); }
    }
    ~AtomicFile() { if (!name.empty()) std::remove(name.c_str()); }
    void commit() {
        stream.close(); require(bool(stream), "closing output failed");
        require(std::rename(name.c_str(),target.c_str()) == 0, "cannot publish output");
        name.clear();
    }
};
void pack(const char *input, const char *output) {
    std::ifstream in(input, std::ios::binary); require(bool(in), "cannot open input");
    AtomicFile file(output);
    write(file.stream, magic);
    Compressor z;
    Bytes directory, body, lengths, line;
    U rows = 0, batch = 0;
    auto flush = [&] {
        if (!batch) return;
        put64(directory, rows-batch); put64(directory, U(file.stream.tellp()));
        require(directory.size()+16 <= max_dir, "archive index exceeds format limit");
        Bytes header; put(header,batch); header += lengths; header += body;
        write(file.stream, z.frame(header));
        lengths.clear(); body.clear(); batch = 0;
    };
    // Read in bounded chunks so a missing line terminator cannot grow memory without bound.
    std::array<char, 65536> buffer{};
    auto add = [&] {
        Bytes encoded = encode(line);
        if (batch && body.size()+encoded.size() > target_frame) flush();
        put(lengths,encoded.size()); body += encoded; ++rows; ++batch;
        if (batch == rows_per_frame) flush();
        line.clear();
    };
    for (;;) {
        in.read(buffer.data(),buffer.size());
        const char *p = buffer.data(), *end = p+in.gcount();
        while (p != end) {
            const char *nl = static_cast<const char*>(std::memchr(p,'\n',end-p));
            const char *next = nl ? nl+1 : end;
            require(line.size()+size_t(next-p) <= max_row, "row exceeds 64 MiB");
            line.append(p, size_t(next-p)); p = next;
            if (nl) add();
        }
        if (in.eof()) break;
        require(bool(in), "input read failed");
    }
    if (!line.empty()) add();
    flush();
    const U directory_pos = U(file.stream.tellp());
    put64(directory,rows); put64(directory,directory_pos);
    write(file.stream,z.frame(directory));
    Bytes footer; put64(footer,directory_pos); footer += trailer; write(file.stream,footer);
    file.commit();
}

struct Entry { U row, pos; };
struct Reader {
    std::ifstream in;
    std::vector<Entry> index;
    Bytes buffer;
    std::vector<View> records;
    size_t current = std::numeric_limits<size_t>::max();
    explicit Reader(const char *path) : in(path,std::ios::binary) {
        require(bool(in), "cannot open archive"); in.seekg(0,std::ios::end);
        auto end = in.tellg(); require(end >= 24, "truncated archive");
        U size = U(end);
        require(read_at(in,0,8) == magic, "wrong archive magic");
        Bytes footer = read_at(in,size-16,16); View f(footer);
        U pos = take64(f); require(f == trailer && pos >= 8 && pos < size-16, "bad archive footer");
        require(size-16-pos <= ZSTD_compressBound(max_dir), "oversize directory");
        Bytes dir = inflate(read_at(in,pos,size-16-pos),max_dir);
        require(dir.size() >= 16 && dir.size()%16 == 0, "bad archive index");
        View d(dir);
        while (!d.empty()) {
            Entry e{take64(d),take64(d)};
            if (index.empty()) require(e.row == 0 && e.pos == 8, "invalid first entry");
            else require(e.row > index.back().row && e.row-index.back().row <= rows_per_frame &&
                         e.pos > index.back().pos && e.pos-index.back().pos <= ZSTD_compressBound(max_frame), "invalid index entry");
            require(e.pos <= pos, "offset outside data");
            index.push_back(e);
        }
        require(index.back().pos == pos, "index boundary mismatch");
    }
    void load(size_t i) {
        if (i == current) return;
        auto a=index.at(i), b=index.at(i+1);
        buffer=inflate(read_at(in,a.pos,b.pos-a.pos),max_frame);
        View v(buffer); U n=take(v); require(n == b.row-a.row, "row count mismatch");
        std::vector<U> lengths; lengths.reserve(n);
        for (U k=0;k<n;++k) lengths.push_back(take(v));
        records.clear();
        for (U len : lengths) records.push_back(slice(v,len));
        require(v.empty(), "trailing bytes in frame"); current=i;
    }
    size_t segment(U row) const {
        require(row < index.back().row, "row index out of range");
        auto it=std::upper_bound(index.begin(),index.end(),row,[](U r,const Entry &e){return r < e.row;});
        return size_t(it-index.begin()-1);
    }
    void scan(std::ostream *output) {
        for (size_t i=0; i+1<index.size(); ++i) {
            load(i);
            for (View record : records) {
                auto line=decode(record); if (output) write(*output,line);
            }
        }
    }
};
} // namespace
int main(int argc, char **argv) {
    try {
        require(argc >= 3, "usage: prefixpack pack INPUT ARCHIVE | unpack ARCHIVE OUTPUT | get ARCHIVE ROW... | verify ARCHIVE");
        View cmd(argv[1]);
        if (cmd == "pack" && argc == 4) pack(argv[2],argv[3]);
        else {
            Reader archive(argv[2]);
            if (cmd == "verify" && argc == 3) archive.scan(nullptr);
            else if (cmd == "unpack" && argc == 4) {
                AtomicFile file(argv[3]); archive.scan(&file.stream); file.commit();
            } else if (cmd == "get" && argc >= 4) {
                std::vector<U> ids;
                for (int i=3;i<argc;++i) {
                    View arg(argv[i]); U row;
                    auto r=std::from_chars(arg.data(),arg.data()+arg.size(),row);
                    require(r.ec == std::errc() && r.ptr == arg.data()+arg.size(), "invalid row index");
                    archive.segment(row); ids.push_back(row);
                }
                for (U row : ids) {
                    auto i=archive.segment(row); archive.load(i);
                    write(std::cout,decode(archive.records[row-archive.index[i].row]));
                }
                std::cout.flush(); require(bool(std::cout), "stdout write failed");
            } else throw std::runtime_error("invalid command or argument count");
        }
    } catch (const std::exception &e) {
        std::cerr << "prefixpack: " << e.what() << '\n'; return 1;
    }
}
