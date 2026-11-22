CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic

all: racecurve

racecurve: src/main.cpp src/racecurve.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/main.cpp -o $@

tests/check: tests/test.cpp tests/reference.hpp src/racecurve.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test.cpp -o $@

tests/benchmark: tests/bench.cpp tests/reference.hpp src/racecurve.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/bench.cpp -o $@

test: racecurve tests/check
	./tests/check
	python3 tests/test_cli.py
	python3 -m unittest -q test_bounds
	python3 -m unittest discover -s integrations -q

bench: tests/benchmark
	./tests/benchmark

clean:
	rm -f racecurve tests/check tests/benchmark benchmarks/stream benchmarks/dense

.PHONY: all test bench clean

benchmarks/stream: benchmarks/stream.cpp src/racecurve.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread benchmarks/stream.cpp -o $@

benchmarks/dense: benchmarks/dense.cpp tests/reference.hpp src/racecurve.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) benchmarks/dense.cpp -o $@
