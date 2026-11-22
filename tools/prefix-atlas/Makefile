CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic

all: prefixpack curve

prefixpack: prefixpack.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) -lzstd -o $@

curve: curve.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) -o $@

test: all
	python3 -m unittest discover -p 'test_*.py'

clean:
	rm -f prefixpack curve

.PHONY: all test clean
