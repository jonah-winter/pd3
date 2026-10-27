CC ?= cc
CXX ?= c++
CFLAGS = -Wall -Wextra -pedantic -O3 -march=native -I. -Iinclude -DNDEBUG
CXXFLAGS = -Wall -Wextra -pedantic -O3 -march=native -I. -Iinclude -I/opt/homebrew/include -DNDEBUG
LDFLAGS = -L/opt/homebrew/lib
LDLIBS = -lbenchmark -lbenchmark_main -lpthread

all: build/main bench

build/main: tests/main.c
	@mkdir -p build
	$(CC) $(CFLAGS) tests/main.c -o build/main

build/bench.o: tests/bench.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c tests/bench.cpp -o build/bench.o

build/bench: build/bench.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o build/bench

bench: build/bench
	./build/bench

clean:
	rm -rf build/

.PHONY: all clean bench
