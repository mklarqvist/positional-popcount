###################################################################
# Copyright (c) 2019
# Author(s): Marcus D. R. Klarqvist, Wojciech Muła, and Daniel Lemire
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
###################################################################

OPTFLAGS  := -O3 -march=native
WARNFLAGS := # -Wall -Wextra -pedantic
CFLAGS     = -std=c99 $(OPTFLAGS) $(DEBUG_FLAGS) $(WARNFLAGS)
CPPFLAGS   = -std=c++0x $(OPTFLAGS) $(DEBUG_FLAGS) $(WARNFLAGS)
CPP_SOURCE = benchmark.cpp benchmark/linux/instrumented_benchmark.cpp
C_SOURCE   = pospopcnt.c example.c
OBJECTS    = $(CPP_SOURCE:.cpp=.o) $(C_SOURCE:.c=.o)

# Default target
all: bench

# Generic rules
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c -o $@ $<

benchmark/linux/instrumented_benchmark.o : benchmark/linux/instrumented_benchmark.cpp
	$(CXX) $(CPPFLAGS) -I. -Ibenchmark/linux -c -o $@ $<

benchmark/linux/instrumented_benchmark_align64.o : benchmark/linux/instrumented_benchmark.cpp
	$(CXX) $(CPPFLAGS) -DALIGN -I. -Ibenchmark/linux -c -o $@ $<

bench: pospopcnt.o benchmark.o
	$(CXX) $(CPPFLAGS) pospopcnt.o benchmark.o -o bench

itest: instrumented_benchmark
	$(CXX) --version
	./instrumented_benchmark

DEPS=benchmark/linux/instrumented_benchmark.cpp benchmark/linux/linux-perf-events.h pospopcnt.h pospopcnt.c pospopcnt.o benchmark/linux/popcnt.h

instrumented_benchmark: $(DEPS) benchmark/linux/instrumented_benchmark.o 
	$(CXX) $(CPPFLAGS) pospopcnt.o benchmark/linux/instrumented_benchmark.o -I. -Ibenchmark/linux -o $@

instrumented_benchmark_align64: $(DEPS) benchmark/linux/instrumented_benchmark_align64.o 
	$(CXX) $(CPPFLAGS) pospopcnt.o benchmark/linux/instrumented_benchmark_align64.o -I. -Ibenchmark/linux -o $@

example: pospopcnt.o example.o
	$(CC) $(CFLAGS) pospopcnt.o example.o -o example

test: bench
	$(CXX) --version
	./bench

clean:
	rm -f $(OBJECTS)
	rm -f bench example instrumented_benchmark

.PHONY: all clean test itest
